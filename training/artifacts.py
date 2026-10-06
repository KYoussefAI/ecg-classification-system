"""Strict, hash-bound artifact contract. Never unpickle a model object."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .config import ARCHITECTURE, CLASSES, LEADS, SPLIT
from .model import ECGNet


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_metadata(m):
    if m.get("schema_version") != 1 or m.get("classes") != CLASSES:
        raise ValueError("Incompatible schema or class order")
    if m.get("architecture", {}).get("name") != ARCHITECTURE:
        raise ValueError("Unsupported architecture")
    if set(m["architecture"].get("config", {})) != {
        "channels",
        "blocks_per_stage",
        "dropout",
    }:
        raise ValueError("Incomplete architecture configuration")
    if m.get("input") != {
        "leads": LEADS,
        "sample_rate": 100,
        "samples": 1000,
        "duration_seconds": 10,
        "units": "mV",
    }:
        raise ValueError("Incompatible input contract")
    if m.get("split") != SPLIT or m.get("threshold_selection") != {
        "fold": 9,
        "objective": "per_class_f1",
    }:
        raise ValueError("Invalid split or threshold provenance")
    thresholds = m.get("thresholds", {})
    if set(thresholds) != set(CLASSES) or not all(
        np.isfinite(thresholds[c]) and 0 <= thresholds[c] <= 1 for c in CLASSES
    ):
        raise ValueError("Invalid class thresholds")
    norm = m.get("normalization", {})
    if (
        norm.get("method") != "train_per_lead_zscore"
        or norm.get("fit_folds") != SPLIT["train"]
        or norm.get("record_count", 0) < 1
    ):
        raise ValueError("Invalid normalization provenance")
    for key in ("mean", "std"):
        values = np.asarray(norm.get(key, []))
        if (
            values.shape != (12,)
            or not np.isfinite(values).all()
            or (key == "std" and (values < 1e-8).any())
        ):
            raise ValueError("Invalid per-lead normalization parameters")
    if m.get("calibration") != {"method": "none", "output": "uncalibrated_model_score"}:
        raise ValueError("Unsupported calibration")
    for key in (
        "model_version",
        "training_date",
        "dataset",
        "seed",
        "best_epoch",
        "git_commit",
        "checkpoint_sha256",
        "data_fingerprint",
        "training_config",
    ):
        if key not in m:
            raise ValueError(f"Missing metadata: {key}")
    if not m["model_version"] or m["best_epoch"] < 1 or not m.get("frozen"):
        raise ValueError("Artifact has not been frozen")


def load_artifact(directory, device="cpu"):
    directory = Path(directory)
    metadata = json.loads(
        (directory / "model_metadata.json").read_text(encoding="utf-8")
    )
    validate_metadata(metadata)
    path = directory / "best_model.pt"
    if sha256(path) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum does not match metadata")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("classes") != CLASSES
        or checkpoint.get("architecture") != metadata["architecture"]
        or checkpoint.get("model_version") != metadata["model_version"]
    ):
        raise ValueError("Checkpoint and metadata are incompatible")
    model = ECGNet(**metadata["architecture"]["config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite weights")
    model.to(device).eval()
    with torch.inference_mode():
        if not torch.isfinite(model(torch.zeros(1, 12, 1000, device=device))).all():
            raise ValueError("Model warmup returned nonfinite logits")
    return model, metadata
