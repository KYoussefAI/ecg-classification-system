"""Fail-closed lifecycle. Synthetic fixtures are never served."""

import json
import logging
import os
from pathlib import Path
import torch
from training.artifacts import load_artifact, sha256
from training.config import CLASSES, DESCRIPTIONS, DISCLAIMER, LEADS
from training.preprocessing import preprocess, validate_signal
from training.runtime import choose_device


class ModelUnavailable(RuntimeError):
    pass


class ModelService:
    _instance = None

    def __init__(self, directory=None):
        self.model, self.metadata, self.curves = None, None, None
        self.state, self.device = "unavailable", torch.device("cpu")
        self.directory = Path(
            directory or os.getenv("MODEL_DIR", "artifacts/ptbxl-resnet1d-v1")
        )
        try:
            torch.set_num_threads(int(os.getenv("MODEL_THREADS", "4")))
            self.device = choose_device(os.getenv("MODEL_DEVICE", "cpu"))
            model, metadata = load_artifact(self.directory, self.device)
            if (
                metadata.get("synthetic") is not False
                or metadata["dataset"]["name"] != "PTB-XL"
            ):
                raise ValueError("Synthetic/fixture artifacts cannot be served")
            report = metadata.get("test_metrics")
            if report is not None:
                if (
                    report.get("fold") != 10
                    or report.get("synthetic") is not False
                    or report.get("checkpoint_sha256") != metadata["checkpoint_sha256"]
                    or report.get("model_version") != metadata["model_version"]
                    or sha256(self.directory / "metrics.json")
                    != metadata.get("metrics_sha256")
                ):
                    raise ValueError("Evaluation provenance mismatch")
                if json.loads((self.directory / "metrics.json").read_text()) != report:
                    raise ValueError("Evaluation metadata mismatch")
                if sha256(self.directory / "curves.json") != metadata.get(
                    "curves_sha256"
                ):
                    raise ValueError("Evaluation curves mismatch")
                self.curves = json.loads((self.directory / "curves.json").read_text())
            self.model, self.metadata, self.state = model, metadata, "ready"
        except FileNotFoundError:
            logging.getLogger(__name__).warning(
                "Model artifact is missing; predictions disabled"
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "Model unavailable; predictions disabled"
            )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def health(self):
        m = self.metadata or {}
        return {
            "status": "healthy" if self.state == "ready" else "degraded",
            "model_state": self.state,
            "model_loaded": self.state == "ready",
            "model_version": m.get("model_version"),
            "architecture": m.get("architecture", {}).get("name"),
            "device": str(self.device),
            "sample_rate": m.get("input", {}).get("sample_rate"),
            "classes": len(m.get("classes", [])),
            "demo_mode": False,
        }

    def public_metadata(self):
        m = self.metadata or {}
        return {
            **self.health(),
            "classes": CLASSES,
            "descriptions": DESCRIPTIONS,
            "input": m.get(
                "input",
                {
                    "leads": LEADS,
                    "samples": 1000,
                    "sample_rate": 100,
                    "duration_seconds": 10,
                    "units": "mV",
                },
            ),
            "thresholds": m.get("thresholds"),
            "dataset": m.get("dataset"),
            "split": m.get("split"),
            "architecture_config": m.get("architecture"),
            "best_epoch": m.get("best_epoch"),
            "test_metrics": m.get("test_metrics"),
            "curves": self.curves,
            "evaluation_status": "Evaluated"
            if m.get("test_metrics")
            else "Awaiting final evaluation",
            "disclaimer": DISCLAIMER,
        }

    def predict(self, signal, *, leads=None, sample_rate=100, units="mV"):
        if self.state != "ready":
            raise ModelUnavailable(
                "No valid trained model is loaded. Prediction is unavailable."
            )
        x = validate_signal(signal, leads=leads, sample_rate=sample_rate, units=units)
        tensor = (
            torch.from_numpy(preprocess(x, self.metadata["normalization"]))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            logits = self.model(tensor)
            if not torch.isfinite(logits).all():
                self.state = "failed"
                raise ModelUnavailable(
                    "Model inference failed; predictions have been disabled"
                )
            scores = logits.sigmoid().cpu().numpy()[0]
        predictions = [
            {
                "class": c,
                "description": DESCRIPTIONS[c],
                "score": float(scores[i]),
                "threshold": self.metadata["thresholds"][c],
                "positive": bool(scores[i] >= self.metadata["thresholds"][c]),
            }
            for i, c in enumerate(CLASSES)
        ]
        positive = [p["class"] for p in predictions if p["positive"]]
        warnings = [
            "Scores are uncalibrated classifier outputs, not clinical disease probabilities."
        ]
        if leads is None:
            warnings.append(
                "Lead metadata absent; the caller asserts the documented standard lead order."
            )
        if "NORM" in positive and len(positive) > 1:
            warnings.append(
                "NORM and abnormal classes both crossed thresholds. Independent multilabel outputs can conflict; no output was overridden."
            )
        return {
            "model_version": self.metadata["model_version"],
            "predictions": predictions,
            "positive_classes": positive,
            "signal_quality": {
                "status": "passed",
                "checks": [
                    "12 leads",
                    "10 s",
                    "100 Hz",
                    "finite samples",
                    "no flatline leads",
                    "amplitude bounds",
                ],
            },
            "preprocessing": {
                "method": self.metadata["normalization"]["method"],
                "fit_folds": self.metadata["normalization"]["fit_folds"],
                **self.metadata["input"],
            },
            "warnings": warnings,
            "disclaimer": DISCLAIMER,
        }
