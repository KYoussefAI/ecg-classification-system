"""One final held-out evaluation, with no fitting or threshold search."""

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .artifacts import load_artifact, sha256, write_json
from .data import ECGDataset, load_metadata, load_partition
from .metrics import bootstrap_auroc, metrics
from .plots import evaluation_plots
from .runtime import choose_device, predict_loader, seed_everything


def finalize_metadata(run, metadata, report):
    metadata["test_metrics"] = report
    metadata["metrics_sha256"] = sha256(run / "metrics.json")
    metadata["curves_sha256"] = sha256(run / "curves.json")
    write_json(run / "model_metadata.json", metadata)


def evaluate(args):
    run = args.run_dir
    seed_everything(42)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    model, metadata = load_artifact(run, device)
    if metadata.get("test_metrics"):
        raise ValueError(
            "This frozen run already has final evaluation results; do not use test results for tuning"
        )
    if (run / "metrics.json").exists():
        if not args.retry_failed:
            raise ValueError(
                "Results already exist; --retry-failed can repair interrupted metadata finalization"
            )
        report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if (
            report.get("checkpoint_sha256") != metadata["checkpoint_sha256"]
            or report.get("model_version") != metadata["model_version"]
            or report.get("fold") != 10
            or report.get("thresholds") != metadata["thresholds"]
            or report.get("synthetic") != metadata["synthetic"]
        ):
            raise ValueError(
                "Existing evaluation results do not match the frozen artifact"
            )
        finalize_metadata(run, metadata, report)
        print(
            "Completed interrupted metadata finalization; test predictions were not recomputed.",
            flush=True,
        )
        return report
    parts, audit = load_metadata(args.data_dir)
    if audit["metadata_sha256"] != metadata["data_fingerprint"]["metadata_sha256"]:
        raise ValueError("Dataset metadata changed since training")
    if args.bootstrap < 0 or args.batch_size < 1:
        raise ValueError("Invalid bootstrap count or batch size")
    # Exclusive marker prevents concurrent evaluations. Failed evaluations may be
    # retried explicitly, preserving the original marker for the audit trail.
    marker = run / "evaluation_started.json"
    if marker.exists():
        if not args.retry_failed:
            raise ValueError(
                "Evaluation previously started; inspect failure then use --retry-failed"
            )
    else:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                },
                stream,
            )
    seen = json.loads((run / "development_waveform_hashes.json").read_text())
    if (
        hashlib.sha256("".join(sorted(seen)).encode()).hexdigest()
        != metadata["data_fingerprint"]["development_waveforms_sha256"]
    ):
        raise ValueError("Development waveform audit changed since training")
    x, y = load_partition(args.data_dir, parts["test"], seen)
    loader = DataLoader(
        ECGDataset(x, y, metadata["normalization"]), batch_size=args.batch_size
    )
    targets, scores, _ = predict_loader(model, loader, device)
    report = metrics(targets, scores, metadata["thresholds"])
    report.update(
        {
            "partition": "test",
            "fold": 10,
            "model_version": metadata["model_version"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "synthetic": metadata["synthetic"],
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if args.bootstrap:
        report["auroc_ci95"] = bootstrap_auroc(
            targets,
            scores,
            parts["test"].patient_id.to_numpy(),
            args.bootstrap,
            metadata["seed"],
        )
    curves = evaluation_plots(run / "plots", targets, scores, report)
    write_json(run / "curves.json", curves)
    np.savez_compressed(
        run / "test_predictions.npz",
        targets=targets,
        scores=scores,
        ids=parts["test"].ecg_id.to_numpy(),
    )
    write_json(run / "metrics.json", report)
    finalize_metadata(run, metadata, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--retry-failed", action="store_true")
    evaluate(p.parse_args())
