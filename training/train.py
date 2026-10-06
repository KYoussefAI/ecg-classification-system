"""Canonical training command. Never loads held-out test waveforms."""

import argparse
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .artifacts import sha256, write_json
from .config import ARCHITECTURE, CLASSES, DEFAULT_MODEL, LEADS, SPLIT
from .data import ECGDataset, load_metadata, load_partition
from .metrics import metrics, select_thresholds
from .model import ECGNet
from .plots import training_plots
from .preprocessing import fit_normalization
from .runtime import choose_device, predict_loader, seed_everything, seed_worker


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--dataset-version", default="1.0.3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--scheduler", choices=["plateau", "none"], default="plateau")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume last_checkpoint.pt in the same run directory",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Small model; permanently marks artifact synthetic and unservable",
    )
    return p


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(args):
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.workers < 0
        or args.patience < 1
        or args.threads < 1
    ):
        raise ValueError(
            "Epochs, batch size, patience and threads must be positive; workers nonnegative"
        )
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative")
    run = args.run_dir
    if (run / "model_metadata.json").exists() or (
        run / "evaluation_started.json"
    ).exists():
        raise ValueError(
            "Frozen runs are immutable. Start a new run; never resume after finalization/evaluation"
        )
    if run.exists() and any(run.iterdir()) and not args.resume:
        raise ValueError(
            "Run directory is not empty; use --resume for an interrupted run or choose a new directory"
        )
    run.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    parts, audit = load_metadata(args.data_dir)
    seen = {}
    x_train, y_train = load_partition(args.data_dir, parts["train"], seen)
    x_val, y_val = load_partition(args.data_dir, parts["validation"], seen)
    normalization = fit_normalization(x_train, parts["train"].strat_fold.tolist())
    # Hash raw train/validation signals to bind resume and later duplicate audit.
    fingerprint = {
        "metadata_sha256": audit["metadata_sha256"],
        "development_waveforms_sha256": __import__("hashlib")
        .sha256("".join(sorted(seen)).encode())
        .hexdigest(),
    }
    counts = y_train.sum(0)
    if ((counts == 0) | (counts == len(y_train))).any():
        raise ValueError("Every training class must contain positives and negatives")
    weights = (len(y_train) - counts) / counts
    print(
        json.dumps(
            {
                "device": str(device),
                "train_counts": dict(zip(CLASSES, counts.tolist())),
                "train_pos_weight": dict(zip(CLASSES, weights.tolist())),
            }
        ),
        flush=True,
    )
    architecture = {
        "name": ARCHITECTURE,
        "config": {**DEFAULT_MODEL, "dropout": args.dropout},
    }
    if args.smoke:
        architecture["config"] = {
            "channels": [4, 8, 16, 32],
            "blocks_per_stage": 1,
            "dropout": args.dropout,
        }
    model = ECGNet(**architecture["config"]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weights, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )
        if args.scheduler == "plateau"
        else None
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and not args.no_amp
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        ECGDataset(x_train, y_train, normalization),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
        worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        ECGDataset(x_val, y_val, normalization),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("resume", "epochs", "run_dir", "data_dir")
    }
    history, best, best_epoch, bad_epochs, start = [], -float("inf"), 0, 0, 1
    training_date = datetime.now(timezone.utc).isoformat()
    if args.resume:
        last = torch.load(
            run / "last_checkpoint.pt", map_location="cpu", weights_only=True
        )
        if (
            last["config"] != config
            or last["data_fingerprint"] != fingerprint
            or last["normalization"] != normalization
        ):
            raise ValueError(
                "Resume configuration or data changed; use a new run directory"
            )
        model.load_state_dict(last["model_state"])
        optimizer.load_state_dict(last["optimizer"])
        if scheduler:
            scheduler.load_state_dict(last["scheduler"])
        scaler.load_state_dict(last["scaler"])
        history, best, best_epoch, bad_epochs = (
            last["history"],
            last["best"],
            last["best_epoch"],
            last["bad_epochs"],
        )
        start, training_date = last["epoch"] + 1, last["training_date"]
        random.setstate(last["python_rng"])
        np.random.set_state(
            (
                last["numpy_rng"][0],
                np.array(last["numpy_rng"][1], dtype=np.uint32),
                *last["numpy_rng"][2:],
            )
        )
        torch.set_rng_state(last["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(last["cuda_rng"])
        generator.set_state(last["loader_rng"])
    write_json(run / "split_audit.json", audit)
    write_json(run / "development_waveform_hashes.json", seen)
    write_json(
        run / "training_config.json",
        {
            **config,
            "epochs": args.epochs,
            "normalization": normalization,
            "pos_weight": weights.tolist(),
            "data_fingerprint": fingerprint,
        },
    )
    # A resumed job already at its stopping condition proceeds to finalization.
    for epoch in range(start, args.epochs + 1):
        if bad_epochs >= args.patience:
            break
        began = time.perf_counter()
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(x)
        targets, scores, val_loss = predict_loader(model, val_loader, device, criterion)
        val_metrics = metrics(targets, scores, dict.fromkeys(CLASSES, 0.5))
        selection = val_metrics["macro_auroc"]
        if selection is None:
            raise ValueError("Validation macro AUROC undefined; check class support")
        lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step(selection)
        if selection > best:
            best, best_epoch, bad_epochs = selection, epoch, 0
            atomic_save(
                {
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "architecture": architecture,
                    "model_version": run.name,
                },
                run / "best_model.pt",
            )
        else:
            bad_epochs += 1
        entry = {
            "epoch": epoch,
            "train_loss": train_loss / len(x_train),
            "validation_loss": val_loss,
            "validation_macro_auroc": selection,
            "validation_macro_auprc": val_metrics["macro_auprc"],
            "learning_rate": lr,
            "duration_seconds": time.perf_counter() - began,
            "best_epoch": best_epoch,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        np_state = np.random.get_state()
        atomic_save(
            {
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best": best,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "config": config,
                "data_fingerprint": fingerprint,
                "normalization": normalization,
                "history": history,
                "training_date": training_date,
                "python_rng": random.getstate(),
                "numpy_rng": [np_state[0], np_state[1].tolist(), *np_state[2:]],
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else [],
                "loader_rng": generator.get_state(),
            },
            run / "last_checkpoint.pt",
        )
        write_json(run / "training_history.json", history)
    if not best_epoch:
        raise ValueError("No valid checkpoint selected")
    model.load_state_dict(
        torch.load(run / "best_model.pt", map_location=device, weights_only=True)[
            "model_state"
        ]
    )
    targets, scores, _ = predict_loader(model, val_loader, device)
    thresholds = select_thresholds(targets, scores, fold=9)
    np.savez_compressed(
        run / "validation_predictions.npz",
        targets=targets,
        scores=scores,
        ids=parts["validation"].ecg_id.to_numpy(),
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    write_json(
        run / "thresholds.json",
        {
            "classes": CLASSES,
            "fold": 9,
            "objective": "per_class_f1",
            "thresholds": thresholds,
        },
    )
    training_plots(
        run / "plots",
        history,
        {"train": counts.tolist(), "validation": y_val.sum(0).tolist()},
    )
    metadata = {
        "schema_version": 1,
        "model_version": run.name,
        "architecture": architecture,
        "classes": CLASSES,
        "thresholds": thresholds,
        "threshold_selection": {"fold": 9, "objective": "per_class_f1"},
        "input": {
            "leads": LEADS,
            "sample_rate": 100,
            "samples": 1000,
            "duration_seconds": 10,
            "units": "mV",
        },
        "normalization": normalization,
        "calibration": {"method": "none", "output": "uncalibrated_model_score"},
        "dataset": {
            "name": "synthetic_fixture" if args.smoke else "PTB-XL",
            "version": args.dataset_version,
        },
        "split": SPLIT,
        "seed": args.seed,
        "training_date": training_date,
        "best_epoch": best_epoch,
        "git_commit": commit,
        "git_dirty": dirty,
        "checkpoint_sha256": sha256(run / "best_model.pt"),
        "data_fingerprint": fingerprint,
        "training_config": config,
        "frozen": True,
        "synthetic": args.smoke,
        "test_metrics": None,
        "software": {"torch": str(torch.__version__), "numpy": np.__version__},
    }
    # Final metadata is the commit marker: incomplete runs cannot be served.
    write_json(run / "model_metadata.json", metadata)
    print(f"Frozen artifact: {run}. Final test evaluation has NOT run.", flush=True)
    return metadata


if __name__ == "__main__":
    train(parser().parse_args())
