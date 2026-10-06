"""Tiny synthetic training + final-evaluation smoke test; never a serving model."""

import argparse
from pathlib import Path
from argparse import Namespace
import numpy as np
import pandas as pd
import wfdb
from training.config import CLASSES, LEADS
from training.train import parser, train
from training.evaluate import evaluate


def create_fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng, rows = np.random.default_rng(421), []
    for fold in range(1, 11):
        for j in range(8):
            index = (fold - 1) * 8 + j + 1
            codes = {c: 100 for i, c in enumerate(CLASSES) if (i + j) % 3 == 0}
            t = np.arange(1000) / 100
            signal = np.column_stack([0.4 * np.sin(t * (4 + i / 3)) for i in range(12)])
            signal += rng.normal(0, 0.04, signal.shape)
            name = f"record_{index:04d}"
            wfdb.wrsamp(
                name,
                fs=100,
                units=["mV"] * 12,
                sig_name=LEADS,
                p_signal=signal,
                fmt=["16"] * 12,
                write_dir=str(root),
            )
            rows.append(
                {
                    "ecg_id": index,
                    "patient_id": index,
                    "strat_fold": fold,
                    "filename_lr": name,
                    "scp_codes": str(codes),
                }
            )
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame(
        {"diagnostic": [1] * 5, "diagnostic_class": CLASSES}, index=CLASSES
    ).to_csv(root / "scp_statements.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    args = p.parse_args()
    if args.output.exists():
        raise ValueError(
            "Choose a new output directory; smoke artifacts are never overwritten"
        )
    data = args.output / "data"
    create_fixture(data)
    run = args.output / "synthetic-run"
    train(
        parser().parse_args(
            [
                "--data-dir",
                str(data),
                "--run-dir",
                str(run),
                "--epochs",
                "2",
                "--batch-size",
                "16",
                "--smoke",
                "--device",
                "cpu",
            ]
        )
    )
    evaluate(
        Namespace(
            data_dir=data,
            run_dir=run,
            device="cpu",
            batch_size=16,
            bootstrap=20,
            retry_failed=False,
        )
    )
    print(
        "Synthetic pipeline checks complete. These metrics are not PTB-XL results; the API rejects this artifact."
    )


if __name__ == "__main__":
    main()
