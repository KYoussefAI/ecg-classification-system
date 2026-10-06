"""Deterministic PTB-XL superclass labels and audited official partitions."""

import ast
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import wfdb
from .artifacts import sha256
from .config import CLASSES, LEADS, SPLIT
from .preprocessing import preprocess, validate_signal


def validate_partitions(frame):
    required = ["ecg_id", "patient_id", "strat_fold", "filename_lr"]
    if frame[required].isna().any().any():
        raise ValueError("Missing record, patient, fold or waveform identifier")
    if frame.ecg_id.duplicated().any() or frame.filename_lr.duplicated().any():
        raise ValueError("Duplicate record ID or waveform path")
    if not set(frame.strat_fold).issubset(range(1, 11)):
        raise ValueError("Invalid official fold membership")
    if (frame.groupby("patient_id").strat_fold.nunique() > 1).any():
        raise ValueError("Patient leakage across folds")
    partitions = {
        name: frame[frame.strat_fold.isin(folds)].copy()
        for name, folds in SPLIT.items()
    }
    if any(part.empty for part in partitions.values()):
        raise ValueError("All three official partitions must be nonempty")
    sets = [set(part.ecg_id) for part in partitions.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Overlapping record IDs")
    return partitions


def load_metadata(root):
    root = Path(root)
    frame = pd.read_csv(root / "ptbxl_database.csv")
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    validate_partitions(frame)
    diagnostic = statements[statements.diagnostic == 1].diagnostic_class.to_dict()

    def label(raw):
        codes = ast.literal_eval(raw)
        if not isinstance(codes, dict):
            raise ValueError("scp_codes must be a dictionary")
        # Code presence is used, including likelihood=0 (unknown likelihood).
        active = {diagnostic[code] for code in codes if code in diagnostic}
        return [int(cls in active) for cls in CLASSES]

    frame["labels"] = frame.scp_codes.map(label)
    # No mapped diagnostic label is unknown, not an all-negative target.
    excluded = frame[frame.labels.map(sum) == 0].ecg_id.tolist()
    frame = (
        frame[frame.labels.map(sum) > 0].sort_values("ecg_id").reset_index(drop=True)
    )
    parts = validate_partitions(frame)
    fingerprint = hashlib.sha256(
        (
            sha256(root / "ptbxl_database.csv") + sha256(root / "scp_statements.csv")
        ).encode()
    ).hexdigest()
    return parts, {
        "metadata_sha256": fingerprint,
        "excluded_unmapped_ids": excluded,
        "partitions": {
            k: {
                "records": len(v),
                "patients": int(v.patient_id.nunique()),
                "ids": v.ecg_id.tolist(),
                "class_counts": np.array(v.labels.tolist()).sum(0).tolist(),
            }
            for k, v in parts.items()
        },
    }


def read_signal(root, filename):
    root = Path(root).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Waveform path escapes dataset root")
    signal, fields = wfdb.rdsamp(str(path))
    # PTB-XL uses lowercase lead names in WFDB; case normalization is explicit.
    names = fields["sig_name"]
    if [name.casefold() for name in names] != [name.casefold() for name in LEADS]:
        raise ValueError(f"Unexpected WFDB lead order: {names}")
    if any(unit != "mV" for unit in fields["units"]):
        raise ValueError("WFDB signals must be in mV")
    return validate_signal(signal, sample_rate=fields["fs"], quality=False)


def load_partition(root, frame, seen=None):
    signals = np.empty((len(frame), 1000, 12), dtype=np.float32)
    seen = {} if seen is None else seen
    for i, row in enumerate(frame.itertuples()):
        signal = read_signal(root, row.filename_lr)
        digest = hashlib.sha256(signal.tobytes()).hexdigest()
        if digest in seen:
            raise ValueError(
                f"Duplicate waveform detected: {seen[digest]} and {row.ecg_id}"
            )
        seen[digest] = int(row.ecg_id)
        signals[i] = signal
    return signals, np.array(frame.labels.tolist(), dtype=np.float32)


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, signals, targets, normalization):
        self.signals, self.targets, self.normalization = signals, targets, normalization

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        return torch.from_numpy(
            preprocess(self.signals[index], self.normalization)
        ), torch.from_numpy(self.targets[index])
