"""One preprocessing implementation. Signals are physical mV, samples × leads."""

import numpy as np
from .config import LEADS, SAMPLE_RATE, SAMPLES, SPLIT


def validate_signal(raw, *, leads=None, sample_rate=100, units="mV", quality=True):
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            "Only 100 Hz signals are supported; resample explicitly before upload"
        )
    if units != "mV":
        raise ValueError("Signal units must be mV; convert explicitly before upload")
    if leads is not None and list(leads) != LEADS:
        raise ValueError("Lead order must be: " + ", ".join(LEADS))
    original = np.asarray(raw)
    if original.dtype.kind not in "fiu":
        raise ValueError("Signal must contain numeric samples, not strings or objects")
    x = np.asarray(raw, dtype=np.float32)
    if x.shape != (SAMPLES, len(LEADS)):
        raise ValueError(
            f"Expected (1000, 12) samples × leads, got {x.shape}; duration must be 10 seconds"
        )
    if not np.isfinite(x).all():
        raise ValueError("Signal contains NaN or infinity")
    if quality:
        flat = np.std(x, axis=0, dtype=np.float64) < 1e-5
        if flat.any():
            raise ValueError(
                "Flatlined or near-zero variance leads: "
                + ", ".join(np.array(LEADS)[flat])
            )
        if np.max(np.abs(x)) > 100:
            raise ValueError(
                "Values exceed the supported ±100 mV range; check units and acquisition"
            )
    return np.ascontiguousarray(x)


def fit_normalization(signals, folds):
    if not len(folds) or not set(folds).issubset(SPLIT["train"]):
        raise ValueError("Normalization may use training folds 1–8 only")
    total = np.zeros(12, dtype=np.float64)
    squares = total.copy()
    n = records = 0
    for raw in signals:
        x = validate_signal(raw, quality=False).astype(np.float64)
        total += x.sum(axis=0)
        squares += (x * x).sum(axis=0)
        n += len(x)
        records += 1
    if records != len(folds):
        raise ValueError("Signal/fold count mismatch")
    mean = total / n
    std = np.sqrt(np.maximum(squares / n - mean**2, 0))
    if (std < 1e-8).any():
        raise ValueError("Degenerate training normalization statistics")
    return {
        "method": "train_per_lead_zscore",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fit_folds": SPLIT["train"],
        "record_count": records,
    }


def preprocess(raw, normalization):
    x = validate_signal(raw, quality=False)
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    return np.ascontiguousarray(((x - mean) / std).T)
