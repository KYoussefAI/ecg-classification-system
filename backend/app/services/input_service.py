"""Bounded adapters producing samples × leads, physical mV."""

import csv
import io
import json
from pathlib import Path
import numpy as np
from training.config import LEADS
from training.preprocessing import validate_signal

MAX_UPLOAD_BYTES = 2 * 1024 * 1024


def parse_file(content, filename, sample_rate=100, units="mV"):
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload must be nonempty and no larger than 2 MiB")
    suffix = Path(filename or "").suffix.lower()
    leads, warnings, synthetic = None, [], False
    if suffix == ".json":
        payload = json.loads(content)
        if isinstance(payload, dict):
            raw = payload["signal_data"]
            leads = payload.get("leads")
            sample_rate = payload.get("sample_rate", sample_rate)
            units = payload.get("units", units)
            synthetic = payload.get("synthetic") is True
        else:
            raw = payload
    elif suffix == ".npy":
        stream = io.BytesIO(content)
        version = np.lib.format.read_magic(stream)
        if version not in ((1, 0), (2, 0)):
            raise ValueError("Unsupported NPY version")
        header = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, _, dtype = header(stream)
        if shape != (1000, 12) or dtype.kind not in "fiu" or dtype.itemsize > 8:
            raise ValueError("NPY must be a numeric (1000, 12) array")
        raw = np.load(io.BytesIO(content), allow_pickle=False)
    elif suffix == ".csv":
        rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
        if not rows:
            raise ValueError("Empty CSV")
        if [item.strip() for item in rows[0]] == LEADS:
            leads, rows = LEADS, rows[1:]
        else:
            try:
                [float(item) for item in rows[0]]
            except ValueError as exc:
                raise ValueError(
                    "CSV header must be exactly: " + ",".join(LEADS)
                ) from exc
        raw = np.asarray(rows, dtype=np.float32)
    else:
        raise ValueError("Supported uploads: .json, .csv and .npy")
    signal = validate_signal(raw, leads=leads, sample_rate=sample_rate, units=units)
    if leads is None:
        warnings.append(
            "No lead names in file: standard lead order is declared by the uploader."
        )
    if synthetic:
        warnings.append("Synthetic demo signal — illustrative, not patient data.")
    return {
        "signal_data": signal.tolist(),
        "leads": leads,
        "sample_rate": sample_rate,
        "units": units,
        "warnings": warnings,
        "synthetic": synthetic,
        "signal_quality": {"status": "passed"},
    }
