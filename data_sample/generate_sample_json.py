"""
Build a deterministic (1000, 12) pseudo–12-lead signal at 100 Hz for API testing.

This is not a real patient recording, but it has clear P/QRS/T structure and
per-lead variation closer to a rhythm strip than pure noise. PTB-XL–trained
models often still give diffuse probabilities on synthetic inputs; use real
exported PTB-XL tensors for meaningful validation.

Output: sample_ecg_synthetic.json next to this script (raw array OR wrapped).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

N = 1000
N_LEADS = 12
FS = 100.0
HR_BPM = 72.0
RNG = np.random.default_rng(42)


def _gaussian(t: np.ndarray, center: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def build_lead(t: np.ndarray, lead_idx: int) -> np.ndarray:
    """One lead: repeating beat pattern + mild baseline wander + tiny noise."""
    period = 60.0 / HR_BPM * FS  # samples per beat
    y = np.zeros_like(t, dtype=np.float64)

    # Precordial leads (V1–V6 approximated as leads 6–11): smaller R, deeper S
    is_prec = lead_idx >= 6
    r_scale = 0.55 if is_prec else 1.0
    s_scale = 0.85 if is_prec else 0.35
    axis_shift = (lead_idx - 5.5) * 0.04

    n_beats = int(math.ceil(N / period)) + 2
    for b in range(-1, n_beats):
        c = b * period + 0.45 * period  # QRS ~ mid-beat
        y += _gaussian(t, c - 0.22 * period, 0.06 * period, 0.12)  # P
        y += _gaussian(t, c - 0.05 * period, 0.018 * period, -0.08 * s_scale)  # Q
        y += _gaussian(t, c, 0.022 * period, 0.95 * r_scale)  # R
        y += _gaussian(t, c + 0.035 * period, 0.025 * period, -0.25 * s_scale)  # S
        y += _gaussian(t, c + 0.28 * period, 0.07 * period, 0.22)  # T

    # Slow baseline + lead axis
    y += 0.08 * np.sin(2 * math.pi * 0.35 * t / N + lead_idx * 0.2)
    y += axis_shift * np.sin(2 * math.pi * t / N)
    y += RNG.normal(0, 0.015, size=N)
    return y


def main() -> None:
    t = np.arange(N, dtype=np.float64)
    leads = np.stack([build_lead(t, li) for li in range(N_LEADS)], axis=1)
    assert leads.shape == (N, N_LEADS)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    # Wrapped format: upload-friendly if the UI unwraps signal_data
    wrapped_path = os.path.join(out_dir, "sample_ecg_synthetic.json")
    raw_path = os.path.join(out_dir, "sample_ecg_synthetic_array.json")

    wrapped = {
        "signal_data": leads.astype(np.float32).tolist(),
        "patient_name": "synthetic_demo",
        "age": 55,
        "sex": "M",
    }
    with open(wrapped_path, "w", encoding="utf-8") as f:
        json.dump(wrapped, f)

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(leads.astype(np.float32).tolist(), f)

    print(f"Wrote {wrapped_path}")
    print(f"Wrote {raw_path}")


if __name__ == "__main__":
    main()
