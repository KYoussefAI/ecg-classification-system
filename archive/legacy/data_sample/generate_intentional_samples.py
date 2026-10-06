"""
Four intentional (1000, 12) pseudo–ECG traces at 100 Hz for API testing.

Lead order: I, II, III, aVR, aVL, aVF, V1–V6 (indices 0–11).

These are *stylized* patterns (not clinical recordings). They are meant to
exercise the pipeline with visibly different morphologies — not to guarantee
specific classifier outputs.

Writes: sample_intentional_1.json … sample_intentional_4.json
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

N = 1000
N_LEADS = 12
FS = 100.0

# I, II, III, aVR, aVL, aVF, V1..V6
LEAD_I, LEAD_II, LEAD_III = 0, 1, 2
LEAD_AVR, LEAD_AVL, LEAD_AVF = 3, 4, 5
LEAD_V1, LEAD_V2, LEAD_V3, LEAD_V6 = 6, 7, 8, 11


def _gaussian(t: np.ndarray, center: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def _beat_train(
    t: np.ndarray,
    period: float,
    rng: np.random.Generator,
    *,
    qrs_sigma_scale: float = 1.0,
    r_height: float = 0.95,
    st_shift: np.ndarray | None = None,
    t_amp_scale: float = 1.0,
    t_invert_leads: frozenset[int] | frozenset = frozenset(),
    noise_std: float = 0.012,
) -> np.ndarray:
    """
    Build (N, 12) signal: one row per time sample, one col per lead.
    st_shift: shape (12,) baseline added after J-point (piecewise per beat); simplified as smooth bumps.
    """
    y = np.zeros((N, N_LEADS), dtype=np.float64)
    if st_shift is None:
        st_shift = np.zeros(N_LEADS, dtype=np.float64)

    n_beats = int(math.ceil(N / period)) + 2
    for lead_idx in range(N_LEADS):
        col = np.zeros(N, dtype=np.float64)
        is_prec = lead_idx >= LEAD_V1
        r_scale = 0.55 if is_prec else 1.0
        s_scale = 0.85 if is_prec else 0.35
        axis_shift = (lead_idx - 5.5) * 0.04

        sig_qrs = 0.022 * period * qrs_sigma_scale
        for b in range(-1, n_beats):
            c = b * period + 0.45 * period
            col += _gaussian(t, c - 0.22 * period, 0.06 * period, 0.12)  # P
            col += _gaussian(t, c - 0.05 * period, 0.018 * period * qrs_sigma_scale, -0.08 * s_scale)
            col += _gaussian(t, c, sig_qrs, r_height * r_scale)
            col += _gaussian(t, c + 0.035 * period * qrs_sigma_scale, 0.025 * period * qrs_sigma_scale, -0.25 * s_scale)
            # T wave
            t_sign = -1.0 if lead_idx in t_invert_leads else 1.0
            col += t_sign * t_amp_scale * _gaussian(t, c + 0.28 * period, 0.07 * period, 0.22)

            # ST segment / early T: slow ramp after QRS (intentional morphology)
            st_k = st_shift[lead_idx]
            if abs(st_k) > 1e-6:
                col += st_k * _gaussian(t, c + 0.12 * period, 0.09 * period, 1.0)

        col += 0.06 * np.sin(2 * math.pi * 0.35 * t / N + lead_idx * 0.2)
        col += axis_shift * np.sin(2 * math.pi * t / N)
        col += rng.normal(0, noise_std, size=N)
        y[:, lead_idx] = col
    return y


def sample_1_normal_sinus() -> np.ndarray:
    """Regular rate, narrow QRS, mild isoelectric ST."""
    rng = np.random.default_rng(101)
    hr = 76.0
    period = 60.0 / hr * FS
    t = np.arange(N, dtype=np.float64)
    st = np.zeros(N_LEADS, dtype=np.float64)
    return _beat_train(
        t, period, rng,
        qrs_sigma_scale=0.85,
        r_height=1.0,
        st_shift=st,
        t_amp_scale=1.0,
        noise_std=0.008,
    ).astype(np.float32)


def sample_2_st_t_wave_focus() -> np.ndarray:
    """Diffuse T peaking + ST-style bumps in chest / lateral leads (ST/T caricature)."""
    rng = np.random.default_rng(202)
    hr = 82.0
    period = 60.0 / hr * FS
    t = np.arange(N, dtype=np.float64)
    st = np.zeros(N_LEADS, dtype=np.float64)
    for li in (LEAD_II, LEAD_III, LEAD_AVF, 9, 10, 11):  # inferior + V4–V6
        st[li] = 0.18
    for li in (LEAD_AVL, LEAD_V1):
        st[li] = -0.12

    y = _beat_train(
        t, period, rng,
        qrs_sigma_scale=0.9,
        r_height=0.9,
        st_shift=st,
        t_amp_scale=1.35,
        noise_std=0.014,
    )
    # Global low-frequency T–U ripple
    for li in range(N_LEADS):
        y[:, li] += 0.07 * np.sin(2 * math.pi * 2.3 * t / N + 0.4 * li)
    return y.astype(np.float32)


def sample_3_wide_qrs_conduction() -> np.ndarray:
    """Slower rate, visibly wider QRS envelope (conduction-delay caricature)."""
    rng = np.random.default_rng(303)
    hr = 58.0
    period = 60.0 / hr * FS
    t = np.arange(N, dtype=np.float64)
    st = np.full(N_LEADS, -0.03, dtype=np.float64)
    y = _beat_train(
        t, period, rng,
        qrs_sigma_scale=1.75,
        r_height=0.85,
        st_shift=st,
        t_amp_scale=0.95,
        noise_std=0.011,
    )
    # Slight notching after R in a few leads
    for li in (LEAD_I, LEAD_V2, LEAD_V3):
        for b in range(-1, int(math.ceil(N / period)) + 2):
            c = b * period + 0.45 * period + 0.05 * period
            y[:, li] += 0.12 * _gaussian(t, c, 0.04 * period, 1.0)
    return y.astype(np.float32)


def sample_4_inferior_st_elevation() -> np.ndarray:
    """Strong post-QRS elevation in II/III/aVF with reciprocal dip in aVL (STEMI-ish toy)."""
    rng = np.random.default_rng(404)
    hr = 88.0
    period = 60.0 / hr * FS
    t = np.arange(N, dtype=np.float64)
    st = np.zeros(N_LEADS, dtype=np.float64)
    for li in (LEAD_II, LEAD_III, LEAD_AVF):
        st[li] = 0.42
    st[LEAD_AVL] = -0.28
    st[LEAD_I] = 0.08

    y = _beat_train(
        t, period, rng,
        qrs_sigma_scale=1.0,
        r_height=1.05,
        st_shift=st,
        t_amp_scale=1.1,
        t_invert_leads=frozenset({LEAD_AVL}),
        noise_std=0.013,
    )
    return y.astype(np.float32)


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    specs = [
        (1, sample_1_normal_sinus(), "intentional_normal_sinus", "M", 42, "Narrow QRS, regular ~76 bpm, low noise"),
        (2, sample_2_st_t_wave_focus(), "intentional_st_t_pattern", "F", 61, "ST/T bumps inferior + V4–V6"),
        (3, sample_3_wide_qrs_conduction(), "intentional_wide_qrs", "M", 67, "Slow rate, widened QRS"),
        (4, sample_4_inferior_st_elevation(), "intentional_inferior_ste", "F", 54, "Inferior STE + aVL dip"),
    ]

    for num, arr, name, sex, age, _hint in specs:
        assert arr.shape == (N, N_LEADS), arr.shape
        path = os.path.join(out_dir, f"sample_intentional_{num}.json")
        payload = {
            "signal_data": arr.tolist(),
            "patient_name": name,
            "age": age,
            "sex": sex,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"Wrote {path}")

    print("Done. Upload sample_intentional_1.json … _4.json in the app.")


if __name__ == "__main__":
    main()
