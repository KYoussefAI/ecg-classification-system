"""Versioned input and output contract; no learned constants live here."""

CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]
DESCRIPTIONS = {
    "CD": "Conduction disturbance",
    "HYP": "Hypertrophy",
    "MI": "Myocardial infarction",
    "NORM": "Normal ECG pattern",
    "STTC": "ST/T change",
}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SAMPLE_RATE = 100
SAMPLES = 1000
SPLIT = {"train": list(range(1, 9)), "validation": [9], "test": [10]}
ARCHITECTURE = "ECGResNet1D-v1"
DEFAULT_MODEL = {"channels": [64, 128, 256, 512], "blocks_per_stage": 2, "dropout": 0.3}
DISCLAIMER = "Research prototype — not a medical device and not intended for clinical diagnosis or emergency decision-making."
