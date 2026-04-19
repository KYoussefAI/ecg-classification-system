"""
ModelService: singleton that loads the trained ECGNet checkpoint
and exposes a predict() method.

If no real checkpoint is found it falls back to a randomly-initialised
model so the API remains runnable for demo / development purposes.
"""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F

from app.models.ecg_model import ECGNet

logger = logging.getLogger(__name__)

# Absolute path to the saved checkpoint
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "model", "ecg_full_model.pth")
)

CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]

CLASS_DESCRIPTIONS = {
    "CD":   "Conduction Disturbance",
    "HYP":  "Hypertrophy",
    "MI":   "Myocardial Infarction",
    "NORM": "Normal ECG",
    "STTC": "ST/T-Change",
}

DEFAULT_THRESHOLDS = {
    "CD":   0.70,
    "HYP":  0.65,
    "MI":   0.45,
    "NORM": 0.55,
    "STTC": 0.65,
}


class ModelService:
    _instance = None

    def __init__(self):
        self.device = torch.device("cpu")  # force CPU for stability
        self.model: ECGNet = None
        self.thresholds: dict = DEFAULT_THRESHOLDS
        self.class_names: list = CLASS_NAMES
        self._load()

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self):
        model_path = os.path.abspath(MODEL_PATH)

        if os.path.exists(model_path):
            logger.info(f"Loading checkpoint from {model_path}")
            try:
                checkpoint = torch.load(
                    model_path,
                    map_location="cpu",
                    weights_only=False
                )

                # ✅ YOUR CASE (from notebook)
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:

                    n_classes = len(checkpoint.get("classes", CLASS_NAMES))

                    self.model = ECGNet(n_classes=n_classes).to(self.device)

                    # ✅ THIS IS THE KEY LINE
                    self.model.load_state_dict(checkpoint["model_state"])

                    self.thresholds = checkpoint.get("thresholds", DEFAULT_THRESHOLDS)
                    self.class_names = list(checkpoint.get("classes", CLASS_NAMES))

                else:
                    # fallback if it's pure state_dict
                    self.model = ECGNet(n_classes=len(CLASS_NAMES)).to(self.device)
                    self.model.load_state_dict(checkpoint)

                self.model.eval()
                logger.info("Checkpoint loaded successfully.")
                return

            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Using random weights.")

        logger.warning("No checkpoint found — using randomly-initialised model (demo mode).")
        self.model = ECGNet(n_classes=len(CLASS_NAMES)).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(signal: np.ndarray) -> np.ndarray:
        return (signal - signal.mean()) / (signal.std() + 1e-8)

    def preprocess(self, signal: np.ndarray) -> torch.Tensor:
        if signal.ndim != 2:
            raise ValueError(f"Expected 2-D signal, got shape {signal.shape}")

        if signal.shape == (12, 1000):
            signal = signal.T
        elif signal.shape != (1000, 12):
            raise ValueError(
                f"Signal must be (1000, 12) or (12, 1000), got {signal.shape}"
            )

        signal = self.normalize(signal)

        tensor = torch.tensor(signal.T, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, signal: np.ndarray) -> dict:
        x = self.preprocess(signal)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        results = []
        for cls, prob in zip(self.class_names, probs):
            threshold = self.thresholds.get(cls, 0.5)
            results.append({
                "class": cls,
                "description": CLASS_DESCRIPTIONS.get(cls, cls),
                "probability": float(round(prob, 4)),
                "positive": bool(prob >= threshold),
            })

        results.sort(key=lambda r: r["probability"], reverse=True)

        positive = [r["class"] for r in results if r["positive"]]
        top = results[0]

        return {
            "predictions": results,
            "top_class": top["class"],
            "confidence": top["probability"],
            "positive_classes": positive,
        }