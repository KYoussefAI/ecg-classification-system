from fastapi import FastAPI, UploadFile, File
import numpy as np
import torch
import torch.nn as nn
import os

# ---- DEVICE ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- PATHS ----
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ecg_full_model.pth")

# ---- LOAD MODEL ----
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

class_names = checkpoint["classes"]
thresholds = checkpoint["thresholds"]

# ---- MODEL ----
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ECGNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.block1 = ResidualBlock(12, 64)
        self.block2 = ResidualBlock(64, 128)
        self.block3 = ResidualBlock(128, 256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

model = ECGNet(len(class_names)).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ---- PREPROCESS ----
def prepare_input(signal):
    x = torch.tensor(signal, dtype=torch.float32)
    x = x.permute(1, 0)  # (12,1000)
    x = x.unsqueeze(0).to(DEVICE)  # (1,12,1000)
    return x

# ---- INTERPRETATION ----
def interpret_results(classes, probs, preds):
    CLASS_DESCRIPTIONS = {
        "MI": "Myocardial Infarction",
        "HYP": "Hypertrophy",
        "CD": "Conduction Disturbance",
        "STTC": "ST/T Changes",
        "NORM": "Normal ECG"
    }

    detected = []

    for cls, p, pred in zip(classes, probs, preds):
        if pred and cls != "NORM":
            detected.append({
                "condition": CLASS_DESCRIPTIONS.get(cls, cls),
                "confidence": round(float(p), 2)
            })

    has_abnormal = any(
        pred and cls != "NORM"
        for cls, pred in zip(classes, preds)
    )

    max_conf = float(np.max(probs))

    if not has_abnormal:
        return {
            "diagnosis": "Normal ECG",
            "confidence": round(max_conf, 2),
            "details": [],
            "probs": probs.tolist(),
            "classes": classes
        }

    return {
        "diagnosis": "Abnormal ECG",
        "confidence": round(max_conf, 2),
        "details": detected,
        "probs": probs.tolist(),
        "classes": classes
    }

# ---- FASTAPI ----
app = FastAPI()

@app.get("/")
def root():
    return {"message": "ECG API running"}

# ---- SINGLE PREDICTION ----
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    signal = np.load(file.file)

    x = prepare_input(signal)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    preds = [
        int(probs[i] >= thresholds[class_names[i]])
        for i in range(len(class_names))
    ]

    return interpret_results(class_names, probs, preds)

# ---- BATCH PREDICTION ----
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    data = np.load(file.file)  # shape (N,1000,12)

    results = []

    for signal in data:
        x = prepare_input(signal)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        preds = [
            int(probs[i] >= thresholds[class_names[i]])
            for i in range(len(class_names))
        ]

        results.append({
            "probs": probs.tolist(),
            "preds": preds
        })

    return results