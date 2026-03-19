import torch
import numpy as np

# ---- CONFIG ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = self.relu(out)
        
        return out


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
        x = self.classifier(x)
        
        return x

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ecg_full_model.pth")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

class_names = checkpoint["classes"]
thresholds = checkpoint["thresholds"]

# rebuild model
model = ECGNet(len(class_names)).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ---- PREPROCESS ----
def prepare_input(signal):
    x = torch.tensor(signal, dtype=torch.float32)
    x = x.permute(1, 0)          # (12,1000)
    x = x.unsqueeze(0).to(DEVICE)
    return x

# ---- PREDICT ----
def predict(signal):
    x = prepare_input(signal)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    preds = [
        int(probs[i] >= thresholds[class_names[i]])
        for i in range(len(class_names))
    ]

    return probs, preds


# ---- TEST ----
if __name__ == "__main__":
    signal = np.load("sample.npy")  # any ECG sample

    probs, preds = predict(signal)

    for cls, p, pred in zip(class_names, probs, preds):
        print(f"{cls}: {p:.3f} -> {pred}")