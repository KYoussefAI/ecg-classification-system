"""
ECG Neural Network model definitions.
Matches the architecture used in the training notebook:
  - 3 residual blocks (12→64→128→256 channels)
  - AdaptiveAvgPool1d
  - Classifier head (256→128→5)
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """1D Residual block for ECG signal processing."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(out_channels)

        # Shortcut projection if channel dims differ
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ECGNet(nn.Module):
    """
    3-block residual network for 12-lead ECG classification.
    Input:  (batch, 12, 1000)  — channels-first
    Output: (batch, n_classes) — raw logits
    """

    def __init__(self, n_classes: int = 5):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
