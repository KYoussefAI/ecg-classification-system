"""Efficient temporal residual network shared by training and serving."""

from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(incoming, outgoing, 7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(outgoing),
            nn.ReLU(),
            nn.Conv1d(outgoing, outgoing, 7, padding=3, bias=False),
            nn.BatchNorm1d(outgoing),
        )
        self.skip = (
            nn.Identity()
            if incoming == outgoing and stride == 1
            else nn.Sequential(
                nn.Conv1d(incoming, outgoing, 1, stride=stride, bias=False),
                nn.BatchNorm1d(outgoing),
            )
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class ECGNet(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512), blocks_per_stage=2, dropout=0.3):
        super().__init__()
        if len(channels) != 4 or any(
            type(c) is not int or c < 1 or c > 512 for c in channels
        ):
            raise ValueError("Expected four channel widths between 1 and 512")
        if (
            type(blocks_per_stage) is not int
            or not 1 <= blocks_per_stage <= 3
            or not 0 <= dropout < 1
        ):
            raise ValueError("Invalid architecture configuration")
        self.stem = nn.Sequential(
            nn.Conv1d(12, channels[0], 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1),
        )
        layers, incoming = [], channels[0]
        for stage, outgoing in enumerate(channels):
            for block in range(blocks_per_stage):
                layers.append(
                    ResidualBlock(incoming, outgoing, 2 if stage and block == 0 else 1)
                )
                incoming = outgoing
        self.stages = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(incoming, 5),
        )

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (12, 1000):
            raise ValueError("Expected tensor (batch, 12, 1000)")
        return self.head(self.stages(self.stem(x)))
