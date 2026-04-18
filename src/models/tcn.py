import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel,
                                           padding=pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel,
                                           padding=pad, dilation=dilation))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        # Causal padding — trim future leakage
        pad = (self.conv1.weight.shape[-1] - 1) * self.conv1.dilation[0]
        out = self.relu(self.dropout(self.conv1(x)[..., :-pad] if pad else self.conv1(x)))
        out = self.relu(self.dropout(self.conv2(out)[..., :-pad] if pad else self.conv2(out)))
        res = self.downsample(x) if self.downsample else x
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size=26, channels=[64,64,64], kernel=3, horizon=1):
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            in_ch = input_size if i == 0 else channels[i-1]
            layers.append(TCNBlock(in_ch, ch, kernel, dilation=2**i))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], horizon)

    def forward(self, x):
        # x: (batch, seq, features) → (batch, features, seq) for Conv1d
        out = self.tcn(x.permute(0, 2, 1))
        return self.fc(out[:, :, -1])