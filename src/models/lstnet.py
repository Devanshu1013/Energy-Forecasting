import torch
import torch.nn as nn

class LSTNet(nn.Module):
    def __init__(self, input_size=26, conv_out=32, conv_kernel=6,
                 gru_hidden=64, skip=24, horizon=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv2d(1, conv_out, (conv_kernel, input_size))
        self.gru = nn.GRU(conv_out, gru_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_hidden, horizon)

    def forward(self, x):
        # x: (batch, seq, features)
        c = self.conv(x.unsqueeze(1))          # (B, conv_out, T', 1)
        c = c.squeeze(-1).permute(0, 2, 1)     # (B, T', conv_out)
        out, _ = self.gru(self.dropout(c))
        return self.fc(out[:, -1, :])