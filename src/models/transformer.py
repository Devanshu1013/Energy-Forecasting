import torch, torch.nn as nn, math

class TransformerModel(nn.Module):
    def __init__(self, input_size=26, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128, horizon=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                   dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.encoder(x)
        return self.fc(out[:, -1, :])