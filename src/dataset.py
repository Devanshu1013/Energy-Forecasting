import numpy as np
import torch
from torch.utils.data import Dataset

class EnergyDataset(Dataset):
    def __init__(self, data, seq_len=48, horizon=1, target_col=0):
        self.X, self.y = [], []
        for i in range(len(data) - seq_len - horizon + 1):
            self.X.append(data[i:i+seq_len])
            self.y.append(data[i+seq_len:i+seq_len+horizon, target_col])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]
    