import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_and_preprocess, train_val_test_split
from src.dataset import EnergyDataset
from src.models.lstm import LSTMModel
from src.models.tcn import TCNModel
from src.models.transformer import TransformerModel
from src.models.lstnet import LSTNet
from src.train import train_model
from src.evaluate import evaluate, plot_results
import torch, time, numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'energydata_complete.csv')
SEQ_LEN = 48   # 8 hours at 10-min intervals
HORIZON = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

data, scaler, cols = load_and_preprocess(DATA_PATH)
train, val, test = train_val_test_split(data)
train_ds = EnergyDataset(train, SEQ_LEN, HORIZON)
val_ds   = EnergyDataset(val,   SEQ_LEN, HORIZON)
test_ds  = EnergyDataset(test,  SEQ_LEN, HORIZON)

MODELS = {
    'LSTM':        LSTMModel(),
    'TCN':         TCNModel(),
    'Transformer': TransformerModel(),
    'LSTNet':      LSTNet(),
}

all_results = {}
for name, model in MODELS.items():
    print(f"\n--- Training {name} ---")
    start = time.time()
    model, history = train_model(model, train_ds, val_ds, device=DEVICE)
    elapsed = time.time() - start
    results = evaluate(model, test_ds, device=DEVICE)
    results['train_time'] = elapsed
    all_results[name] = results
    print(f"{name}: MAE={results['MAE']:.4f}  RMSE={results['RMSE']:.4f}  Time={elapsed:.1f}s")

plot_results(all_results)