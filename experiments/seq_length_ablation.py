import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from src.data_loader import load_and_preprocess, train_val_test_split
from src.dataset import EnergyDataset
from src.models.lstm import LSTMModel
from src.models.tcn import TCNModel
from src.models.transformer import TransformerModel
from src.models.lstnet import LSTNet
from src.train import train_model
from src.evaluate import evaluate

#  Config 
DATA_PATH   = 'data/energydata_complete.csv'
HORIZON     = 1
EPOCHS      = 40
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR    = 'results/figures'
os.makedirs(SAVE_DIR, exist_ok=True)

# Sequence lengths to test (in timesteps, each = 10 min)
# 12=2h, 24=4h, 48=8h, 96=16h, 144=24h
SEQ_LENGTHS = [12, 24, 48, 96, 144]

MODEL_CONFIGS = {
    'LSTM':        lambda: LSTMModel(input_size=26, hidden=128, layers=2, horizon=HORIZON),
    'TCN':         lambda: TCNModel(input_size=26, channels=[64,64,64], horizon=HORIZON),
    'Transformer': lambda: TransformerModel(input_size=26, d_model=64, nhead=4, horizon=HORIZON),
    'LSTNet':      lambda: LSTNet(input_size=26, horizon=HORIZON),
}

#  Load & split data once 
print("Loading data...")
data, scaler, cols = load_and_preprocess(DATA_PATH)
train_data, val_data, test_data = train_val_test_split(data)

#  Run ablation 
# results[model_name][seq_len] = {'MAE': ..., 'RMSE': ..., 'time': ...}
results = {name: {} for name in MODEL_CONFIGS}

for seq_len in SEQ_LENGTHS:
    print(f"\n{'='*55}")
    print(f"  Sequence length: {seq_len} steps  ({seq_len*10} minutes)")
    print(f"{'='*55}")

    train_ds = EnergyDataset(train_data, seq_len=seq_len, horizon=HORIZON)
    val_ds   = EnergyDataset(val_data,   seq_len=seq_len, horizon=HORIZON)
    test_ds  = EnergyDataset(test_data,  seq_len=seq_len, horizon=HORIZON)

    for name, build_model in MODEL_CONFIGS.items():
        print(f"\n  -> Training {name}...")
        model = build_model()
        start = time.time()
        model, history = train_model(
            model, train_ds, val_ds,
            epochs=EPOCHS, lr=1e-3, batch=64, device=DEVICE
        )
        elapsed = time.time() - start
        metrics = evaluate(model, test_ds, device=DEVICE)
        metrics['time'] = elapsed
        results[name][seq_len] = metrics

        print(f"     MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"Time={elapsed:.1f}s")

#  Build summary DataFrame 
rows = []
for name in MODEL_CONFIGS:
    for seq_len in SEQ_LENGTHS:
        m = results[name][seq_len]
        rows.append({
            'Model':   name,
            'SeqLen':  seq_len,
            'Minutes': seq_len * 10,
            'MAE':     round(m['MAE'],  4),
            'RMSE':    round(m['RMSE'], 4),
            'Time_s':  round(m['time'], 1),
        })

summary = pd.DataFrame(rows)
summary.to_csv('results/seq_length_ablation.csv', index=False)
print("\n\nSummary saved to results/seq_length_ablation.csv")
print(summary.to_string(index=False))

#  Plot 1: MAE vs Sequence Length (line per model) 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {'LSTM': 'steelblue', 'TCN': 'coral', 'Transformer': 'seagreen', 'LSTNet': 'mediumpurple'}
minute_labels = [f"{s*10}m" for s in SEQ_LENGTHS]

for name in MODEL_CONFIGS:
    maes  = [results[name][s]['MAE']  for s in SEQ_LENGTHS]
    rmses = [results[name][s]['RMSE'] for s in SEQ_LENGTHS]
    axes[0].plot(minute_labels, maes,  marker='o', label=name, color=colors[name], linewidth=2)
    axes[1].plot(minute_labels, rmses, marker='o', label=name, color=colors[name], linewidth=2)

for ax, metric in zip(axes, ['MAE', 'RMSE']):
    ax.set_title(f'{metric} vs sequence length', fontsize=12)
    ax.set_xlabel('Sequence length (minutes)')
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, alpha=0.4)

plt.suptitle('Sequence length ablation study', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/seq_length_ablation_lines.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved -> seq_length_ablation_lines.png")

#  Plot 2: Heatmap — MAE across models × seq lengths 
pivot_mae  = summary.pivot(index='Model', columns='Minutes', values='MAE')
pivot_rmse = summary.pivot(index='Model', columns='Minutes', values='RMSE')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
import seaborn as sns

sns.heatmap(pivot_mae,  annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0], linewidths=0.5)
axes[0].set_title('MAE heatmap (lower = better)')
axes[0].set_xlabel('Sequence length (minutes)')

sns.heatmap(pivot_rmse, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1], linewidths=0.5)
axes[1].set_title('RMSE heatmap (lower = better)')
axes[1].set_xlabel('Sequence length (minutes)')

plt.suptitle('Sequence length × model performance', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/seq_length_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved -> seq_length_heatmap.png")

#  Plot 3: Training time vs sequence length 
fig, ax = plt.subplots(figsize=(9, 4))
for name in MODEL_CONFIGS:
    times = [results[name][s]['time'] for s in SEQ_LENGTHS]
    ax.plot(minute_labels, times, marker='s', label=name, color=colors[name], linewidth=2)

ax.set_title('Training time vs sequence length')
ax.set_xlabel('Sequence length (minutes)')
ax.set_ylabel('Time (seconds)')
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/seq_length_training_time.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved -> seq_length_training_time.png")

#  Best sequence length per model 
print("\n--- Best sequence length per model (lowest MAE) ---")
for name in MODEL_CONFIGS:
    best_seq = min(SEQ_LENGTHS, key=lambda s: results[name][s]['MAE'])
    best_mae = results[name][best_seq]['MAE']
    print(f"  {name:<14}: seq_len={best_seq:3d} ({best_seq*10} min)  MAE={best_mae:.4f}")