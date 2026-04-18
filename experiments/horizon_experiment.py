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
DATA_PATH = 'data/energydata_complete.csv'
SEQ_LEN   = 48     # fixed at best default (8 hours)
EPOCHS    = 40
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR  = 'results/figures'
os.makedirs(SAVE_DIR, exist_ok=True)

# Forecasting horizons (timesteps × 10 min each)
# 1=10min, 3=30min, 6=1hr, 12=2hr, 24=4hr
HORIZONS = [1, 3, 6, 12, 24]

def build_models(horizon):
    return {
        'LSTM':        LSTMModel(input_size=26, hidden=128, layers=2, horizon=horizon),
        'TCN':         TCNModel(input_size=26, channels=[64,64,64], horizon=horizon),
        'Transformer': TransformerModel(input_size=26, d_model=64, nhead=4, horizon=horizon),
        'LSTNet':      LSTNet(input_size=26, horizon=horizon),
    }

#  Load & split data once 
print("Loading data...")
data, scaler, cols = load_and_preprocess(DATA_PATH)
train_data, val_data, test_data = train_val_test_split(data)

#  Run horizon experiments 
# results[model_name][horizon] = {'MAE': ..., 'RMSE': ..., 'preds': ..., 'actuals': ...}
results = {name: {} for name in build_models(1)}

for horizon in HORIZONS:
    print(f"\n{'='*55}")
    print(f"  Horizon: {horizon} step(s)  ({horizon*10} minutes ahead)")
    print(f"{'='*55}")

    train_ds = EnergyDataset(train_data, seq_len=SEQ_LEN, horizon=horizon)
    val_ds   = EnergyDataset(val_data,   seq_len=SEQ_LEN, horizon=horizon)
    test_ds  = EnergyDataset(test_data,  seq_len=SEQ_LEN, horizon=horizon)

    models = build_models(horizon)

    for name, model in models.items():
        print(f"\n  → Training {name} (horizon={horizon})...")
        start = time.time()
        model, history = train_model(
            model, train_ds, val_ds,
            epochs=EPOCHS, lr=1e-3, batch=64, device=DEVICE
        )
        elapsed = time.time() - start
        metrics = evaluate(model, test_ds, device=DEVICE)
        metrics['time']    = elapsed
        metrics['history'] = history
        results[name][horizon] = metrics

        print(f"     MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"Time={elapsed:.1f}s")

#  Build summary DataFrame 
rows = []
for name in results:
    for h in HORIZONS:
        m = results[name][h]
        rows.append({
            'Model':   name,
            'Horizon': h,
            'Minutes': h * 10,
            'MAE':     round(m['MAE'],  4),
            'RMSE':    round(m['RMSE'], 4),
            'Time_s':  round(m['time'], 1),
        })

summary = pd.DataFrame(rows)
summary.to_csv('results/horizon_experiment.csv', index=False)
print("\n\nSummary saved to results/horizon_experiment.csv")
print(summary.to_string(index=False))

#  Plot 1: MAE & RMSE degradation curves 
colors = {'LSTM': 'steelblue', 'TCN': 'coral', 'Transformer': 'seagreen', 'LSTNet': 'mediumpurple'}
minute_labels = [f"{h*10}m" for h in HORIZONS]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name in results:
    maes  = [results[name][h]['MAE']  for h in HORIZONS]
    rmses = [results[name][h]['RMSE'] for h in HORIZONS]
    axes[0].plot(minute_labels, maes,  marker='o', label=name, color=colors[name], linewidth=2)
    axes[1].plot(minute_labels, rmses, marker='o', label=name, color=colors[name], linewidth=2)

for ax, metric in zip(axes, ['MAE', 'RMSE']):
    ax.set_title(f'{metric} vs forecasting horizon', fontsize=12)
    ax.set_xlabel('Forecasting horizon (minutes ahead)')
    ax.set_ylabel(metric)
    ax.legend(); ax.grid(True, alpha=0.4)

plt.suptitle('Multi-horizon forecasting performance', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/horizon_degradation_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → horizon_degradation_curves.png")

#  Plot 2: Heatmap 
import seaborn as sns

pivot_mae  = summary.pivot(index='Model', columns='Minutes', values='MAE')
pivot_rmse = summary.pivot(index='Model', columns='Minutes', values='RMSE')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.heatmap(pivot_mae,  annot=True, fmt='.4f', cmap='Blues', ax=axes[0], linewidths=0.5)
axes[0].set_title('MAE heatmap (lower = better)')
axes[0].set_xlabel('Horizon (minutes)')

sns.heatmap(pivot_rmse, annot=True, fmt='.4f', cmap='Blues', ax=axes[1], linewidths=0.5)
axes[1].set_title('RMSE heatmap (lower = better)')
axes[1].set_xlabel('Horizon (minutes)')

plt.suptitle('Model × horizon performance matrix', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/horizon_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → horizon_heatmap.png")

#  Plot 3: Predicted vs Actual (short-term vs long-term, best model) 
# Uses horizon=1 and horizon=24 for visual comparison
best_model_name = summary[summary['Horizon'] == 1].sort_values('MAE').iloc[0]['Model']
print(f"\nBest model at horizon=1: {best_model_name}")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
sample = 200  # show first 200 test points

for ax, h, title in zip(
    axes,
    [1, 24],
    [f'Short-term forecast (10 min ahead)', f'Long-term forecast (4 hours ahead)']
):
    preds   = results[best_model_name][h]['preds'][:sample, 0]
    actuals = results[best_model_name][h]['actuals'][:sample, 0]
    ax.plot(actuals, label='Actual',    linewidth=1.5, color='steelblue')
    ax.plot(preds,   label='Predicted', linewidth=1.5, color='coral', linestyle='--')
    ax.set_title(f'{best_model_name} — {title}', fontsize=11)
    ax.set_ylabel('Normalized energy')
    ax.legend(); ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Test sample index')
plt.suptitle(f'Predicted vs actual — {best_model_name}', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/horizon_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → horizon_pred_vs_actual.png")

#  Plot 4: Performance degradation (% increase in MAE relative to horizon=1) 
fig, ax = plt.subplots(figsize=(9, 4))

for name in results:
    base_mae = results[name][1]['MAE']
    degradation = [(results[name][h]['MAE'] - base_mae) / base_mae * 100
                   for h in HORIZONS]
    ax.plot(minute_labels, degradation, marker='o', label=name,
            color=colors[name], linewidth=2)

ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_title('MAE degradation relative to 10-min horizon')
ax.set_xlabel('Forecasting horizon (minutes)')
ax.set_ylabel('% increase in MAE from baseline')
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/horizon_degradation_pct.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → horizon_degradation_pct.png")

#  Final summary 
print("\n--- Best model per horizon (lowest MAE) ---")
for h in HORIZONS:
    subset = summary[summary['Horizon'] == h].sort_values('MAE')
    best = subset.iloc[0]
    print(f"  {h:2d} step ({h*10:3d} min): {best['Model']:<14}  MAE={best['MAE']:.4f}  RMSE={best['RMSE']:.4f}")

print("\n--- Most robust model (lowest avg MAE across all horizons) ---")
avg_mae = summary.groupby('Model')['MAE'].mean().sort_values()
for model, mae in avg_mae.items():
    print(f"  {model:<14}: avg MAE = {mae:.4f}")