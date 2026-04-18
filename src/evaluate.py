import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def evaluate(model, test_ds, device='cpu'):
    model.eval()
    preds, actuals = [], []
    loader = torch.utils.data.DataLoader(test_ds, batch_size=128)
    with torch.no_grad():
        for X, y in loader:
            preds.append(model(X.to(device)).cpu().numpy())
            actuals.append(y.numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    return {'MAE': mae, 'RMSE': rmse, 'preds': preds, 'actuals': actuals}

def plot_results(results_dict, save_path='results/comparison.png'):
    """Bar chart comparing MAE/RMSE across all models."""
    models = list(results_dict.keys())
    maes  = [results_dict[m]['MAE'] for m in models]
    rmses = [results_dict[m]['RMSE'] for m in models]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x, maes);  axes[0].set_xticks(x); axes[0].set_xticklabels(models); axes[0].set_title('MAE')
    axes[1].bar(x, rmses); axes[1].set_xticks(x); axes[1].set_xticklabels(models); axes[1].set_title('RMSE')
    plt.tight_layout(); plt.savefig(save_path); plt.show()