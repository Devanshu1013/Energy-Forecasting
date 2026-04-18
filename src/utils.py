import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


#  Reproducibility 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed set to {seed}")


#  Device 

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: GPU — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Device: Apple MPS (M-series chip)")
    else:
        device = torch.device('cpu')
        print("Device: CPU")
    return device


#  Normalization helpers 

def inverse_transform_target(scaler, predictions, target_col=0, n_features=26):
    predictions = np.array(predictions)
    original_shape = predictions.shape
    predictions = predictions.reshape(-1, 1)

    # Build a dummy matrix of zeros to inverse-transform just the target col
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, target_col] = predictions[:, 0]
    inversed = scaler.inverse_transform(dummy)[:, target_col]
    return inversed.reshape(original_shape)


def compute_metrics_original_scale(actuals, preds, scaler, target_col=0, n_features=26):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    act_orig  = inverse_transform_target(scaler, actuals,  target_col, n_features)
    pred_orig = inverse_transform_target(scaler, preds,    target_col, n_features)
    mae  = mean_absolute_error(act_orig, pred_orig)
    rmse = np.sqrt(mean_squared_error(act_orig, pred_orig))
    return {'MAE_wh': round(mae, 4), 'RMSE_wh': round(rmse, 4),
            'actuals_wh': act_orig, 'preds_wh': pred_orig}


#  Model helpers 

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")
    return {'total': total, 'trainable': trainable}


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")


def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded ← {path}")
    return model


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':      epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss':   val_loss,
    }, path)
    print(f"Checkpoint saved → {path}  (epoch={epoch}, val_loss={val_loss:.4f})")


def load_checkpoint(model, optimizer, path, device='cpu'):
    """Load checkpoint and return epoch + val_loss."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"Checkpoint loaded ← {path}  (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return ckpt['epoch'], ckpt['val_loss']


#  Early stopping 

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, save_path='results/best_model.pt'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.save_path  = save_path
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_model(model, self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                if self.best_state:
                    model.load_state_dict(self.best_state)
                return True
        return False

    def reset(self):
        self.best_loss = float('inf')
        self.counter   = 0
        self.best_state = None


#  Plotting helpers 

def plot_loss_curves(history, model_name='Model', save_path=None):
    """Plot train vs validation loss curves."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history['train_loss'], label='Train loss', linewidth=1.5, color='steelblue')
    ax.plot(history['val_loss'],   label='Val loss',   linewidth=1.5, color='coral', linestyle='--')
    ax.set_title(f'{model_name} — loss curves')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss curve saved → {save_path}")
    plt.show()


def plot_all_loss_curves(histories, save_path=None):
    colors = {'LSTM': 'steelblue', 'TCN': 'coral',
              'Transformer': 'seagreen', 'LSTNet': 'mediumpurple'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for name, hist in histories.items():
        c = colors.get(name, 'gray')
        axes[0].plot(hist['train_loss'], label=name, color=c, linewidth=1.5)
        axes[1].plot(hist['val_loss'],   label=name, color=c, linewidth=1.5, linestyle='--')

    for ax, title in zip(axes, ['Training loss', 'Validation loss']):
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss'); ax.legend(); ax.grid(True, alpha=0.4)

    plt.suptitle('Convergence comparison — all models', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss curves saved → {save_path}")
    plt.show()


def plot_predictions(actuals, preds, model_name='Model',
                     n_samples=200, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(actuals[:n_samples], label='Actual',    linewidth=1.5, color='steelblue')
    ax.plot(preds[:n_samples],   label='Predicted', linewidth=1.5, color='coral', linestyle='--')
    ax.set_title(f'{model_name} — predicted vs actual')
    ax.set_xlabel('Sample index'); ax.set_ylabel('Normalized energy')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prediction plot saved → {save_path}")
    plt.show()


def plot_error_distribution(actuals, preds, model_name='Model', save_path=None):
    errors = np.array(actuals).flatten() - np.array(preds).flatten()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(errors, bins=60, color='steelblue', edgecolor='white', linewidth=0.3)
    axes[0].axvline(0, color='red', linewidth=1, linestyle='--')
    axes[0].set_title(f'{model_name} — error distribution')
    axes[0].set_xlabel('Prediction error'); axes[0].set_ylabel('Frequency')

    axes[1].scatter(actuals[:500], preds[:500], alpha=0.3, s=8, color='coral')
    lims = [min(actuals[:500].min(), preds[:500].min()),
            max(actuals[:500].max(), preds[:500].max())]
    axes[1].plot(lims, lims, 'k--', linewidth=1)
    axes[1].set_title(f'{model_name} — actual vs predicted scatter')
    axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Predicted')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Error plot saved → {save_path}")
    plt.show()


def plot_model_comparison_bar(results_dict, save_path=None):
    models = list(results_dict.keys())
    maes   = [results_dict[m]['MAE']  for m in models]
    rmses  = [results_dict[m]['RMSE'] for m in models]
    colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars0 = axes[0].bar(x, maes,  color=colors[:len(models)], alpha=0.8, edgecolor='white')
    bars1 = axes[1].bar(x, rmses, color=colors[:len(models)], alpha=0.8, edgecolor='white')

    for bars, ax, metric in zip([bars0, bars1], axes, ['MAE', 'RMSE']):
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
        ax.set_title(f'{metric} comparison', fontsize=12)
        ax.set_ylabel(metric); ax.grid(True, axis='y', alpha=0.4)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{bar.get_height():.4f}',
                    ha='center', va='bottom', fontsize=9)

    plt.suptitle('Model performance comparison', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Bar chart saved → {save_path}")
    plt.show()


#  Results table 

def build_results_table(results_dict):
    """
    Build a clean summary DataFrame from results dict.
    results_dict: { model_name: {'MAE': ..., 'RMSE': ..., 'time': ..., 'params': ...} }
    """
    rows = []
    for name, m in results_dict.items():
        rows.append({
            'Model':        name,
            'MAE':          round(m.get('MAE',    0), 4),
            'RMSE':         round(m.get('RMSE',   0), 4),
            'Train_time_s': round(m.get('time',   0), 1),
            'Params':       m.get('params', {}).get('trainable', 'N/A'),
        })
    df = pd.DataFrame(rows).sort_values('MAE')
    df['Rank'] = range(1, len(df) + 1)
    return df[['Rank', 'Model', 'MAE', 'RMSE', 'Train_time_s', 'Params']]


def print_results_table(results_dict):
    """Print a formatted results table to console."""
    df = build_results_table(results_dict)
    print("\n" + "="*60)
    print("         FINAL RESULTS SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    best = df.iloc[0]['Model']
    print(f"\n  Best model: {best}  (MAE={df.iloc[0]['MAE']})")
    return df


#  Directory setup 

def setup_dirs():
    """Create all required project directories."""
    dirs = [
        'data',
        'results',
        'results/figures',
        'results/checkpoints',
        'notebooks',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Project directories ready.")