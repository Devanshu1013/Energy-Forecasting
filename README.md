# DS8013 — Deep Learning Project
## A Comparative Study of Deep Learning Architectures for Multivariate Energy Time-Series Forecasting

**Course:** DS8013 — Deep Learning  
**Semester:** Semester 2  
**Institution:** Toronto Metropolitan University (TMU)  
**Students:** Nishi Patel (501356244) · Devanshu Prajapati (501389606)

---

## What This Project Does

This project compares four deep learning models to predict household energy consumption using real sensor data. We use past 8 hours of data (48 timesteps) to predict the next energy reading.

The four models we compare are:
- **LSTM** — Long Short-Term Memory (baseline)
- **TCN** — Temporal Convolutional Network
- **Transformer** — Attention-based model
- **LSTNet** — Hybrid CNN + RNN model

---

## Dataset

**Name:** UCI Appliances Energy Prediction Dataset  
**Source:** https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction  
**Size:** 19,735 rows × 29 columns  
**Interval:** Every 10 minutes  
**Period:** January 2016 – May 2016  
**Target:** `Appliances` column (energy consumption in Wh)  

The dataset includes indoor temperature and humidity sensors (T1–T9, RH_1–RH_9), outdoor weather data (temperature, wind speed, visibility), and the target appliance energy consumption.

---

## Project Structure

```
energy_forecasting/
│
├── data/
│   └── energydata_complete.csv        # Raw dataset (download separately)
│
├── notebooks/
│   └── EDA.py                         # Exploratory data analysis + 5 graphs
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Load, clean, normalize, split data
│   ├── dataset.py                     # Sliding window sequence builder
│   ├── train.py                       # Training loop with early stopping
│   ├── evaluate.py                    # MAE, RMSE, comparison charts
│   ├── utils.py                       # Helper functions, plotting, checkpoints
│   └── models/
│       ├── __init__.py
│       ├── lstm.py                    # LSTM model
│       ├── tcn.py                     # Temporal Convolutional Network
│       ├── transformer.py             # Transformer model
│       └── lstnet.py                  # LSTNet hybrid model
│
├── experiments/
│   ├── run_all_models.py              # Main script — trains all 4 models
│   ├── horizon_experiment.py          # Multi-horizon forecasting experiment
│   └── seq_length_ablation.py         # Sequence length ablation study
│
├── results/
│   ├── figures/                       # All saved charts
│   └── checkpoints/                   # Saved model weights
│
└── requirements.txt
```

---

## How to Run

### Step 1 — Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
```

### Step 2 — Download the dataset

Place `energydata_complete.csv` inside the `data/` folder.  
Download from: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

### Step 3 — Run EDA (optional but recommended first)

```bash
cd energy_forecasting
python notebooks/EDA.py
```

This generates 5 graphs saved to `results/figures/`.

### Step 4 — Train all models

```bash
python experiments/run_all_models.py
```

This is the main script. It automatically:
1. Loads and preprocesses the data
2. Creates sliding window sequences
3. Trains all 4 models (LSTM, TCN, Transformer, LSTNet)
4. Evaluates each model on the test set
5. Saves a comparison bar chart to `results/`

### Step 5 — Run extension experiments (optional)

```bash
# Multi-horizon forecasting (predict 1, 3, 6, 12 steps ahead)
python experiments/horizon_experiment.py

# Sequence length ablation (test window sizes 12, 24, 48, 96)
python experiments/seq_length_ablation.py
```

---

## How the Code Works

### Data Pipeline

**`data_loader.py`**  
Reads the CSV, sorts by date (chronological order is critical for time-series), drops the `rv1` and `rv2` noise columns, fills missing values, and normalizes all features to [0, 1] using MinMaxScaler.

**`dataset.py`**  
Converts the normalized data into sliding window sequences. For each position `i`, it takes 48 timesteps as input (`X`) and the next timestep as the target (`y`). The window slides one step at a time, creating overlapping training samples.

```
Input  (X): [t1, t2, t3, ... t48]  →  Target (y): [t49]
Input  (X): [t2, t3, t4, ... t49]  →  Target (y): [t50]
...
```

Data is split **chronologically** — 70% train, 15% validation, 15% test. Shuffling is never used because future data must not leak into training.

### Models

**`models/lstm.py`** — LSTM processes the sequence step by step, maintaining a memory cell that learns what to remember and forget. Uses 2 stacked layers with 128 hidden units.

**`models/tcn.py`** — TCN uses dilated 1D convolutions that skip timesteps with increasing gaps (dilation = 1, 2, 4, 8...). This lets it see long histories efficiently without recurrence.

**`models/transformer.py`** — The Transformer uses self-attention to look at all 48 timesteps simultaneously. Each timestep can directly attend to any other timestep regardless of distance.

**`models/lstnet.py`** — LSTNet is a hybrid: Conv2D captures short-term local patterns across features, then a GRU processes those patterns for long-range memory.

### Training

**`train.py`**  
Uses the Adam optimizer with a learning rate of 0.001. `ReduceLROnPlateau` automatically halves the learning rate if validation loss does not improve for 5 epochs. Gradient clipping (max norm = 1.0) prevents exploding gradients. The best model weights are saved and restored at the end.

### Evaluation

**`evaluate.py`**  
Runs inference on the test set and computes:
- **MAE** (Mean Absolute Error) — average prediction error
- **RMSE** (Root Mean Squared Error) — penalizes large errors more

Lower is better for both metrics.

---

## Key Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `SEQ_LEN` | 48 | Use past 8 hours to predict |
| `HORIZON` | 1 | Predict 1 step (10 min) ahead |
| `EPOCHS` | 50 | Maximum training epochs |
| `BATCH_SIZE` | 64 | Samples per gradient update |
| `LEARNING_RATE` | 0.001 | Adam optimizer initial LR |
| Train split | 70% | ~13,800 samples |
| Val split | 15% | ~2,950 samples |
| Test split | 15% | ~2,950 samples |

---

## Results

After running `run_all_models.py`, results are printed to the console and a comparison chart is saved to `results/comparison.png`.

Example output format:
```
--- Training LSTM ---
Epoch 10: train=0.0182  val=0.0201
Epoch 20: train=0.0154  val=0.0178
...
LSTM:        MAE=0.0312  RMSE=0.0445  Time=42.3s
TCN:         MAE=0.0287  RMSE=0.0401  Time=28.1s
Transformer: MAE=0.0301  RMSE=0.0423  Time=35.6s
LSTNet:      MAE=0.0295  RMSE=0.0412  Time=31.4s
```

---

## Extension Experiments

**Multi-Horizon Forecasting** (`horizon_experiment.py`)  
Tests how model accuracy degrades as we predict further into the future (1, 3, 6, 12 steps = 10 min, 30 min, 1 hr, 2 hr ahead). Results saved to `results/horizon_results.csv`.

**Sequence Length Ablation** (`seq_length_ablation.py`)  
Tests whether giving the model more history (12, 24, 48, 96 timesteps) improves accuracy. Results saved to `results/seq_length_results.csv`.

---

## Requirements

```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
statsmodels>=0.14.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Notes

- The project uses **PyTorch** for all deep learning models
- All random seeds are fixed to 42 for reproducibility (`utils.py`)
- GPU is used automatically if available, otherwise falls back to CPU
- All charts are saved to `results/figures/` automatically
- Model checkpoints are saved to `results/checkpoints/`
