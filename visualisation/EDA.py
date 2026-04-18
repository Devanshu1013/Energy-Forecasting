import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ── Paths (works regardless of where you run from) ────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, 'data', 'energydata_complete.csv')
FIGURES_PATH = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(FIGURES_PATH, exist_ok=True)

print(f"Loading data from: {DATA_PATH}")

# ── 1. Load data ──────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

print("Shape:", df.shape)
print("\nDate range:", df['date'].min(), "->", df['date'].max())
print("\nFirst 3 rows:")
print(df.head(3))

# ── 2. Data types & missing values ───────────────────────
print("\n=== Data Types & Non-Null Counts ===")
print(df.info())

print("\n=== Missing Values ===")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

# ── 3. Descriptive statistics ─────────────────────────────
stats_df = df.drop(columns=['date']).describe().T
stats_df['cv'] = stats_df['std'] / stats_df['mean']
stats_df = stats_df[['mean', 'std', 'min', 'max', 'cv']].round(3)
print("\n=== Descriptive Statistics ===")
print(stats_df.to_string())

# ── 4. Target variable distribution ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(df['Appliances'], bins=80, color='steelblue',
             edgecolor='white', linewidth=0.3)
axes[0].set_title('Appliances energy — distribution')
axes[0].set_xlabel('Wh')
axes[0].set_ylabel('Frequency')

axes[1].boxplot(df['Appliances'], patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6))
axes[1].set_title('Appliances energy — boxplot')
axes[1].set_ylabel('Wh')

axes[2].hist(np.log1p(df['Appliances']), bins=80, color='coral',
             edgecolor='white', linewidth=0.3)
axes[2].set_title('Log(1 + Appliances) — distribution')
axes[2].set_xlabel('log(Wh)')

plt.suptitle('Target variable: Appliances energy consumption',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'target_distribution.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSkewness: {df['Appliances'].skew():.3f}")
print(f"Kurtosis: {df['Appliances'].kurt():.3f}")

# ── 5. Full time series plot ──────────────────────────────
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(df['date'], df['Appliances'], linewidth=0.5,
        color='steelblue', alpha=0.8)
ax.set_title('Appliances energy consumption over time')
ax.set_xlabel('Date')
ax.set_ylabel('Wh')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'timeseries_full.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# ── 6. One-week sample ────────────────────────────────────
week = df[(df['date'] >= '2016-02-01') & (df['date'] < '2016-02-08')]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(week['date'], week['Appliances'], linewidth=1.2,
        color='steelblue', marker='o', markersize=2, alpha=0.8)
ax.set_title('Appliances energy — one week sample (Feb 2016)')
ax.set_xlabel('Date')
ax.set_ylabel('Wh')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'timeseries_week.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# ── 7. Temporal patterns ──────────────────────────────────
df['hour']      = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek
df['month']     = df['date'].dt.month

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

hourly = df.groupby('hour')['Appliances'].mean()
axes[0].bar(hourly.index, hourly.values, color='steelblue', alpha=0.8)
axes[0].set_title('Average consumption by hour')
axes[0].set_xlabel('Hour of day')
axes[0].set_ylabel('Avg Wh')

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
daily = df.groupby('dayofweek')['Appliances'].mean()
axes[1].bar(daily.index, daily.values, color='coral', alpha=0.8)
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(days)
axes[1].set_title('Average consumption by day of week')
axes[1].set_ylabel('Avg Wh')

monthly = df.groupby('month')['Appliances'].mean()
axes[2].bar(monthly.index, monthly.values, color='seagreen', alpha=0.8)
axes[2].set_title('Average consumption by month')
axes[2].set_xlabel('Month')
axes[2].set_ylabel('Avg Wh')

plt.suptitle('Temporal patterns in energy consumption',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'temporal_patterns.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# ── 8. Correlation heatmap ────────────────────────────────
numeric_df = df.drop(columns=['date', 'hour', 'dayofweek', 'month'])
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 8})
ax.set_title('Feature correlation matrix', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'correlation_heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 10 features correlated with Appliances:")
print(corr['Appliances'].abs().sort_values(ascending=False).head(11)[1:])

# ── 9. Scatter plots — top features vs target ─────────────
top_features = corr['Appliances'].abs().sort_values(
    ascending=False).index[1:7].tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feat in enumerate(top_features):
    axes[i].scatter(df[feat], df['Appliances'],
                    alpha=0.15, s=4, color='steelblue')
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel('Appliances (Wh)')
    axes[i].set_title(f'{feat} vs Appliances')

plt.suptitle('Top 6 features vs target variable', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'scatter_top_features.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# ── 10. ACF / PACF ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(df['Appliances'],  lags=144, ax=axes[0])
plot_pacf(df['Appliances'], lags=72,  ax=axes[1])
axes[0].set_title('Autocorrelation (ACF) — 24h window')
axes[1].set_title('Partial autocorrelation (PACF) — 12h window')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'acf_pacf.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# ── 11. Sensor feature distributions ─────────────────────
sensor_cols = [c for c in df.columns
               if c not in ['date', 'hour', 'dayofweek', 'month',
                             'Appliances', 'lights', 'rv1', 'rv2']]

cols_per_row = 4
rows = (len(sensor_cols) + cols_per_row - 1) // cols_per_row

fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, rows * 3))
axes = axes.flatten()

for i, col in enumerate(sensor_cols):
    axes[i].hist(df[col], bins=40, color='mediumpurple',
                 edgecolor='white', linewidth=0.3)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_ylabel('Freq')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Sensor feature distributions', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'sensor_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# ── 12. Outlier detection ─────────────────────────────────
numeric_df2 = df.drop(columns=['date', 'hour', 'dayofweek', 'month'])
z_scores = np.abs(stats.zscore(numeric_df2))

# Convert to DataFrame so we can use sort_values
outlier_counts = pd.Series(
    (z_scores > 3).sum(axis=0),
    index=numeric_df2.columns
)

print("\nOutliers per feature (|z| > 3):")
print(outlier_counts[outlier_counts > 0].sort_values(ascending=False))
print(f"\nTotal outlier rows (any feature): "
      f"{(z_scores > 3).any(axis=1).sum()}")
print(f"That is {(z_scores > 3).any(axis=1).mean() * 100:.2f}% of data")

# ── 13. Final preprocessing & scaling ────────────────────
drop_cols = ['date', 'rv1', 'rv2', 'hour', 'dayofweek', 'month']
clean_df = df.drop(columns=drop_cols)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(clean_df.values)
scaled_df = pd.DataFrame(scaled, columns=clean_df.columns)

print("\nFinal feature set shape:", scaled_df.shape)
print("Features:", clean_df.columns.tolist())
print("\nScaled data preview:")
print(scaled_df.head(3))

# ── 14. EDA summary ───────────────────────────────────────
print("=" * 55)
print("           EDA SUMMARY")
print("=" * 55)
print(f"  Total samples     : {len(df):,}")
print(f"  Features (input)  : {scaled_df.shape[1] - 1}")
print(f"  Target column     : Appliances (Wh)")
print(f"  Sampling interval : 10 minutes")
print(f"  Date range        : "
      f"{df['date'].min().date()} -> {df['date'].max().date()}")
print(f"  Target mean       : {df['Appliances'].mean():.2f} Wh")
print(f"  Target std        : {df['Appliances'].std():.2f} Wh")
print(f"  Target skewness   : {df['Appliances'].skew():.3f}")
print(f"  Missing values    : {df.isnull().sum().sum()}")
print("=" * 55)
print("\n  Recommended SEQ_LEN : 48  (8 hours)")
print("  Recommended HORIZON  : 1, 3, 6, 12 steps")
print("  Train/Val/Test split : 70% / 15% / 15%")
print(f"\nAll figures saved to: {FIGURES_PATH}")