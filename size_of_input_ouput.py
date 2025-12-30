import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.seasonal import STL

# -----------------------------
# PARAMETERS
# -----------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

LATENT_DIM_TOTAL = 100
LATENT_TREND = 32
LATENT_SEASONAL = 32
LATENT_RESID = LATENT_DIM_TOTAL - (LATENT_TREND + LATENT_SEASONAL)
LATENT_COVAR = 16
synthetic_length = 168
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Synthetic time series generation
# -----------------------------
n_synthetic = 5000

def generate_synthetic_series(n, length):
    series = []
    for _ in range(n):
        trend = 0.05 * np.arange(length) + np.random.normal(0, 0.1, length)
        seasonal = 0.5 * np.sin(2 * np.pi * np.arange(length) / 24) + np.random.normal(0, 0.05, length)
        resid = np.random.normal(0, 0.1, length)
        series.append(trend + seasonal + resid)
    return np.array(series, dtype=np.float32)

synthetic_series = generate_synthetic_series(n_synthetic, synthetic_length)
print(f"Synthetic series shape: {synthetic_series.shape}")

# -----------------------------
# STL decomposition
# -----------------------------
def stl_decompose_batch(series, period=24):
    trend_list, seasonal_list, resid_list = [], [], []
    for x in series:
        x = np.nan_to_num(x)
        try:
            stl = STL(x, period=period, robust=True)
            res = stl.fit()
            trend_list.append(res.trend.astype(np.float32))
            seasonal_list.append(res.seasonal.astype(np.float32))
            resid_list.append(res.resid.astype(np.float32))
        except:
            trend_list.append(x * 0)
            seasonal_list.append(x * 0)
            resid_list.append(x)
    return np.stack(trend_list), np.stack(seasonal_list), np.stack(resid_list)

trend_syn, seasonal_syn, resid_syn = stl_decompose_batch(synthetic_series)
print(f"Trend synthetic shape: {trend_syn.shape}")
print(f"Seasonal synthetic shape: {seasonal_syn.shape}")
print(f"Residual synthetic shape: {resid_syn.shape}")

# -----------------------------
# Normalization
# -----------------------------
def normalize_rows(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (X - means) / stds

trend_syn = normalize_rows(trend_syn)
seasonal_syn = normalize_rows(seasonal_syn)
resid_syn = normalize_rows(resid_syn)
print(f"Normalized trend synthetic shape: {trend_syn.shape}")
print(f"Normalized seasonal synthetic shape: {seasonal_syn.shape}")
print(f"Normalized residual synthetic shape: {resid_syn.shape}")

# -----------------------------
# Simple AE class
# -----------------------------
class SimpleAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        mid = max(64, input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, input_dim)
        )
        # Print input and output dimensions
        print(f"Autoencoder initialized with input_dim={input_dim}, latent_dim={latent_dim}, output_dim={input_dim}")

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_feather("train_final.feather")

# -----------------------------
# Covariates AE (per building)
# -----------------------------
cov_df = df.groupby('building_id').agg({
    'air_temperature': 'mean',
    'primary_use': 'first'
}).reset_index()

primary_use_ohe = OneHotEncoder(sparse_output=False)
primary_use_encoded = primary_use_ohe.fit_transform(cov_df[['primary_use']])
air_temp = cov_df[['air_temperature']].values.astype(np.float32)
covariates_per_building = np.concatenate([primary_use_encoded, air_temp], axis=1)
print(f"Number of unique primary_use categories: {len(primary_use_ohe.categories_[0])}")
print(f"Covariates per building shape: {covariates_per_building.shape}")

# -----------------------------
# Initialize AEs
# -----------------------------
print("\n--- Autoencoder Dimensions ---")
ae_trend = SimpleAE(synthetic_length, LATENT_TREND)
ae_season = SimpleAE(synthetic_length, LATENT_SEASONAL)
ae_resid = SimpleAE(synthetic_length, LATENT_RESID)
ae_covariates = SimpleAE(covariates_per_building.shape[1], LATENT_COVAR)