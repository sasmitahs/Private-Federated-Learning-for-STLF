import os
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
from Models import model_fn
from my_utils import load_energy_data_feather, get_weights, set_weights
from AggregationStrategy import average_weights
# -----------------------------
# PARAMETERS
# -----------------------------
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", DEVICE)
# AE / synthetic params
LATENT_DIM_TOTAL = 100
LATENT_TREND = 32
LATENT_SEASONAL = 32
LATENT_RESID = LATENT_DIM_TOTAL - (LATENT_TREND + LATENT_SEASONAL)
LATENT_COVAR = 16
AE_EPOCHS_TREND = 12
AE_EPOCHS_SEASON = 10
AE_EPOCHS_RESID = 8
AE_EPOCHS_COVAR = 6
AE_BATCH = 64
AE_LR = 1e-3
PATIENCE = 3 # early stopping
K_FORCED = 20
# Training / FL params
MODEL_NAMES = ["gru","simple_cnn","moe_lstm", "lstm", "simple_ann", "cnn_gru_no_cov"]
NUM_CLIENTS = 1410
CLIENT_FRAC = 0.15
NUM_ROUNDS = 40
LOCAL_EPOCHS = 5
LR = 0.001
DATA_FILE = "train_final.feather"
# FedProx proximal coefficient
MU = 0.01
# -----------------------------
# Load data
# -----------------------------
df = pd.read_feather("train_final.feather")
# -----------------------------
# Synthetic time series generation
# -----------------------------
n_synthetic = 5000
synthetic_length = 168
def generate_synthetic_series(n, length):
    series = []
    for _ in range(n):
        trend = 0.05 * np.arange(length) + np.random.normal(0, 0.1, length)
        seasonal = 0.5 * np.sin(2 * np.pi * np.arange(length) / 24) + np.random.normal(0, 0.05, length)
        resid = np.random.normal(0, 0.1, length)
        series.append(trend + seasonal + resid)
    return np.array(series, dtype=np.float32)
synthetic_series = generate_synthetic_series(n_synthetic, synthetic_length)
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
        except Exception:
            trend_list.append(x * 0)
            seasonal_list.append(x * 0)
            resid_list.append(x)
    return np.stack(trend_list), np.stack(seasonal_list), np.stack(resid_list)
trend_syn, seasonal_syn, resid_syn = stl_decompose_batch(synthetic_series)
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
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
# -----------------------------
# AE Training with Early Stopping
# -----------------------------
def train_ae(ae, X_train, X_val=None, epochs=10, batch_size=64, lr=1e-3, device=DEVICE, verbose=False, patience=3):
    ae = ae.to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    has_val = X_val is not None
    if has_val:
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    best_val_loss = float('inf')
    wait = 0
    for ep in range(epochs):
        ae.train()
        running = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = ae(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            running += loss.item() * batch.shape[0]
        running /= len(X_train)
        val_loss = None
        if has_val:
            ae.eval()
            with torch.no_grad():
                recon_val = ae(X_val_t)
                val_loss = loss_fn(recon_val, X_val_t).item()
        if verbose:
            if has_val:
                print(f"AE epoch {ep+1}/{epochs} train_loss={running:.6f} val_loss={val_loss:.6f}")
            else:
                print(f"AE epoch {ep+1}/{epochs} train_loss={running:.6f}")
        if has_val:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {ep+1}")
                break
    ae.eval()
    return ae
# -----------------------------
# Train AEs for time series components
# -----------------------------
ae_trend = SimpleAE(synthetic_length, LATENT_TREND)
ae_season = SimpleAE(synthetic_length, LATENT_SEASONAL)
ae_resid = SimpleAE(synthetic_length, LATENT_RESID)
ae_trend = train_ae(ae_trend, trend_syn, epochs=AE_EPOCHS_TREND, batch_size=AE_BATCH, lr=AE_LR, verbose=True)
ae_season = train_ae(ae_season, seasonal_syn, epochs=AE_EPOCHS_SEASON, batch_size=AE_BATCH, lr=AE_LR, verbose=True)
ae_resid = train_ae(ae_resid, resid_syn, epochs=AE_EPOCHS_RESID, batch_size=AE_BATCH, lr=AE_LR, verbose=True)
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
ae_covariates = SimpleAE(covariates_per_building.shape[1], LATENT_COVAR)
ae_covariates = train_ae(ae_covariates, covariates_per_building, epochs=AE_EPOCHS_COVAR, batch_size=AE_BATCH, lr=AE_LR, verbose=True)
# -----------------------------
# Prepare real series for encoding
# -----------------------------
client_ids = df['building_id'].unique()
series_dict = {}
for cid in client_ids:
    series = df[df['building_id'] == cid].sort_values('timestamp')['meter_reading'].values.astype(np.float32)
    series = np.nan_to_num(series, nan=0.0)
    series_dict[cid] = series
resized_series = []
for cid in client_ids:
    s = series_dict[cid]
    if len(s) > synthetic_length:
        resized_series.append(s[:synthetic_length])
    else:
        pad_len = synthetic_length - len(s)
        resized_series.append(np.pad(s, (0, pad_len)))
resized_series = np.array(resized_series)
trend_real, seasonal_real, resid_real = stl_decompose_batch(resized_series)
trend_real = normalize_rows(trend_real)
seasonal_real = normalize_rows(seasonal_real)
resid_real = normalize_rows(resid_real)
# -----------------------------
# Encode all embeddings
# -----------------------------
with torch.no_grad():
    Z_tr = ae_trend.encoder(torch.tensor(trend_real, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_se = ae_season.encoder(torch.tensor(seasonal_real, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_re = ae_resid.encoder(torch.tensor(resid_real, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_cov = ae_covariates.encoder(torch.tensor(covariates_per_building, dtype=torch.float32).to(DEVICE)).cpu().numpy()
encodings = np.concatenate([Z_tr, Z_se, Z_re, Z_cov], axis=1)
encodings_scaled = StandardScaler().fit_transform(encodings)
# -----------------------------
# KMeans clustering
# -----------------------------
K = min(K_FORCED, len(client_ids))
km = KMeans(n_clusters=K, n_init=20, random_state=SEED)
labels = km.fit_predict(encodings_scaled)
clusters = {f"cluster_{k}": [client_ids[i] for i in range(len(client_ids)) if labels[i] == k] for k in range(K)}
for name, ids in clusters.items():
    print(f"{name}: {len(ids)} buildings; sample: {ids[:5]}")
CLUSTERS = clusters
# Create results directory
for model_name in MODEL_NAMES:
    for cluster_id in CLUSTERS:
        cluster_name = f"cluster_{cluster_id}"
        os.makedirs(os.path.join("results", model_name, cluster_name), exist_ok=True)
# -----------------------------
# FedProx local training
# -----------------------------
def train_local_fedprox(
    model: nn.Module,
    train_loader: DataLoader,
    global_weights: List[torch.Tensor],
    mu: float,
    epochs: int,
    lr: float,
    device: torch.device,
    verbose: bool = False
) -> Tuple[List[torch.Tensor], List[float]]:
    """Train locally with FedProx proximal term."""
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Convert global_weights to tensors if they are numpy arrays, then to device
    if isinstance(global_weights[0], np.ndarray):
        global_weights_device = [torch.tensor(w, dtype=torch.float32).to(device) for w in global_weights]
    else:
        global_weights_device = [w.clone().to(device) for w in global_weights]
    loss_history = []
    for ep in range(epochs):
        running_loss = 0.0
        n_samples = 0
        for batch in train_loader:
            # Handle multi-input models (x_ts, x_cov, primary_use, y)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 4:  # x_ts, x_cov, primary_use, y
                    x_ts = batch[0].to(device)
                    x_cov = batch[1].to(device)
                    primary_use = batch[2].to(device)
                    y = batch[3].to(device)
                    optimizer.zero_grad()
                    preds = model(x_ts, x_cov, primary_use)
                elif len(batch) == 2:  # x, y
                    x, y = batch[0].to(device), batch[1].to(device)
                    optimizer.zero_grad()
                    preds = model(x)
                else:
                    raise ValueError(f"Unexpected batch structure with {len(batch)} elements")
            else:
                raise ValueError("Batch must be a tuple or list")
            # Supervised loss
            loss = criterion(preds, y)
            # FedProx proximal term: (mu/2) * ||w - w_global||^2
            prox_term = 0.0
            for p, w_global in zip(model.parameters(), global_weights_device):
                prox_term += torch.sum((p - w_global) ** 2)
            prox_term = (mu / 2.0) * prox_term
            total_loss = loss + prox_term
            total_loss.backward()
            optimizer.step()
            batch_size = y.shape[0]
            running_loss += loss.item() * batch_size
            n_samples += batch_size
        epoch_loss = running_loss / max(1, n_samples)
        loss_history.append(epoch_loss)
        if verbose:
            print(f" FedProx epoch {ep+1}/{epochs}, supervised_loss={epoch_loss:.6f}, prox_term={prox_term.item():.6f}")
    updated_weights = get_weights(model)
    return updated_weights, loss_history
# -----------------------------
# Global training loop with RANDOM SAMPLING
# -----------------------------
for model_name in MODEL_NAMES:
    print(f"Starting experiment with model: {model_name}")
    # One global model for all clusters
    global_model = model_fn(model_name).to(DEVICE)
    global_weights = get_weights(global_model)
    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} ---")
        sampled_clients = []
        # 📌 RANDOM stratified sampling across all clusters
        for cluster_name, client_ids_in_cluster in CLUSTERS.items():
            if len(client_ids_in_cluster) == 0:
                continue
            # Number of clients to sample from this cluster
            n_sample = max(1, int(CLIENT_FRAC * len(client_ids_in_cluster)))
            # RANDOM sampling (uniform probability)
            sampled_from_cluster = random.sample(list(client_ids_in_cluster), n_sample)
            sampled_clients.extend(sampled_from_cluster)
        print(f"Sampled total {len(sampled_clients)} clients across all clusters (random sampling)")
        # --- Local training on sampled clients ---
        local_weights = []
        for cid in tqdm(sampled_clients):
            model = model_fn(model_name).to(DEVICE)
            set_weights(model, global_weights)
            train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)
            updated_weights, loss_history = train_local_fedprox(
                model=model,
                train_loader=train_loader,
                global_weights=global_weights,
                mu=MU,
                epochs=LOCAL_EPOCHS,
                lr=LR,
                device=DEVICE,
                verbose=False
            )
            local_weights.append(updated_weights)
        # --- Standard FedAvg aggregation (uniform weights) ---
        global_weights = average_weights(local_weights)
        set_weights(global_model, global_weights)
        # --- Save checkpoint ---
        checkpoint_dir = os.path.join("results", model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_round_{rnd+1}_random_cluster_fedprox.pt")
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"✅ Saved global model to {checkpoint_path}")