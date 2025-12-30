# train_poc.py
import os
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm

# === Your project imports (assumed present) ===
# from Models import MoELSTM, LSTMModel, train_model, model_fn
# from my_utils import train_model, load_energy_data_feather, get_weights, set_weights
# from AggregationStrategy import sync_aggregate, average_weights, sync_aggregate_norm, sync_aggregate_softmax, fedavgm_update
# (Make sure these imports point to your actual modules)
from Models import model_fn  # expected model constructor
from my_utils import load_energy_data_feather

# -----------------------------
# PARAMETERS
# -----------------------------
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", DEVICE)

# AE / synthetic params (kept from your original script)
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
PATIENCE = 3  # early stopping

K_FORCED = 20

# Training / FL params
MODEL_NAMES = ["dual_cnn_gru_fcnn","dual_cnn_ann_fcnn","dual_simple_ann_fcnn"] # keep as in your script
NUM_CLIENTS = 1410
CLIENT_FRAC = 0.15
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
LR = 0.001
DATA_FILE = "train_final.feather"

# FedProx proximal coefficient
MU = 0.01
MIN_SAMPLING_PROB = 0.05

# -----------------------------
# Load data / preprocess (kept from your original script)
# -----------------------------
# load the feather dataframe
df = pd.read_feather("train_final.feather")

# Synthetic TS generation (kept)
n_synthetic = 5000
synthetic_length = 168

def generate_synthetic_series(n, length):
    series = []
    for _ in range(n):
        trend = 0.05*np.arange(length) + np.random.normal(0, 0.1, length)
        seasonal = 0.5*np.sin(2*np.pi*np.arange(length)/24) + np.random.normal(0, 0.05, length)
        resid = np.random.normal(0, 0.1, length)
        series.append(trend + seasonal + resid)
    return np.array(series, dtype=np.float32)

synthetic_series = generate_synthetic_series(n_synthetic, synthetic_length)

# STL decomposition helpers
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
        except Exception as e:
            trend_list.append(x*0)
            seasonal_list.append(x*0)
            resid_list.append(x)
    return np.stack(trend_list), np.stack(seasonal_list), np.stack(resid_list)

def normalize_rows(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds==0] = 1.0
    return (X - means)/stds

# Decompose synthetic
trend_syn, seasonal_syn, resid_syn = stl_decompose_batch(synthetic_series)
trend_syn = normalize_rows(trend_syn)
seasonal_syn = normalize_rows(seasonal_syn)
resid_syn = normalize_rows(resid_syn)

# Simple AE (kept)
class SimpleAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        mid = max(64, input_dim//2)
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

def train_ae(ae, X_train, X_val=None, epochs=10, batch_size=64, lr=1e-3, device=DEVICE, verbose=False, patience=3):
    ae = ae.to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if X_val is not None:
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
        if X_val is not None:
            ae.eval()
            with torch.no_grad():
                recon_val = ae(X_val_t)
                val_loss = loss_fn(recon_val, X_val_t).item()
            if verbose:
                print(f"AE epoch {ep+1}/{epochs} train_loss={running:.6f} val_loss={val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {ep+1}")
                    break
        else:
            if verbose:
                print(f"AE epoch {ep+1}/{epochs} train_loss={running:.6f}")
    
    ae.eval()
    return ae

# Train AEs for time series components
ae_trend = SimpleAE(synthetic_length, LATENT_TREND)
ae_season = SimpleAE(synthetic_length, LATENT_SEASONAL)
ae_resid = SimpleAE(synthetic_length, LATENT_RESID)

ae_trend = train_ae(ae_trend, trend_syn, epochs=AE_EPOCHS_TREND, batch_size=AE_BATCH, lr=AE_LR, verbose=True)
ae_season = train_ae(ae_season, seasonal_syn, epochs=AE_EPOCHS_SEASON, batch_size=AE_BATCH, lr=AE_LR, verbose=True)
ae_resid = train_ae(ae_resid, resid_syn, epochs=AE_EPOCHS_RESID, batch_size=AE_BATCH, lr=AE_LR, verbose=True)

# -----------------------------
# Covariates AE
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
    series = df[df['building_id']==cid].sort_values('timestamp')['meter_reading'].values.astype(np.float32)
    series = np.nan_to_num(series, nan=0.0)
    series_dict[cid] = series

resized_series = []
for cid in client_ids:
    s = series_dict[cid]
    if len(s) > synthetic_length:
        resized_series.append(s[:synthetic_length])
    else:
        pad_len = synthetic_length - len(s)
        resized_series.append(np.pad(s, (0,pad_len)))
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

clusters = {f"cluster_{k}": [client_ids[i] for i in range(len(client_ids)) if labels[i]==k] for k in range(K)}
for name, ids in clusters.items():
    print(f"{name}: {len(ids)} buildings; sample: {ids[:5]}")

# -----------------------------
# Helper functions
# -----------------------------
def get_tensor_weights(model: nn.Module) -> List[torch.Tensor]:
    """Extract model parameters as a list of tensors."""
    return [p.detach().clone() for p in model.parameters()]

def set_tensor_weights(model: nn.Module, weights: List[torch.Tensor]):
    """Set model parameters from a list of tensors."""
    for p, w in zip(model.parameters(), weights):
        p.data = w.clone().to(p.device)

def aggregate_weights(weight_lists: List[List[torch.Tensor]], client_weights: List[float]) -> List[torch.Tensor]:
    """Weighted aggregation of model weights."""
    aggregated = []
    for i in range(len(weight_lists[0])):
        weighted_sum = sum(w[i].to(DEVICE) * weight for w, weight in zip(weight_lists, client_weights))
        aggregated.append(weighted_sum)
    return aggregated

# -----------------------------
# FedProx local training
# -----------------------------
def train_local_fedprox(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    global_weights: List[torch.Tensor],
    mu: float,
    epochs: int,
    lr: float,
    device: torch.device,
    verbose: bool = False
) -> tuple[List[torch.Tensor], List[float]]:
    """Train locally with FedProx proximal term."""
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Move global weights to device for proximal term computation
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

            batch_size = x_ts.shape[0] if 'x_ts' in locals() else x.shape[0]
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        epoch_loss = running_loss / max(1, n_samples)
        loss_history.append(epoch_loss)
        if verbose:
            print(f"  FedProx epoch {ep+1}/{epochs}, supervised_loss={epoch_loss:.6f}, prox_term={prox_term.item():.6f}")

    updated_weights = get_tensor_weights(model)
    return updated_weights, loss_history

# -----------------------------
# Difficulty tracker
# -----------------------------
class TimeSeriesDifficultyWeight:
    def __init__(self, num_clients, accumulate_iters=20, device=None):
        self.num_clients = num_clients
        self.device = device if device is not None else DEVICE
        self.last_loss = torch.ones(num_clients).float().to(self.device)
        self.learn_score = torch.zeros(num_clients).float().to(self.device)
        self.unlearn_score = torch.zeros(num_clients).float().to(self.device)
        self.ema_difficulty = torch.ones(num_clients).float().to(self.device)
        self.accumulate_iters = accumulate_iters

    def update(self, cid: int, loss_history: List[float]) -> float:
        current_loss = torch.tensor(loss_history[-1], dtype=torch.float32).to(self.device)
        previous_loss = self.last_loss[cid]
        delta = current_loss - previous_loss
        ratio = torch.log((current_loss + 1e-8) / (previous_loss + 1e-8))
        learn = torch.where(delta < 0, -delta * ratio, torch.tensor(0.0, device=self.device))
        unlearn = torch.where(delta >= 0, delta * ratio, torch.tensor(0.0, device=self.device))
        momentum = (self.accumulate_iters - 1) / self.accumulate_iters
        self.learn_score[cid] = momentum * self.learn_score[cid] + (1 - momentum) * learn
        self.unlearn_score[cid] = momentum * self.unlearn_score[cid] + (1 - momentum) * unlearn
        diff_ratio = (self.unlearn_score[cid] + 1e-8) / (self.learn_score[cid] + 1e-8)
        difficulty = diff_ratio
        self.ema_difficulty[cid] = momentum * self.ema_difficulty[cid] + (1 - momentum) * difficulty
        self.last_loss[cid] = current_loss
        return self.ema_difficulty[cid].item()

    def get_normalized_weights(self, client_ids: List[int]) -> List[float]:
        weights = [self.ema_difficulty[cid].item() for cid in client_ids]
        total = sum(weights)
        if total == 0:
            return [1.0 / len(client_ids)] * len(client_ids)
        return [w / total for w in weights]

    def get_sampling_probabilities(self, min_prob=0.05):
        difficulty = self.ema_difficulty
        inv_difficulty = 1.0 / (difficulty + 1e-6)
        inv_difficulty = inv_difficulty / inv_difficulty.sum()
        probs = torch.clamp(inv_difficulty, min=min_prob)
        return (probs / probs.sum()).cpu().numpy()

# -----------------------------
# Prepare flattened client list and difficulty tracker
# -----------------------------
all_client_ids = sorted(set([cid for ids in clusters.values() for cid in ids]))
num_all_clients = len(all_client_ids)
print("Total unique clients:", num_all_clients)

id2idx = {cid: idx for idx, cid in enumerate(all_client_ids)}
idx2id = {idx: cid for cid, idx in id2idx.items()}
difficulty_tracker = TimeSeriesDifficultyWeight(num_clients=len(all_client_ids))

# -----------------------------
# GLOBAL TRAINING LOOP (FedProx with difficulty-aware sampling and aggregation)
# -----------------------------
for model_name in MODEL_NAMES:
    print(f"Starting experiment with model: {model_name}")

    # create one global model
    global_model = model_fn(model_name).to(DEVICE)
    global_weights = get_tensor_weights(global_model)

    # optional: per-round checkpoints dir
    checkpoint_dir = os.path.join("results", model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} ---")

        sampling_probs = difficulty_tracker.get_sampling_probabilities(min_prob=MIN_SAMPLING_PROB)
        sampling_probs = sampling_probs / sampling_probs.sum()

        n_sample = max(1, int(CLIENT_FRAC * len(all_client_ids)))
        sampled_indices = np.random.choice(len(all_client_ids), size=n_sample, replace=False, p=sampling_probs)
        sampled_clients = [idx2id[idx] for idx in sampled_indices]
        print(f"Sampled {len(sampled_clients)} clients for this round.")

        local_weights_list = []
        for cid in tqdm(sampled_clients, desc="Local FedProx training"):
            try:
                local_model = model_fn(model_name).to(DEVICE)
                set_tensor_weights(local_model, global_weights)
                train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)

                updated_weights, loss_history = train_local_fedprox(
                    model=local_model,
                    train_loader=train_loader,
                    global_weights=global_weights,
                    mu=MU,
                    epochs=LOCAL_EPOCHS,
                    lr=LR,
                    device=DEVICE,
                    verbose=False
                )
                local_weights_list.append(updated_weights)
                difficulty_tracker.update(id2idx[cid], loss_history)
            except Exception as e:
                print(f"Warning: failed training for client {cid}: {e}")

        if len(local_weights_list) == 0:
            print("No successful client updates this round — skipping aggregation")
        else:
            # Difficulty-aware weighted aggregation
            sampled_internal_indices = [id2idx[cid] for cid in sampled_clients]
            normalized_w = difficulty_tracker.get_normalized_weights(sampled_internal_indices)
            global_weights = aggregate_weights(local_weights_list, normalized_w)
            set_tensor_weights(global_model, global_weights)

        # save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_round_{rnd+1}_poc_cluster_fedprox.pt")
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"Saved global model to {checkpoint_path}")

print("Training finished.")