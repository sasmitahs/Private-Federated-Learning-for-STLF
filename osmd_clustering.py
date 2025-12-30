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

# Project imports
from Models import model_fn  # Expected model constructor
from my_utils import train_model, load_energy_data_feather, get_weights, set_weights
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
PATIENCE = 3  # Early stopping

K_FORCED = 20

# Training / FL params
MODEL_NAMES = ["dual_cnn_gru_fcnn"]
NUM_CLIENTS = 1410
CLIENT_FRAC = 0.15
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
LR = 0.001
DATA_FILE = "train_final.feather"

# OSMD params
ETA0 = 0.5  # Initial learning rate
GAMMA = 0.01  # Uniform mixing for exploration
MIN_P = 1e-4  # Minimum probability
MAX_WEIGHT = 1e3  # Clip importance weights

# -----------------------------
# Load data / preprocess
# -----------------------------
df = pd.read_feather(DATA_FILE)

# Synthetic TS generation
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
            trend_list.append(x * 0)
            seasonal_list.append(x * 0)
            resid_list.append(x)
    return np.stack(trend_list), np.stack(seasonal_list), np.stack(resid_list)

def normalize_rows(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (X - means) / stds

# Decompose synthetic
trend_syn, seasonal_syn, resid_syn = stl_decompose_batch(synthetic_series)
trend_syn = normalize_rows(trend_syn)
seasonal_syn = normalize_rows(seasonal_syn)
resid_syn = normalize_rows(resid_syn)

# Simple AE
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

# Covariates AE
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

# Prepare real series for encoding
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

# Encode all embeddings
with torch.no_grad():
    Z_tr = ae_trend.encoder(torch.tensor(trend_real, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_se = ae_season.encoder(torch.tensor(seasonal_real, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_re = ae_resid.encoder(torch.tensor(resid_real, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_cov = ae_covariates.encoder(torch.tensor(covariates_per_building, dtype=torch.float32).to(DEVICE)).cpu().numpy()

encodings = np.concatenate([Z_tr, Z_se, Z_re, Z_cov], axis=1)
encodings_scaled = StandardScaler().fit_transform(encodings)

# KMeans clustering
K = min(K_FORCED, len(client_ids))
km = KMeans(n_clusters=K, n_init=20, random_state=SEED)
labels = km.fit_predict(encodings_scaled)

clusters = {f"cluster_{k}": [client_ids[i] for i in range(len(client_ids)) if labels[i] == k] for k in range(K)}
for name, ids in clusters.items():
    print(f"{name}: {len(ids)} buildings; sample: {ids[:5]}")

# Compute similarity-based base_probs
def compute_similarity_probs(encodings_scaled, labels, client_ids, clusters):
    base_probs = np.zeros(len(client_ids))
    for k in range(K):
        cluster_cids = clusters[f"cluster_{k}"]
        if len(cluster_cids) == 0:
            continue
        # Get indices of clients in this cluster
        cluster_indices = [np.where(client_ids == cid)[0][0] for cid in cluster_cids]
        cluster_encodings = encodings_scaled[cluster_indices]
        # Compute median embedding
        median_embedding = np.median(cluster_encodings, axis=0)
        # Compute Euclidean distances to median
        distances = np.linalg.norm(cluster_encodings - median_embedding, axis=1)
        # Assign probabilities proportional to distance (more dissimilar -> higher prob)
        cluster_probs = distances / (distances.sum() + 1e-6)  # Normalize within cluster
        for idx, cid in zip(cluster_indices, cluster_cids):
            base_probs[idx] = cluster_probs[cluster_indices.index(idx)] / len(clusters)  # Equal weight per cluster
    base_probs /= base_probs.sum()  # Normalize across all clients
    return base_probs

base_probs = compute_similarity_probs(encodings_scaled, labels, client_ids, clusters)

# -----------------------------
# OSMD Adaptive Sampler
# -----------------------------
class OSMDAdaptiveSampler:
    def __init__(
        self,
        client_ids: List[Any],
        base_probs: Optional[np.ndarray] = None,
        eta0: float = 0.5,
        gamma: float = 0.01,
        min_p: float = 1e-4,
        max_weight: float = 1e3,
        device: torch.device = DEVICE,
        n_eval_batches: int = 1,
    ):
        self.client_ids = list(client_ids)
        self.K = len(self.client_ids)
        if base_probs is None:
            base_probs = np.ones(self.K) / self.K
        self.p = np.array(base_probs, dtype=float)
        self.eta0 = float(eta0)
        self.gamma = float(gamma)
        self.min_p = float(min_p)
        self.max_weight = float(max_weight)
        self.t = 1
        self.device = device
        self.n_eval_batches = max(1, int(n_eval_batches))
        self.cid2pos = {cid: i for i, cid in enumerate(self.client_ids)}
        self.loss_cache = {cid: 1e6 for cid in self.client_ids}

    def sample_candidates_with_replacement(self, d: int) -> tuple[List[Any], np.ndarray]:
        idx = np.random.choice(self.K, size=d, replace=True, p=self.p)
        return [self.client_ids[i] for i in idx], idx

    def estimate_loss_for_client(self, cid: Any, model_ctor, global_weights, filepath: str) -> float:
        try:
            train_loader, _ = load_energy_data_feather(cid, filepath=filepath)
        except Exception:
            return float(self.loss_cache.get(cid, 1e6))

        model = model_ctor().to(self.device)
        try:
            set_weights(model, global_weights)
        except Exception:
            pass

        model.eval()
        loss_fn = nn.MSELoss()
        total_loss = 0.0
        seen = 0
        try:
            it = iter(train_loader)
            for _ in range(self.n_eval_batches):
                batch = next(it)
                if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                    x_ts, x_cov, primary_use, y = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device), batch[3].to(self.device)
                else:
                    return float(self.loss_cache.get(cid, 1e6))

                with torch.no_grad():
                    preds = model(x_ts, x_cov, primary_use)
                    loss = loss_fn(preds, y).item()
                total_loss += loss
                seen += 1
        except StopIteration:
            pass
        except Exception:
            return float(self.loss_cache.get(cid, 1e6))

        if seen == 0:
            return float(self.loss_cache.get(cid, 1e6))
        est = float(total_loss / seen)
        self.loss_cache[cid] = est
        return est

    def compute_weights_delta_norm2(self, global_weights: Dict[str, torch.Tensor], updated_weights: Dict[str, torch.Tensor]) -> float:
        norm2 = 0.0
        for key in global_weights:
            delta = updated_weights[key] - global_weights[key]
            norm2 += torch.norm(delta, p=2).item() ** 2
        return norm2

    def osmd_update(self, sampled_indices: np.ndarray, rewards: List[float]):
        hat_r = np.zeros(self.K, dtype=float)
        for idx_i, r in zip(sampled_indices, rewards):
            hat_r[idx_i] += r / max(self.p[idx_i], self.min_p)
        l_hat = -hat_r
        eta_t = self.eta0 / np.sqrt(self.t)
        log_p = np.log(np.maximum(self.p, self.min_p)) + eta_t * hat_r
        max_log = np.max(log_p)
        p_unnorm = np.exp(log_p - max_log)
        p_new = p_unnorm / np.sum(p_unnorm)
        uniform = np.ones(self.K) / self.K
        p_mixed = (1.0 - self.gamma) * p_new + self.gamma * uniform
        p_mixed = np.clip(p_mixed, self.min_p, None)
        p_mixed /= p_mixed.sum()
        self.p = p_mixed
        self.t += 1

    def select_active(self, candidate_ids: List[Any], loss_dict: Dict[Any, float]) -> List[Any]:
        sorted_cand = sorted(candidate_ids, key=lambda cid: loss_dict.get(cid, float(-1e9)), reverse=True)
        return sorted_cand[:m]

    def get_probability(self, cid: Any) -> float:
        pos = self.cid2pos[cid]
        return self.p[pos]

# -----------------------------
# Prepare client list and sampler
# -----------------------------
all_client_ids = sorted(client_ids)
num_all_clients = len(all_client_ids)
print("Total unique clients:", num_all_clients)

m = max(1, int(CLIENT_FRAC * num_all_clients))
initial_d = min(num_all_clients, max(m * 5, m + 10))
sampler = OSMDAdaptiveSampler(
    client_ids=all_client_ids,
    base_probs=base_probs,
    eta0=ETA0,
    gamma=GAMMA,
    min_p=MIN_P,
    max_weight=MAX_WEIGHT,
    device=DEVICE,
    n_eval_batches=1
)

# -----------------------------
# Utility: model constructor wrapper
# -----------------------------
def make_model_for_name(name: str):
    def _ctor():
        model = model_fn(name).to(DEVICE)
        return model
    return _ctor

# -----------------------------
# GLOBAL TRAINING LOOP
# -----------------------------
for model_name in MODEL_NAMES:
    print(f"Starting experiment with model: {model_name}")

    global_model = model_fn(model_name).to(DEVICE)
    global_weights = get_weights(global_model)

    checkpoint_dir = os.path.join("results", model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_ctor = make_model_for_name(model_name)

    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} --- d={initial_d} m={m}")

        # Sample d candidates with replacement
        candidates, idxs = sampler.sample_candidates_with_replacement(d=initial_d)
        print(f"Sampled {len(candidates)} candidates")

        # Estimate loss and compute rewards
        loss_dict = {}
        rewards = []
        sampled_indices = []
        unique_candidates = set(candidates)
        for cid in tqdm(unique_candidates, desc="Estimating candidate losses and rewards"):
            try:
                # Loss estimation (for top-m selection)
                est_loss = sampler.estimate_loss_for_client(cid, model_ctor, global_weights, filepath=DATA_FILE)
                loss_dict[cid] = est_loss

                # Reward computation (1 epoch local training)
                model = model_fn(model_name).to(DEVICE)
                set_weights(model, global_weights)
                train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)
                updated_weights, _ = train_model(
                    model, train_loader,
                    device=DEVICE, learning_rate=LR,
                    loss_fn=None, optimizer_class=optim.Adam,
                    epochs=1  # Lightweight for reward
                )
                delta_norm2 = sampler.compute_weights_delta_norm2(global_weights, updated_weights)
                reward = delta_norm2
            except Exception as e:
                print(f"Warning: failed processing for client {cid}: {e}")
                loss_dict[cid] = sampler.loss_cache.get(cid, 1e6)
                reward = 0.0

            # Assign reward to all instances of this client in candidates
            for i, cand_id in enumerate(candidates):
                if cand_id == cid:
                    rewards.append(reward)
                    sampled_indices.append(idxs[i])

        # Update sampling distribution
        sampler.osmd_update(np.array(sampled_indices), rewards)

        # Select top-m by loss
        active_clients = sampler.select_active(candidates, loss_dict)
        print(f"Selected {len(active_clients)} active clients")

        # Local training
        local_weights = []
        local_probs = []
        successful_clients = []
        for cid in tqdm(active_clients, desc="Local training"):
            try:
                model = model_fn(model_name).to(DEVICE)
                set_weights(model, global_weights)
                train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)

                updated_weights, loss_history = train_model(
                    model, train_loader,
                    device=DEVICE, learning_rate=LR,
                    loss_fn=None, optimizer_class=optim.Adam,
                    epochs=LOCAL_EPOCHS
                )
                local_weights.append(updated_weights)
                local_probs.append(sampler.get_probability(cid))
                successful_clients.append(cid)

                if isinstance(loss_history, (list, tuple)) and len(loss_history) > 0:
                    sampler.loss_cache[cid] = float(loss_history[-1])
            except Exception as e:
                print(f"Warning: failed training for client {cid}: {e}")

        if len(local_weights) == 0:
            print("No successful client updates this round — skipping aggregation")
        else:
            # Importance-weighted aggregation
            local_probs = np.array(local_probs)
            weights = np.clip(1.0 / np.maximum(local_probs, MIN_P), 0, MAX_WEIGHT)
            weights /= weights.sum()  # Normalize
            global_weights = average_weights(local_weights, weights=weights)
            set_weights(global_model, global_weights)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_round_{rnd+1}_osmd.pt")
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"Saved global model to {checkpoint_path}")

print("Training finished.")