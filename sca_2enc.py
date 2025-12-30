import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from Models import MoELSTM
import os
from collections import OrderedDict
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import random
from Models import MoELSTM, LSTMModel, train_model
from Preprocess import (
    compute_metrics,
    convert_timeseries_to_numpy,
    create_dataloader,
    load_building_series,
    split_series_list,
)
import pandas as pd
from collections import defaultdict
import os
import torch
import torch.optim as optim
from tqdm import tqdm


from Models import model_fn
from tqdm import tqdm
from my_utils import train_model, load_energy_data_feather, get_weights, set_weights
from AggregationStrategy import sync_aggregate,average_weights,sync_aggregate_norm,sync_aggregate_softmax, fedavgm_update
df = pd.read_feather("train_final.feather")


import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.seasonal import STL
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

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

AE_EPOCHS_TREND = 12
AE_EPOCHS_SEASON = 10
AE_EPOCHS_RESID = 8
AE_EPOCHS_COVAR = 6

AE_BATCH = 64
AE_LR = 1e-3
PATIENCE = 3  # early stopping patience

K_FORCED = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Synthetic time series generation
# -----------------------------
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
            trend_list.append(x*0)
            seasonal_list.append(x*0)
            resid_list.append(x)
    return np.stack(trend_list), np.stack(seasonal_list), np.stack(resid_list)

trend_syn, seasonal_syn, resid_syn = stl_decompose_batch(synthetic_series)

# -----------------------------
# Normalization
# -----------------------------
def normalize_rows(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds==0] = 1.0
    return (X - means)/stds

trend_syn = normalize_rows(trend_syn)
seasonal_syn = normalize_rows(seasonal_syn)
resid_syn = normalize_rows(resid_syn)

# -----------------------------
# Simple AE class
# -----------------------------
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
    
    if X_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    
    best_val_loss = float('inf')
    wait = 0
    
    for ep in range(epochs):
        # Training
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
        
        # Validation
        val_loss = None
        if X_val is not None:
            ae.eval()
            with torch.no_grad():
                recon_val = ae(X_val_t)
                val_loss = loss_fn(recon_val, X_val_t).item()
            if verbose:
                print(f"AE epoch {ep+1}/{epochs} train_loss={running:.6f} val_loss={val_loss:.6f}")
            
            # Early stopping
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
# Aggregate numeric and categorical covariates per building
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

# Concatenate embeddings per building
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


# Config
# List of models to experiment with
MODEL_NAMES = ["dual_gru_fcnn"]

# Config
NUM_CLIENTS = 1410
CLIENT_FRAC = 0.15
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE ="train_final.feather" # "meter_0_data_cleaned.feather"


import os
import torch
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np

CLUSTERS = clusters

import torch
import numpy as np
from typing import List

class TimeSeriesDifficultyWeight:
    def __init__(self, num_clients, accumulate_iters=20, device=None):
        self.num_clients = num_clients
        self.device = device if device is not None else DEVICE  # fallback to global
        self.last_loss = torch.ones(num_clients).float().to(self.device)
        self.learn_score = torch.zeros(num_clients).float().to(self.device)
        self.unlearn_score = torch.zeros(num_clients).float().to(self.device)
        self.ema_difficulty = torch.ones(num_clients).float().to(self.device)
        self.accumulate_iters = accumulate_iters

    def update(self, cid: int, loss_history: List[float]) -> float:
        """
        Update difficulty based on loss trend for a client.
        Expects a list of per-epoch losses.
        """
        current_loss = torch.tensor(loss_history[-1], dtype=torch.float32).to(self.device)
        previous_loss = self.last_loss[cid]
        delta = current_loss - previous_loss
        ratio = torch.log((current_loss + 1e-8) / (previous_loss + 1e-8))

        learn = torch.where(delta < 0, -delta * ratio, torch.tensor(0.0, device=self.device))
        unlearn = torch.where(delta >= 0, delta * ratio, torch.tensor(0.0, device=self.device))

        # EMA update
        momentum = (self.accumulate_iters - 1) / self.accumulate_iters
        self.learn_score[cid] = momentum * self.learn_score[cid] + (1 - momentum) * learn
        self.unlearn_score[cid] = momentum * self.unlearn_score[cid] + (1 - momentum) * unlearn

        # Difficulty score (ratio of forgetting vs learning)
        diff_ratio = (self.unlearn_score[cid] + 1e-8) / (self.learn_score[cid] + 1e-8)
        difficulty = diff_ratio

        # Smooth difficulty over rounds
        self.ema_difficulty[cid] = momentum * self.ema_difficulty[cid] + (1 - momentum) * difficulty

        self.last_loss[cid] = current_loss
        return self.ema_difficulty[cid].item()

    def get_normalized_weights(self, client_ids: List[int]) -> List[float]:
        """Return normalized weights proportional to difficulty for given client IDs."""
        weights = [self.ema_difficulty[cid].item() for cid in client_ids]
        total = sum(weights)
        if total == 0:
            return [1.0 / len(client_ids)] * len(client_ids)
        return [w / total for w in weights]

    def get_sampling_probabilities(self, min_prob=0.05):
        """
        Default: favors *easy* clients (inverse difficulty).
        """
        difficulty = self.ema_difficulty
        inv_difficulty = 1.0 / (difficulty + 1e-6)
        inv_difficulty = inv_difficulty / inv_difficulty.sum()
        probs = torch.clamp(inv_difficulty, min=min_prob)
        return (probs / probs.sum()).cpu().numpy()

    def get_hard_sampling_probabilities(self, min_prob=0.05):
        """
        Favors *hard* clients (direct difficulty).
        """
        difficulty = self.ema_difficulty
        probs = difficulty / difficulty.sum()
        probs = torch.clamp(probs, min=min_prob)
        return (probs / probs.sum()).cpu().numpy()

    def get_mixed_sampling_probabilities(self, alpha=0.7, min_prob=0.01):
        """
        Mix between hard (alpha=1.0) and easy (alpha=0.0) sampling.
        """
        diff = self.ema_difficulty
        inv_diff = 1.0 / (diff + 1e-6)

        hard_probs = diff / diff.sum()
        easy_probs = inv_diff / inv_diff.sum()

        probs = alpha * hard_probs + (1 - alpha) * easy_probs
        probs = torch.clamp(probs, min=min_prob)
        return (probs / probs.sum()).cpu().numpy()

    def get_top_clients(self, top_k=10):
        """
        Return indices and scores of top-k hardest clients.
        """
        scores = self.ema_difficulty.detach().cpu().numpy()
        idx = np.argsort(-scores)[:top_k]
        return idx, scores[idx]

def train_model_scaffold(
    local_model,
    train_loader,         # DataLoader (yields either 4-tuples or 2-tuples)
    global_weights,       # list of numpy arrays (get_weights(global_model))
    server_c,             # list of numpy arrays (server control variates)
    client_ci,            # list of numpy arrays (client control variates for this client)
    device=DEVICE,
    learning_rate=LR,
    loss_fn=None,
    optimizer_class=optim.Adam,
    epochs=LOCAL_EPOCHS
):
    """
    Train a local_model with SCAFFOLD correction.
    Returns:
      delta_y: list of numpy arrays (local_weights - global_weights)
      delta_c: list of numpy arrays (new_ci - old ci)
      new_ci: list of numpy arrays (updated client control variates)
      local_weights: list of numpy arrays (final local weights)
      loss_history: list of per-epoch losses
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_model.to(device)
    loss_fn = loss_fn or nn.MSELoss()
    optimizer = optimizer_class(local_model.parameters(), lr=learning_rate)
    loss_history = []

    local_model.train()
    total_steps = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            # support both 4-tuple and 2-tuple dataloaders
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                X_ts, X_air_temp, X_primary_use, y_batch = batch
                X_ts = X_ts.to(device)
                X_air_temp = X_air_temp.to(device)
                X_primary_use = X_primary_use.to(device)
                y_batch = y_batch.to(device)
                # call model with three inputs
                optimizer.zero_grad()
                output = local_model(X_ts, X_air_temp, X_primary_use)
            else:
                # fallback: (X_batch, y_batch)
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                if y_batch.dim() == 3 and y_batch.shape[-1] == 1:
                    y_batch = y_batch.squeeze(-1)
                optimizer.zero_grad()
                output = local_model(X_batch)

            # compute loss
            try:
                loss = loss_fn(output, y_batch)
            except TypeError:
                # if custom loss expects extra args, try passing input as well
                try:
                    loss = loss_fn(output, y_batch, X_ts if 'X_ts' in locals() else X_batch)
                except Exception:
                    loss = loss_fn(output, y_batch)

            loss.backward()
            optimizer.step()

            # SCAFFOLD correction: adjust parameters using (server_c - client_ci)
            # server_c and client_ci are lists of numpy arrays (same shape as weights)
            with torch.no_grad():
                for p, sc_np, ci_np in zip(local_model.parameters(), server_c, client_ci):
                    # convert numpy arrays to tensor on the param device & dtype
                    sc_t = torch.tensor(sc_np, dtype=p.dtype, device=p.device)
                    ci_t = torch.tensor(ci_np, dtype=p.dtype, device=p.device)
                    # p.data <- p.data - lr * (sc - ci)
                    p.data.add_( - learning_rate * (sc_t - ci_t) )

            epoch_loss += loss.item()
            total_steps += 1

        # avoid empty loader division
        if len(train_loader) > 0:
            loss_history.append(epoch_loss / len(train_loader))
        else:
            loss_history.append(epoch_loss)

    # final local weights (numpy arrays)
    local_weights = get_weights(local_model)

    # compute delta_y (local - global)
    delta_y = [lw - gw for lw, gw in zip(local_weights, global_weights)]

    # guard K
    K = max(1, total_steps)

    # update client control variates (numpy arithmetic)
    new_ci = []
    delta_c = []
    for gw, lw, ci_np, sc_np in zip(global_weights, local_weights, client_ci, server_c):
        # All are numpy arrays
        # ci_new = ci - sc + (gw - lw) / (K * lr)
        ci_new = ci_np - sc_np + (gw - lw) / (K * learning_rate)
        new_ci.append(ci_new)
        delta_c.append(ci_new - ci_np)

    return delta_y, delta_c, new_ci, local_weights, loss_history


for model_name in MODEL_NAMES:
    difficulty_tracker = TimeSeriesDifficultyWeight(num_clients=NUM_CLIENTS)
    print(f"Starting experiment with model: {model_name}")

    global_model = model_fn(model_name).to(DEVICE)
    global_weights = get_weights(global_model)
    print(f"Using device: {DEVICE}")

    server_c = [np.zeros_like(w) for w in global_weights]
    client_cs = {cid: [np.zeros_like(w) for w in global_weights] for cid in range(NUM_CLIENTS)}

    for rnd in range(NUM_ROUNDS):
        print(f"Round {rnd+1}/{NUM_ROUNDS}")

        # === Difficulty-aware sampling ===
        sampling_probs = difficulty_tracker.get_sampling_probabilities(min_prob=0.05)
        sampled_clients = np.random.choice(
            np.arange(NUM_CLIENTS),
            size=int(CLIENT_FRAC * NUM_CLIENTS),
            replace=False,
            p=sampling_probs
        )
        print(f"Sampled {len(sampled_clients)} clients")

        local_weight_deltas = []
        local_c_deltas = []

        for cid in tqdm(sampled_clients):
            local_model = model_fn(model_name).to(DEVICE)
            set_weights(local_model, global_weights)

            train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)

            delta_y, delta_c, new_ci, local_weights, loss_history = train_model_scaffold(
                local_model, train_loader,
                global_weights=global_weights,
                server_c=server_c,
                client_ci=client_cs[cid],
                device=DEVICE,
                learning_rate=LR,
                loss_fn=None,
                optimizer_class=optim.Adam,
                epochs=LOCAL_EPOCHS
            )

            # === Difficulty update ===
            difficulty_tracker.update(cid, loss_history)

            local_weight_deltas.append(delta_y)
            local_c_deltas.append(delta_c)
            client_cs[cid] = new_ci

        # === Aggregate and update global weights ===
        mean_delta_y = average_weights(local_weight_deltas)
        global_weights = [gw + mean_delta_y[i] for i, gw in enumerate(global_weights)]

        mean_delta_c = average_weights(local_c_deltas)
        frac = len(sampled_clients) / NUM_CLIENTS
        server_c = [sc + frac * mean_delta_c[i] for i, sc in enumerate(server_c)]

        set_weights(global_model, global_weights)

        ckpt_path = os.path.join("results", model_name, f"{model_name}_round_{rnd+1}_AEpublic_k-means_2enc_sca.pt")
        torch.save(global_model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")
