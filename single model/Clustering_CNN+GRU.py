import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import STL
from sklearn.cluster import KMeans
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from collections import OrderedDict
from tqdm import tqdm
import os
import random
from typing import List, Tuple, Optional, Dict

# Assuming these are defined elsewhere in your codebase
from Models import MoELSTM, LSTMModel, train_model, model_fn
from Preprocess import compute_metrics, convert_timeseries_to_numpy, create_dataloader, load_building_series, split_series_list
from my_utils import train_model, load_energy_data_feather, get_weights, set_weights
from AggregationStrategy import sync_aggregate, average_weights, sync_aggregate_norm, sync_aggregate_softmax, fedavgm_update
df = pd.read_feather("train_final.feather")

# Set random seeds
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

# Parameters
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
PATIENCE = 3

K_FORCED = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Synthetic time series generation
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

# STL decomposition
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

# Normalization
def normalize_rows(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (X - means) / stds

trend_syn = normalize_rows(trend_syn)
seasonal_syn = normalize_rows(seasonal_syn)
resid_syn = normalize_rows(resid_syn)

# CNN+GRU Autoencoder for Time Series
class CNNGRU_AE(nn.Module):
    def __init__(self, input_dim, latent_dim, gru_hidden=64, gru_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        
        # CNN Encoder: Extract local features
        self.cnn_encoder = nn.Sequential(
            # Input shape: (batch_size, 1, input_dim)
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample by 2
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample by 2
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Calculate CNN output dimension
        self.cnn_output_dim = input_dim // 4  # After two pooling layers
        
        # GRU Encoder: Capture temporal dependencies
        self.gru_encoder = nn.GRU(
            input_size=128,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.2 if gru_layers > 1 else 0
        )
        
        # Latent projection
        self.fc_latent = nn.Sequential(
            nn.Linear(gru_hidden, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: Reconstruct from latent space
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, gru_hidden),
            nn.ReLU()
        )
        
        # GRU Decoder
        self.gru_decoder = nn.GRU(
            input_size=gru_hidden,
            hidden_size=128,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.2 if gru_layers > 1 else 0
        )
        
        # CNN Decoder (transpose convolutions)
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # Upsample by 2
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # Upsample by 2
            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=7, stride=1, padding=3),
        )
        
    def encode(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # CNN feature extraction
        cnn_out = self.cnn_encoder(x)  # (batch_size, 128, cnn_output_dim)
        
        # Prepare for GRU: (batch_size, seq_len, features)
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch_size, cnn_output_dim, 128)
        
        # GRU encoding
        gru_out, h_n = self.gru_encoder(cnn_out)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch_size, gru_hidden)
        
        # Project to latent space
        latent = self.fc_latent(last_hidden)  # (batch_size, latent_dim)
        
        return latent
    
    def decode(self, z):
        # z shape: (batch_size, latent_dim)
        batch_size = z.size(0)
        
        # Decode from latent
        h = self.fc_decode(z)  # (batch_size, gru_hidden)
        
        # Prepare for GRU decoder: repeat to create sequence
        h_seq = h.unsqueeze(1).repeat(1, self.cnn_output_dim, 1)  # (batch_size, cnn_output_dim, gru_hidden)
        
        # GRU decoding
        gru_out, _ = self.gru_decoder(h_seq)  # (batch_size, cnn_output_dim, 128)
        
        # Prepare for CNN decoder
        gru_out = gru_out.permute(0, 2, 1)  # (batch_size, 128, cnn_output_dim)
        
        # CNN reconstruction
        recon = self.cnn_decoder(gru_out)  # (batch_size, 1, input_dim)
        recon = recon.squeeze(1)  # (batch_size, input_dim)
        
        return recon
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

# AE Training with Early Stopping
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

# Train CNN+GRU AEs for time series components
print("Training Trend CNN+GRU Autoencoder...")
ae_trend = CNNGRU_AE(synthetic_length, LATENT_TREND, gru_hidden=64, gru_layers=2)
ae_trend = train_ae(ae_trend, trend_syn, epochs=AE_EPOCHS_TREND, batch_size=AE_BATCH, lr=AE_LR, verbose=True)

print("\nTraining Seasonal CNN+GRU Autoencoder...")
ae_season = CNNGRU_AE(synthetic_length, LATENT_SEASONAL, gru_hidden=64, gru_layers=2)
ae_season = train_ae(ae_season, seasonal_syn, epochs=AE_EPOCHS_SEASON, batch_size=AE_BATCH, lr=AE_LR, verbose=True)

print("\nTraining Residual CNN+GRU Autoencoder...")
ae_resid = CNNGRU_AE(synthetic_length, LATENT_RESID, gru_hidden=64, gru_layers=2)
ae_resid = train_ae(ae_resid, resid_syn, epochs=AE_EPOCHS_RESID, batch_size=AE_BATCH, lr=AE_LR, verbose=True)

# Covariates AE (per building) - Keep original architecture for non-temporal data
cov_df = df.groupby('building_id').agg({
    'air_temperature': 'mean',
    'primary_use': 'first'
}).reset_index()

primary_use_ohe = OneHotEncoder(sparse_output=False)
primary_use_encoded = primary_use_ohe.fit_transform(cov_df[['primary_use']])
air_temp = cov_df[['air_temperature']].values.astype(np.float32)
covariates_per_building = np.concatenate([primary_use_encoded, air_temp], axis=1)

class SimpleAECov(nn.Module):
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

print("\nTraining Covariates Autoencoder...")
ae_covariates = SimpleAECov(covariates_per_building.shape[1], LATENT_COVAR)
ae_covariates = train_ae(ae_covariates, covariates_per_building, epochs=AE_EPOCHS_COVAR, batch_size=AE_BATCH, lr=AE_LR, verbose=True)

# Prepare real series for encoding
df = pd.read_feather("train_final.feather")
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

print("\nDecomposing real series...")
trend_real, seasonal_real, resid_real = stl_decompose_batch(resized_series)
trend_real = normalize_rows(trend_real)
seasonal_real = normalize_rows(seasonal_real)
resid_real = normalize_rows(resid_real)

# Encode all embeddings using CNN+GRU encoders
print("\nEncoding real series components...")
with torch.no_grad():
    Z_tr = ae_trend.encode(torch.tensor(trend_real, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_se = ae_season.encode(torch.tensor(seasonal_real, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_re = ae_resid.encode(torch.tensor(resid_real, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_cov = ae_covariates.encoder(torch.tensor(covariates_per_building, dtype=torch.float32).to(DEVICE)).cpu().numpy()

# Concatenate embeddings per building
encodings = np.concatenate([Z_tr, Z_se, Z_re, Z_cov], axis=1)
encodings_scaled = StandardScaler().fit_transform(encodings)

# KMeans clustering
print(f"\nPerforming KMeans clustering with K={K_FORCED}...")
K = min(K_FORCED, len(client_ids))
km = KMeans(n_clusters=K, n_init=20, random_state=SEED)
labels = km.fit_predict(encodings_scaled)

clusters = {f"cluster_{k}": [client_ids[i] for i in range(len(client_ids)) if labels[i] == k] for k in range(K)}

print("\nCluster Distribution:")
for name, ids in clusters.items():
    print(f"{name}: {len(ids)} buildings; sample: {ids[:5]}")
    
    
MODEL_NAMES = ["gru"]

# Config
NUM_CLIENTS = 1410
CLIENT_FRAC = 0.15
NUM_ROUNDS = 40
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
# Define your clusters manually (or randomly for now)
# CLUSTERS = {
#     "cluster_0": list(range(0, 30)),    # Clients 0–29
#     "cluster_1": list(range(30, 60)),   # Clients 30–59
#     "cluster_2": list(range(60, 90)),   # Clients 60–89
# }

# Create results directory
for model_name in MODEL_NAMES:
    for cluster_id in CLUSTERS:
        cluster_name = f"cluster_{cluster_id}"  # force string
        os.makedirs(os.path.join("results", model_name, cluster_name), exist_ok=True)

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
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# --- Build ID mappings (real client IDs <-> tracker indices) ---
all_client_ids = sorted(set([cid for ids in CLUSTERS.values() for cid in ids]))
id2idx = {cid: idx for idx, cid in enumerate(all_client_ids)}
idx2id = {idx: cid for cid, idx in id2idx.items()}

# Initialize difficulty tracker with number of unique clients
difficulty_tracker = TimeSeriesDifficultyWeight(num_clients=len(all_client_ids))

# --- Global training loop ---
for model_name in MODEL_NAMES:
    print(f"Starting experiment with model: {model_name}")

    # One global model for all clusters
    global_model = model_fn(model_name).to(DEVICE)
    global_weights = get_weights(global_model)

    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} ---")

        sampled_clients = []

        # 📌 Stratified sampling across all clusters
        for cluster_name, client_ids in CLUSTERS.items():
            if len(client_ids) == 0:
                continue

            # Convert real IDs -> internal indices
            cluster_indices = np.array([id2idx[int(cid)] for cid in client_ids])

            # Difficulty-aware sampling probabilities
            cluster_probs = difficulty_tracker.get_sampling_probabilities(min_prob=0.05)[cluster_indices]
            cluster_probs = cluster_probs / cluster_probs.sum()

            # Number of clients to sample from this cluster
            n_sample = max(1, int(CLIENT_FRAC * len(cluster_indices)))
            sampled_indices = np.random.choice(
                cluster_indices,
                size=n_sample,
                replace=False,
                p=cluster_probs
            )

            # Convert back to real client IDs
            sampled_clients.extend([idx2id[idx] for idx in sampled_indices])

        print(f"Sampled total {len(sampled_clients)} clients across all clusters")

        # --- Local training on sampled clients ---
        local_weights = []
        for cid in tqdm(sampled_clients):
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

            # Update difficulty tracker (use internal index)
            difficulty_tracker.update(id2idx[cid], loss_history)

        # --- Difficulty-aware weighted aggregation ---
        normalized_weights = difficulty_tracker.get_normalized_weights([id2idx[cid] for cid in sampled_clients])
        global_weights = average_weights(local_weights, client_weights=normalized_weights)
        set_weights(global_model, global_weights)

        # --- Save checkpoint ---
        checkpoint_dir = os.path.join("results", model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_round_{rnd+1}_AEpublic_CNN+GRU.pt")
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"✅ Saved global model to {checkpoint_path}")
