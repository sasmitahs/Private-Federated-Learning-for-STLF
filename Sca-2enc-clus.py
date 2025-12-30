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
from Models import model_fn
from tqdm import tqdm
from my_utils import train_model, load_energy_data_feather, get_weights, set_weights
import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.seasonal import STL
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

# Fixed AggregationStrategy functions
def average_weights(local_weights: List[List[torch.Tensor]], 
                   client_weights: Optional[List[float]] = None) -> List[torch.Tensor]:
    """
    Average the weights from multiple clients with optional client-specific weighting.
    
    Args:
        local_weights: List of weight lists, where each weight list contains tensors for one client
        client_weights: Optional list of weights for each client (should sum to 1.0)
    
    Returns:
        List of averaged tensors
    """
    if not local_weights:
        raise ValueError("local_weights cannot be empty")
    
    num_clients = len(local_weights)
    
    # Default to uniform weighting if no weights provided
    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients
    else:
        # Ensure weights are normalized
        total_weight = sum(client_weights)
        if total_weight == 0:
            client_weights = [1.0 / num_clients] * num_clients
        else:
            client_weights = [w / total_weight for w in client_weights]
    
    if len(client_weights) != num_clients:
        raise ValueError(f"Number of client weights ({len(client_weights)}) must match number of clients ({num_clients})")
    
    # Get the number of layers from the first client
    num_layers = len(local_weights[0])
    
    # Initialize list to store averaged weights
    averaged_weights = []
    
    for layer_idx in range(num_layers):
        # Extract weights for this layer from all clients
        layer_weights = []
        for client_idx in range(num_clients):
            weight_tensor = local_weights[client_idx][layer_idx]
            # Move tensor to CPU if it's on CUDA
            if weight_tensor.is_cuda:
                weight_tensor = weight_tensor.cpu()
            layer_weights.append(weight_tensor)
        
        # Stack tensors and convert to numpy for weighted averaging
        try:
            layer_stack = np.stack([w.detach().numpy() for w in layer_weights])
        except Exception as e:
            print(f"Error stacking layer {layer_idx}: {e}")
            print(f"Layer shapes: {[w.shape for w in layer_weights]}")
            raise
        
        # Apply weighted average
        client_weights_np = np.array(client_weights).reshape(-1, 1)
        
        # Ensure proper broadcasting for multi-dimensional tensors
        while client_weights_np.ndim < layer_stack.ndim:
            client_weights_np = client_weights_np[..., np.newaxis]
        
        weighted_avg = np.sum(layer_stack * client_weights_np, axis=0)
        
        # Convert back to tensor and preserve original device
        original_device = local_weights[0][layer_idx].device
        averaged_tensor = torch.tensor(weighted_avg, dtype=local_weights[0][layer_idx].dtype)
        averaged_tensor = averaged_tensor.to(original_device)
        
        averaged_weights.append(averaged_tensor)
    
    return averaged_weights


def sync_aggregate(local_weights: List[List[torch.Tensor]], 
                  client_weights: Optional[List[float]] = None) -> List[torch.Tensor]:
    """
    Synchronous aggregation using simple averaging.
    Alias for average_weights for backward compatibility.
    """
    return average_weights(local_weights, client_weights)


def sync_aggregate_norm(local_weights: List[List[torch.Tensor]], 
                       client_weights: Optional[List[float]] = None) -> List[torch.Tensor]:
    """
    Synchronous aggregation with L2 normalization of weights before averaging.
    """
    if not local_weights:
        raise ValueError("local_weights cannot be empty")
    
    # Normalize each client's weights by L2 norm
    normalized_weights = []
    for client_weights_list in local_weights:
        client_norm_weights = []
        for weight_tensor in client_weights_list:
            # Move to CPU if on CUDA
            if weight_tensor.is_cuda:
                weight_tensor = weight_tensor.cpu()
            
            # Calculate L2 norm and normalize
            l2_norm = torch.norm(weight_tensor.float())
            if l2_norm > 0:
                normalized_tensor = weight_tensor / l2_norm
            else:
                normalized_tensor = weight_tensor
            
            client_norm_weights.append(normalized_tensor)
        normalized_weights.append(client_norm_weights)
    
    # Apply standard averaging to normalized weights
    return average_weights(normalized_weights, client_weights)


def sync_aggregate_softmax(local_weights: List[List[torch.Tensor]], 
                          temperature: float = 1.0,
                          client_weights: Optional[List[float]] = None) -> List[torch.Tensor]:
    """
    Synchronous aggregation using softmax-weighted averaging based on weight magnitudes.
    
    Args:
        local_weights: List of weight lists from clients
        temperature: Temperature parameter for softmax (lower = more concentrated)
        client_weights: Optional base weights for clients
    """
    if not local_weights:
        raise ValueError("local_weights cannot be empty")
    
    num_clients = len(local_weights)
    num_layers = len(local_weights[0])
    
    # Calculate magnitude-based weights for each client
    magnitude_weights = []
    for client_weights_list in local_weights:
        total_magnitude = 0.0
        for weight_tensor in client_weights_list:
            if weight_tensor.is_cuda:
                weight_tensor = weight_tensor.cpu()
            total_magnitude += torch.norm(weight_tensor.float()).item()
        magnitude_weights.append(total_magnitude)
    
    # Apply softmax to magnitude weights
    magnitude_weights = np.array(magnitude_weights)
    softmax_weights = np.exp(magnitude_weights / temperature)
    softmax_weights = softmax_weights / np.sum(softmax_weights)
    
    # Combine with client_weights if provided
    if client_weights is not None:
        client_weights = np.array(client_weights)
        client_weights = client_weights / np.sum(client_weights)  # Normalize
        final_weights = softmax_weights * client_weights
        final_weights = final_weights / np.sum(final_weights)  # Re-normalize
    else:
        final_weights = softmax_weights
    
    return average_weights(local_weights, final_weights.tolist())


def fedavgm_update(local_weights: List[List[torch.Tensor]], 
                  global_weights: List[torch.Tensor],
                  momentum_buffer: List[torch.Tensor],
                  momentum: float = 0.9,
                  client_weights: Optional[List[float]] = None) -> tuple:
    """
    FedAvgM (Federated Averaging with Momentum) aggregation.
    
    Args:
        local_weights: List of weight lists from clients
        global_weights: Current global model weights
        momentum_buffer: Momentum buffer from previous round
        momentum: Momentum coefficient (default: 0.9)
        client_weights: Optional weights for clients
    
    Returns:
        Tuple of (new_global_weights, updated_momentum_buffer)
    """
    if not local_weights:
        raise ValueError("local_weights cannot be empty")
    
    # Calculate standard federated average
    avg_weights = average_weights(local_weights, client_weights)
    
    # Calculate weight update (difference from current global weights)
    weight_updates = []
    for avg_w, global_w in zip(avg_weights, global_weights):
        # Ensure both tensors are on the same device
        if avg_w.device != global_w.device:
            avg_w = avg_w.to(global_w.device)
        update = avg_w - global_w
        weight_updates.append(update)
    
    # Update momentum buffer
    new_momentum_buffer = []
    for update, buffer in zip(weight_updates, momentum_buffer):
        if update.device != buffer.device:
            update = update.to(buffer.device)
        new_buffer = momentum * buffer + update
        new_momentum_buffer.append(new_buffer)
    
    # Apply momentum update to global weights
    new_global_weights = []
    for global_w, buffer in zip(global_weights, new_momentum_buffer):
        if buffer.device != global_w.device:
            buffer = buffer.to(global_w.device)
        new_weight = global_w + buffer
        new_global_weights.append(new_weight)
    
    return new_global_weights, new_momentum_buffer


df = pd.read_feather("train_final.feather")

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
MODEL_NAMES = ["gru"]
NUM_CLIENTS = 1410
CLIENT_FRAC = 0.15
NUM_ROUNDS = 40
LOCAL_EPOCHS = 5
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE ="train_final.feather"

CLUSTERS = clusters

# -----------------------------
# SCAFFOLD Training Function
# -----------------------------
def train_model_scaffold(model, train_loader, device, learning_rate, loss_fn, optimizer_class, local_epochs, global_c, local_c):
    """
    SCAFFOLD training function that returns updated weights, loss history, and control variate updates.
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    model.train()
    
    # Fix GRU weight memory warning
    if hasattr(model, 'flatten_parameters'):
        model.flatten_parameters()
    
    loss_history = []
    
    # Store initial weights for control variate update
    initial_weights = [p.data.clone().to(device) for p in model.parameters()]
    
    for epoch in range(local_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                X, y = batch
                X, y = X.to(device), y.to(device)
            else:
                # Handle single tensor case
                X = batch.to(device) if hasattr(batch, 'to') else batch
                y = X  # For time series prediction, often y is derived from X
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X)
            loss = loss_fn(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # SCAFFOLD correction: Apply control variate correction to gradients
            with torch.no_grad():
                for param, gc, lc in zip(model.parameters(), global_c, local_c):
                    if param.grad is not None:
                        param.grad.data = param.grad.data - gc + lc
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_epoch_loss)
    
    # Calculate control variate update (delta_c)
    final_weights = [p.data.clone().to(device) for p in model.parameters()]
    delta_c = []
    
    for initial, final, gc, lc in zip(initial_weights, final_weights, global_c, local_c):
        # SCAFFOLD control variate update formula
        delta = (initial - final) / (local_epochs * learning_rate) - gc + lc
        delta_c.append(delta)
    
    # Return updated model weights
    updated_weights = [p.data.clone() for p in model.parameters()]
    
    return updated_weights, loss_history, delta_c

# -----------------------------
# Time Series Difficulty Weight Class
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

# --- Build ID mappings ---
all_client_ids = sorted(set([cid for ids in CLUSTERS.values() for cid in ids]))
id2idx = {cid: idx for idx, cid in enumerate(all_client_ids)}
idx2id = {idx: cid for cid, idx in id2idx.items()}

difficulty_tracker = TimeSeriesDifficultyWeight(num_clients=len(all_client_ids))

# --- Global training loop ---
for model_name in MODEL_NAMES:
    print(f"Starting experiment with model: {model_name}")

    global_model = model_fn(model_name).to(DEVICE)
    
    # Fix GRU weight memory warning for global model
    if hasattr(global_model, 'flatten_parameters'):
        global_model.flatten_parameters()
        
    global_weights = get_weights(global_model)
    
    # Initialize control variates
    global_c = [torch.zeros_like(p.data).to(DEVICE) for p in global_model.parameters()]
    local_c = {cid: [torch.zeros_like(p.data).to(DEVICE) for p in global_model.parameters()] 
               for cid in all_client_ids}

    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} ---")

        sampled_clients = []

        # Stratified sampling across all clusters
        for cluster_name, client_ids in CLUSTERS.items():
            if len(client_ids) == 0:
                continue

            cluster_indices = np.array([id2idx[int(cid)] for cid in client_ids])
            cluster_probs = difficulty_tracker.get_sampling_probabilities(min_prob=0.05)[cluster_indices]
            cluster_probs = cluster_probs / cluster_probs.sum()

            n_sample = max(1, int(CLIENT_FRAC * len(cluster_indices)))
            sampled_indices = np.random.choice(
                cluster_indices,
                size=n_sample,
                replace=False,
                p=cluster_probs
            )

            sampled_clients.extend([idx2id[idx] for idx in sampled_indices])

        print(f"Sampled total {len(sampled_clients)} clients across all clusters")

        # Local training with SCAFFOLD
        local_weights = []
        delta_c_list = []

        for cid in tqdm(sampled_clients):
            model = model_fn(model_name).to(DEVICE)
            
            # Fix GRU weight memory warning for local model
            if hasattr(model, 'flatten_parameters'):
                model.flatten_parameters()
            
            set_weights(model, global_weights)

            train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)

            # Use SCAFFOLD training
            updated_weights, loss_history, delta_c = train_model_scaffold(
                model, train_loader,
                device=DEVICE,
                learning_rate=LR,
                loss_fn=None,
                optimizer_class=optim.Adam,
                local_epochs=LOCAL_EPOCHS,
                global_c=global_c,
                local_c=local_c[cid]
            )

            local_weights.append(updated_weights)
            delta_c_list.append(delta_c)
            difficulty_tracker.update(id2idx[cid], loss_history)

        # Aggregate global model and control variates
        normalized_weights = difficulty_tracker.get_normalized_weights([id2idx[cid] for cid in sampled_clients])
        global_weights = average_weights(local_weights, client_weights=normalized_weights)
        set_weights(global_model, global_weights)

        # Fix GRU weights after updating
        if hasattr(global_model, 'flatten_parameters'):
            global_model.flatten_parameters()

        # Update global control variate
        for i, p in enumerate(global_c):
            global_c[i] += sum([delta_c[i] * normalized_weights[j] for j, delta_c in enumerate(delta_c_list)])

        # Update local control variates for sampled clients
        for j, cid in enumerate(sampled_clients):
            local_c[cid] = [lc + delta for lc, delta in zip(local_c[cid], delta_c_list[j])]

        # Save checkpoint

        # Save checkpoint
        checkpoint_dir = os.path.join("results", model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_round_{rnd+1}_AEpublic_k-means_2enc_sca.pt")
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"✅ Saved global model to {checkpoint_path}")