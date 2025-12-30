# train_4_models_ae_cluster_difficulty_fedavg.py
# AE-based clustering + Difficulty-aware sampling + FedAvg (no FedProx)
# Trains 4 independent runs → saves models ROUND BY ROUND

import os
import random
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm

# Your project imports
from Models import model_fn
from my_utils import train_model, load_energy_data_feather, get_weights, set_weights
from AggregationStrategy import average_weights  # We'll use this for clean FedAvg

# =============================
# CONFIGURATION
# =============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", DEVICE)

MODEL_NAMES = ["dual_cnn_gru_fcnn"]
NUM_ROUNDS = 40
LOCAL_EPOCHS = 5
LR = 0.001
CLIENT_FRAC = 0.15
DATA_FILE = "train_final.feather"
MIN_SAMPLING_PROB = 0.05

# AE & Synthetic
n_synthetic = 5000
synthetic_length = 168
LATENT_TREND = 32
LATENT_SEASONAL = 32
LATENT_RESID = 36  # 100 - 32 - 32
LATENT_COVAR = 16
K_FORCED = 20

# Number of independent runs
NUM_RUNS = 4
RUN_SEEDS = [42, 123, 999, 2025]

# SAVE EVERY N ROUNDS (set to 1 to save every round, or 5 to save every 5 rounds, etc.)
SAVE_INTERVAL = 1  # Save every round

# =============================
# SYNTHETIC + STL + NORMALIZE
# =============================
def generate_synthetic_series(n, length):
    series = []
    for _ in range(n):
        trend = 0.05 * np.arange(length) + np.random.normal(0, 0.1, length)
        seasonal = 0.5 * np.sin(2 * np.pi * np.arange(length) / 24) + np.random.normal(0, 0.05, length)
        resid = np.random.normal(0, 0.1, length)
        series.append(trend + seasonal + resid)
    return np.array(series, dtype=np.float32)

def stl_decompose_batch(series, period=24):
    trend, seasonal, resid = [], [], []
    for x in series:
        x = np.nan_to_num(x)
        try:
            stl = STL(x, period=period, robust=True).fit()
            trend.append(stl.trend.astype(np.float32))
            seasonal.append(stl.seasonal.astype(np.float32))
            resid.append(stl.resid.astype(np.float32))
        except:
            trend.append(np.zeros_like(x))
            seasonal.append(np.zeros_like(x))
            resid.append(x)
    return np.stack(trend), np.stack(seasonal), np.stack(resid)

def normalize_rows(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std

# =============================
# SIMPLE AUTOENCODER
# =============================
class SimpleAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        mid = max(64, input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid), nn.ReLU(),
            nn.Linear(mid, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid), nn.ReLU(),
            nn.Linear(mid, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_ae(ae, X_train, epochs=10, batch_size=64, lr=1e-3, patience=3):
    ae.to(DEVICE)
    opt = optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float()), batch_size=batch_size, shuffle=True)
    best_loss = float('inf')
    wait = 0
    for ep in range(epochs):
        ae.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            recon = ae(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        epoch_loss = total_loss / len(X_train)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return ae

# =============================
# DIFFICULTY TRACKER
# =============================
class TimeSeriesDifficultyWeight:
    def __init__(self, num_clients, device=DEVICE):
        self.device = device
        self.last_loss = torch.ones(num_clients, device=device)
        self.learn_score = torch.zeros(num_clients, device=device)
        self.unlearn_score = torch.zeros(num_clients, device=device)
        self.ema_difficulty = torch.ones(num_clients, device=device)
        self.momentum = 0.95

    def update(self, cid: int, loss_history: List[float]):
        if not loss_history:
            return
        cur = torch.tensor(loss_history[-1], device=self.device)
        prev = self.last_loss[cid]
        delta = cur - prev
        ratio = torch.log((cur + 1e-8) / (prev + 1e-8))
        learn = torch.where(delta < 0, -delta * ratio, torch.tensor(0.0, device=self.device))
        unlearn = torch.where(delta >= 0, delta * ratio, torch.tensor(0.0, device=self.device))

        self.learn_score[cid] = self.momentum * self.learn_score[cid] + (1 - self.momentum) * learn
        self.unlearn_score[cid] = self.momentum * self.unlearn_score[cid] + (1 - self.momentum) * unlearn
        diff = (self.unlearn_score[cid] + 1e-8) / (self.learn_score[cid] + 1e-8)
        self.ema_difficulty[cid] = self.momentum * self.ema_difficulty[cid] + (1 - self.momentum) * diff
        self.last_loss[cid] = cur

    def get_sampling_probabilities(self, min_prob=0.05):
        inv = 1.0 / (self.ema_difficulty + 1e-6)
        probs = inv / inv.sum()
        probs = torch.clamp(probs, min=min_prob)
        return (probs / probs.sum()).cpu().numpy()

    def get_normalized_weights(self, indices):
        w = self.ema_difficulty[indices].cpu().numpy()
        total = w.sum()
        return [wi / total for wi in w] if total > 0 else [1/len(indices)] * len(indices)

# =============================
# MAIN LOOP: 4 INDEPENDENT RUNS
# =============================
for run_idx in range(1, NUM_RUNS + 1):
    print("\n" + "="*80)
    print(f"STARTING RUN {run_idx}/{NUM_RUNS} | SEED = {RUN_SEEDS[run_idx-1]}")
    print("="*80)

    # SET SEED
    SEED = RUN_SEEDS[run_idx - 1]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # LOAD DATA
    df = pd.read_feather(DATA_FILE)
    client_ids = sorted(df['building_id'].unique())
    num_clients = len(client_ids)
    print(f"Found {num_clients} clients")

    # SYNTHETIC + AE TRAINING (once per run)
    print("\n[AE Training Phase]")
    synthetic_series = generate_synthetic_series(n_synthetic, synthetic_length)
    trend_syn, seasonal_syn, resid_syn = stl_decompose_batch(synthetic_series)
    trend_syn = normalize_rows(trend_syn)
    seasonal_syn = normalize_rows(seasonal_syn)
    resid_syn = normalize_rows(resid_syn)

    print("Training AE for Trend...")
    ae_trend = train_ae(SimpleAE(synthetic_length, LATENT_TREND), trend_syn, epochs=12)
    print("Training AE for Seasonal...")
    ae_season = train_ae(SimpleAE(synthetic_length, LATENT_SEASONAL), seasonal_syn, epochs=10)
    print("Training AE for Residual...")
    ae_resid = train_ae(SimpleAE(synthetic_length, LATENT_RESID), resid_syn, epochs=8)

    # Covariates AE
    print("Training AE for Covariates...")
    cov_df = df.groupby('building_id').agg({'air_temperature': 'mean', 'primary_use': 'first'}).reset_index()
    ohe = OneHotEncoder(sparse_output=False)
    pu_enc = ohe.fit_transform(cov_df[['primary_use']]).astype(np.float32)
    temp = cov_df['air_temperature'].values.reshape(-1, 1).astype(np.float32)
    covariates = np.hstack([pu_enc, temp])
    ae_cov = train_ae(SimpleAE(covariates.shape[1], LATENT_COVAR), covariates, epochs=6)

    # Real series → embeddings
    print("\n[Encoding Real Client Data]")
    series_dict = {cid: np.nan_to_num(df[df['building_id']==cid].sort_values('timestamp')['meter_reading'].values.astype(np.float32))
                   for cid in client_ids}
    resized = [s[:synthetic_length] if len(s) > synthetic_length else np.pad(s, (0, synthetic_length - len(s))) for s in series_dict.values()]
    resized = np.array(resized, dtype=np.float32)

    trend_r, seasonal_r, resid_r = stl_decompose_batch(resized)
    trend_r = normalize_rows(trend_r)
    seasonal_r = normalize_rows(seasonal_r)
    resid_r = normalize_rows(resid_r)

    with torch.no_grad():
        Z_tr = ae_trend.encoder(torch.from_numpy(trend_r).to(DEVICE)).cpu().numpy()
        Z_se = ae_season.encoder(torch.from_numpy(seasonal_r).to(DEVICE)).cpu().numpy()
        Z_re = ae_resid.encoder(torch.from_numpy(resid_r).to(DEVICE)).cpu().numpy()
        Z_cov = ae_cov.encoder(torch.from_numpy(covariates).to(DEVICE)).cpu().numpy()
    encodings = np.hstack([Z_tr, Z_se, Z_re, Z_cov])
    encodings_scaled = StandardScaler().fit_transform(encodings)

    # Clustering (for reporting only)
    K = min(K_FORCED, num_clients)
    labels = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit_predict(encodings_scaled)
    print(f"Clustered {num_clients} clients into {K} groups")

    # Mapping
    cid2idx = {cid: i for i, cid in enumerate(client_ids)}

    # Initialize tracker
    tracker = TimeSeriesDifficultyWeight(num_clients=num_clients)

    # Train each model name
    for model_name in MODEL_NAMES:
        print(f"\n{'='*80}")
        print(f"Training {model_name} (Run {run_idx})")
        print(f"{'='*80}")
        
        global_model = model_fn(model_name).to(DEVICE)
        global_weights = get_weights(global_model)

        # Create results directory for this run
        results_dir = f"results_ae_cluster_difficulty_fedavg/run_{run_idx}"
        os.makedirs(results_dir, exist_ok=True)

        # Training rounds
        for rnd in range(NUM_ROUNDS):
            print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} ---")
            
            # Difficulty-aware sampling
            probs = tracker.get_sampling_probabilities(min_prob=MIN_SAMPLING_PROB)
            n_sample = max(1, int(CLIENT_FRAC * num_clients))
            chosen_idx = np.random.choice(num_clients, size=n_sample, replace=False, p=probs)
            chosen_clients = [client_ids[i] for i in chosen_idx]

            print(f"Selected {len(chosen_clients)} clients for training")

            local_weights_list = []
            client_indices = []

            # Local training
            for cid in tqdm(chosen_clients, desc=f"R{rnd+1} Local Training", leave=False):
                try:
                    model = model_fn(model_name).to(DEVICE)
                    set_weights(model, global_weights)
                    loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)

                    updated_weights, loss_hist = train_model(
                        model, loader, device=DEVICE, learning_rate=LR,
                        optimizer_class=optim.Adam, epochs=LOCAL_EPOCHS
                    )

                    local_weights_list.append(updated_weights)
                    client_indices.append(cid2idx[cid])
                    tracker.update(cid2idx[cid], loss_hist)
                except Exception as e:
                    print(f"[WARN] Client {cid} failed: {e}")

            # FedAvg aggregation
            if local_weights_list:
                # FedAvg with difficulty-aware weights
                agg_weights = tracker.get_normalized_weights(client_indices)
                global_weights = average_weights(local_weights_list, client_weights=agg_weights)
                set_weights(global_model, global_weights)
                print(f"✓ Aggregated {len(local_weights_list)} client models")
            else:
                print("[WARN] No valid client updates in this round")

            # Save model checkpoint every SAVE_INTERVAL rounds
            if (rnd + 1) % SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(
                    results_dir, 
                    f"{model_name}_round_{rnd+1}_ae_cluster_difficulty_fedavg.pt"
                )
                torch.save({
                    'round': rnd + 1,
                    'model_state_dict': global_model.state_dict(),
                    'run_id': run_idx,
                    'seed': SEED
                }, checkpoint_path)
                print(f"✓ Saved checkpoint: Round {rnd+1} → {checkpoint_path}")

        # Save final model (separate from round checkpoints)
        final_path = os.path.join(results_dir, f"{model_name}_final_run{run_idx}.pt")
        torch.save({
            'round': NUM_ROUNDS,
            'model_state_dict': global_model.state_dict(),
            'run_id': run_idx,
            'seed': SEED
        }, final_path)
        print(f"\n{'='*60}")
        print(f"✓ RUN {run_idx} | {model_name} COMPLETE")
        print(f"✓ Final model saved: {final_path}")
        print(f"{'='*60}")

print("\n" + "="*80)
print("ALL 4 RUNS COMPLETED!")
print("="*80)
print(f"Models saved in: results_ae_cluster_difficulty_fedavg/")
print(f"  - run_1/ through run_{NUM_RUNS}/")
print(f"  - Each round saved as: <model>_round_<N>_ae_cluster_difficulty_fedavg.pt")
print(f"  - Final models: <model>_final_run<X>.pt")
print("="*80)