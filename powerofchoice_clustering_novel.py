# train_4_models_poc.py
# Power-of-Choice + AE clustering
# Trains 4 independent runs → saves models ROUND BY ROUND

import os
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm

# === Your project imports ===
from Models import model_fn
from my_utils import train_model, load_energy_data_feather, get_weights, set_weights
from AggregationStrategy import average_weights

# -----------------------------
# PARAMETERS
# -----------------------------
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
PATIENCE = 3
K_FORCED = 20

# Training / FL params
MODEL_NAME = "dual_cnn_gru_fcnn"
NUM_CLIENTS = 1410
CLIENT_FRAC = 0.15
NUM_ROUNDS = 40
LOCAL_EPOCHS = 5
LR = 0.001
DATA_FILE = "train_final.feather"

# Number of independent runs
NUM_RUNS = 4
RUN_SEEDS = [42, 123, 999, 2024]  # different seed for each run

# SAVE EVERY N ROUNDS (set to 1 to save every round)
SAVE_INTERVAL = 1  # Save every round

# -----------------------------
# Re-usable components (AE, STL, etc.)
# -----------------------------
def generate_synthetic_series(n, length):
    series = []
    for _ in range(n):
        trend = 0.05 * np.arange(length) + np.random.normal(0, 0.1, length)
        seasonal = 0.5 * np.sin(2 * np.pi * np.arange(length) / 24) + np.random.normal(0, 0.05, length)
        resid = np.random.normal(0, 0.1, length)
        series.append(trend + seasonal + resid)
    return np.array(series, dtype=np.float32)

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

def normalize_rows(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (X - means) / stds

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
        z = self.encoder(x)
        return self.decoder(z)

def train_ae(ae, X_train, epochs=10, batch_size=64, lr=1e-3, device=DEVICE, patience=3):
    ae = ae.to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best_loss = float('inf')
    wait = 0
    for ep in range(epochs):
        ae.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = ae(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(X_train)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return ae

# -----------------------------
# Power-of-Choice Selector (FIXED for multi-input model)
# -----------------------------
class PowerOfChoiceSelector:
    def __init__(self, all_client_ids, base_probs=None, d=20, m=5, min_d=None,
                 decay_d=True, decay_rate=0.98, device=DEVICE, n_eval_batches=1):
        self.client_ids = list(all_client_ids)
        self.K = len(self.client_ids)
        self.base_probs = np.ones(self.K)/self.K if base_probs is None else np.array(base_probs)
        self.d = int(max(d, m))
        self.m = int(m)
        self.min_d = int(min_d) if min_d is not None else self.m
        self.decay_d = decay_d
        self.decay_rate = decay_rate
        self.round = 0
        self.device = device
        self.n_eval_batches = max(1, int(n_eval_batches))
        self.cid2pos = {cid: i for i, cid in enumerate(self.client_ids)}
        self.loss_cache = {cid: 1e6 for cid in self.client_ids}

    def sample_candidates(self):
        return list(np.random.choice(self.client_ids, size=min(self.d, self.K), replace=False, p=self.base_probs))

    def estimate_loss_for_client(self, cid, model_ctor, global_weights, filepath):
        try:
            train_loader, _ = load_energy_data_feather(cid, filepath=filepath)
        except Exception as e:
            print(f"Failed to load data for client {cid}: {e}")
            return float(self.loss_cache.get(cid, 1e6))

        model = model_ctor().to(self.device)
        try:
            set_weights(model, global_weights)
        except Exception as e:
            print(f"Failed to set weights for client {cid}: {e}")

        model.eval()
        loss_fn = nn.MSELoss()
        total = 0.0
        seen = 0

        try:
            it = iter(train_loader)
            for _ in range(self.n_eval_batches):
                batch = next(it)

                # Handle the 4-item batch from load_energy_data_feather
                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    X_ts, X_air_temp, X_primary_use, y = [b.to(self.device) for b in batch]
                else:
                    # Fallback or error
                    return float(self.loss_cache.get(cid, 1e6))

                # primary_use is likely categorical → convert to long
                X_primary_use = X_primary_use.long()

                with torch.no_grad():
                    output = model(X_ts, X_air_temp, X_primary_use)  # Correct 3-input call
                    loss = loss_fn(output, y).item()

                total += loss
                seen += 1
                if seen >= self.n_eval_batches:
                    break

        except StopIteration:
            pass
        except Exception as e:
            print(f"Error during loss estimation for client {cid}: {e}")
            return float(self.loss_cache.get(cid, 1e6))

        if seen == 0:
            return float(self.loss_cache.get(cid, 1e6))

        est = total / seen
        self.loss_cache[cid] = est
        return est

    def select_active(self, candidate_ids, loss_dict):
        sorted_cand = sorted(candidate_ids, key=lambda c: loss_dict.get(c, -1e9), reverse=True)
        return sorted_cand[:self.m]

    def step(self):
        self.round += 1
        if self.decay_d:
            new_d = max(self.min_d, int(self.d * (self.decay_rate ** self.round)))
            new_d = max(new_d, self.m)
            new_d = min(new_d, len(self.client_ids))
            self.d = new_d

# -----------------------------
# MAIN LOOP – TRAIN 4 INDEPENDENT MODELS
# -----------------------------
for run_idx in range(1, NUM_RUNS + 1):
    print(f"\n{'='*80}")
    print(f"STARTING RUN {run_idx}/{NUM_RUNS} (seed = {RUN_SEEDS[run_idx-1]})")
    print(f"{'='*80}")

    # --------------------- SET SEED FOR THIS RUN ---------------------
    SEED = RUN_SEEDS[run_idx - 1]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --------------------- DATA & PRE-PROCESSING (once per run) ---------------------
    print("\n[Loading Data & Pre-processing]")
    df = pd.read_feather("train_final.feather")

    # Synthetic data + STL + AEs
    print("Generating synthetic series...")
    n_synthetic = 5000
    synthetic_length = 168
    synthetic_series = generate_synthetic_series(n_synthetic, synthetic_length)
    trend_syn, seasonal_syn, resid_syn = stl_decompose_batch(synthetic_series)
    trend_syn = normalize_rows(trend_syn)
    seasonal_syn = normalize_rows(seasonal_syn)
    resid_syn = normalize_rows(resid_syn)

    print("Training Autoencoders...")
    print("  - Training AE for Trend...")
    ae_trend = train_ae(SimpleAE(synthetic_length, LATENT_TREND), trend_syn,
                        epochs=AE_EPOCHS_TREND, batch_size=AE_BATCH, lr=AE_LR)
    print("  - Training AE for Seasonal...")
    ae_season = train_ae(SimpleAE(synthetic_length, LATENT_SEASONAL), seasonal_syn,
                         epochs=AE_EPOCHS_SEASON, batch_size=AE_BATCH, lr=AE_LR)
    print("  - Training AE for Residual...")
    ae_resid = train_ae(SimpleAE(synthetic_length, LATENT_RESID), resid_syn,
                        epochs=AE_EPOCHS_RESID, batch_size=AE_BATCH, lr=AE_LR)

    # Covariates AE
    print("  - Training AE for Covariates...")
    cov_df = df.groupby('building_id').agg({
        'air_temperature': 'mean',
        'primary_use': 'first'
    }).reset_index()
    primary_use_ohe = OneHotEncoder(sparse_output=False)
    primary_use_encoded = primary_use_ohe.fit_transform(cov_df[['primary_use']]).astype(np.float32)
    air_temp = cov_df[['air_temperature']].values.astype(np.float32)
    covariates_per_building = np.concatenate([primary_use_encoded, air_temp], axis=1)

    ae_covariates = train_ae(SimpleAE(covariates_per_building.shape[1], LATENT_COVAR),
                             covariates_per_building, epochs=AE_EPOCHS_COVAR,
                             batch_size=AE_BATCH, lr=AE_LR)

    # Real series preparation
    print("Encoding real client data...")
    client_ids = df['building_id'].unique()
    series_dict = {cid: np.nan_to_num(df[df['building_id']==cid]
                     .sort_values('timestamp')['meter_reading'].values.astype(np.float32))
                   for cid in client_ids}

    resized_series = []
    for cid in client_ids:
        s = series_dict[cid]
        if len(s) > synthetic_length:
            resized_series.append(s[:synthetic_length])
        else:
            resized_series.append(np.pad(s, (0, synthetic_length - len(s))))
    resized_series = np.array(resized_series)

    trend_real, seasonal_real, resid_real = stl_decompose_batch(resized_series)
    trend_real = normalize_rows(trend_real)
    seasonal_real = normalize_rows(seasonal_real)
    resid_real = normalize_rows(resid_real)

    # Encode everything
    with torch.no_grad():
        Z_tr = ae_trend.encoder(torch.tensor(trend_real).to(DEVICE)).cpu().numpy()
        Z_se = ae_season.encoder(torch.tensor(seasonal_real).to(DEVICE)).cpu().numpy()
        Z_re = ae_resid.encoder(torch.tensor(resid_real).to(DEVICE)).cpu().numpy()
        Z_cov = ae_covariates.encoder(torch.tensor(covariates_per_building).to(DEVICE)).cpu().numpy()

    encodings = np.concatenate([Z_tr, Z_se, Z_re, Z_cov], axis=1)
    encodings_scaled = StandardScaler().fit_transform(encodings)

    # Clustering
    print("Performing K-means clustering...")
    K = min(K_FORCED, len(client_ids))
    km = KMeans(n_clusters=K, n_init=20, random_state=SEED)
    labels = km.fit_predict(encodings_scaled)
    clusters = {f"cluster_{k}": [client_ids[i] for i in range(len(client_ids)) if labels[i]==k]
                for k in range(K)}
    print(f"Clustered {len(client_ids)} clients into {K} groups")

    # --------------------- CLIENT SELECTION SETUP ---------------------
    all_client_ids = sorted(client_ids)
    m = max(1, int(CLIENT_FRAC * len(all_client_ids)))
    initial_d = min(len(all_client_ids), max(m * 5, m + 10))

    selector = PowerOfChoiceSelector(
        all_client_ids=all_client_ids,
        d=initial_d,
        m=m,
        min_d=m,
        decay_d=True,
        decay_rate=0.98,
        device=DEVICE,
        n_eval_batches=1
    )

    def model_ctor():
        return model_fn(MODEL_NAME).to(DEVICE)

    # --------------------- GLOBAL MODEL & TRAINING LOOP ---------------------
    print(f"\n{'='*80}")
    print(f"Starting Federated Training - Run {run_idx}")
    print(f"{'='*80}")
    
    global_model = model_fn(MODEL_NAME).to(DEVICE)
    global_weights = get_weights(global_model)
    
    # Create checkpoint directory for this run
    checkpoint_dir = f"results_poc/run_{run_idx}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} (Run {run_idx}) | d={selector.d} m={selector.m} ---")
        
        # Power-of-Choice: Sample candidates
        candidates = selector.sample_candidates()
        print(f"Sampled {len(candidates)} candidate clients")
        
        # Estimate loss for each candidate
        loss_dict = {}
        for cid in tqdm(candidates, desc="Loss estimation", leave=False):
            loss_dict[cid] = selector.estimate_loss_for_client(cid, model_ctor, global_weights, DATA_FILE)

        # Select top-m clients with highest loss
        active_clients = selector.select_active(candidates, loss_dict)
        print(f"Selected {len(active_clients)} clients for training")

        # Local training on selected clients
        local_weights = []
        for cid in tqdm(active_clients, desc="Local training", leave=False):
            try:
                model = model_fn(MODEL_NAME).to(DEVICE)
                set_weights(model, global_weights)
                train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)
                updated_weights, loss_hist = train_model(
                    model, train_loader, device=DEVICE, learning_rate=LR,
                    loss_fn=None, optimizer_class=optim.Adam, epochs=LOCAL_EPOCHS
                )
                local_weights.append(updated_weights)
                if loss_hist:
                    selector.loss_cache[cid] = float(loss_hist[-1])
            except Exception as e:
                print(f"[WARN] Client {cid} failed: {e}")

        # Aggregate local models
        if local_weights:
            global_weights = average_weights(local_weights)
            set_weights(global_model, global_weights)
            print(f"✓ Aggregated {len(local_weights)} client models")
        else:
            print("[WARN] No valid client updates in this round")

        # Update selector (decay d if configured)
        selector.step()

        # Save model checkpoint every SAVE_INTERVAL rounds
        if (rnd + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{MODEL_NAME}_round_{rnd+1}_poc.pt"
            )
            torch.save({
                'round': rnd + 1,
                'model_state_dict': global_model.state_dict(),
                'run_id': run_idx,
                'seed': SEED,
                'd': selector.d,
                'm': selector.m
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: Round {rnd+1} → {checkpoint_path}")

    # --------------------- SAVE FINAL MODEL FOR THIS RUN ---------------------
    final_path = os.path.join(checkpoint_dir, f"{MODEL_NAME}_final_run{run_idx}.pt")
    torch.save({
        'round': NUM_ROUNDS,
        'model_state_dict': global_model.state_dict(),
        'run_id': run_idx,
        'seed': SEED
    }, final_path)
    print(f"\n{'='*60}")
    print(f"✓ RUN {run_idx} COMPLETE")
    print(f"✓ Final model saved: {final_path}")
    print(f"{'='*60}")

print("\n" + "="*80)
print("ALL 4 RUNS COMPLETED!")
print("="*80)
print(f"Models saved in: results_poc/")
print(f"  - run_1/ through run_{NUM_RUNS}/")
print(f"  - Each round saved as: <model>_round_<N>_poc.pt")
print(f"  - Final models: <model>_final_run<X>.pt")
print("="*80)