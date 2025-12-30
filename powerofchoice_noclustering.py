# train_4_models_poc_nocluster.py
# Trains the same model 4 times with different seeds using Power-of-Choice (no clustering)
# Saves models ROUND BY ROUND

import os
import random
import numpy as np
import pandas as pd
from typing import List, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

# Project-specific imports
from Models import model_fn
from my_utils import train_model, load_energy_data_feather, get_weights, set_weights
from AggregationStrategy import average_weights

# =============================
# CONFIGURATION
# =============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", DEVICE)

MODEL_NAME = "dual_cnn_gru_fcnn"
NUM_ROUNDS = 40
LOCAL_EPOCHS = 5
LR = 0.001
CLIENT_FRAC = 0.15
DATA_FILE = "train_final.feather"
SYNTHETIC_LENGTH = 168

# Number of independent runs
NUM_RUNS = 4
RUN_SEEDS = [42, 123, 2025, 8888]   # different seed per run

# SAVE EVERY N ROUNDS (set to 1 to save every round)
SAVE_INTERVAL = 1  # Save every round

# =============================
# Power-of-Choice Selector (unchanged)
# =============================
class PowerOfChoiceSelector:
    def __init__(
        self,
        all_client_ids: List[Any],
        base_probs: Optional[np.ndarray] = None,
        d: int = 20,
        m: int = 5,
        min_d: Optional[int] = None,
        decay_d: bool = True,
        decay_rate: float = 0.98,
        device: torch.device = DEVICE,
        n_eval_batches: int = 1,
    ):
        self.client_ids = list(all_client_ids)
        self.K = len(self.client_ids)
        self.base_probs = np.ones(self.K)/self.K if base_probs is None else np.array(base_probs, dtype=float)
        assert len(self.base_probs) == self.K

        self.d = int(max(d, m))
        self.m = int(m)
        self.min_d = int(min_d) if min_d is not None else self.m
        self.decay_d = decay_d
        self.decay_rate = decay_rate
        self.round = 0
        self.device = device
        self.n_eval_batches = max(1, int(n_eval_batches))
        self.loss_cache = {cid: 1e6 for cid in self.client_ids}  # large init → picked early

    def sample_candidates(self) -> List[Any]:
        size = min(self.d, self.K)
        return list(np.random.choice(self.client_ids, size=size, replace=False, p=self.base_probs))

    def estimate_loss_for_client(self, cid, model_ctor, global_weights, filepath):
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
                # Expected format from load_energy_data_feather: (x_ts, x_cov, primary_use, y)
                if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                    x_ts, x_cov, pu, y = (t.to(self.device) for t in batch[:4])
                else:
                    return float(self.loss_cache.get(cid, 1e6))

                with torch.no_grad():
                    preds = model(x_ts, x_cov, pu)
                    loss = loss_fn(preds, y).item()
                total_loss += loss
                seen += 1
        except StopIteration:
            pass
        except Exception:
            return float(self.loss_cache.get(cid, 1e6))

        if seen == 0:
            return float(self.loss_cache.get(cid, 1e6))

        est_loss = total_loss / seen
        self.loss_cache[cid] = est_loss
        return est_loss

    def select_active(self, candidate_ids, loss_dict):
        sorted_ids = sorted(candidate_ids, key=lambda c: loss_dict.get(c, -1e9), reverse=True)
        return sorted_ids[:self.m]

    def step(self):
        self.round += 1
        if self.decay_d:
            new_d = int(self.d * (self.decay_rate ** self.round))
            new_d = max(self.min_d, new_d)
            new_d = min(new_d, self.K)
            self.d = new_d


# =============================
# MAIN LOOP – 4 INDEPENDENT RUNS
# =============================
for run_idx in range(1, NUM_RUNS + 1):
    print("\n" + "="*80)
    print(f"STARTING RUN {run_idx}/{NUM_RUNS}  |  SEED = {RUN_SEEDS[run_idx-1]}")
    print("="*80)

    # ------------------ SET SEED ------------------
    SEED = RUN_SEEDS[run_idx - 1]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ------------------ LOAD & PREPROCESS DATA (once per run) ------------------
    print("\n[Loading Data & Pre-processing]")
    df = pd.read_feather(DATA_FILE)
    client_ids = df['building_id'].unique()
    num_clients = len(client_ids)
    print(f"Total clients: {num_clients}")

    # Covariates (air_temperature + primary_use one-hot)
    print("Processing covariates...")
    cov_df = df.groupby('building_id').agg({
        'air_temperature': 'mean',
        'primary_use': 'first'
    }).reset_index()

    ohe = OneHotEncoder(sparse_output=False)
    primary_use_encoded = ohe.fit_transform(cov_df[['primary_use']])
    air_temp_mean = cov_df['air_temperature'].values.astype(np.float32)

    # Time series: resize to fixed length 168
    print("Processing time series...")
    series_dict = {}
    for cid in client_ids:
        series = df[df['building_id'] == cid].sort_values('timestamp')['meter_reading'].values.astype(np.float32)
        series = np.nan_to_num(series, nan=0.0)
        if len(series) > SYNTHETIC_LENGTH:
            series = series[:SYNTHETIC_LENGTH]
        else:
            series = np.pad(series, (0, SYNTHETIC_LENGTH - len(series)))
        series_dict[cid] = series

    # Normalize each series independently
    def normalize_rows(X):
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (X - mean) / std

    all_series = np.stack([series_dict[cid] for cid in client_ids])
    normalized_series = normalize_rows(all_series)  # [N, 168]
    print("✓ Data pre-processing complete")

    # ------------------ CLIENT SELECTION SETUP ------------------
    m = max(1, int(CLIENT_FRAC * num_clients))
    initial_d = min(num_clients, max(m * 5, m + 10))

    selector = PowerOfChoiceSelector(
        all_client_ids=client_ids,
        d=initial_d,
        m=m,
        min_d=m,
        decay_d=True,
        decay_rate=0.98,
        device=DEVICE,
        n_eval_batches=1
    )
    print(f"Power-of-Choice initialized: d={initial_d}, m={m}")

    def model_ctor():
        return model_fn(MODEL_NAME).to(DEVICE)

    # ------------------ GLOBAL MODEL & TRAINING LOOP ------------------
    print(f"\n{'='*80}")
    print(f"Starting Federated Training - Run {run_idx}")
    print(f"{'='*80}")
    
    global_model = model_fn(MODEL_NAME).to(DEVICE)
    global_weights = get_weights(global_model)

    # Create checkpoint directory for this run
    results_dir = f"results_nocluster/run_{run_idx}"
    os.makedirs(results_dir, exist_ok=True)

    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} (Run {run_idx}) | d={selector.d} m={selector.m} ---")

        # Sample candidates
        candidates = selector.sample_candidates()
        print(f"Sampled {len(candidates)} candidate clients")
        
        # Estimate loss for candidates
        loss_dict = {}
        for cid in tqdm(candidates, desc="Estimating losses", leave=False):
            loss_dict[cid] = selector.estimate_loss_for_client(cid, model_ctor, global_weights, DATA_FILE)

        # Select top-m clients with highest loss
        active_clients = selector.select_active(candidates, loss_dict)
        print(f"Selected {len(active_clients)} clients for training")

        # Local training
        local_weights_list = []
        for cid in tqdm(active_clients, desc="Local training", leave=False):
            try:
                model = model_fn(MODEL_NAME).to(DEVICE)
                set_weights(model, global_weights)
                train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)

                updated_weights, loss_hist = train_model(
                    model, train_loader,
                    device=DEVICE,
                    learning_rate=LR,
                    loss_fn=None,
                    optimizer_class=optim.Adam,
                    epochs=LOCAL_EPOCHS
                )
                local_weights_list.append(updated_weights)

                if loss_hist and len(loss_hist) > 0:
                    selector.loss_cache[cid] = float(loss_hist[-1])
            except Exception as e:
                print(f"[WARN] Client {cid} failed: {e}")

        # Aggregate local models
        if local_weights_list:
            global_weights = average_weights(local_weights_list)
            set_weights(global_model, global_weights)
            print(f"✓ Aggregated {len(local_weights_list)} client models")
        else:
            print("[WARN] No valid client updates in this round")

        # Update selector (decay d)
        selector.step()

        # Save model checkpoint every SAVE_INTERVAL rounds
        if (rnd + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(
                results_dir,
                f"{MODEL_NAME}_round_{rnd+1}_poc_nocluster.pt"
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

    # ------------------ SAVE FINAL MODEL FOR THIS RUN ------------------
    final_path = os.path.join(results_dir, f"{MODEL_NAME}_final_run{run_idx}.pt")
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
print(f"Models saved in: results_nocluster/")
print(f"  - run_1/ through run_{NUM_RUNS}/")
print(f"  - Each round saved as: <model>_round_<N>_poc_nocluster.pt")
print(f"  - Final models: <model>_final_run<X>.pt")
print("="*80)
print("\nModel paths:")
for i in range(1, NUM_RUNS + 1):
    print(f"  results_nocluster/run_{i}/{MODEL_NAME}_final_run{i}.pt")