# train_4_models_difficulty_no_cluster_no_ae.py
# Trains 4 independent runs of your difficulty-aware FL (no clustering, no AE)
# Saves models ROUND BY ROUND
# Each run uses different seed → different sampling → different final model

import os
import random
from collections import OrderedDict
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# === Your project imports ===
from Models import model_fn
from my_utils import train_model, load_energy_data_feather, get_weights, set_weights
from AggregationStrategy import average_weights

# =============================
# CONFIGURATION
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

MODEL_NAME = "dual_cnn_gru_fcnn"
NUM_ROUNDS = 40
LOCAL_EPOCHS = 5
LR = 0.001
CLIENT_FRAC = 0.15
DATA_FILE = "train_final.feather"
RESULTS_DIR = "results_difficulty_no_cluster_no_ae"

# Number of independent runs
NUM_RUNS = 4
RUN_SEEDS = [42, 123, 8888, 2025]  # different seed for each run

# SAVE EVERY N ROUNDS (set to 1 to save every round)
SAVE_INTERVAL = 1  # Save every round

# =============================
# Difficulty Tracker (your original class - cleaned & fixed)
# =============================
class TimeSeriesDifficultyWeight:
    def __init__(self, num_clients, accumulate_iters=20, device=None):
        self.num_clients = num_clients
        self.device = device if device is not None else DEVICE
        self.last_loss = torch.ones(num_clients, device=self.device)
        self.learn_score = torch.zeros(num_clients, device=self.device)
        self.unlearn_score = torch.zeros(num_clients, device=self.device)
        self.ema_difficulty = torch.ones(num_clients, device=self.device)
        self.accumulate_iters = accumulate_iters
        self.momentum = (accumulate_iters - 1) / accumulate_iters

    def update(self, cid: int, loss_history: List[float]):
        if not loss_history:
            return 1.0
        current_loss = torch.tensor(loss_history[-1], dtype=torch.float32, device=self.device)
        previous_loss = self.last_loss[cid]

        delta = current_loss - previous_loss
        ratio = torch.log((current_loss + 1e-8) / (previous_loss + 1e-8))

        learn = torch.where(delta < 0, -delta * ratio, torch.tensor(0.0, device=self.device))
        unlearn = torch.where(delta >= 0, delta * ratio, torch.tensor(0.0, device=self.device))

        self.learn_score[cid] = self.momentum * self.learn_score[cid] + (1 - self.momentum) * learn
        self.unlearn_score[cid] = self.momentum * self.unlearn_score[cid] + (1 - self.momentum) * unlearn

        diff_ratio = (self.unlearn_score[cid] + 1e-8) / (self.learn_score[cid] + 1e-8)
        self.ema_difficulty[cid] = self.momentum * self.ema_difficulty[cid] + (1 - self.momentum) * diff_ratio
        self.last_loss[cid] = current_loss

        return self.ema_difficulty[cid].item()

    def get_sampling_probabilities(self, min_prob=0.05):
        inv_diff = 1.0 / (self.ema_difficulty + 1e-6)
        probs = inv_diff / inv_diff.sum()
        probs = torch.clamp(probs, min=min_prob)
        return (probs / probs.sum()).cpu().numpy()

    def get_normalized_weights(self, client_indices: List[int]) -> List[float]:
        weights = self.ema_difficulty[client_indices].cpu().numpy()
        total = weights.sum()
        if total == 0:
            return [1.0 / len(client_indices)] * len(client_indices)
        return (weights / total).tolist()


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

    # ------------------ LOAD DATA & CLIENTS ------------------
    print("\n[Loading Data & Discovering Clients]")
    df = pd.read_feather(DATA_FILE)
    client_ids = sorted(df['building_id'].unique())
    num_clients = len(client_ids)
    print(f"Found {num_clients} unique clients.")

    # Mapping: real client ID → index in difficulty tracker
    id2idx = {cid: idx for idx, cid in enumerate(client_ids)}
    idx2id = {idx: cid for cid, idx in id2idx.items()}

    # Initialize fresh difficulty tracker for this run
    difficulty_tracker = TimeSeriesDifficultyWeight(num_clients=num_clients, device=DEVICE)

    # ------------------ GLOBAL MODEL ------------------
    print(f"\n{'='*80}")
    print(f"Starting Federated Training - Run {run_idx}")
    print(f"{'='*80}")
    
    global_model = model_fn(MODEL_NAME).to(DEVICE)
    global_weights = get_weights(global_model)

    # Results directory for this run
    run_dir = os.path.join(RESULTS_DIR, f"run_{run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    # ------------------ TRAINING LOOP ------------------
    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} (Run {run_idx}) ---")

        # Sample clients using difficulty-aware inverse probabilities
        sampling_probs = difficulty_tracker.get_sampling_probabilities(min_prob=0.05)
        n_sample = max(1, int(CLIENT_FRAC * num_clients))
        sampled_indices = np.random.choice(
            len(client_ids), size=n_sample, replace=False, p=sampling_probs
        )
        sampled_clients = [idx2id[idx] for idx in sampled_indices]

        print(f"Sampled {len(sampled_clients)} clients based on difficulty")

        local_weights_list = []
        client_indices_for_agg = []

        for cid in tqdm(sampled_clients, desc="Local training", leave=False):
            try:
                local_model = model_fn(MODEL_NAME).to(DEVICE)
                set_weights(local_model, global_weights)

                train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)

                updated_weights, loss_history = train_model(
                    local_model, train_loader,
                    device=DEVICE,
                    learning_rate=LR,
                    loss_fn=None,
                    optimizer_class=optim.Adam,
                    epochs=LOCAL_EPOCHS
                )

                local_weights_list.append(updated_weights)
                client_indices_for_agg.append(id2idx[cid])

                # Update difficulty tracker
                difficulty_tracker.update(id2idx[cid], loss_history)

            except Exception as e:
                print(f"[WARN] Client {cid} failed: {e}")

        # Aggregate local models
        if local_weights_list:
            # Weighted aggregation based on difficulty
            agg_weights = difficulty_tracker.get_normalized_weights(client_indices_for_agg)
            global_weights = average_weights(local_weights_list, client_weights=agg_weights)
            set_weights(global_model, global_weights)
            print(f"✓ Aggregated {len(local_weights_list)} client models with difficulty-aware weights")
        else:
            print("[WARN] No clients trained this round.")

        # Save model checkpoint every SAVE_INTERVAL rounds
        if (rnd + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(
                run_dir,
                f"{MODEL_NAME}_round_{rnd+1}_difficulty_no_cluster_no_ae.pt"
            )
            torch.save({
                'round': rnd + 1,
                'model_state_dict': global_model.state_dict(),
                'run_id': run_idx,
                'seed': SEED,
                'num_clients_trained': len(local_weights_list)
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: Round {rnd+1} → {checkpoint_path}")

    # ------------------ SAVE FINAL MODEL ------------------
    final_path = os.path.join(run_dir, f"{MODEL_NAME}_final_run{run_idx}.pt")
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

# =============================
# FINAL SUMMARY
# =============================
print("\n" + "="*80)
print("ALL 4 RUNS COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"Models saved in: {RESULTS_DIR}/")
print(f"  - run_1/ through run_{NUM_RUNS}/")
print(f"  - Each round saved as: <model>_round_<N>_difficulty_no_cluster_no_ae.pt")
print(f"  - Final models: <model>_final_run<X>.pt")
print("="*80)
print("\nModel paths:")
for i in range(1, NUM_RUNS + 1):
    print(f"  {RESULTS_DIR}/run_{i}/{MODEL_NAME}_final_run{i}.pt")
