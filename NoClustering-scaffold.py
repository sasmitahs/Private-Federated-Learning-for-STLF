import os
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# === Local utility imports ===
from Models import model_fn
from my_utils import load_energy_data_feather, get_weights, set_weights

# -----------------------------
# CONFIG
# -----------------------------
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAMES = ["dual_cnn_gru_fcnn"]
CLIENT_FRAC = 0.15
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
LR = 1e-3
SERVER_LR = 1.0  # Server learning rate for SCAFFOLD
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE = "train_final.feather"
RESULTS_DIR = "results"

MIN_SAMPLING_PROB = 0.05

# -----------------------------
# Helper functions
# -----------------------------
def average_weights(weight_lists: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Average a list of weight lists (numpy arrays)."""
    avg = []
    for i in range(len(weight_lists[0])):
        avg.append(np.mean([w[i] for w in weight_lists], axis=0))
    return avg

def aggregate_weights_with_difficulty(
    weight_lists: List[List[np.ndarray]], 
    client_weights: List[float]
) -> List[np.ndarray]:
    """Weighted aggregation of model weights using difficulty scores."""
    aggregated = []
    for i in range(len(weight_lists[0])):
        weighted_sum = sum(w[i] * weight for w, weight in zip(weight_lists, client_weights))
        aggregated.append(weighted_sum)
    return aggregated

# -----------------------------
# SCAFFOLD local training
# -----------------------------
def train_model_scaffold(
    local_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    global_weights: List[np.ndarray],
    server_c: List[np.ndarray],
    client_ci: List[np.ndarray],
    device: torch.device,
    learning_rate: float,
    epochs: int,
    loss_fn: nn.Module = None,
    optimizer_class = optim.Adam,
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Train client with SCAFFOLD correction.
    Returns: delta_y, delta_c, new_ci, local_weights, loss_history
    """
    local_model.to(device)
    loss_fn = loss_fn or nn.MSELoss()
    optimizer = optimizer_class(local_model.parameters(), lr=learning_rate)
    loss_history = []
    local_model.train()
    total_steps = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            # Handle multi-input models (x_ts, x_cov, primary_use, y)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 4:  # x_ts, x_cov, primary_use, y
                    x_ts = batch[0].to(device)
                    x_cov = batch[1].to(device)
                    primary_use = batch[2].to(device)
                    y = batch[3].to(device)
                    
                    if y.dim() == 3 and y.shape[-1] == 1:
                        y = y.squeeze(-1)
                    
                    optimizer.zero_grad()
                    output = local_model(x_ts, x_cov, primary_use)
                    
                elif len(batch) == 2:  # x, y
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    
                    if y.dim() == 3 and y.shape[-1] == 1:
                        y = y.squeeze(-1)
                    
                    optimizer.zero_grad()
                    output = local_model(x)
                else:
                    raise ValueError(f"Unexpected batch structure with {len(batch)} elements")
            else:
                raise ValueError("Batch must be a tuple or list")

            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            # ✅ SCAFFOLD correction: adjust each param after normal SGD step
            with torch.no_grad():
                for p, sc_np, ci_np in zip(local_model.parameters(), server_c, client_ci):
                    sc_tensor = torch.tensor(sc_np, dtype=p.dtype, device=p.device)
                    ci_tensor = torch.tensor(ci_np, dtype=p.dtype, device=p.device)
                    p.data -= learning_rate * (sc_tensor - ci_tensor)

            epoch_loss += loss.item()
            total_steps += 1
            n_batches += 1

        avg_epoch_loss = epoch_loss / max(1, n_batches)
        loss_history.append(avg_epoch_loss)
        
        if verbose:
            print(f"  SCAFFOLD epoch {epoch+1}/{epochs}, loss={avg_epoch_loss:.6f}")

    # Compute deltas
    local_weights = get_weights(local_model)
    delta_y = [lw - gw for lw, gw in zip(local_weights, global_weights)]

    # Update client control variate
    K = total_steps
    new_ci = []
    delta_c = []
    
    for gw, lw, ci, sc in zip(global_weights, local_weights, client_ci, server_c):
        ci_new = ci - sc + (gw - lw) / (K * learning_rate)
        new_ci.append(ci_new)
        delta_c.append(ci_new - ci)

    return delta_y, delta_c, new_ci, local_weights, loss_history

# -----------------------------
# Difficulty tracker (unchanged from FedProx)
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
# Build client list
# -----------------------------
print("Discovering clients from data...")
df = pd.read_feather(DATA_FILE)
client_ids = sorted(df['building_id'].unique())
print(f"Found {len(client_ids)} clients.")
CLUSTERS = {"all_clients": list(client_ids)}

all_client_ids = sorted(CLUSTERS["all_clients"])
id2idx = {cid: idx for idx, cid in enumerate(all_client_ids)}
idx2id = {idx: cid for cid, idx in id2idx.items()}
difficulty_tracker = TimeSeriesDifficultyWeight(num_clients=len(all_client_ids))

for model_name in MODEL_NAMES:
    os.makedirs(os.path.join(RESULTS_DIR, model_name, "all_clients"), exist_ok=True)

# -----------------------------
# Global SCAFFOLD training loop
# -----------------------------
for model_name in MODEL_NAMES:
    print(f"\nStarting SCAFFOLD experiment: {model_name}")
    print(f"Using device: {DEVICE}")

    # Initialize global model and control variates
    global_model = model_fn(model_name).to(DEVICE)
    global_weights = get_weights(global_model)
    
    # Initialize server control variate (c) and client control variates (c_i)
    server_c = [np.zeros_like(w) for w in global_weights]
    client_cs = {idx2id[idx]: [np.zeros_like(w) for w in global_weights] 
                 for idx in range(len(all_client_ids))}

    for rnd in range(NUM_ROUNDS):
        print(f"\n=== Round {rnd+1}/{NUM_ROUNDS} ===")

        # Sample clients using difficulty-aware sampling
        cluster_indices = np.array([id2idx[cid] for cid in CLUSTERS["all_clients"]])
        sampling_probs = difficulty_tracker.get_sampling_probabilities(min_prob=MIN_SAMPLING_PROB)[cluster_indices]
        sampling_probs = sampling_probs / sampling_probs.sum()

        n_sample = max(1, int(CLIENT_FRAC * len(cluster_indices)))
        sampled_indices = np.random.choice(cluster_indices, size=n_sample, replace=False, p=sampling_probs)
        sampled_clients = [idx2id[idx] for idx in sampled_indices]
        print(f"Sampled {len(sampled_clients)} clients for this round.")

        # Local training with SCAFFOLD
        local_weight_deltas = []
        local_c_deltas = []
        
        for cid in tqdm(sampled_clients, desc="Local SCAFFOLD training"):
            local_model = model_fn(model_name).to(DEVICE)
            set_weights(local_model, global_weights)

            train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)
            
            delta_y, delta_c, new_ci, local_weights, loss_history = train_model_scaffold(
                local_model=local_model,
                train_loader=train_loader,
                global_weights=global_weights,
                server_c=server_c,
                client_ci=client_cs[cid],
                device=DEVICE,
                learning_rate=LR,
                epochs=LOCAL_EPOCHS,
                loss_fn=None,
                optimizer_class=optim.Adam,
                verbose=False
            )
            
            local_weight_deltas.append(delta_y)
            local_c_deltas.append(delta_c)
            client_cs[cid] = new_ci
            
            # Update difficulty tracker
            difficulty_tracker.update(id2idx[cid], loss_history)

        # === Aggregate weight deltas with difficulty weighting ===
        sampled_internal_indices = [id2idx[cid] for cid in sampled_clients]
        normalized_w = difficulty_tracker.get_normalized_weights(sampled_internal_indices)
        
        # Weighted aggregation of deltas
        mean_delta_y = aggregate_weights_with_difficulty(local_weight_deltas, normalized_w)
        
        # Update global weights
        global_weights = [
            gw + SERVER_LR * dy for gw, dy in zip(global_weights, mean_delta_y)
        ]

        # === Aggregate control variate updates ===
        mean_delta_c = average_weights(local_c_deltas)
        frac = len(sampled_clients) / len(all_client_ids)
        server_c = [
            sc + frac * dc for sc, dc in zip(server_c, mean_delta_c)
        ]

        # Update global model
        set_weights(global_model, global_weights)

        # Save checkpoint
        checkpoint_dir = os.path.join(RESULTS_DIR, model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f"{model_name}_round_{rnd+1}_no-cluster_no-AE_SCAFFOLD.pt"
        )
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"Saved global checkpoint: {checkpoint_path}")

print("\nSCAFFOLD training completed.")