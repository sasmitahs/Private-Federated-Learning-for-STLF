import os
import random
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# === Imports from your local modules (unchanged) ===
# Make sure these modules are on PYTHONPATH / same folder
from Models import model_fn # factory returning a model by name
from my_utils import load_energy_data_feather, get_weights, set_weights
from AggregationStrategy import average_weights
# -----------------------------
# PARAMETERS (tweak as needed)
# -----------------------------
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
# Federated / training config
MODEL_NAMES = ["dual_cnn_gru_fcnn","dual_cnn_ann_fcnn","dual_simple_ann_fcnn"]
NUM_CLIENTS = 1410 # will be inferred from data
CLIENT_FRAC = 0.15
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE = "train_final.feather" # path to the dataframe file used by load_energy_data_feather
RESULTS_DIR = "results"

# FedProx proximal coefficient
MU = 0.01

# -----------------------------
# Load data and build client list
# -----------------------------
print("Loading dataframe to discover clients...")
df = pd.read_feather(DATA_FILE) # must contain 'building_id' column as before
client_ids = sorted(df['building_id'].unique())
NUM_CLIENTS = len(client_ids)
print(f"Found {NUM_CLIENTS} unique clients (buildings).")
# For simplicity: treat all clients as a single cluster
CLUSTERS = {"all_clients": list(client_ids)}
# Create results directories for models
for model_name in MODEL_NAMES:
    os.makedirs(os.path.join(RESULTS_DIR, model_name, "all_clients"), exist_ok=True)
# -----------------------------
# Difficulty Tracker (kept for potential future use, but not used for sampling/aggregation)
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
    def get_hard_sampling_probabilities(self, min_prob=0.05):
        difficulty = self.ema_difficulty
        probs = difficulty / difficulty.sum()
        probs = torch.clamp(probs, min=min_prob)
        return (probs / probs.sum()).cpu().numpy()
    def get_mixed_sampling_probabilities(self, alpha=0.7, min_prob=0.01):
        diff = self.ema_difficulty
        inv_diff = 1.0 / (diff + 1e-6)
        hard_probs = diff / diff.sum()
        easy_probs = inv_diff / inv_diff.sum()
        probs = alpha * hard_probs + (1 - alpha) * easy_probs
        probs = torch.clamp(probs, min=min_prob)
        return (probs / probs.sum()).cpu().numpy()
    def get_top_clients(self, top_k=10):
        scores = self.ema_difficulty.detach().cpu().numpy()
        idx = np.argsort(-scores)[:top_k]
        return idx, scores[idx]
# Build ID mappings (real client IDs <-> tracker indices)
all_client_ids = sorted(CLUSTERS["all_clients"])
id2idx = {cid: idx for idx, cid in enumerate(all_client_ids)}
idx2id = {idx: cid for cid, idx in id2idx.items()}
# Initialize difficulty tracker
difficulty_tracker = TimeSeriesDifficultyWeight(num_clients=len(all_client_ids))
# -----------------------------
# FedProx local training
# -----------------------------
def train_local_fedprox(
    model: nn.Module,
    train_loader: DataLoader,
    global_weights: List[torch.Tensor],
    mu: float,
    epochs: int,
    lr: float,
    device: torch.device,
    verbose: bool = False
) -> Tuple[List[torch.Tensor], List[float]]:
    """Train locally with FedProx proximal term."""
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert global_weights to tensors if they are numpy arrays, then to device
    if isinstance(global_weights[0], np.ndarray):
        global_weights_device = [torch.tensor(w, dtype=torch.float32).to(device) for w in global_weights]
    else:
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

            batch_size = y.shape[0]
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        epoch_loss = running_loss / max(1, n_samples)
        loss_history.append(epoch_loss)
        if verbose:
            print(f"  FedProx epoch {ep+1}/{epochs}, supervised_loss={epoch_loss:.6f}, prox_term={prox_term.item():.6f}")

    updated_weights = get_weights(model)
    return updated_weights, loss_history

# -----------------------------
# Global training loop (single "cluster" containing all clients)
# -----------------------------
for model_name in MODEL_NAMES:
    print(f"Starting experiment with model: {model_name}")
    # Initialize a single global model for all clients
    global_model = model_fn(model_name).to(DEVICE)
    global_weights = get_weights(global_model)
    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} ---")
        sampled_clients = []
        # Because we have a single cluster, sample from the single set "all_clients"
        client_ids_in_cluster = np.array([id2idx[cid] for cid in CLUSTERS["all_clients"]])
        # Random uniform sampling probabilities across all clients
        n_sample = max(1, int(CLIENT_FRAC * len(client_ids_in_cluster)))
        sampled_indices = np.random.choice(client_ids_in_cluster, size=n_sample, replace=False)
        sampled_clients = [idx2id[idx] for idx in sampled_indices]
        print(f"Sampled total {len(sampled_clients)} clients (from all_clients)")
        # --- Local training on sampled clients ---
        local_weights = []
        for cid in tqdm(sampled_clients, desc="Local FedProx training"):
            # create a fresh model and load global weights
            local_model = model_fn(model_name).to(DEVICE)
            set_weights(local_model, global_weights)
            # load client's training data (your helper must return train_loader, val_loader or similar)
            train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)
            # train locally with FedProx
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
            local_weights.append(updated_weights)
            # update difficulty tracker (internal index) - kept for consistency, though not used
            difficulty_tracker.update(id2idx[cid], loss_history)
        # --- Uniform weighted aggregation (random sampling implies equal weights) ---
        num_sampled = len(sampled_clients)
        if num_sampled > 0:
            uniform_weights = [1.0 / num_sampled] * num_sampled
            global_weights = average_weights(local_weights, client_weights=uniform_weights)
            set_weights(global_model, global_weights)
        # --- Save checkpoint ---
        checkpoint_dir = os.path.join(RESULTS_DIR, model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_round_{rnd+1}_no-cluster_random_fedprox.pt")
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"✅ Saved global model to {checkpoint_path}")
print("Training completed.")