import os
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Models import model_fn
from my_utils import load_energy_data_feather, get_weights, set_weights
from AggregationStrategy import average_weights
# -----------------------------
# PARAMETERS
# -----------------------------
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
# Training / FL params
MODEL_NAMES = ["gru","simple_cnn","moe_lstm", "lstm", "simple_ann", "cnn_gru_no_cov" ]
NUM_CLIENTS = 1410
CLIENT_FRAC = 0.15
NUM_ROUNDS = 40
LOCAL_EPOCHS = 5
LR = 0.001
DATA_FILE = "train_final.feather"
# FedProx proximal coefficient
MU = 0.01
# -----------------------------
# Load data
# -----------------------------
df = pd.read_feather(DATA_FILE)
client_ids = df['building_id'].unique()
CLUSTERS = {"all_clients": list(client_ids)} # Treat all clients as one cluster
# -----------------------------
# Difficulty Weight Tracker
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
            print(f" FedProx epoch {ep+1}/{epochs}, supervised_loss={epoch_loss:.6f}, prox_term={prox_term.item():.6f}")
    updated_weights = get_weights(model)
    return updated_weights, loss_history

# -----------------------------
# Create results directory
# -----------------------------
for model_name in MODEL_NAMES:
    cluster_name = "all_clients"
    os.makedirs(os.path.join("results", model_name, cluster_name), exist_ok=True)

# -----------------------------
# Global training loop
# -----------------------------
all_client_ids = sorted(client_ids)
id2idx = {cid: idx for idx, cid in enumerate(all_client_ids)}
idx2id = {idx: cid for cid, idx in id2idx.items()}
difficulty_tracker = TimeSeriesDifficultyWeight(num_clients=len(all_client_ids))
for model_name in MODEL_NAMES:
    print(f"Starting experiment with model: {model_name}")
    global_model = model_fn(model_name).to(DEVICE)
    global_weights = get_weights(global_model)
    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} ---")
        # Sample clients from the single cluster
        cluster_indices = np.array([id2idx[int(cid)] for cid in CLUSTERS["all_clients"]])
        cluster_probs = difficulty_tracker.get_sampling_probabilities(min_prob=0.05)[cluster_indices]
        cluster_probs = cluster_probs / cluster_probs.sum()
        n_sample = max(1, int(CLIENT_FRAC * len(cluster_indices)))
        sampled_indices = np.random.choice(
            cluster_indices,
            size=n_sample,
            replace=False,
            p=cluster_probs
        )
        sampled_clients = [idx2id[idx] for idx in sampled_indices]
        print(f"Sampled {len(sampled_clients)} clients")
        # Local training
        local_weights = []
        for cid in tqdm(sampled_clients):
            model = model_fn(model_name).to(DEVICE)
            set_weights(model, global_weights)
            train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)
            updated_weights, loss_history = train_local_fedprox(
                model=model,
                train_loader=train_loader,
                global_weights=global_weights,
                mu=MU,
                epochs=LOCAL_EPOCHS,
                lr=LR,
                device=DEVICE,
                verbose=False
            )
            local_weights.append(updated_weights)
            difficulty_tracker.update(id2idx[cid], loss_history)
        # Difficulty-aware weighted aggregation
        normalized_weights = difficulty_tracker.get_normalized_weights([id2idx[cid] for cid in sampled_clients])
        global_weights = average_weights(local_weights, client_weights=normalized_weights)
        set_weights(global_model, global_weights)
        # Save checkpoint
        checkpoint_dir = os.path.join("results", model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_round_{rnd+1}_das_no_cluster_fedprox.pt")
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"✅ Saved global model to {checkpoint_path}")