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

MODEL_NAMES = ["dual_cnn_gru_fcnn","dual_cnn_ann_fcnn","dual_simple_ann_fcnn"]
CLIENT_FRAC = 0.15
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE = "train_final.feather"
RESULTS_DIR = "results"

# FedProx proximal coefficient
MU = 0.01
MIN_SAMPLING_PROB = 0.05

# -----------------------------
# Helper functions
# -----------------------------
def get_tensor_weights(model: nn.Module) -> List[torch.Tensor]:
    """Extract model parameters as a list of tensors."""
    return [p.detach().clone() for p in model.parameters()]

def set_tensor_weights(model: nn.Module, weights: List[torch.Tensor]):
    """Set model parameters from a list of tensors."""
    for p, w in zip(model.parameters(), weights):
        p.data = w.clone().to(p.device)

def aggregate_weights(weight_lists: List[List[torch.Tensor]], client_weights: List[float]) -> List[torch.Tensor]:
    """Weighted aggregation of model weights."""
    aggregated = []
    for i in range(len(weight_lists[0])):
        weighted_sum = sum(w[i].to(DEVICE) * weight for w, weight in zip(weight_lists, client_weights))
        aggregated.append(weighted_sum)
    return aggregated

# -----------------------------
# FedProx local training
# -----------------------------
def train_local_fedprox(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
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

    # Move global weights to device for proximal term computation
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

            batch_size = x_ts.shape[0] if 'x_ts' in locals() else x.shape[0]
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        epoch_loss = running_loss / max(1, n_samples)
        loss_history.append(epoch_loss)
        if verbose:
            print(f"  FedProx epoch {ep+1}/{epochs}, supervised_loss={epoch_loss:.6f}, prox_term={prox_term.item():.6f}")

    updated_weights = get_tensor_weights(model)
    return updated_weights, loss_history

# -----------------------------
# Difficulty tracker
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
# Global FedProx training loop
# -----------------------------
for model_name in MODEL_NAMES:
    print(f"Starting FedProx experiment: {model_name}")

    global_model = model_fn(model_name).to(DEVICE)
    global_weights = get_tensor_weights(global_model)

    for rnd in range(NUM_ROUNDS):
        print(f"\n=== Round {rnd+1}/{NUM_ROUNDS} ===")

        cluster_indices = np.array([id2idx[cid] for cid in CLUSTERS["all_clients"]])
        sampling_probs = difficulty_tracker.get_sampling_probabilities(min_prob=MIN_SAMPLING_PROB)[cluster_indices]
        sampling_probs = sampling_probs / sampling_probs.sum()

        n_sample = max(1, int(CLIENT_FRAC * len(cluster_indices)))
        sampled_indices = np.random.choice(cluster_indices, size=n_sample, replace=False, p=sampling_probs)
        sampled_clients = [idx2id[idx] for idx in sampled_indices]
        print(f"Sampled {len(sampled_clients)} clients for this round.")

        local_weights_list = []
        for cid in tqdm(sampled_clients, desc="Local FedProx training"):
            local_model = model_fn(model_name).to(DEVICE)
            set_tensor_weights(local_model, global_weights)

            train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)
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
            local_weights_list.append(updated_weights)
            difficulty_tracker.update(id2idx[cid], loss_history)

        # Difficulty-aware weighted aggregation
        sampled_internal_indices = [id2idx[cid] for cid in sampled_clients]
        normalized_w = difficulty_tracker.get_normalized_weights(sampled_internal_indices)
        global_weights = aggregate_weights(local_weights_list, normalized_w)
        set_tensor_weights(global_model, global_weights)

        checkpoint_dir = os.path.join(RESULTS_DIR, model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_round_{rnd+1}_no-cluster_no-AE_FedProx.pt")
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"Saved global checkpoint: {checkpoint_path}")

print("FedProx training completed.")