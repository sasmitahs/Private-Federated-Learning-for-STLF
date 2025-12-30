import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
from typing import List, Tuple
from tqdm import tqdm
from my_utils import load_energy_data_feather, get_weights, set_weights
from Models import model_fn

# -----------------------------
# PARAMETERS
# -----------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_CLIENTS = 1410
CLIENT_FRAC = 0.15
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
LR = 0.001
SERVER_LR = 1.0  # Server learning rate for SCAFFOLD
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_FILE = "train_final.feather"
MODEL_NAMES = ["dual_cnn_gru_fcnn"]

# Create results directory
for model_name in MODEL_NAMES:
    os.makedirs(os.path.join("results", model_name, "single_cluster"), exist_ok=True)

# -----------------------------
# Helper Functions for Weight Aggregation
# -----------------------------
def average_weight_deltas(delta_lists: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Average a list of delta weight lists (numpy arrays)."""
    avg = []
    for i in range(len(delta_lists[0])):
        avg.append(np.mean([d[i] for d in delta_lists], axis=0))
    return avg

def aggregate_deltas_with_difficulty(
    delta_lists: List[List[np.ndarray]], 
    client_weights: List[float]
) -> List[np.ndarray]:
    """Weighted aggregation of weight deltas using difficulty scores."""
    aggregated = []
    for i in range(len(delta_lists[0])):
        weighted_sum = sum(d[i] * weight for d, weight in zip(delta_lists, client_weights))
        aggregated.append(weighted_sum)
    return aggregated

# -----------------------------
# SCAFFOLD Local Training Function
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
    optimizer_class=optim.Adam,
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

    for ep in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
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

            # SCAFFOLD correction: adjust each param after normal SGD step
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
            print(f"  SCAFFOLD epoch {ep+1}/{epochs}, loss={avg_epoch_loss:.6f}")

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
# Difficulty Tracker
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
        """Update difficulty based on loss trend for a client."""
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
        """Return normalized weights proportional to difficulty for given client IDs."""
        weights = [self.ema_difficulty[cid].item() for cid in client_ids]
        total = sum(weights)
        if total == 0:
            return [1.0 / len(client_ids)] * len(client_ids)
        return [w / total for w in weights]

    def get_sampling_probabilities(self, min_prob=0.05):
        """Favors easy clients (inverse difficulty)."""
        difficulty = self.ema_difficulty
        inv_difficulty = 1.0 / (difficulty + 1e-6)
        inv_difficulty = inv_difficulty / inv_difficulty.sum()
        probs = torch.clamp(inv_difficulty, min=min_prob)
        return (probs / probs.sum()).cpu().numpy()

# -----------------------------
# Main Training Loop
# -----------------------------
# Load client IDs from data
df = pd.read_feather(DATA_FILE)
all_client_ids = sorted(df['building_id'].unique())

# Build ID mappings (real client IDs <-> tracker indices)
id2idx = {cid: idx for idx, cid in enumerate(all_client_ids)}
idx2id = {idx: cid for cid, idx in id2idx.items()}

# Initialize difficulty tracker
difficulty_tracker = TimeSeriesDifficultyWeight(num_clients=len(all_client_ids))

for model_name in MODEL_NAMES:
    print(f"\n{'='*60}")
    print(f"Starting SCAFFOLD experiment with model: {model_name}")
    print(f"Using device: {DEVICE}")
    print(f"{'='*60}")

    # Initialize global model
    global_model = model_fn(model_name).to(DEVICE)
    global_weights = get_weights(global_model)

    # Initialize server and client control variates
    server_c = [np.zeros_like(w) for w in global_weights]
    client_cs = {cid: [np.zeros_like(w) for w in global_weights] for cid in all_client_ids}

    for rnd in range(NUM_ROUNDS):
        print(f"\n=== Round {rnd+1}/{NUM_ROUNDS} ===")

        # Sample clients using difficulty-aware probabilities
        probs = difficulty_tracker.get_sampling_probabilities(min_prob=0.05)
        n_sample = max(1, int(CLIENT_FRAC * len(all_client_ids)))
        sampled_indices = np.random.choice(
            range(len(all_client_ids)),
            size=n_sample,
            replace=False,
            p=probs
        )
        sampled_clients = [idx2id[idx] for idx in sampled_indices]
        print(f"Sampled {len(sampled_clients)} clients")

        # Local SCAFFOLD training
        local_weight_deltas = []
        local_c_deltas = []

        for cid in tqdm(sampled_clients, desc="Local SCAFFOLD training"):
            model = model_fn(model_name).to(DEVICE)
            set_weights(model, global_weights)

            train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)

            delta_y, delta_c, new_ci, local_weights, loss_history = train_model_scaffold(
                local_model=model,
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

        # Aggregate weight deltas with difficulty weighting
        sampled_internal_indices = [id2idx[cid] for cid in sampled_clients]
        normalized_weights = difficulty_tracker.get_normalized_weights(sampled_internal_indices)
        mean_delta_y = aggregate_deltas_with_difficulty(local_weight_deltas, normalized_weights)

        # Update global weights
        global_weights = [
            gw + SERVER_LR * dy for gw, dy in zip(global_weights, mean_delta_y)
        ]

        # Aggregate control variate updates
        mean_delta_c = average_weight_deltas(local_c_deltas)
        frac = len(sampled_clients) / len(all_client_ids)
        server_c = [
            sc + frac * dc for sc, dc in zip(server_c, mean_delta_c)
        ]

        # Update global model
        set_weights(global_model, global_weights)

        # Save checkpoint
        checkpoint_dir = os.path.join("results", model_name, "single_cluster")
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{model_name}_round_{rnd+1}_SCAFFOLD_nocluster_correct.pt"
        )
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"✅ Saved global model to {checkpoint_path}")

print("\n" + "="*60)
print("SCAFFOLD training completed!")
print("="*60)