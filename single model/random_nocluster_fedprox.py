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
MODEL_NAMES = ["gru","simple_cnn","moe_lstm", "lstm", "simple_ann", "cnn_gru_no_cov", ]
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
# Global training loop with RANDOM SAMPLING
# -----------------------------
for model_name in MODEL_NAMES:
    print(f"Starting experiment with model: {model_name}")
    global_model = model_fn(model_name).to(DEVICE)
    global_weights = get_weights(global_model)
    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} ---")
        # Sample clients randomly from the single cluster
        all_clients = CLUSTERS["all_clients"]
        n_sample = max(1, int(CLIENT_FRAC * len(all_clients)))
        # RANDOM sampling (uniform probability)
        sampled_clients = random.sample(all_clients, n_sample)
        print(f"Sampled {len(sampled_clients)} clients (random sampling)")
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
        # Standard FedAvg aggregation (uniform weights)
        global_weights = average_weights(local_weights)
        set_weights(global_model, global_weights)
        # Save checkpoint
        checkpoint_dir = os.path.join("results", model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_round_{rnd+1}_random_no_cluster_fedprox.pt")
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"✅ Saved global model to {checkpoint_path}")