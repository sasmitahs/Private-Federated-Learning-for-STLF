import os
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Models import model_fn
from my_utils import train_model, load_energy_data_feather, get_weights, set_weights
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
MODEL_NAMES = ["simple_cnn","cnn_gru_no_cov","cnn_gru", "simple_ann", "gru","lstm","moe_lstm"]
NUM_CLIENTS = 1410
CLIENT_FRAC = 0.15
NUM_ROUNDS = 40
LOCAL_EPOCHS = 5
LR = 0.001
DATA_FILE = "train_final.feather"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_feather(DATA_FILE)
client_ids = df['building_id'].unique()
CLUSTERS = {"all_clients": list(client_ids)}  # Treat all clients as one cluster

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

            updated_weights, loss_history = train_model(
                model, train_loader,
                device=DEVICE, learning_rate=LR,
                loss_fn=None, optimizer_class=optim.Adam,
                epochs=LOCAL_EPOCHS
            )
            local_weights.append(updated_weights)

        # Standard FedAvg aggregation (uniform weights)
        global_weights = average_weights(local_weights)
        set_weights(global_model, global_weights)

        # Save checkpoint
        checkpoint_dir = os.path.join("results", model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_round_{rnd+1}_random_no_cluster.pt")
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"✅ Saved global model to {checkpoint_path}")