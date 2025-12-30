
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from Models import model_fn
from tqdm import tqdm
from my_utils import train_model, load_energy_data_feather, load_energy_data_hour_feather, get_weights, set_weights
from AggregationStrategy import sync_aggregate, average_weights, sync_aggregate_norm, sync_aggregate_softmax, fedavgm_update
import os

# Config
MODEL_NAMES = ["dual_gru_fcnn"]
NUM_CLIENTS = 1410
CLIENT_FRAC = 0.15
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE = "train_final.feather"

for model_name in MODEL_NAMES:
    print(f"Starting experiment with model: {model_name}")

    # Directory to save checkpoints
    model_dir = os.path.join("results", model_name)
    os.makedirs(model_dir, exist_ok=True)
    global_model = model_fn(model_name).to(DEVICE)

    sampled_clients = list(range(NUM_CLIENTS))

    for cid in tqdm(sampled_clients, desc="Training clients"):
        # Use hour variant for models with _hour suffix
        if model_name.endswith("_hour"):
            train_loader, _ = load_energy_data_hour_feather(cid, filepath=DATA_FILE)
        else:
            train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)

        updated_weights, fin_loss = train_model(
            global_model,
            train_loader,
            device=DEVICE,
            learning_rate=LR,
            loss_fn=None,
            optimizer_class=optim.Adam,
            epochs=LOCAL_EPOCHS
        )

    checkpoint_path = os.path.join(model_dir, f"{model_name}_global_model.pt")
    torch.save(global_model.state_dict(), checkpoint_path)
    print(f"Saved global model to {checkpoint_path}")