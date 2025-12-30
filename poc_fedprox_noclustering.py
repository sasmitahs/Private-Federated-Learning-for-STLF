import os
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
# Project imports
from Models import model_fn
from my_utils import load_energy_data_feather, get_weights, set_weights
from AggregationStrategy import average_weights # Assuming this does FedAvg
# -----------------------------
# PARAMETERS
# -----------------------------
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", DEVICE)
MODEL_NAMES = ["dual_cnn_gru_fcnn","dual_cnn_ann_fcnn","dual_simple_ann_fcnn"]
NUM_CLIENTS = 1410
CLIENT_FRAC = 0.15
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
LR = 0.001
DATA_FILE = "train_final.feather"
RESULTS_DIR = "results"
# FedProx parameter
MU = 0.01 # Proximal term coefficient
# -----------------------------
# Helper function to convert batch to tensors
# -----------------------------
def batch_to_tensor(batch_item, device):
    """Convert batch item to tensor and move to device."""
    if isinstance(batch_item, torch.Tensor):
        return batch_item.to(device)
    elif isinstance(batch_item, np.ndarray):
        return torch.from_numpy(batch_item).to(device)
    else:
        return torch.tensor(batch_item).to(device)
# -----------------------------
# Helper function to ensure list of tensors
# -----------------------------
def ensure_tensor_weights(weights_list, device):
    """Ensure a list of weights are tensors on the given device."""
    tensor_weights = []
    for w in weights_list:
        if isinstance(w, np.ndarray):
            w_tensor = torch.from_numpy(w)
        elif isinstance(w, torch.Tensor):
            w_tensor = w.clone()
        else:
            w_tensor = torch.tensor(w)
        tensor_weights.append(w_tensor.to(device))
    return tensor_weights
# -----------------------------
# FedProx Local Training Function
# -----------------------------
def train_local_fedprox(
    model: nn.Module,
    train_loader: DataLoader,
    global_weights: List[torch.Tensor],
    mu: float,
    epochs: int,
    lr: float,
    device: torch.device
) -> Tuple[List[torch.Tensor], List[float]]:
    """Train locally using FedProx (with proximal term)."""
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Move global weights to device, ensuring they are tensors
    global_weights_dev = ensure_tensor_weights(global_weights, device)
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_samples = 0
        for batch in train_loader:
            optimizer.zero_grad()
            # Handle batch format: (x_ts, x_cov, primary_use, y)
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                # Convert each batch element to tensor properly
                x_ts = batch_to_tensor(batch[0], device)
                x_cov = batch_to_tensor(batch[1], device)
                primary_use = batch_to_tensor(batch[2], device)
                y = batch_to_tensor(batch[3], device)
                preds = model(x_ts, x_cov, primary_use)
            else:
                raise ValueError("Expected batch with 4 elements: x_ts, x_cov, primary_use, y")
            # Supervised loss
            loss = criterion(preds, y)
            # Proximal term: (mu/2) * ||w - w_global||^2
            prox_term = 0.0
            for p, w_g in zip(model.parameters(), global_weights_dev):
                prox_term += torch.sum((p - w_g) ** 2)
            prox_term = (mu / 2.0) * prox_term
            total_loss = loss + prox_term
            total_loss.backward()
            optimizer.step()
            batch_size = y.size(0)
            epoch_loss += loss.item() * batch_size
            n_samples += batch_size
        avg_loss = epoch_loss / max(n_samples, 1)
        loss_history.append(avg_loss)
    updated_weights = get_weights(model)
    return updated_weights, loss_history
# -----------------------------
# Power-of-Choice Selector
# -----------------------------
class PowerOfChoiceSelector:
    def __init__(
        self,
        all_client_ids: List[Any],
        base_probs: Optional[np.ndarray] = None,
        d: int = 20,
        m: int = 5,
        min_d: Optional[int] = None,
        decay_d: bool = True,
        decay_rate: float = 0.99,
        device: torch.device = DEVICE,
        n_eval_batches: int = 1,
    ):
        self.client_ids = list(all_client_ids)
        self.K = len(self.client_ids)
        if base_probs is None:
            base_probs = np.ones(self.K) / self.K
        self.base_probs = np.array(base_probs, dtype=float)
        self.base_probs /= self.base_probs.sum()
        self.d = int(max(d, m))
        self.m = int(m)
        self.min_d = int(min_d) if min_d is not None else self.m
        self.decay_d = decay_d
        self.decay_rate = decay_rate
        self.round = 0
        self.device = device
        self.n_eval_batches = max(1, int(n_eval_batches))
        self.cid2pos = {cid: i for i, cid in enumerate(self.client_ids)}
        self.loss_cache: Dict[Any, float] = {}
    def sample_candidates(self) -> List[Any]:
        size = min(self.d, self.K)
        return list(np.random.choice(self.client_ids, size=size, replace=False, p=self.base_probs))
    def estimate_loss_for_client(self, cid: Any, model, global_weights, filepath: str) -> float:
        try:
            train_loader, _ = load_energy_data_feather(cid, filepath=filepath)
        except:
            return 1e6
        try:
            # Ensure global_weights are tensors before setting
            global_weights_tensors = []
            for w in global_weights:
                if isinstance(w, np.ndarray):
                    global_weights_tensors.append(torch.from_numpy(w))
                else:
                    global_weights_tensors.append(w.clone() if isinstance(w, torch.Tensor) else torch.tensor(w))
            set_weights(model, global_weights_tensors)
        except:
            pass
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for _ in range(self.n_eval_batches):
                try:
                    batch = next(iter(train_loader))
                    if len(batch) != 4:
                        continue
                    # Convert batch items to tensors properly
                    x_ts = batch_to_tensor(batch[0], self.device)
                    x_cov = batch_to_tensor(batch[1], self.device)
                    primary_use = batch_to_tensor(batch[2], self.device)
                    y = batch_to_tensor(batch[3], self.device)
                    preds = model(x_ts, x_cov, primary_use)
                    loss = criterion(preds, y).item()
                    total_loss += loss
                    count += 1
                except Exception as e:
                    print(f" Error evaluating client {cid}: {e}")
                    break
        return total_loss / max(count, 1) if count > 0 else 1e6
    def select_active(self, candidate_ids: List[Any], loss_dict: Dict[Any, float]) -> List[Any]:
        sorted_ids = sorted(candidate_ids, key=lambda c: loss_dict.get(c, -1e9), reverse=True)
        return sorted_ids[:self.m]
    def step(self):
        self.round += 1
        if self.decay_d:
            new_d = max(self.min_d, int(self.d * (self.decay_rate ** self.round)))
            new_d = max(new_d, self.m)
            new_d = min(new_d, self.K)
            self.d = new_d
# -----------------------------
# Load Data & Preprocess
# -----------------------------
df = pd.read_feather(DATA_FILE)
client_ids = sorted(df['building_id'].unique())
print(f"Total clients: {len(client_ids)}")
# Preprocessing
cov_df = df.groupby('building_id').agg({
    'air_temperature': 'mean',
    'primary_use': 'first'
}).reset_index()
primary_use_ohe = OneHotEncoder(sparse_output=False)
primary_use_encoded = primary_use_ohe.fit_transform(cov_df[['primary_use']])
air_temp = cov_df[['air_temperature']].values.astype(np.float32)
covariates_per_building = np.concatenate([primary_use_encoded, air_temp], axis=1)
# Resize series to 168
synthetic_length = 168
resized_series = []
for cid in client_ids:
    series = df[df['building_id'] == cid].sort_values('timestamp')['meter_reading'].values.astype(np.float32)
    series = np.nan_to_num(series, nan=0.0)
    if len(series) > synthetic_length:
        series = series[:synthetic_length]
    else:
        series = np.pad(series, (0, synthetic_length - len(series)))
    resized_series.append(series)
resized_series = np.array(resized_series, dtype=np.float32)
def normalize_rows(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (X - means) / stds
normalized_series = normalize_rows(resized_series)
# -----------------------------
# Setup Selector
# -----------------------------
all_client_ids = client_ids
base_probs = np.ones(len(all_client_ids)) / len(all_client_ids)
m = max(1, int(CLIENT_FRAC * len(all_client_ids)))
initial_d = min(len(all_client_ids), max(m * 5, m + 10))
selector = PowerOfChoiceSelector(
    all_client_ids=all_client_ids,
    base_probs=base_probs,
    d=initial_d,
    m=m,
    min_d=m,
    decay_d=True,
    decay_rate=0.98,
    device=DEVICE,
    n_eval_batches=1
)
# -----------------------------
# Model Constructor
# -----------------------------
def make_model(name):
    return model_fn(name).to(DEVICE)
# -----------------------------
# MAIN FEDPROX + PoC LOOP
# -----------------------------
for model_name in MODEL_NAMES:
    print(f"\n=== Starting FedProx + PoC with {model_name} ===")
    global_model = make_model(model_name)
    global_weights = get_weights(global_model)
    os.makedirs(os.path.join(RESULTS_DIR, model_name), exist_ok=True)
    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} | d={selector.d} | m={selector.m} ---")
        # Step 1: Sample candidates
        candidates = selector.sample_candidates()
        print(f" Sampled {len(candidates)} candidates")
        # Step 2: Estimate loss on one batch
        temp_model = make_model(model_name)
        loss_dict = {}
        for cid in tqdm(candidates, desc="Evaluating candidates"):
            loss = selector.estimate_loss_for_client(cid, temp_model, global_weights, DATA_FILE)
            loss_dict[cid] = loss
            selector.loss_cache[cid] = loss
        # Step 3: Select top-m
        active_clients = selector.select_active(candidates, loss_dict)
        print(f" Selected {len(active_clients)} clients: {active_clients[:5]}...")
        # Step 4: Local FedProx training
        local_weights_list = []
        successful_clients = []
        for cid in tqdm(active_clients, desc="Local FedProx"):
            try:
                local_model = make_model(model_name)
                set_weights(local_model, global_weights)
                train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)
                updated_weights, loss_hist = train_local_fedprox(
                    model=local_model,
                    train_loader=train_loader,
                    global_weights=global_weights,
                    mu=MU,
                    epochs=LOCAL_EPOCHS,
                    lr=LR,
                    device=DEVICE
                )
                local_weights_list.append(updated_weights)
                successful_clients.append(cid)
                selector.loss_cache[cid] = loss_hist[-1]
            except Exception as e:
                print(f" [Failed] Client {cid}: {e}")
        # Step 5: Aggregate (FedAvg)
        if len(local_weights_list) > 0:
            global_weights = average_weights(local_weights_list)
            set_weights(global_model, global_weights)
            print(f" Aggregated updates from {len(local_weights_list)} clients")
        else:
            print(" No updates this round")
        # Step 6: Save checkpoint
        ckpt_path = os.path.join(RESULTS_DIR, model_name, f"{model_name}_round_{rnd+1}_poc_nocluster_fedprox.pt")
        torch.save(global_model.state_dict(), ckpt_path)
        print(f" Saved: {ckpt_path}")
        # Step 7: Update selector
        selector.step()
print("\nFedProx + Power-of-Choice Training Completed!")