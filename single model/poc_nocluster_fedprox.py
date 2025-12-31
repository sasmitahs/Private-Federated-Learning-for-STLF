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
print("Device:", DEVICE)
# Training / FL params
MODEL_NAMES = ["simple_ann"]
NUM_CLIENTS = 1410
CLIENT_FRAC = 0.15
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
LR = 0.001
DATA_FILE = "train_final.feather"
# FedProx proximal coefficient
MU = 0.01
# -----------------------------
# Load data
# -----------------------------
df = pd.read_feather("train_final.feather")
all_client_ids = sorted(df['building_id'].unique())
num_all_clients = len(all_client_ids)
print("Total unique clients:", num_all_clients)
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
        decay_rate: float = 0.98,
        device: torch.device = DEVICE,
        n_eval_batches: int = 1,
    ):
        self.client_ids = list(all_client_ids)
        self.K = len(self.client_ids)
        if base_probs is None:
            base_probs = np.ones(self.K) / self.K
        assert len(base_probs) == self.K
        self.base_probs = np.array(base_probs, dtype=float)
        self.d = int(max(d, m))
        self.m = int(m)
        self.min_d = int(min_d) if min_d is not None else self.m
        self.decay_d = decay_d
        self.decay_rate = decay_rate
        self.round = 0
        self.device = device
        self.n_eval_batches = max(1, int(n_eval_batches))
        self.cid2pos = {cid: i for i, cid in enumerate(self.client_ids)}
        self.loss_cache = {cid: 1e6 for cid in self.client_ids}

    def sample_candidates(self) -> List[Any]:
        chosen = np.random.choice(
            self.client_ids,
            size=min(self.d, self.K),
            replace=False,
            p=self.base_probs
        )
        return list(chosen)

    def estimate_loss_for_client(self, cid: Any, model_ctor, global_weights, filepath: str) -> float:
        try:
            train_loader, _ = load_energy_data_feather(cid, filepath=filepath)
        except Exception:
            return float(self.loss_cache.get(cid, 1e6))
        model = model_ctor().to(self.device) if callable(model_ctor) else model_ctor
        try:
            set_weights(model, global_weights)
        except Exception:
            pass
        model.eval()
        loss_fn = nn.MSELoss()
        total_loss = 0.0
        seen = 0
        try:
            it = iter(train_loader)
            for _ in range(self.n_eval_batches):
                batch = next(it)
                # Handle multi-input models (x_ts, x_cov, primary_use, y)
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 4:  # x_ts, x_cov, primary_use, y
                        x_ts = batch[0].to(self.device)
                        x_cov = batch[1].to(self.device)
                        primary_use = batch[2].to(self.device)
                        y = batch[3].to(self.device)
                        with torch.no_grad():
                            preds = model(x_ts, x_cov, primary_use)
                    elif len(batch) == 2:  # x, y
                        x = batch[0].to(self.device)
                        y = batch[1].to(self.device)
                        with torch.no_grad():
                            preds = model(x)
                    else:
                        # Fallback
                        x = batch[0].to(self.device)
                        y = batch[1].to(self.device)
                        with torch.no_grad():
                            preds = model(x)
                else:
                    x = batch.to(self.device)
                    y = batch.to(self.device)
                    with torch.no_grad():
                        preds = model(x)
                loss = loss_fn(preds, y).item()
                total_loss += loss
                seen += 1
        except StopIteration:
            pass
        except Exception:
            return float(self.loss_cache.get(cid, 1e6))
        if seen == 0:
            return float(self.loss_cache.get(cid, 1e6))
        est = float(total_loss / seen)
        self.loss_cache[cid] = est
        return est

    def select_active(self, candidate_ids: List[Any], loss_dict: Dict[Any, float]) -> List[Any]:
        sorted_cand = sorted(candidate_ids, key=lambda cid: loss_dict.get(cid, float(-1e9)), reverse=True)
        active = sorted_cand[:self.m]
        return active

    def step(self):
        self.round += 1
        if self.decay_d:
            new_d = max(self.min_d, int(self.d * (self.decay_rate ** self.round)))
            new_d = max(new_d, self.m)
            new_d = min(new_d, self.K)
            self.d = new_d
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
# Prepare selector
# -----------------------------
base_probs = np.ones(num_all_clients) / num_all_clients
m = max(1, int(CLIENT_FRAC * num_all_clients))
initial_d = min(num_all_clients, max(m * 5, m + 10))
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
# Utility: model constructor wrapper
# -----------------------------
def make_model_for_name(name: str):
    def _ctor():
        model = model_fn(name).to(DEVICE)
        return model
    return _ctor
# -----------------------------
# Global Training Loop with Power-of-Choice
# -----------------------------
for model_name in MODEL_NAMES:
    print(f"Starting experiment with model: {model_name}")
    global_model = model_fn(model_name).to(DEVICE)
    global_weights = get_weights(global_model)
    checkpoint_dir = os.path.join("results", model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_ctor = make_model_for_name(model_name)
    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd+1}/{NUM_ROUNDS} --- d={selector.d} m={selector.m}")
        # 1) Sample candidate set
        candidates = selector.sample_candidates()
        print(f"Sampled {len(candidates)} candidates")
        # 2) Estimate loss for candidates
        loss_dict = {}
        for cid in tqdm(candidates, desc="Estimating candidate losses"):
            est_loss = selector.estimate_loss_for_client(cid, model_ctor, global_weights, filepath=DATA_FILE)
            loss_dict[cid] = est_loss
        # 3) Select top-m clients by loss
        active_clients = selector.select_active(candidates, loss_dict)
        print(f"Selected {len(active_clients)} active clients")
        # 4) Local training
        local_weights = []
        successful_clients = []
        for cid in tqdm(active_clients, desc="Local training"):
            try:
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
                successful_clients.append(cid)
                # Update loss cache with final training loss
                if isinstance(loss_history, (list, tuple)) and len(loss_history) > 0:
                    selector.loss_cache[cid] = float(loss_history[-1])
            except Exception as e:
                print(f"Warning: failed training for client {cid}: {e}")
        # 5) Aggregate
        if len(local_weights) == 0:
            print("No successful client updates this round — skipping aggregation")
        else:
            global_weights = average_weights(local_weights)
            set_weights(global_model, global_weights)
        # 6) Step selector (decay d)
        selector.step()
        # 7) Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_round_{rnd+1}_poc_no-clustering_fedprox.pt")
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"Saved global model to {checkpoint_path}")
print("Training finished.")