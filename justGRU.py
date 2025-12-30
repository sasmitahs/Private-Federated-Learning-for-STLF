import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE = "train_final.feather"

BATCH_SIZE = 64
HISTORY = 24 * 7    # past 7 days hourly
HORIZON = 24        # forecast next 24 hours
LR = 1e-3
EPOCHS = 40
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------
# Utilities
# -----------------------
def load_data(feather_path):
    df = pd.read_feather(feather_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['building_id', 'timestamp']).reset_index(drop=True)
    df['meter_reading'] = df['meter_reading'].fillna(0.0)
    return df

# -----------------------
# Dataset
# -----------------------
class SimpleTimeSeriesDataset(Dataset):
    def __init__(self, df, history=HISTORY, horizon=HORIZON):
        self.history = history
        self.horizon = horizon
        self.df = df.reset_index(drop=True).copy()

        self.start_idx = 0
        self.end_idx = len(self.df) - (history + horizon) + 1
        self.indices = list(range(max(0, self.end_idx)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        window = self.df.iloc[i:i+self.history]
        future = self.df.iloc[i+self.history:i+self.history+self.horizon]

        x_raw = window['meter_reading'].values.astype(np.float32)
        y = future['meter_reading'].values.astype(np.float32)

        return {
            'x_raw': x_raw,
            'y': y
        }

def collate_fn(batch):
    x_raw = torch.as_tensor(np.stack([b['x_raw'] for b in batch]), dtype=torch.float32)
    y = torch.as_tensor(np.stack([b['y'] for b in batch]), dtype=torch.float32)
    return {'x_raw': x_raw, 'y': y}

# -----------------------
# Model
# -----------------------
class SimpleGRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, out_horizon=HORIZON):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, out_horizon)

    def forward(self, x):
        x = x.unsqueeze(-1)  # Add feature dimension: (batch, history) -> (batch, history, 1)
        out, _ = self.gru(x)  # out: (batch, history, hidden_dim * 2)
        out = out[:, -1, :]   # Take last time step: (batch, hidden_dim * 2)
        out = self.fc(out)    # (batch, horizon)
        return out

# -----------------------
# Training / Eval
# -----------------------
def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

def train_model(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    for ep in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            for k in batch: batch[k] = batch[k].to(device)
            pred = model(batch['x_raw'])
            if torch.isnan(pred).any():
                print("Warning: NaNs in predictions")
            loss = mse_loss(pred, batch['y'])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        val_loss = evaluate_model(model, val_loader, device)
        print(f"Epoch {ep+1}/{epochs} | Train MSE {np.mean(losses):.6f} | Val MSE {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
    return model, best_val

def evaluate_model(model, dataloader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            for k in batch: batch[k] = batch[k].to(device)
            pred = model(batch['x_raw'])
            losses.append(mse_loss(pred, batch['y']).item())
    return np.mean(losses) if losses else float('nan')

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    df = load_data(DATA_FILE)
    df_subset = df.sample(frac=0.15, random_state=SEED).reset_index(drop=True)

    dataset = SimpleTimeSeriesDataset(df_subset)
    if len(dataset) == 0:
        print("No samples in dataset, exiting.")
        exit()

    n_total = len(dataset)
    n_train = int(0.15 * n_total)
    n_val = int(0.015 * n_total)
    n_test = n_total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test],
                                             generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = SimpleGRUModel()
    model, best_val = train_model(model, train_loader, val_loader, EPOCHS, LR, DEVICE)
    test_loss = evaluate_model(model, test_loader, DEVICE)
    print(f"Simple GRU Test MSE: {test_loss:.6f}")

    results = [{'config': 'simple_gru', 'val_loss': best_val, 'test_loss': test_loss}]
    pd.DataFrame(results).to_csv('simple_gru_results.csv', index=False)
    print("Saved results to simple_gru_results.csv")