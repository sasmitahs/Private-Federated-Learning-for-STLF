import os
import math
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE = "train_final.feather"
BATCH_SIZE = 64
HISTORY = 24 * 7  # past 7 days hourly
HORIZON = 24  # forecast next 24 hours
TIME_EMB_H = 4
TIME_EMB_DR = 8
LR = 1e-3
EPOCHS = 40
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# -----------------------
# Utilities
# -----------------------
def load_data(feather_path):
    df = pd.read_feather(feather_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['building_id', 'timestamp']).reset_index(drop=True)
    # Fill NaNs
    df['meter_reading'] = df['meter_reading'].fillna(0.0)
    df['air_temperature'] = df['air_temperature'].fillna(df['air_temperature'].mean())
    return df

def timestamp_to_float(ts):
    vals = ts.values
    secs = vals.astype('datetime64[s]').astype(np.int64).astype(np.float32)
    # normalize to reduce scale
    secs = (secs - secs.mean()) / 1e6
    return secs

def simple_ma_decompose(series, period=24):
    trend = pd.Series(series).rolling(window=period, min_periods=1, center=True).mean().values.astype(np.float32)
    seasonal = np.zeros_like(trend, dtype=np.float32)
    resid = (series - trend).astype(np.float32)
    return trend, seasonal, resid

def simple_ema_decompose(series, span=24):
    trend = pd.Series(series).ewm(span=span, adjust=False).mean().values.astype(np.float32)
    seasonal = np.zeros_like(trend, dtype=np.float32)
    resid = (series - trend).astype(np.float32)
    return trend, seasonal, resid

# -----------------------
# Dataset
# -----------------------
class STLFWindowDataset(Dataset):
    def __init__(self, df, history=HISTORY, horizon=HORIZON, use_decomp=True, use_covariates=True, use_multi_decomp=True):
        self.history = history
        self.horizon = horizon
        self.use_decomp = use_decomp
        self.use_covariates = use_covariates
        self.use_multi_decomp = use_multi_decomp
        self.df = df.reset_index(drop=True).copy()
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['dow'] = self.df['timestamp'].dt.dayofweek
        self.df['pu_id'] = pd.factorize(self.df['primary_use'])[0]
        self.start_idx = 0
        self.end_idx = len(self.df) - (history + horizon) + 1
        self.indices = list(range(max(0, self.end_idx)))
        self.decomp_functions = [
            ('simple_ma', lambda s: simple_ma_decompose(s, period=24)),
            ('simple_ema', lambda s: simple_ema_decompose(s, span=24)),
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        window = self.df.iloc[i:i+self.history]
        future = self.df.iloc[i+self.history:i+self.history+self.horizon]
        series = window['meter_reading'].values.astype(np.float32)
        x_raw = series

        if self.use_decomp:
            decomps = []
            if self.use_multi_decomp:
                for name, func in self.decomp_functions:
                    trend, seasonal, resid = func(series)
                    trend = np.nan_to_num(trend, nan=np.nanmean(trend) if np.any(~np.isnan(trend)) else 0.0)
                    seasonal = np.nan_to_num(seasonal, nan=0.0)
                    resid = np.nan_to_num(resid, nan=0.0)
                    decomps.append(np.stack([trend, seasonal, resid], axis=1))
            else:
                # Use only simple_ma_decompose if use_multi_decomp is False
                trend, seasonal, resid = simple_ma_decompose(series)
                trend = np.nan_to_num(trend, nan=np.nanmean(trend) if np.any(~np.isnan(trend)) else 0.0)
                seasonal = np.nan_to_num(seasonal, nan=0.0)
                resid = np.nan_to_num(resid, nan=0.0)
                decomps.append(np.stack([trend, seasonal, resid], axis=1))
            x_decomp = np.stack(decomps, axis=-1).astype(np.float32)  # (history, 3, num_decomps)
        else:
            x_decomp = np.zeros((self.history, 3, 1), dtype=np.float32)

        # Covariates
        if self.use_covariates:
            covs = []
            covs.append(window['air_temperature'].values.astype(np.float32))
            hours = window['hour'].values
            covs.append(np.stack([np.sin(2*np.pi*hours/24),
                                  np.cos(2*np.pi*hours/24)], axis=-1).astype(np.float32))
            if 'dow' in window.columns:
                dow = np.eye(7)[window['dow'].values].astype(np.float32)
                covs.append(dow)
            else:
                covs.append(np.zeros((self.history, 7), dtype=np.float32))
            cov_list = []
            for c in covs:
                if c.ndim == 1:
                    cov_list.append(c.reshape(-1, 1))
                else:
                    cov_list.append(c)
            covariate_array = np.concatenate(cov_list, axis=-1).astype(np.float32)
        else:
            covariate_array = np.zeros((self.history, 0), dtype=np.float32)

        y = future['meter_reading'].values.astype(np.float32)
        ts_window = timestamp_to_float(window['timestamp'])
        ts_future = timestamp_to_float(future['timestamp'])

        return {
            'x_raw': x_raw,
            'x_decomp': x_decomp,
            'covariates': covariate_array,
            'ts_window': ts_window,
            'ts_future': ts_future,
            'y': y
        }

    @staticmethod
    def collate_fn(batch):
        x_raw = torch.as_tensor(np.stack([b['x_raw'] for b in batch]), dtype=torch.float32)
        x_decomp = torch.as_tensor(np.stack([b['x_decomp'] for b in batch]), dtype=torch.float32)
        covariates = torch.as_tensor(np.stack([b['covariates'] for b in batch]), dtype=torch.float32)
        ts_window = torch.as_tensor(np.stack([b['ts_window'] for b in batch]), dtype=torch.float32)
        ts_future = torch.as_tensor(np.stack([b['ts_future'] for b in batch]), dtype=torch.float32)
        y = torch.as_tensor(np.stack([b['y'] for b in batch]), dtype=torch.float32)
        return {
            'x_raw': x_raw,
            'x_decomp': x_decomp,
            'covariates': covariates,
            'ts_window': ts_window,
            'ts_future': ts_future,
            'y': y
        }

# -----------------------
# Model Components
# -----------------------
class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear = nn.Linear(1, emb_dim)

    def forward(self, t):
        return self.linear(t.unsqueeze(-1))

class RawEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_emb_dim=0):
        super().__init__()
        self.gru = nn.GRU(input_dim+time_emb_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x, time_emb):
        enc_in = torch.cat([x, time_emb], dim=-1) if time_emb is not None else x
        out, _ = self.gru(enc_in)
        return out

class CovariateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, cov):
        out, _ = self.gru(cov)
        return out

class Decoder(nn.Module):
    def __init__(self, in_dim, out_horizon):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_horizon)

    def forward(self, h):
        return self.fc(h)

class DualStreamModel(nn.Module):
    def __init__(self, time_emb_h, time_emb_dr, raw_hidden, cov_hidden, cov_dim,
                 use_time_attn=True, use_decomp=True, single_stream=False, use_covariates=True, num_decomps=2):
        super().__init__()
        self.use_time_attn = use_time_attn
        self.use_decomp = use_decomp
        self.single_stream = single_stream
        self.use_covariates = use_covariates
        self.num_decomps = num_decomps
        in_raw = 1 + (3 if use_decomp else 0)
        self.time_emb = TimeEmbedding(time_emb_h) if use_time_attn else None
        self.raw_encoder = RawEncoder(in_raw, raw_hidden, time_emb_dim=(time_emb_h if use_time_attn else 0))
        self.cov_encoder = CovariateEncoder(cov_dim, cov_hidden) if (use_covariates and cov_dim > 0) else None
        self.decomp_weights = nn.Parameter(torch.ones(self.num_decomps)) if use_decomp else None
        dec_in = raw_hidden*2 if single_stream or not use_covariates else raw_hidden*2 + cov_hidden*2
        self.decoder = Decoder(dec_in, HORIZON)

    def forward(self, x_raw, x_decomp, covariates, ts_window, ts_future):
        xr = x_raw.unsqueeze(-1)
        if self.use_decomp:
            softmax_w = F.softmax(self.decomp_weights, dim=0) if self.decomp_weights is not None else None
            x_decomp_weighted = torch.matmul(x_decomp, softmax_w) if softmax_w is not None else x_decomp.squeeze(-1)  # (b, t, 3, num_decomps) @ (num_decomps,) -> (b, t, 3)
            x_in = torch.cat([xr, x_decomp_weighted], dim=-1)
        else:
            x_in = xr
        time_emb = self.time_emb(ts_window) if self.use_time_attn else None
        h_raw = self.raw_encoder(x_in, time_emb)
        h_raw = h_raw.mean(dim=1)
        if self.cov_encoder is not None:
            h_cov = self.cov_encoder(covariates)
            h_cov = h_cov.mean(dim=1)
            h = torch.cat([h_raw, h_cov], dim=-1)
        else:
            h = h_raw
        return self.decoder(h)

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
            for k in batch:
                batch[k] = batch[k].to(device)
            pred = model(batch['x_raw'], batch['x_decomp'], batch['covariates'],
                         batch['ts_window'], batch['ts_future'])
            if torch.isnan(pred).any():
                print("Warning: NaNs in predictions")
            loss = mse_loss(pred, batch['y'])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Fixed typo here
            opt.step()
            losses.append(loss.item())
        val_loss = evaluate_model(model, val_loader, device)
        print(f"Epoch {ep+1}/{epochs} | Train {np.mean(losses):.6f} | Val {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
    return model, best_val

def evaluate_model(model, dataloader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            for k in batch:
                batch[k] = batch[k].to(device)
            pred = model(batch['x_raw'], batch['x_decomp'], batch['covariates'],
                         batch['ts_window'], batch['ts_future'])
            losses.append(mse_loss(pred, batch['y']).item())
    return np.mean(losses) if losses else float('nan')

# -----------------------
# Ablation Runner
# -----------------------
def run_ablation(df, ablation_configs, out_csv="ablation_results_central.csv"):
    # Subsample 15% of the dataset
    df_subset = df.sample(frac=0.15, random_state=SEED).reset_index(drop=True)
    results = []
    for name, cfg in ablation_configs.items():
        print("\n========== Running ablation:", name, "==========")
        use_multi_decomp = cfg.get('use_multi_decomp', False)
        dataset = STLFWindowDataset(df_subset, use_decomp=cfg['use_decomp'], use_covariates=cfg['use_covariates'], use_multi_decomp=use_multi_decomp)
        if len(dataset) == 0:
            print(f"No samples for config {name}, skipping.")
            results.append({'config': name, 'test_loss': float('nan')})
            continue
        n_total = len(dataset)
        n_train = int(0.15 * n_total)
        n_val = int(0.015 * n_total)
        n_test = n_total - n_train - n_val
        train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test],
                                                 generator=torch.Generator().manual_seed(SEED))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn)
        sample_item = dataset[0]
        covdim = sample_item['covariates'].shape[1] if sample_item['covariates'].ndim > 1 else 0
        num_decomps = 2 if cfg['use_decomp'] and use_multi_decomp else 1
        model = DualStreamModel(time_emb_h=TIME_EMB_H, time_emb_dr=TIME_EMB_DR,
                                raw_hidden=128, cov_hidden=64, cov_dim=covdim,
                                use_time_attn=cfg['use_time_attn'],
                                use_decomp=cfg['use_decomp'],
                                single_stream=cfg['single_stream'],
                                use_covariates=cfg['use_covariates'],
                                num_decomps=num_decomps)
        model, best_val = train_model(model, train_loader, val_loader, EPOCHS, LR, DEVICE)
        test_loss = evaluate_model(model, test_loader, DEVICE)
        print(f"Ablation {name}: Test MSE = {test_loss:.6f}")
        results.append({'config': name, 'val_loss': best_val, 'test_loss': test_loss})
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("Saved results to", out_csv)

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    df = load_data(DATA_FILE)
    ablation_configs = {
        "base": {"use_time_attn": True, "use_decomp": True, "single_stream": False, "use_covariates": True, "use_multi_decomp": True},
        "no_covariates": {"use_time_attn": True, "use_decomp": True, "single_stream": False, "use_covariates": False, "use_multi_decomp": True},
        "single_stream": {"use_time_attn": True, "use_decomp": True, "single_stream": True, "use_covariates": True, "use_multi_decomp": True},
        "single_decomp": {"use_time_attn": True, "use_decomp": True, "single_stream": False, "use_covariates": True, "use_multi_decomp": False},
    }
    run_ablation(df, ablation_configs, out_csv="ablation_results_central_decomp.csv")