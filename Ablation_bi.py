import os
import math
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler  # Updated import
from tqdm import tqdm

# Optional decomposition
try:
    from statsmodels.tsa.seasonal import STL
    HAVE_STL = True
except Exception:
    HAVE_STL = False

# -----------------------
# Config
# -----------------------
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE = "train_final.feather"

BATCH_SIZE = 16  # Reduced batch size
HISTORY = 24 * 7
HORIZON = 24
TIME_EMB_H = 4
TIME_EMB_DR = 8
LR = 1e-3
EPOCHS = 20
SEED = 42
ACCUM_STEPS = 4  # Gradient accumulation steps

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

def decompose_series(series, period=24):
    if HAVE_STL:
        series = np.nan_to_num(series)
        stl = STL(series, period=period, robust=True)
        res = stl.fit()
        return res.trend.astype(np.float32), res.seasonal.astype(np.float32), res.resid.astype(np.float32)
    else:
        trend = pd.Series(series).rolling(window=period, min_periods=1, center=True).mean().values.astype(np.float32)
        seasonal = np.zeros_like(trend, dtype=np.float32)
        resid = (series - trend).astype(np.float32)
        return trend, seasonal, resid

# -----------------------
# Dataset
# -----------------------
class STLFWindowDataset(Dataset):
    def __init__(self, df, history=HISTORY, horizon=HORIZON, use_decomp=True, use_covariates=True):
        self.history = history
        self.horizon = horizon
        self.use_decomp = use_decomp
        self.use_covariates = use_covariates
        self.df = df.reset_index(drop=True).copy()

        if use_decomp:
            trend, seasonal, resid = decompose_series(self.df['meter_reading'].values, period=24)
            self.df['trend'] = trend
            self.df['seasonal'] = seasonal
            self.df['resid'] = resid
        else:
            self.df['trend'] = 0.0
            self.df['seasonal'] = 0.0
            self.df['resid'] = 0.0

        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['dow'] = self.df['timestamp'].dt.dayofweek
        self.df['pu_id'] = pd.factorize(self.df['primary_use'])[0]

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
        x_decomp = window[['trend', 'seasonal', 'resid']].values.astype(np.float32)

        # Covariates
        if self.use_covariates:
            covs = []
            covs.append(window['air_temperature'].values.astype(np.float32))
            hours = window['hour'].values
            covs.append(np.stack([np.sin(2*np.pi*hours/24),
                                 np.cos(2*np.pi*hours/24)], axis=-1).astype(np.float32))
            # Make sure day-of-week exists
            if 'dow' in window.columns:
                dow = np.eye(7)[window['dow'].values].astype(np.float32)
                covs.append(dow)
            else:
                covs.append(np.zeros((self.history, 7), dtype=np.float32))
            # Only stack the covariates that exist
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

def collate_fn(batch):
    x_raw = torch.as_tensor(np.stack([b['x_raw'] for b in batch]), dtype=torch.float32)
    x_decomp = torch.as_tensor(np.stack([b['x_decomp'] for b in batch]), dtype=torch.float32)
    covariates = torch.as_tensor(np.stack([b['covariates'] for b in batch]), dtype=torch.float32)
    ts_window = torch.as_tensor(np.stack([b['ts_window'] for b in batch]), dtype=torch.float32)
    ts_future = torch.as_tensor(np.stack([b['ts_future'] for b in batch]), dtype=torch.float32)
    y = torch.as_tensor(np.stack([b['y'] for b in batch]), dtype=torch.float32)
    return {'x_raw': x_raw, 'x_decomp': x_decomp, 'covariates': covariates,
            'ts_window': ts_window, 'ts_future': ts_future, 'y': y}

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
    def __init__(self, input_dim, hidden_dim, time_emb_dim=0, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_dim+time_emb_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, time_emb):
        enc_in = torch.cat([x, time_emb], dim=-1) if time_emb is not None else x
        out, _ = self.gru(enc_in)
        return out

class CovariateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)

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
                 use_time_attn=True, use_decomp=True, single_stream=False, use_covariates=True, bidirectional=False):
        super().__init__()
        self.use_time_attn = use_time_attn
        self.use_decomp = use_decomp
        self.single_stream = single_stream
        self.use_covariates = use_covariates
        self.bidirectional = bidirectional

        in_raw = 1 + (3 if use_decomp else 0)
        self.time_emb = TimeEmbedding(time_emb_h) if use_time_attn else None
        self.raw_encoder = RawEncoder(in_raw, raw_hidden, time_emb_dim=(time_emb_h if use_time_attn else 0), bidirectional=bidirectional)
        self.cov_encoder = CovariateEncoder(cov_dim, cov_hidden, bidirectional=bidirectional) if (use_covariates and cov_dim>0) else None

        # Adjust decoder input size based on bidirectional flag
        hidden_multiplier = 2 if bidirectional else 1
        dec_in = (raw_hidden * hidden_multiplier) if single_stream or not use_covariates else (raw_hidden * hidden_multiplier + cov_hidden * hidden_multiplier)
        self.decoder = Decoder(dec_in, HORIZON)

    def forward(self, x_raw, x_decomp, covariates, ts_window, ts_future):
        xr = x_raw.unsqueeze(-1)
        x_in = torch.cat([xr, x_decomp], dim=-1) if self.use_decomp else xr

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

def train_model(model, train_loader, val_loader, epochs, lr, device, accum_steps=4):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    best_val = float('inf')
    for ep in range(epochs):
        model.train()
        losses = []
        opt.zero_grad()
        for i, batch in enumerate(train_loader):
            for k in batch: batch[k] = batch[k].to(device)
            with torch.amp.autocast(device_type='cuda'):  # Updated
                pred = model(batch['x_raw'], batch['x_decomp'], batch['covariates'],
                             batch['ts_window'], batch['ts_future'])
                if torch.isnan(pred).any():
                    print("Warning: NaNs in predictions")
                loss = mse_loss(pred, batch['y']) / accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            losses.append(loss.item() * accum_steps)
        if (i + 1) % accum_steps != 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
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
            for k in batch: batch[k] = batch[k].to(device)
            with torch.amp.autocast(device_type='cuda'):  # Updated
                pred = model(batch['x_raw'], batch['x_decomp'], batch['covariates'],
                             batch['ts_window'], batch['ts_future'])
                losses.append(mse_loss(pred, batch['y']).item())
    return np.mean(losses) if losses else float('nan')

# -----------------------
# Ablation Runner
# -----------------------
def run_ablation(df, ablation_configs, out_csv="ablation_results_central.csv"):
    torch.cuda.empty_cache()  # Clear GPU memory
    df_subset = df.sample(frac=0.15, random_state=SEED).reset_index(drop=True)
    results = []
    for name, cfg in ablation_configs.items():
        print("\n========== Running ablation:", name, "==========")

        dataset = STLFWindowDataset(df_subset, use_decomp=cfg['use_decomp'], use_covariates=cfg['use_covariates'])
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

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

        sample_item = dataset[0]
        covdim = sample_item['covariates'].shape[1] if sample_item['covariates'].ndim > 1 else 0

        model = DualStreamModel(
            time_emb_h=TIME_EMB_H,
            time_emb_dr=TIME_EMB_DR,
            raw_hidden=64,  # Reduced hidden size
            cov_hidden=32,  # Reduced hidden size
            cov_dim=covdim,
            use_time_attn=cfg['use_time_attn'],
            use_decomp=cfg['use_decomp'],
            single_stream=cfg['single_stream'],
            use_covariates=cfg['use_covariates'],
            bidirectional=cfg.get('bidirectional', False)  # Default to False
        )
        model, best_val = train_model(model, train_loader, val_loader, EPOCHS, LR, DEVICE, ACCUM_STEPS)
        test_loss = evaluate_model(model, test_loader, DEVICE)
        print(f"Ablation {name}: Test MSE = {test_loss:.6f}")
        results.append({'config': name, 'val_loss': best_val, 'test_loss': test_loss})

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("Saved results to", out_csv)

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    torch.cuda.empty_cache()
    df = load_data(DATA_FILE)
    ablation_configs = {
        "base": {"use_time_attn": True, "use_decomp": True, "single_stream": False, "use_covariates": True, "bidirectional": False},
        "no_time_attn": {"use_time_attn": False, "use_decomp": True, "single_stream": False, "use_covariates": True, "bidirectional": False},
        "no_decomp": {"use_time_attn": True, "use_decomp": False, "single_stream": False, "use_covariates": True, "bidirectional": False},
        "no_covariates": {"use_time_attn": True, "use_decomp": True, "single_stream": False, "use_covariates": False, "bidirectional": False},
        "single_stream": {"use_time_attn": True, "use_decomp": True, "single_stream": True, "use_covariates": True, "bidirectional": False},
        "uni_gru": {"use_time_attn": True, "use_decomp": True, "single_stream": False, "use_covariates": True, "bidirectional": False}
    }
    run_ablation(df, ablation_configs, out_csv="ablation_bi_results_central.csv")