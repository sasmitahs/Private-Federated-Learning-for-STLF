import os
import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

# Optional decomposition
try:
    from statsmodels.tsa.seasonal import STL
    HAVE_STL = True
except Exception:
    HAVE_STL = False

# -----------------------
# Logging Setup
# -----------------------
log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Save logs to file
        logging.StreamHandler()             # Print logs to console
    ]
)
logging.info(f"Logging to file: {log_filename}")


# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE = "train_final.feather"

BATCH_SIZE = 64
HISTORY = 24 * 7    # past 7 days hourly
HORIZON = 24        # forecast next 24 hours
TIME_EMB_H = 4
LR = 1e-3
EPOCHS = 40
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------
# Utilities
# -----------------------
def load_data(feather_path):
    df = pd.read_feather(feather_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    df["meter_reading"] = df["meter_reading"].fillna(0.0)
    df["air_temperature"] = df["air_temperature"].fillna(df["air_temperature"].mean())
    return df

def timestamp_to_float(ts):
    vals = ts.values
    secs = vals.astype("datetime64[s]").astype(np.int64).astype(np.float32)
    secs = (secs - secs.mean()) / 1e6
    return secs

def decompose_series(series, period=24):
    if HAVE_STL:
        series = np.nan_to_num(series)
        stl = STL(series, period=period, robust=True)
        res = stl.fit()
        return (
            res.trend.astype(np.float32),
            res.seasonal.astype(np.float32),
            res.resid.astype(np.float32),
        )
    else:
        trend = (
            pd.Series(series)
            .rolling(window=period, min_periods=1, center=True)
            .mean()
            .values.astype(np.float32)
        )
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

        # Per-building decomposition
        self.df["trend"] = 0.0
        self.df["seasonal"] = 0.0
        self.df["resid"] = 0.0
        if use_decomp:
            for bid, grp in self.df.groupby("building_id"):
                idx = grp.index
                trend, seasonal, resid = decompose_series(grp["meter_reading"].values, period=24)
                self.df.loc[idx, "trend"] = trend
                self.df.loc[idx, "seasonal"] = seasonal
                self.df.loc[idx, "resid"] = resid

        self.df["hour"] = self.df["timestamp"].dt.hour
        self.df["dow"] = self.df["timestamp"].dt.dayofweek
        self.df["pu_id"] = pd.factorize(self.df["primary_use"])[0]

        # Build valid window start indices per building
        self.indices = []
        for bid, grp in self.df.groupby("building_id"):
            n = len(grp)
            max_start = n - (history + horizon) + 1
            if max_start > 0:
                start_idx = grp.index.min()
                self.indices.extend(list(range(start_idx, start_idx + max_start)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        window = self.df.iloc[i : i + self.history]
        future = self.df.iloc[i + self.history : i + self.history + self.horizon]

        x_raw = window["meter_reading"].values.astype(np.float32)
        x_decomp = window[["trend", "seasonal", "resid"]].values.astype(np.float32)

        if self.use_covariates:
            covs = []
            covs.append(window["air_temperature"].values.astype(np.float32))
            hours = window["hour"].values
            covs.append(
                np.stack(
                    [np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)],
                    axis=-1,
                ).astype(np.float32)
            )
            dow = np.eye(7)[window["dow"].values].astype(np.float32)
            covs.append(dow)

            cov_list = []
            for c in covs:
                if c.ndim == 1:
                    cov_list.append(c.reshape(-1, 1))
                else:
                    cov_list.append(c)
            covariate_array = np.concatenate(cov_list, axis=-1).astype(np.float32)
        else:
            covariate_array = np.zeros((self.history, 0), dtype=np.float32)

        y = future["meter_reading"].values.astype(np.float32)
        ts_window = timestamp_to_float(window["timestamp"])
        ts_future = timestamp_to_float(future["timestamp"])

        return {
            "x_raw": x_raw,
            "x_decomp": x_decomp,
            "covariates": covariate_array,
            "ts_window": ts_window,
            "ts_future": ts_future,
            "y": y,
        }

def collate_fn(batch):
    x_raw = torch.as_tensor(np.stack([b["x_raw"] for b in batch]), dtype=torch.float32)
    x_decomp = torch.as_tensor(np.stack([b["x_decomp"] for b in batch]), dtype=torch.float32)
    covariates = torch.as_tensor(np.stack([b["covariates"] for b in batch]), dtype=torch.float32)
    ts_window = torch.as_tensor(np.stack([b["ts_window"] for b in batch]), dtype=torch.float32)
    ts_future = torch.as_tensor(np.stack([b["ts_future"] for b in batch]), dtype=torch.float32)
    y = torch.as_tensor(np.stack([b["y"] for b in batch]), dtype=torch.float32)
    return {
        "x_raw": x_raw,
        "x_decomp": x_decomp,
        "covariates": covariates,
        "ts_window": ts_window,
        "ts_future": ts_future,
        "y": y,
    }

# -----------------------
# Model Components
# -----------------------
class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(1, emb_dim)

    def forward(self, t):
        return self.linear(t.unsqueeze(-1))

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_emb_dim=0):
        super().__init__()
        self.gru = nn.GRU(input_dim + time_emb_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x, time_emb):
        enc_in = torch.cat([x, time_emb], dim=-1) if time_emb is not None else x
        out, _ = self.gru(enc_in)
        return out

class CNNGRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_emb_dim=0):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim + time_emb_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gru = nn.GRU(32, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x, time_emb):
        enc_in = torch.cat([x, time_emb], dim=-1) if time_emb is not None else x
        enc_in = enc_in.permute(0, 2, 1)
        out = self.relu(self.conv1(enc_in))
        out = self.pool(out)
        out = self.relu(self.conv2(out))
        out = self.pool(out)
        out = out.permute(0, 2, 1)
        out, _ = self.gru(out)
        return out

class CovariateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, cov):
        out, _ = self.gru(cov)
        return out

class FCDecoder(nn.Module):
    def __init__(self, in_dim, out_horizon):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_horizon)

    def forward(self, h):
        return self.fc(h)

class GRUDecoder(nn.Module):
    def __init__(self, in_dim, out_horizon):
        super().__init__()
        self.hidden_size = 128
        self.gru = nn.GRU(input_size=in_dim, hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)
        self.hidden_projection = nn.Linear(in_dim, self.hidden_size)
        self.out_horizon = out_horizon

    def forward(self, h):
        batch_size = h.size(0)
        hidden = self.hidden_projection(h).unsqueeze(0)
        decoder_input = torch.zeros(batch_size, 1, self.gru.input_size, device=h.device)
        outputs = []
        for _ in range(self.out_horizon):
            output, hidden = self.gru(decoder_input, hidden)
            output = self.fc(output.squeeze(1))
            outputs.append(output)
            decoder_input = torch.zeros(batch_size, 1, self.gru.input_size, device=h.device)
        return torch.cat(outputs, dim=1)

class DualStreamModel(nn.Module):
    def __init__(self, time_emb_h, raw_hidden, cov_hidden, cov_dim,
                 use_time_attn=True, use_decomp=True, single_stream=False, use_covariates=True,
                 encoder_type="gru", decoder_type="fc"):
        super().__init__()
        self.use_time_attn = use_time_attn
        self.use_decomp = use_decomp
        self.single_stream = single_stream
        self.use_covariates = use_covariates and (not single_stream)
        self.encoder_type = encoder_type

        in_raw = 1 + (3 if use_decomp else 0)
        time_emb_dim = time_emb_h if use_time_attn else 0
        if encoder_type == "gru":
            self.raw_encoder = GRUEncoder(in_raw, raw_hidden, time_emb_dim=time_emb_dim)
        elif encoder_type == "cnn_gru":
            self.raw_encoder = CNNGRUEncoder(in_raw, raw_hidden, time_emb_dim=time_emb_dim)
        else:
            raise ValueError("Unknown encoder_type")

        self.time_emb = TimeEmbedding(time_emb_h) if use_time_attn else None
        self.cov_encoder = CovariateEncoder(cov_dim, cov_hidden) if (self.use_covariates and cov_dim > 0) else None

        if self.single_stream:
            dec_in = raw_hidden * 2
        elif self.use_covariates:
            dec_in = raw_hidden * 2 + cov_hidden * 2
        else:
            dec_in = raw_hidden * 2

        if decoder_type == "fc":
            self.decoder = FCDecoder(dec_in, HORIZON)
        elif decoder_type == "gru":
            self.decoder = GRUDecoder(dec_in, HORIZON)
        else:
            raise ValueError("Unknown decoder_type")

    def forward(self, x_raw, x_decomp, covariates, ts_window, ts_future):
        xr = x_raw.unsqueeze(-1)
        x_in = torch.cat([xr, x_decomp], dim=-1) if self.use_decomp else xr

        time_emb = self.time_emb(ts_window) if self.use_time_attn else None
        h_raw = self.raw_encoder(x_in, time_emb)
        h_raw = h_raw.mean(dim=1)

        if self.use_covariates and self.cov_encoder is not None:
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

def train_model(model, train_loader, val_loader, epochs, lr, device, model_name="model"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_epoch = -1
    for ep in range(epochs):
        model.train()
        losses = []
        for batch_idx, batch in enumerate(train_loader):
            for k in batch:
                batch[k] = batch[k].to(device)
            pred = model(batch["x_raw"], batch["x_decomp"], batch["covariates"],
                         batch["ts_window"], batch["ts_future"])
            if torch.isnan(pred).any():
                logging.warning(f"[{model_name}] NaNs in predictions at epoch {ep+1}, batch {batch_idx}")
            loss = mse_loss(pred, batch["y"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        val_loss = evaluate_mse(model, val_loader, device)
        logging.info(f"[{model_name}] Epoch {ep+1}/{epochs} | Train {np.mean(losses):.6f} | Val {val_loss:.6f} | Best {best_val:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep + 1
            logging.info(f"[{model_name}] New best val {best_val:.6f} at epoch {best_epoch}, saving checkpoint.")
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": best_val,
                },
                f"best_{model_name}.pth",
            )
    return model, best_val, best_epoch

def evaluate_mse(model, dataloader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            for k in batch:
                batch[k] = batch[k].to(device)
            pred = model(batch["x_raw"], batch["x_decomp"], batch["covariates"],
                         batch["ts_window"], batch["ts_future"])
            losses.append(mse_loss(pred, batch["y"]).item())
    return np.mean(losses) if losses else float("nan")

def evaluate_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            for k in batch:
                batch[k] = batch[k].to(device)
            pred = model(batch["x_raw"], batch["x_decomp"], batch["covariates"],
                         batch["ts_window"], batch["ts_future"])
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch["y"].cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    maes, rmses, smapes = [], [], []
    for i in range(len(preds)):
        p, t = preds[i], targets[i]
        mae = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))
        smape = 100 * np.mean(2 * np.abs(p - t) / (np.abs(p) + np.abs(t) + 1e-8))
        maes.append(mae)
        rmses.append(rmse)
        smapes.append(smape)
    return {
        "median_mae": np.median(maes),
        "median_rmse": np.median(rmses),
        "median_smape": np.median(smapes),
    }

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    df = load_data(DATA_FILE)

    # Split per-building (no leakage)
    train_dfs, val_dfs, test_dfs = [], [], []
    for bid, group in df.groupby("building_id"):
        group = group.sort_values("timestamp")
        len_g = len(group)
        train_len = int(0.15 * len_g)
        val_len = int(0.015 * len_g)
        test_len = len_g - train_len - val_len
        train_dfs.append(group.iloc[:train_len])
        val_dfs.append(group.iloc[train_len : train_len + val_len])
        test_dfs.append(group.iloc[train_len + val_len :])
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)

    models_to_run = {
        "decoder_gru_base": {"encoder_type": "gru", "decoder_type": "gru", "use_time_attn": True, "use_decomp": True, "single_stream": False, "use_covariates": True},
        "cnn_gru_no_cov": {"encoder_type": "cnn_gru", "decoder_type": "fc", "use_time_attn": True, "use_decomp": True, "single_stream": False, "use_covariates": False},
        "single_stream_fc": {"encoder_type": "gru", "decoder_type": "fc", "use_time_attn": False, "use_decomp": True, "single_stream": True, "use_covariates": False},
    }

    for model_name, params in models_to_run.items():
        logging.info(f"--- Training {model_name} ---")
        train_ds = STLFWindowDataset(train_df, use_decomp=params["use_decomp"], use_covariates=params["use_covariates"])
        val_ds = STLFWindowDataset(val_df, use_decomp=params["use_decomp"], use_covariates=params["use_covariates"])
        test_ds = STLFWindowDataset(test_df, use_decomp=params["use_decomp"], use_covariates=params["use_covariates"])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        cov_dim = train_ds[0]["covariates"].shape[1] if params["use_covariates"] else 0
        model = DualStreamModel(time_emb_h=TIME_EMB_H,
                                raw_hidden=64,
                                cov_hidden=32,
                                cov_dim=cov_dim,
                                use_time_attn=params["use_time_attn"],
                                use_decomp=params["use_decomp"],
                                single_stream=params["single_stream"],
                                use_covariates=params["use_covariates"],
                                encoder_type=params["encoder_type"],
                                decoder_type=params["decoder_type"])

        model, best_val, best_epoch = train_model(model, train_loader, val_loader, EPOCHS, LR, DEVICE, model_name=model_name)
        logging.info(f"[{model_name}] Best validation loss {best_val:.6f} at epoch {best_epoch}")

        checkpoint = torch.load(f"best_{model_name}.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        val_metrics = evaluate_metrics(model, val_loader, DEVICE)
        test_metrics = evaluate_metrics(model, test_loader, DEVICE)
        logging.info(f"[{model_name}] Val metrics: {val_metrics}")
        logging.info(f"[{model_name}] Test metrics: {test_metrics}")
