#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full prediction → metric → summary pipeline
Works for *any* model / strategy you put in MODELS / STRATEGIES.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from tqdm import tqdm
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# ----------------------------------------------------------------------
# --------------------------- USER CONFIG -------------------------------
# ----------------------------------------------------------------------
BASE_RESULTS_DIR   = "/home/user/DPFL-Sasmita/FL-Baseline-Codes1/results"
BASE_OUTPUT_DIR    = "/home/user/DPFL-Sasmita/FL-Baseline-Codes1/predictions_combined"
METRICS_DIR        = "/home/user/DPFL-Sasmita/FL-Baseline-Codes1/metrics_combined"
FEATHER_PATH       = "/home/user/DPFL-Sasmita/FL-Baseline-Codes1/train_final.feather"

# <<<---  PUT YOUR MODELS & STRATEGIES HERE  --->>>
MODELS = ["gru","simple_cnn","moe_lstm", "lstm", "cnn_gru_no_cov" ]

STRATEGIES = [
    "das_cluster_fedprox"
    # "random_cluster_fedprox",
    # "random_no_cluster_fedprox",
    # "das_no_cluster_fedprox",
]

ROUNDS      = [40]               # only round 40 is used in the original table
INPUT_LEN   = 168
OUTPUT_LEN  = 24
CID_RANGE   = range(101, 1409)   # tighten if you know the exact range

# ----------------------------------------------------------------------
# Import the **model factory** that returns a fresh nn.Module
# ----------------------------------------------------------------------
# The factory must have the signature:
#   model_fn(name: str) -> nn.Module
from Models import model_fn   # <-- adjust import path if needed

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

LOG_FILE = os.path.join(METRICS_DIR, f"full_pipeline_{TIMESTAMP}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger()

# ----------------------------------------------------------------------
# -------------------------- METRIC HELPERS ----------------------------
# ----------------------------------------------------------------------
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)) * 100

def mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def nrmse_range(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r = np.max(y_true) - np.min(y_true)
    return rmse / r if r != 0 else np.nan

def nmae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mean_y = np.mean(y_true)
    return 100 * mae / mean_y if mean_y != 0 else np.nan

def nmbe(y_true, y_pred):
    bias = np.mean(y_true - y_pred)
    mean_y = np.mean(y_true)
    return 100 * bias / mean_y if mean_y != 0 else np.nan

def nrmse_mean(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_y = np.mean(y_true)
    return 100 * rmse / mean_y if mean_y != 0 else np.nan

# ----------------------------------------------------------------------
# ----------------------- ROLLING FORECAST ----------------------------
# ----------------------------------------------------------------------
@torch.no_grad()
def rolling_forecast_on_test(cid: int, model: nn.Module,
                            input_len: int = INPUT_LEN,
                            output_len: int = OUTPUT_LEN) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    """Return list of (pred_ts, true_ts) for every forecast window."""
    df = pd.read_feather(FEATHER_PATH)
    df = df[df['building_id'] == cid].copy()
    df['meter_reading'] = df['meter_reading'].fillna(0)
    if df.empty:
        raise ValueError(f"CID {cid}: no rows in feather")

    ts = TimeSeries.from_dataframe(df, time_col='timestamp',
                                   value_cols='meter_reading',
                                   fill_missing_dates=True, freq='h')
    _, test = ts.split_before(0.75)

    scaler = MinMaxScaler(feature_range=(0.1, 1))
    transformer = Scaler(scaler)
    test_scaled = transformer.fit_transform(test)

    values = test_scaled.values().squeeze()
    times  = test_scaled.time_index

    preds, trues = [], []
    model.eval()
    device = next(model.parameters()).device

    step = output_len
    for i in range(0, len(values) - input_len - output_len + 1, step):
        x = values[i:i+input_len]
        y = values[i+input_len:i+input_len+output_len]
        t = times[i+input_len:i+input_len+output_len]

        inp = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        out = model(inp)

        # squeeze possible extra dims
        if out.dim() == 3:
            out = out.squeeze(0).squeeze(-1)
        else:
            out = out.squeeze(0)

        pred_ts = TimeSeries.from_times_and_values(t, out.cpu().numpy())
        true_ts = TimeSeries.from_times_and_values(t, y)

        preds.append(transformer.inverse_transform(pred_ts))
        trues.append(transformer.inverse_transform(true_ts))

    return preds, trues


# ----------------------------------------------------------------------
# ------------------- PREDICTION CSV WRITER ---------------------------
# ----------------------------------------------------------------------
def write_prediction_csv(cid: int, model_name: str, strategy: str,
                         rounds: List[int], model_dir: str, out_csv: str):
    rows = []
    for rnd in rounds:
        pth = os.path.join(model_dir, f"{model_name}_round_{rnd}_{strategy}.pt")
        if not os.path.exists(pth):
            log.warning(f"Model missing → {pth}")
            continue

        model = model_fn(model_name)
        state = torch.load(pth, weights_only=True)
        model.load_state_dict(state)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()

        pred_list, true_list = rolling_forecast_on_test(cid, model)

        for p_ts, t_ts in zip(pred_list, true_list):
            df_p = pd.DataFrame({"timestamp": p_ts.time_index,
                                 "pred": p_ts.values().squeeze()})
            df_t = pd.DataFrame({"timestamp": t_ts.time_index,
                                 "true": t_ts.values().squeeze()})
            merged = pd.merge(df_t, df_p, on="timestamp", how="inner")
            merged["round"] = rnd
            rows.append(merged[["timestamp", "true", "pred", "round"]])

    if rows:
        final = pd.concat(rows, ignore_index=True)
        final.to_csv(out_csv, index=False)
        log.info(f"Predictions → {out_csv} ({len(final)} rows)")
    else:
        # empty CSV with header – metric step will write NaNs
        pd.DataFrame(columns=["timestamp","true","pred","round"]).to_csv(out_csv, index=False)
        log.warning(f"No forecast windows for CID {cid} → empty CSV")


# ----------------------------------------------------------------------
# -------------------- METRIC CSV WRITER ------------------------------
# ----------------------------------------------------------------------
def compute_and_save_metrics(pred_csv: str, metric_csv: str):
    df = pd.read_csv(pred_csv)
    if df.empty:
        metrics = {m: np.nan for m in [
            "MAE","MSE","RMSE","MAPE (%)","SMAPE (%)",
            "NRMSE_range","NMAE (%)","NMBE (%)","NRMSE_mean (%)"
        ]}
        metrics["round"] = ROUNDS[0]
        pd.DataFrame([metrics]).to_csv(metric_csv, index=False)
        return

    metric_list = []
    for rnd in df["round"].unique():
        sub = df[df["round"] == rnd].copy()
        sub.fillna(0.005, inplace=True)
        y_t, y_p = sub["true"].values, sub["pred"].values

        mae   = mean_absolute_error(y_t, y_p)
        mse   = mean_squared_error(y_t, y_p)
        rmse  = np.sqrt(mse)
        mape_v  = mape(y_t, y_p)
        smape_v = smape(y_t, y_p)
        nrmse_r = nrmse_range(y_t, y_p)
        nmae_v  = nmae(y_t, y_p)
        nmbe_v  = nmbe(y_t, y_p)
        nrmse_m = nrmse_mean(y_t, y_p)

        metric_list.append({
            "round": rnd,
            "MAE": mae, "MSE": mse, "RMSE": rmse,
            "MAPE (%)": mape_v, "SMAPE (%)": smape_v,
            "NRMSE_range": nrmse_r,
            "NMAE (%)": nmae_v, "NMBE (%)": nmbe_v,
            "NRMSE_mean (%)": nrmse_m
        })

    pd.DataFrame(metric_list).to_csv(metric_csv, index=False)


# ----------------------------------------------------------------------
# -------------------------- MAIN LOOP ---------------------------------
# ----------------------------------------------------------------------
def main():
    log.info("=== START FULL PIPELINE ===")
    total_jobs = len(MODELS) * len(STRATEGIES) * len(CID_RANGE)
    pbar = tqdm(total=total_jobs, desc="Overall progress")

    for model in MODELS:
        model_dir = os.path.join(BASE_RESULTS_DIR, model)
        if not os.path.isdir(model_dir):
            log.error(f"Model directory missing: {model_dir}")
            continue

        for strat in STRATEGIES:
            log.info(f"\n--- Processing {model} | {strat} ---")
            for cid in CID_RANGE:
                pred_csv   = os.path.join(BASE_OUTPUT_DIR,
                                         f"{cid}_{model}_{strat}.csv")
                metric_csv = os.path.join(METRICS_DIR,
                                         f"cid{cid}_{model}_{strat}_metrics.csv")

                # ---- 1. predictions -------------------------------------------------
                try:
                    write_prediction_csv(cid, model, strat,
                                         ROUNDS, model_dir, pred_csv)
                except Exception as e:
                    log.error(f"PREDICTION FAILED cid{cid} {model} {strat}: {e}")

                # ---- 2. metrics ----------------------------------------------------
                try:
                    compute_and_save_metrics(pred_csv, metric_csv)
                except Exception as e:
                    log.error(f"METRIC FAILED cid{cid} {model} {strat}: {e}")

                pbar.update(1)

    pbar.close()
    log.info("=== PREDICTION + METRIC PHASE DONE ===")

    # ------------------------------------------------------------------
    # ----------------------- BUILD SUMMARY TABLE -----------------------
    # ------------------------------------------------------------------
    METRIC_NAMES = [
        "MAE","MSE","RMSE","MAPE (%)","SMAPE (%)",
        "NRMSE_range","NMAE (%)","NMBE (%)","NRMSE_mean (%)"
    ]

    summary_rows = []
    for model in MODELS:
        for strat in STRATEGIES:
            metric_vals = {m: [] for m in METRIC_NAMES}
            valid = 0

            for cid in CID_RANGE:
                mfile = os.path.join(METRICS_DIR,
                                    f"cid{cid}_{model}_{strat}_metrics.csv")
                if not os.path.exists(mfile):
                    continue
                try:
                    df = pd.read_csv(mfile)
                    if df.empty or df[METRIC_NAMES].isna().all().all():
                        continue
                    row = df.iloc[0]                     # round 40 is the only one
                    valid += 1
                    for m in METRIC_NAMES:
                        v = row[m]
                        if pd.notna(v):
                            metric_vals[m].append(v)
                except Exception:
                    continue

            row = {"Model": model, "Strategy": strat, "Valid_Clients": valid}
            for m in METRIC_NAMES:
                vals = np.array(metric_vals[m])
                if len(vals):
                    row[f"Median_{m}"] = np.median(vals)
                    row[f"Mean_{m}"]   = np.mean(vals)
                    row[f"Var_{m}"]    = np.var(vals, ddof=1)
                else:
                    row[f"Median_{m}"] = np.nan
                    row[f"Mean_{m}"]   = np.nan
                    row[f"Var_{m}"]    = np.nan
            summary_rows.append(row)

    out_summary = os.path.join(METRICS_DIR,
                               f"all_metrics_summary_FULL_{TIMESTAMP}.csv")
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    log.info(f"FINAL SUMMARY → {out_summary}")

    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("ALL DONE")
    print("="*80)
    print(f"  • Predictions : {BASE_OUTPUT_DIR}")
    print(f"  • Per-CID metrics : {METRICS_DIR}")
    print(f"  • Summary table   : {out_summary}")
    print(f"  • Log file        : {LOG_FILE}")
    print("="*80)


if __name__ == "__main__":
    main()