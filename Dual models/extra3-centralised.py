import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


# ==================== METRIC FUNCTIONS ====================
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.where(denominator == 0, 0,
                            np.abs(y_true - y_pred) / denominator)) * 100


def mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def nrmse_range(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    true_range = np.max(y_true) - np.min(y_true)
    return rmse / true_range if true_range != 0 else np.nan


def nmae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mean_y = np.mean(y_true)
    return 100 * mae / mean_y if mean_y != 0 else np.nan


def nmbe(y_true, y_pred):
    mean_bias = np.mean(y_true - y_pred)
    mean_y = np.mean(y_true)
    return 100 * mean_bias / mean_y if mean_y != 0 else np.nan


def nrmse_mean(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_y = np.mean(y_true)
    return 100 * rmse / mean_y if mean_y != 0 else np.nan


# ==================== SINGLE-MODEL METRICS ====================
def evaluate_forecast_metrics(csv_path: str) -> pd.DataFrame:
    """
    Compute **one** set of metrics from a prediction CSV (no rounds).
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    df = df.fillna(0.005)
    y_true = df["true"].values
    y_pred = df["pred"].values

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape_val = mape(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    nrmse_range_val = nrmse_range(y_true, y_pred)
    nmae_val = nmae(y_true, y_pred)
    nmbe_val = nmbe(y_true, y_pred)
    nrmse_mean_val = nrmse_mean(y_true, y_pred)

    metrics_df = pd.DataFrame([{
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape_val,
        "SMAPE (%)": smape_val,
        "NRMSE_range": nrmse_range_val,
        "NMAE (%)": nmae_val,
        "NMBE (%)": nmbe_val,
        "NRMSE_mean (%)": nrmse_mean_val
    }])
    return metrics_df


# ==================== RECOMPUTE METRICS FROM EXISTING CSVs ====================
def recompute_metrics_from_existing(
    MODELS: List[str],
    STRATEGIES: List[str],
    BASE_OUTPUT_DIR: str,
    METRICS_DIR: str,
    CID: range
):
    """
    Walk through already-generated prediction CSVs and (re)write metric files.
    """
    os.makedirs(METRICS_DIR, exist_ok=True)

    for cid in tqdm(CID, desc="Recomputing metrics"):
        for model_name in MODELS:
            for strategy in STRATEGIES:
                pred_csv = os.path.join(BASE_OUTPUT_DIR,
                                        f"{cid}_{model_name}_{strategy}.csv")
                metric_csv = os.path.join(METRICS_DIR,
                                          f"cid{cid}_{model_name}_{strategy}_metrics.csv")

                if not os.path.exists(pred_csv):
                    # silently skip – no prediction file
                    continue

                try:
                    metrics_df = evaluate_forecast_metrics(pred_csv)
                    metrics_df.insert(0, "building_id", cid)
                    metrics_df.insert(1, "model", model_name)
                    metrics_df.insert(2, "strategy", strategy)
                    metrics_df.to_csv(metric_csv, index=False)
                except Exception as e:
                    print(f"[ERROR] CID={cid} {model_name}/{strategy}: {e}")


# ==================== SUMMARY (median / mean / variance) ====================
def compute_and_log_median_metrics(
    MODELS: List[str],
    STRATEGIES: List[str],
    METRICS_DIR: str,
    CID: range,
    OUTPUT_FILE: str = None
):
    if OUTPUT_FILE is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_FILE = os.path.join(METRICS_DIR,
                                   f"ashrae_median_metrics_summary_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(OUTPUT_FILE), logging.StreamHandler()],
        force=True
    )

    logging.info("=" * 80)
    logging.info("ASHRAE METRICS SUMMARY – SINGLE MODEL (NO ROUNDS)")
    logging.info("=" * 80)
    logging.info(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logging.info(f"Metrics dir: {METRICS_DIR}")
    logging.info(f"Clients: {CID.start} – {CID.stop - 1}")
    logging.info("=" * 80)

    summary_rows = []

    for model_name in MODELS:
        for strategy in STRATEGIES:
            logging.info(f"\n{'='*40} {model_name} | {strategy} {'='*40}")

            nmae_l, nmbe_l, nrmse_l = [], [], []
            valid = missing = empty = err = 0

            for cid in CID:
                mcsv = os.path.join(METRICS_DIR,
                                    f"cid{cid}_{model_name}_{strategy}_metrics.csv")
                if not os.path.exists(mcsv):
                    missing += 1
                    continue

                try:
                    df = pd.read_csv(mcsv)
                    if df.empty:
                        empty += 1
                        continue

                    row = df.iloc[0]
                    nmae = row.get('NMAE (%)', np.nan)
                    nmbe = row.get('NMBE (%)', np.nan)
                    nrmse = row.get('NRMSE_mean (%)', np.nan)

                    if pd.notna(nmae): nmae_l.append(nmae)
                    if pd.notna(nmbe): nmbe_l.append(nmbe)
                    if pd.notna(nrmse): nrmse_l.append(nrmse)
                    valid += 1
                except Exception:
                    err += 1

            if nmae_l:
                med_nmae = np.median(nmae_l)
                med_nmbe = np.median(nmbe_l)
                med_nrmse = np.median(nrmse_l)
                mean_nmae = np.mean(nmae_l)
                mean_nmbe = np.mean(nmbe_l)
                mean_nrmse = np.mean(nrmse_l)
                var_nmae = np.var(nmae_l, ddof=1)
                var_nmbe = np.var(nmbe_l, ddof=1)
                var_nrmse = np.var(nrmse_l, ddof=1)

                logging.info(f"Valid: {valid} | Missing: {missing} | Empty: {empty} | Err: {err}")
                logging.info(f"Median  – NMAE: {med_nmae: .4f}  NMBE: {med_nmbe: .4f}  NRMSE: {med_nrmse: .4f}")
                logging.info(f"Mean    – NMAE: {mean_nmae: .4f}  NMBE: {mean_nmbe: .4f}  NRMSE: {mean_nrmse: .4f}")
                logging.info(f"Variance– NMAE: {var_nmae: .4f}  NMBE: {var_nmbe: .4f}  NRMSE: {var_nrmse: .4f}")

                summary_rows.append({
                    "Model": model_name, "Strategy": strategy,
                    "Valid": valid,
                    "Median_NMAE": med_nmae, "Median_NMBE": med_nmbe, "Median_NRMSE": med_nrmse,
                    "Mean_NMAE": mean_nmae, "Mean_NMBE": mean_nmbe, "Mean_NRMSE": mean_nrmse,
                    "Var_NMAE": var_nmae, "Var_NMBE": var_nmbe, "Var_NRMSE": var_nrmse
                })
            else:
                logging.info("No valid metric files for this combination.")
                summary_rows.append({
                    "Model": model_name, "Strategy": strategy, "Valid": 0,
                    "Median_NMAE": np.nan, "Median_NMBE": np.nan, "Median_NRMSE": np.nan,
                    "Mean_NMAE": np.nan, "Mean_NMBE": np.nan, "Mean_NRMSE": np.nan,
                    "Var_NMAE": np.nan, "Var_NMBE": np.nan, "Var_NRMSE": np.nan
                })

    # ----- save CSV summary -----
    summary_df = pd.DataFrame(summary_rows)
    csv_out = OUTPUT_FILE.replace('.log', '_summary.csv')
    summary_df.to_csv(csv_out, index=False)

    logging.info("\n" + "="*80)
    logging.info("SUMMARY TABLE")
    logging.info(summary_df.to_string(index=False))
    logging.info(f"CSV summary: {csv_out}")
    logging.info(f"Log file   : {OUTPUT_FILE}")
    logging.info("="*80)

    return summary_df


# ==================== CONFIG & RUN ====================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    #  USER SETTINGS – adjust to your environment
    # ------------------------------------------------------------------
    MODELS      = ["dual_cnn_gru_fcnn","dual_cnn_ann_fcnn","dual_simple_ann_fcnn"]             # add more if needed
    STRATEGIES  = ["global"]
    BASE_OUTPUT_DIR = "predictions_global"      # <-- folder with existing CSVs
    METRICS_DIR     = "metrics_global"           # <-- where metric CSVs will be (re)written
    CID         = range(100, 1409)                    # client range

    # ------------------------------------------------------------------
    #  1. Re-compute metrics from existing prediction CSVs
    # ------------------------------------------------------------------
    print("Re-computing metrics from existing prediction files...")
    recompute_metrics_from_existing(
        MODELS=MODELS,
        STRATEGIES=STRATEGIES,
        BASE_OUTPUT_DIR=BASE_OUTPUT_DIR,
        METRICS_DIR=METRICS_DIR,
        CID=CID
    )

    # ------------------------------------------------------------------
    #  2. Build median / mean / variance summary
    # ------------------------------------------------------------------
    print("\nBuilding ASHRAE summary...")
    summary = compute_and_log_median_metrics(
        MODELS=MODELS,
        STRATEGIES=STRATEGIES,
        METRICS_DIR=METRICS_DIR,
        CID=CID
    )

    print("\nFinished! Check the files in:", METRICS_DIR)