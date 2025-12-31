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
    os.makedirs(METRICS_DIR, exist_ok=True)
    for cid in tqdm(CID, desc="Recomputing metrics"):
        for model_name in MODELS:
            for strategy in STRATEGIES:
                pred_csv = os.path.join(BASE_OUTPUT_DIR, f"{cid}_{model_name}_{strategy}.csv")
                metric_csv = os.path.join(METRICS_DIR, f"cid{cid}_{model_name}_{strategy}_metrics.csv")
                if not os.path.exists(pred_csv):
                    continue
                try:
                    metrics_df = evaluate_forecast_metrics(pred_csv)
                    metrics_df.insert(0, "building_id", cid)
                    metrics_df.insert(1, "model", model_name)
                    metrics_df.insert(2, "strategy", strategy)
                    metrics_df.to_csv(metric_csv, index=False)
                except Exception as e:
                    print(f"[ERROR] CID={cid} {model_name}/{strategy}: {e}")

# ==================== EXTENDED SUMMARY WITH ALL MEDIAN METRICS ====================
def compute_and_log_full_median_metrics(
    MODELS: List[str],
    STRATEGIES: List[str],
    METRICS_DIR: str,
    CID: range,
    OUTPUT_FILE: str = None
):
    if OUTPUT_FILE is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_FILE = os.path.join(METRICS_DIR, f"ashrae_full_median_metrics_summary_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(OUTPUT_FILE), logging.StreamHandler()],
        force=True
    )

    logging.info("=" * 100)
    logging.info("ASHRAE FULL METRICS SUMMARY (Median over all buildings)")
    logging.info("=" * 100)
    logging.info(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logging.info(f"Metrics dir: {METRICS_DIR}")
    logging.info(f"Buildings: {CID.start} – {CID.stop - 1}")
    logging.info("=" * 100)

    summary_rows = []

    for model_name in MODELS:
        for strategy in STRATEGIES:
            logging.info(f"\n{'='*50} {model_name.upper()} | {strategy.upper()} {'='*50}")

            # Lists to collect metrics across buildings
            mae_l, mse_l, rmse_l = [], [], []
            mape_l, smape_l = [], []
            nrmse_range_l, nmae_l, nmbe_l, nrmse_mean_l = [], [], [], []

            valid = missing = empty = err = 0

            for cid in CID:
                mcsv = os.path.join(METRICS_DIR, f"cid{cid}_{model_name}_{strategy}_metrics.csv")
                if not os.path.exists(mcsv):
                    missing += 1
                    continue
                try:
                    df = pd.read_csv(mcsv)
                    if df.empty:
                        empty += 1
                        continue
                    row = df.iloc[0]

                    # Append all metrics if they are valid
                    for key, container in [
                        ("MAE", mae_l),
                        ("MSE", mse_l),
                        ("RMSE", rmse_l),
                        ("MAPE (%)", mape_l),
                        ("SMAPE (%)", smape_l),
                        ("NRMSE_range", nrmse_range_l),
                        ("NMAE (%)", nmae_l),
                        ("NMBE (%)", nmbe_l),
                        ("NRMSE_mean (%)", nrmse_mean_l),
                    ]:
                        val = row.get(key, np.nan)
                        if pd.notna(val):
                            container.append(val)

                    valid += 1
                except Exception as e:
                    err += 1
                    print(f"[Read error] {mcsv}: {e}")

            # Compute medians (and optionally means/variances if you want them later)
            def safe_median(lst): return np.median(lst) if lst else np.nan
            def safe_mean(lst):   return np.mean(lst) if lst else np.nan
            def safe_var(lst):    return np.var(lst, ddof=1) if lst else np.nan

            med_mae   = safe_median(mae_l)
            med_mse   = safe_median(mse_l)
            med_rmse  = safe_median(rmse_l)
            med_mape  = safe_median(mape_l)
            med_smape = safe_median(smape_l)
            med_nrmse_range = safe_median(nrmse_range_l)
            med_nmae  = safe_median(nmae_l)
            med_nmbe  = safe_median(nmbe_l)
            med_nrmse_mean = safe_median(nrmse_mean_l)

            logging.info(f"Valid buildings : {valid} | Missing: {missing} | Empty: {empty} | Errors: {err}")
            logging.info(f"Median MAE       : {med_mae:8.4f}")
            logging.info(f"Median MSE       : {med_mse:8.4f}")
            logging.info(f"Median RMSE      : {med_rmse:8.4f}")
            logging.info(f"Median MAPE (%)  : {med_mape:8.4f}")
            logging.info(f"Median SMAPE (%) : {med_smape:8.4f}")
            logging.info(f"Median NRMSE_range : {med_nrmse_range:8.4f}")
            logging.info(f"Median NMAE (%)  : {med_nmae:8.4f}")
            logging.info(f"Median NMBE (%)  : {med_nmbe:8.4f}")
            logging.info(f"Median NRMSE_mean(%) : {med_nrmse_mean:8.4f}")

            summary_rows.append({
                "Model": model_name,
                "Strategy": strategy,
                "Valid": valid,
                "Median_MAE": med_mae,
                "Median_MSE": med_mse,
                "Median_RMSE": med_rmse,
                "Median_MAPE": med_mape,
                "Median_SMAPE": med_smape,
                "Median_NRMSE_range": med_nrmse_range,
                "Median_NMAE": med_nmae,
                "Median_NMBE": med_nmbe,
                "Median_NRMSE_mean": med_nrmse_mean,
            })

    # Save CSV summary
    summary_df = pd.DataFrame(summary_rows)
    csv_out = OUTPUT_FILE.replace('.log', '_full_summary.csv')
    summary_df.to_csv(csv_out, index=False)

    logging.info("\n" + "="*100)
    logging.info("FULL MEDIAN SUMMARY TABLE")
    logging.info(summary_df.to_string(index=False, float_format="{:.4f}".format))
    logging.info(f"\nCSV summary saved to: {csv_out}")
    logging.info(f"Log file saved to     : {OUTPUT_FILE}")
    logging.info("="*100)

    return summary_df

# ==================== CONFIG & RUN ====================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # USER SETTINGS – adjust to your paths
    # ------------------------------------------------------------------
    MODELS = ["lstm", "gru", "simple_ann", "moe_lstm", "simple_cnn", "cnn_gru_no_cov"]
    STRATEGIES = ["global_model_30epochs"]
    BASE_OUTPUT_DIR = "predictions40-50-168-T"
    METRICS_DIR = "metrics40-50-168-T"
    CID = range(100, 1409)   # 100 to 1408 inclusive

    # 1. Re-compute individual metric files (safe to run multiple times)
    print("Step 1 – Recomputing per-building metrics...")
    recompute_metrics_from_existing(
        MODELS=MODELS,
        STRATEGIES=STRATEGIES,
        BASE_OUTPUT_DIR=BASE_OUTPUT_DIR,
        METRICS_DIR=METRICS_DIR,
        CID=CID
    )

    # 2. Compute and print FULL median summary (including MAE, MSE, RMSE, etc.)
    print("\nStep 2 – Building full median metrics summary...")
    summary = compute_and_log_full_median_metrics(
        MODELS=MODELS,
        STRATEGIES=STRATEGIES,
        METRICS_DIR=METRICS_DIR,
        CID=CID
    )

    print("\nAll done! Check the log and CSV in:", METRICS_DIR)