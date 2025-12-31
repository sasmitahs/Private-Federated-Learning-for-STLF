import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import logging
from datetime import datetime
from typing import List


# ============================================================
# METRIC FUNCTIONS (ASHRAE + Standard)
# ============================================================

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


# ============================================================
# EVALUATE SINGLE PREDICTION CSV
# ============================================================

def evaluate_autoarima_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")

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


# ============================================================
# RECOMPUTE METRICS FROM EXISTING PREDICTION CSVs
# ============================================================

def recompute_autoarima_metrics(
    PREDICTIONS_DIR: str,
    METRICS_DIR: str,
    CID: range
):
    os.makedirs(METRICS_DIR, exist_ok=True)
    print(f"Recomputing metrics from: {PREDICTIONS_DIR}")
    print(f"Saving to: {METRICS_DIR}")

    for cid in tqdm(CID, desc="Processing Buildings"):
        pred_csv = os.path.join(PREDICTIONS_DIR, f"predictions_building_{cid}.csv")
        metric_csv = os.path.join(METRICS_DIR, f"cid{cid}_autoarima_metrics.csv")

        if not os.path.exists(pred_csv):
            continue  # Skip if prediction missing

        try:
            metrics_df = evaluate_autoarima_csv(pred_csv)
            metrics_df.insert(0, "building_id", cid)
            metrics_df.insert(1, "model", "autoarima")
            metrics_df.to_csv(metric_csv, index=False)
        except Exception as e:
            print(f"[ERROR] CID={cid}: {e}")


# ============================================================
# SUMMARY: MEDIAN / MEAN / VARIANCE
# ============================================================

def compute_and_log_summary(
    METRICS_DIR: str,
    CID: range,
    OUTPUT_FILE: str = None
):
    if OUTPUT_FILE is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_FILE = os.path.join(METRICS_DIR, f"autoarima_ashrae_summary_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(OUTPUT_FILE), logging.StreamHandler()],
        force=True
    )

    logging.info("=" * 80)
    logging.info("AUTOARIMA ASHRAE METRICS SUMMARY (Single Model)")
    logging.info("=" * 80)
    logging.info(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logging.info(f"Metrics Dir: {METRICS_DIR}")
    logging.info(f"Buildings: {CID.start} – {CID.stop - 1}")
    logging.info("=" * 80)

    nmae_list, nmbe_list, nrmse_list = [], [], []
    valid = missing = empty = err = 0

    for cid in CID:
        mcsv = os.path.join(METRICS_DIR, f"cid{cid}_autoarima_metrics.csv")
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

            if pd.notna(nmae): nmae_list.append(nmae)
            if pd.notna(nmbe): nmbe_list.append(nmbe)
            if pd.notna(nrmse): nrmse_list.append(nrmse)
            valid += 1
        except Exception:
            err += 1

    if nmae_list:
        med_nmae = np.median(nmae_list)
        med_nmbe = np.median(nmbe_list)
        med_nrmse = np.median(nrmse_list)
        mean_nmae = np.mean(nmae_list)
        mean_nmbe = np.mean(nmbe_list)
        mean_nrmse = np.mean(nrmse_list)
        var_nmae = np.var(nmae_list, ddof=1)
        var_nmbe = np.var(nmbe_list, ddof=1)
        var_nrmse = np.var(nrmse_list, ddof=1)

        logging.info(f"Valid: {valid} | Missing: {missing} | Empty: {empty} | Error: {err}")
        logging.info(f"Median  – NMAE: {med_nmae:.4f}  NMBE: {med_nmbe:.4f}  NRMSE: {med_nrmse:.4f}")
        logging.info(f"Mean    – NMAE: {mean_nmae:.4f}  NMBE: {mean_nmbe:.4f}  NRMSE: {mean_nrmse:.4f}")
        logging.info(f"Variance– NMAE: {var_nmae:.4f}  NMBE: {var_nmbe:.4f}  NRMSE: {var_nrmse:.4f}")

        summary_df = pd.DataFrame([{
            "Model": "autoarima",
            "Valid_Files": valid,
            "Median_NMAE": med_nmae,
            "Median_NMBE": med_nmbe,
            "Median_NRMSE_mean": med_nrmse,
            "Mean_NMAE": mean_nmae,
            "Mean_NMBE": mean_nmbe,
            "Mean_NRMSE_mean": mean_nrmse,
            "Var_NMAE": var_nmae,
            "Var_NMBE": var_nmbe,
            "Var_NRMSE_mean": var_nrmse
        }])
    else:
        logging.info("No valid metric files found.")
        summary_df = pd.DataFrame([{
            "Model": "autoarima", "Valid_Files": 0,
            "Median_NMAE": np.nan, "Median_NMBE": np.nan, "Median_NRMSE_mean": np.nan,
            "Mean_NMAE": np.nan, "Mean_NMBE": np.nan, "Mean_NRMSE_mean": np.nan,
            "Var_NMAE": np.nan, "Var_NMBE": np.nan, "Var_NRMSE_mean": np.nan
        }])

    # Save CSV
    csv_out = OUTPUT_FILE.replace('.log', '_summary.csv')
    summary_df.to_csv(csv_out, index=False)

    logging.info("\n" + "="*80)
    logging.info("SUMMARY TABLE")
    logging.info(summary_df.to_string(index=False))
    logging.info(f"CSV: {csv_out}")
    logging.info(f"Log: {OUTPUT_FILE}")
    logging.info("="*80)

    return summary_df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # ------------------ CONFIG ------------------
    PREDICTIONS_DIR = "predictions_autoarima_weekly"   # ← Your existing CSVs
    METRICS_DIR     = "metrics_autoarima_weekly"       # ← Will be created
    CID             = range(0, 1409)                   # Adjust as needed

    # ------------------ STEP 1: Recompute Metrics ------------------
    print("Recomputing ASHRAE metrics from existing predictions...")
    recompute_autoarima_metrics(
        PREDICTIONS_DIR=PREDICTIONS_DIR,
        METRICS_DIR=METRICS_DIR,
        CID=CID
    )

    # ------------------ STEP 2: Generate Summary ------------------
    print("\nGenerating summary...")
    summary = compute_and_log_summary(
        METRICS_DIR=METRICS_DIR,
        CID=CID
    )

    print(f"\nDone! Check results in: {METRICS_DIR}")