import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==================== METRIC FUNCTIONS ====================
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)) * 100

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def nrmse_range(y_true, y_pred):
    """Normalized Root Mean Squared Error (range normalization)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    true_range = np.max(y_true) - np.min(y_true)
    if true_range == 0:
        return np.nan
    return rmse / true_range

def nmae(y_true, y_pred):
    """Normalized Mean Absolute Error (mean normalization, ASHRAE)."""
    mae = mean_absolute_error(y_true, y_pred)
    mean_y = np.mean(y_true)
    if mean_y == 0:
        return np.nan
    return 100 * mae / mean_y

def nmbe(y_true, y_pred):
    """Normalized Mean Bias Error (mean normalization, ASHRAE)."""
    mean_bias = np.mean(y_true - y_pred)
    mean_y = np.mean(y_true)
    if mean_y == 0:
        return np.nan
    return 100 * mean_bias / mean_y

def nrmse_mean(y_true, y_pred):
    """Normalized Root Mean Squared Error (mean normalization, ASHRAE)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_y = np.mean(y_true)
    if mean_y == 0:
        return np.nan
    return 100 * rmse / mean_y

# ==================== METRICS EVALUATION ====================
def evaluate_forecast_metrics_per_round(csv_path):
    """
    Compute MAE, MSE, RMSE, MAPE, SMAPE, NRMSE (range), and ASHRAE metrics: NMAE, NMBE, NRMSE (mean) per round.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty or invalid")

    metrics_list = []

    for rnd in sorted(df['round'].unique()):
        df_rnd = df[df['round'] == rnd]
        df_rnd = df_rnd.fillna(0.005)
        y_true = df_rnd["true"].values
        y_pred = df_rnd["pred"].values

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape_val = mape(y_true, y_pred)
        smape_val = smape(y_true, y_pred)
        nrmse_range_val = nrmse_range(y_true, y_pred)

        # ASHRAE metrics
        nmae_val = nmae(y_true, y_pred)
        nmbe_val = nmbe(y_true, y_pred)
        nrmse_mean_val = nrmse_mean(y_true, y_pred)

        metrics_list.append({
            "round": rnd,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE (%)": mape_val,
            "SMAPE (%)": smape_val,
            "NRMSE_range": nrmse_range_val,
            "NMAE (%)": nmae_val,
            "NMBE (%)": nmbe_val,
            "NRMSE_mean (%)": nrmse_mean_val
        })

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df

# ==================== RECOMPUTE METRICS FOR ALL (IF NEEDED) ====================
def recompute_metrics_for_all(BASE_OUTPUT_DIR: str, METRICS_DIR: str, MODELS: List[str], STRATEGIES: List[str], CID: range, ROUNDS: List[int]):
    """
    Recompute and save updated metrics CSV files including the new ASHRAE metrics.
    Assumes predictions CSV files exist in BASE_OUTPUT_DIR.
    """
    os.makedirs(METRICS_DIR, exist_ok=True)

    for cid in CID:
        for model_name in MODELS:
            for strategy in STRATEGIES:
                output_csv = os.path.join(BASE_OUTPUT_DIR, f"{cid}_{model_name}_{strategy}.csv")
                metrics_csv = os.path.join(METRICS_DIR, f"cid{cid}_{model_name}_{strategy}_metrics.csv")

                if not os.path.exists(output_csv):
                    print(f"[WARN] Predictions CSV missing: {output_csv}")
                    continue

                try:
                    metrics_df = evaluate_forecast_metrics_per_round(output_csv)
                    metrics_df.to_csv(metrics_csv, index=False)
                    print(f"[INFO] Updated metrics saved for CID {cid}, {model_name}, {strategy}")
                except Exception as e:
                    print(f"[ERROR] Failed to compute metrics for {output_csv}: {e}")

# ==================== MEDIAN/MEAN/VARIANCE SUMMARY ====================
def compute_and_log_median_metrics(
    MODELS: List[str],
    STRATEGIES: List[str],
    METRICS_DIR: str,
    CID: range,
    OUTPUT_FILE: str = None
):
    """
    Compute median, mean, and variance of metrics from individual client metric files and log to file.
    Focuses on ASHRAE metrics: NMAE, NMBE, NRMSE_mean.
    
    Args:
        MODELS: List of model names
        STRATEGIES: List of strategy names
        METRICS_DIR: Directory containing individual metric files
        CID: Range of client IDs
        OUTPUT_FILE: Path to output log file (optional, auto-generated if None)
    """
    
    # Set up output file
    if OUTPUT_FILE is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_FILE = os.path.join(METRICS_DIR, f"ashrae_median_metrics_summary_{timestamp}.log")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(OUTPUT_FILE),
            logging.StreamHandler()
        ],
        force=True  # Override any existing config
    )
    
    logging.info("=" * 80)
    logging.info("ASHRAE METRICS SUMMARY (NMAE, NMBE, NRMSE_mean)")
    logging.info("=" * 80)
    logging.info(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Metrics Directory: {METRICS_DIR}")
    logging.info(f"Number of Models: {len(MODELS)}")
    logging.info(f"Number of Strategies: {len(STRATEGIES)}")
    logging.info(f"Client ID Range: {CID.start} to {CID.stop - 1}")
    logging.info("=" * 80)
    
    # Create summary dataframe
    summary_rows = []
    
    for model_name in MODELS:
        for strategy in STRATEGIES:
            logging.info(f"\n{'=' * 80}")
            logging.info(f"Model: {model_name} | Strategy: {strategy}")
            logging.info(f"{'=' * 80}")
            
            nmae_list = []
            nmbe_list = []
            nrmse_mean_list = []
            
            valid_files = 0
            missing_files = 0
            empty_files = 0
            invalid_files = 0
            
            for cid in CID:
                metrics_csv = os.path.join(METRICS_DIR, f"cid{cid}_{model_name}_{strategy}_metrics.csv")
                
                if not os.path.exists(metrics_csv):
                    missing_files += 1
                    continue
                
                try:
                    metrics_df = pd.read_csv(metrics_csv)
                    
                    if metrics_df.empty:
                        empty_files += 1
                        continue
                    
                    # Check if ASHRAE metrics are NaN
                    if metrics_df[['NMAE (%)', 'NMBE (%)', 'NRMSE_mean (%)']].isna().all().all():
                        invalid_files += 1
                        continue
                    
                    metrics = metrics_df.iloc[0]
                    
                    nmae = metrics.get('NMAE (%)', np.nan)
                    nmbe = metrics.get('NMBE (%)', np.nan)
                    nrmse_mean = metrics.get('NRMSE_mean (%)', np.nan)
                    
                    if not pd.isna(nmae): nmae_list.append(nmae)
                    if not pd.isna(nmbe): nmbe_list.append(nmbe)
                    if not pd.isna(nrmse_mean): nrmse_mean_list.append(nrmse_mean)
                    
                    valid_files += 1
                    
                except Exception as e:
                    invalid_files += 1
                    logging.debug(f"Error reading CID {cid}: {e}")
                    continue
            
            # Compute statistics if lists are non-empty
            if nmae_list:
                # Medians
                median_nmae = np.median(nmae_list)
                median_nmbe = np.median(nmbe_list)
                median_nrmse_mean = np.median(nrmse_mean_list)
                
                # Means
                mean_nmae = np.mean(nmae_list)
                mean_nmbe = np.mean(nmbe_list)
                mean_nrmse_mean = np.mean(nrmse_mean_list)
                
                # Variances (sample variance)
                var_nmae = np.var(nmae_list, ddof=1)
                var_nmbe = np.var(nmbe_list, ddof=1)
                var_nrmse_mean = np.var(nrmse_mean_list, ddof=1)
                
                logging.info(f"\nFile Statistics:")
                logging.info(f"  Valid files: {valid_files}")
                logging.info(f"  Missing files: {missing_files}")
                logging.info(f"  Empty files: {empty_files}")
                logging.info(f"  Invalid/Error files: {invalid_files}")
                logging.info(f"  Total expected: {len(CID)}")
                
                logging.info(f"\nASHRAE Median Metrics (%):")
                logging.info(f"  NMAE:     {median_nmae:.4f}")
                logging.info(f"  NMBE:     {median_nmbe:.4f}")
                logging.info(f"  NRMSE:    {median_nrmse_mean:.4f}")
                
                logging.info(f"\nASHRAE Mean Metrics (%):")
                logging.info(f"  NMAE:     {mean_nmae:.4f}")
                logging.info(f"  NMBE:     {mean_nmbe:.4f}")
                logging.info(f"  NRMSE:    {mean_nrmse_mean:.4f}")
                
                logging.info(f"\nASHRAE Variance Metrics (%²):")
                logging.info(f"  NMAE:     {var_nmae:.4f}")
                logging.info(f"  NMBE:     {var_nmbe:.4f}")
                logging.info(f"  NRMSE:    {var_nrmse_mean:.4f}")
                
                # Add to summary
                summary_rows.append({
                    'Model': model_name,
                    'Strategy': strategy,
                    'Valid_Files': valid_files,
                    'Median_NMAE': median_nmae,
                    'Median_NMBE': median_nmbe,
                    'Median_NRMSE_mean': median_nrmse_mean,
                    'Mean_NMAE': mean_nmae,
                    'Mean_NMBE': mean_nmbe,
                    'Mean_NRMSE_mean': mean_nrmse_mean,
                    'Var_NMAE': var_nmae,
                    'Var_NMBE': var_nmbe,
                    'Var_NRMSE_mean': var_nrmse_mean
                })
            else:
                logging.info(f"\nNo valid ASHRAE metric files found!")
                logging.info(f"  Missing files: {missing_files}")
                logging.info(f"  Empty files: {empty_files}")
                logging.info(f"  Invalid/Error files: {invalid_files}")
                
                summary_rows.append({
                    'Model': model_name,
                    'Strategy': strategy,
                    'Valid_Files': 0,
                    'Median_NMAE': np.nan,
                    'Median_NMBE': np.nan,
                    'Median_NRMSE_mean': np.nan,
                    'Mean_NMAE': np.nan,
                    'Mean_NMBE': np.nan,
                    'Mean_NRMSE_mean': np.nan,
                    'Var_NMAE': np.nan,
                    'Var_NMBE': np.nan,
                    'Var_NRMSE_mean': np.nan
                })
    
    # Save summary as CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_FILE.replace('.log', '_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    
    logging.info(f"\n{'=' * 80}")
    logging.info("ASHRAE SUMMARY TABLE")
    logging.info(f"{'=' * 80}")
    logging.info(f"\n{summary_df.to_string(index=False)}")
    
    logging.info(f"\n{'=' * 80}")
    logging.info(f"Summary saved to: {summary_csv}")
    logging.info(f"Log saved to: {OUTPUT_FILE}")
    logging.info(f"{'=' * 80}")
    
    return summary_df


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    STRATEGIES = ["poc_correct","poc_nocluster","no-cluster_no-AE_correct","CNN+GRU-Clustering","CNN-clustering","AEpublic_k-means_2enc_correct",]
    MODELS = ["dual_cnn_gru_fcnn","dual_simple_ann_fcnn","dual_cnn_ann_fcnn"]
    BASE_OUTPUT_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/predictions40-50-168-T"  # Path to predictions CSVs
    METRICS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/metrics40-50-168-T"
    CID = range(101, 1409)
    ROUNDS = [40]
    
    # Step 1: Recompute metrics including ASHRAE ones (run this if CSVs don't have them yet)
    print("Recomputing metrics with ASHRAE additions...")
    recompute_metrics_for_all(BASE_OUTPUT_DIR, METRICS_DIR, MODELS, STRATEGIES, CID, ROUNDS)
    
    # Step 2: Compute and log median/mean/variance for ASHRAE metrics
    print("\nComputing summary...")
    summary_df = compute_and_log_median_metrics(
        MODELS=MODELS,
        STRATEGIES=STRATEGIES,
        METRICS_DIR=METRICS_DIR,
        CID=CID
    )
    
    print("\nDone! Check the output files in the metrics directory.")