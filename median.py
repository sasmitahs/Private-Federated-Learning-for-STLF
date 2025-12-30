import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List

def compute_and_log_median_metrics(
    MODELS: List[str],
    STRATEGIES: List[str],
    METRICS_DIR: str,
    CID: range,
    OUTPUT_FILE: str = None
):
    """
    Compute median metrics from individual client metric files and log to file.
    
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
        OUTPUT_FILE = os.path.join(METRICS_DIR, f"median_metrics_summary_{timestamp}.log")
    
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
    logging.info("MEDIAN METRICS SUMMARY")
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
            
            mae_list = []
            mse_list = []
            rmse_list = []
            mape_list = []
            smape_list = []
            nrmse_list = []
            
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
                    
                    # Check if all metrics are NaN
                    if metrics_df[['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'SMAPE (%)', 'NRMSE']].isna().all().all():
                        invalid_files += 1
                        continue
                    
                    metrics = metrics_df.iloc[0]
                    
                    mae = metrics.get('MAE', np.nan)
                    mse = metrics.get('MSE', np.nan)
                    rmse = metrics.get('RMSE', np.nan)
                    mape = metrics.get('MAPE (%)', np.nan)
                    smape = metrics.get('SMAPE (%)', np.nan)
                    nrmse = metrics.get('NRMSE', np.nan)
                    
                    if not pd.isna(mae): mae_list.append(mae)
                    if not pd.isna(mse): mse_list.append(mse)
                    if not pd.isna(rmse): rmse_list.append(rmse)
                    if not pd.isna(mape): mape_list.append(mape)
                    if not pd.isna(smape): smape_list.append(smape)
                    if not pd.isna(nrmse): nrmse_list.append(nrmse)
                    
                    valid_files += 1
                    
                except Exception as e:
                    invalid_files += 1
                    logging.debug(f"Error reading CID {cid}: {e}")
                    continue
            
            # Compute medians
            if mae_list:
                median_mae = np.median(mae_list)
                median_mse = np.median(mse_list)
                median_rmse = np.median(rmse_list)
                median_mape = np.median(mape_list)
                median_smape = np.median(smape_list)
                median_nrmse = np.median(nrmse_list) if nrmse_list else np.nan
                
                # Compute mean and std for reference
                mean_mae = np.mean(mae_list)
                std_mae = np.std(mae_list)
                mean_rmse = np.mean(rmse_list)
                std_rmse = np.std(rmse_list)
                
                logging.info(f"\nFile Statistics:")
                logging.info(f"  Valid files: {valid_files}")
                logging.info(f"  Missing files: {missing_files}")
                logging.info(f"  Empty files: {empty_files}")
                logging.info(f"  Invalid/Error files: {invalid_files}")
                logging.info(f"  Total expected: {len(CID)}")
                
                logging.info(f"\nMedian Metrics:")
                logging.info(f"  MAE:          {median_mae:.4f}")
                logging.info(f"  MSE:          {median_mse:.4f}")
                logging.info(f"  RMSE:         {median_rmse:.4f}")
                logging.info(f"  MAPE (%):     {median_mape:.4f}")
                logging.info(f"  SMAPE (%):    {median_smape:.4f}")
                logging.info(f"  NRMSE:        {median_nrmse:.4f}")
                
                logging.info(f"\nAdditional Statistics:")
                logging.info(f"  Mean MAE:     {mean_mae:.4f} ± {std_mae:.4f}")
                logging.info(f"  Mean RMSE:    {mean_rmse:.4f} ± {std_rmse:.4f}")
                logging.info(f"  Min MAE:      {np.min(mae_list):.4f}")
                logging.info(f"  Max MAE:      {np.max(mae_list):.4f}")
                logging.info(f"  Min RMSE:     {np.min(rmse_list):.4f}")
                logging.info(f"  Max RMSE:     {np.max(rmse_list):.4f}")
                
                # Add to summary
                summary_rows.append({
                    'Model': model_name,
                    'Strategy': strategy,
                    'Valid_Files': valid_files,
                    'Median_MAE': median_mae,
                    'Median_MSE': median_mse,
                    'Median_RMSE': median_rmse,
                    'Median_MAPE': median_mape,
                    'Median_SMAPE': median_smape,
                    'Median_NRMSE': median_nrmse,
                    'Mean_MAE': mean_mae,
                    'Std_MAE': std_mae,
                    'Mean_RMSE': mean_rmse,
                    'Std_RMSE': std_rmse
                })
            else:
                logging.info(f"\nNo valid metric files found!")
                logging.info(f"  Missing files: {missing_files}")
                logging.info(f"  Empty files: {empty_files}")
                logging.info(f"  Invalid/Error files: {invalid_files}")
                
                summary_rows.append({
                    'Model': model_name,
                    'Strategy': strategy,
                    'Valid_Files': 0,
                    'Median_MAE': np.nan,
                    'Median_MSE': np.nan,
                    'Median_RMSE': np.nan,
                    'Median_MAPE': np.nan,
                    'Median_SMAPE': np.nan,
                    'Median_NRMSE': np.nan,
                    'Mean_MAE': np.nan,
                    'Std_MAE': np.nan,
                    'Mean_RMSE': np.nan,
                    'Std_RMSE': np.nan
                })
    
    # Save summary as CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_FILE.replace('.log', '_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    
    logging.info(f"\n{'=' * 80}")
    logging.info("SUMMARY TABLE")
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
    STRATEGIES = ["no-cluster_random", "random_sampling"]
    MODELS = ["dual_simple_ann_fcnn", "dual_cnn_ann_fcnn","dual_cnn_gru_fcnn"]
    METRICS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/metrics40-50-168-T"
    CID = range(101, 1409)
    
    # Compute and log median metrics
    summary_df = compute_and_log_median_metrics(
        MODELS=MODELS,
        STRATEGIES=STRATEGIES,
        METRICS_DIR=METRICS_DIR,
        CID=CID
    )
    
    print("\nDone! Check the output files in the metrics directory.")