import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
log_dir = '/home/user/DPFL-Sasmita/FL-Baseline-Codes1/logs/'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join(log_dir, f"metrics_log_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# ============================================================================
# Metrics Computation
# ============================================================================

def evaluate_and_compute_mean_metrics(
    METRICS_DIR: str,
    MODELS: list,
    STRATEGIES: list,
    OUTPUT_DIR: str,
    CID: range = range(101, 1409)
):
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # List to store results for CSV
    results_list = []

    for model_name in MODELS:
        for strategy in STRATEGIES:
            logging.info(f"\n=== Processing Model: {model_name}, Strategy: {strategy} ===")
            
            missing_metric_files = []
            empty_metric_files = []
            nan_metric_files = []
            invalid_metric_files = []
            mae_list = []
            mse_list = []
            rmse_list = []
            mape_list = []
            smape_list = []

            for cid in tqdm(CID, desc="Processing Clients"):
                logging.info(f"\nProcessing Client ID: {cid}")
                metrics_csv = os.path.join(METRICS_DIR, f"cid{cid}_{model_name}_{strategy}_metrics.csv")

                try:
                    if not os.path.exists(metrics_csv):
                        missing_metric_files.append(cid)
                        logging.info(f"CID {cid}: Metric file: Missing")
                        continue

                    metrics_df = pd.read_csv(metrics_csv)
                    if metrics_df.empty:
                        empty_metric_files.append(cid)
                        logging.info(f"CID {cid}: Metric file: Empty")
                        continue

                    if metrics_df[['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'SMAPE (%)']].isna().all().all():
                        nan_metric_files.append(cid)
                        logging.info(f"CID {cid}: Metric file: All NaN")
                        continue

                    metrics = metrics_df.iloc[0]
                    mae = metrics.get('MAE', np.nan)
                    mse = metrics.get('MSE', np.nan)
                    rmse = metrics.get('RMSE', np.nan)
                    mape = metrics.get('MAPE (%)', np.nan)
                    smape = metrics.get('SMAPE (%)', np.nan)

                    if not pd.isna(mae): mae_list.append(mae)
                    if not pd.isna(mse): mse_list.append(mse)
                    if not pd.isna(rmse): rmse_list.append(rmse)
                    if not pd.isna(mape): mape_list.append(mape)
                    if not pd.isna(smape): smape_list.append(smape)

                except Exception as e:
                    invalid_metric_files.append(cid)
                    logging.info(f"[ERROR] CID={cid}: Error reading metrics file: {e}")
                    continue

            if mae_list:
                mean_mae = np.mean(mae_list)
                var_mae = np.var(mae_list)
                mean_mse = np.mean(mse_list)
                var_mse = np.var(mse_list)
                mean_rmse = np.mean(rmse_list)
                var_rmse = np.var(rmse_list)
                mean_mape = np.mean(mape_list)
                var_mape = np.var(mape_list)
                mean_smape = np.mean(smape_list)
                var_smape = np.var(smape_list)
                
                logging.info(f"\nResults for Model: {model_name}, Strategy: {strategy}")
                logging.info(f"Number of processed metric files: {len(mae_list)}")
                logging.info(f"MAE: {mean_mae:.4f} ± {var_mae:.4f}")
                logging.info(f"MSE: {mean_mse:.4f} ± {var_mse:.4f}")
                logging.info(f"RMSE: {mean_rmse:.4f} ± {var_rmse:.4f}")
                logging.info(f"MAPE (%): {mean_mape:.4f} ± {var_mape:.4f}")
                logging.info(f"SMAPE (%): {mean_smape:.4f} ± {var_smape:.4f}")

                # Append results to list for CSV
                results_list.append({
                    'Model': model_name,
                    'Strategy': strategy,
                    'Processed_Files': len(mae_list),
                    'MAE_mean': mean_mae,
                    'MAE_var': var_mae,
                    'MSE_mean': mean_mse,
                    'MSE_var': var_mse,
                    'RMSE_mean': mean_rmse,
                    'RMSE_var': var_rmse,
                    'MAPE_mean (%)': mean_mape,
                    'MAPE_var (%)': var_mape,
                    'SMAPE_mean (%)': mean_smape,
                    'SMAPE_var (%)': var_smape
                })
            else:
                logging.info(f"\nNo valid metric files found for model {model_name} and strategy {strategy}.")
                # Append empty results to maintain consistency
                results_list.append({
                    'Model': model_name,
                    'Strategy': strategy,
                    'Processed_Files': 0,
                    'MAE_mean': np.nan,
                    'MAE_var': np.nan,
                    'MSE_mean': np.nan,
                    'MSE_var': np.nan,
                    'RMSE_mean': np.nan,
                    'RMSE_var': np.nan,
                    'MAPE_mean (%)': np.nan,
                    'MAPE_var (%)': np.nan,
                    'SMAPE_mean (%)': np.nan,
                    'SMAPE_var (%)': np.nan
                })

            unprocessed_cids = set(missing_metric_files + empty_metric_files + nan_metric_files + invalid_metric_files)

            if unprocessed_cids:
                logging.info(f"\nNumber of unprocessed clients: {len(unprocessed_cids)}")
                logging.info("Unprocessed client IDs:")
                logging.info(" ".join(map(str, sorted(unprocessed_cids))))
            else:
                logging.info("\nAll clients were processed successfully.")

            logging.info("\nSummary of Issues:")
            logging.info(f"Missing metric files: {len(missing_metric_files)}")
            logging.info(f"Empty metric files: {len(empty_metric_files)}")
            logging.info(f"NaN metric files: {len(nan_metric_files)}")
            logging.info(f"Invalid metric files: {len(invalid_metric_files)}")

    # Save results to CSV
    results_df = pd.DataFrame(results_list)
    output_csv = os.path.join(OUTPUT_DIR, f"metrics_summary_{timestamp}.csv")
    results_df.to_csv(output_csv, index=False)
    logging.info(f"\nResults saved to {output_csv}")

# ============================================================================
# Run Evaluation
# ============================================================================

if __name__ == "__main__":
    METRICS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes1/metrics40-50-168-T"
    OUTPUT_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes1/results"
    STRATEGIES = ["poc", "poc_no-clustering", "das_no_cluster", "AEpublic_k-means_2enc", "global_model_correct"]
    MODELS = ["simple_cnn"]
    CID = range(101, 1409)

    evaluate_and_compute_mean_metrics(
        METRICS_DIR=METRICS_DIR,
        MODELS=MODELS,
        STRATEGIES=STRATEGIES,
        OUTPUT_DIR=OUTPUT_DIR,
        CID=CID
    )