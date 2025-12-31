import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Set up logging
log_dir = '/home/user/DPFL-Sasmita/FL-Baseline-Codes1/logs/'
os.makedirs(log_dir, exist_ok=True)

# Create a timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join(log_dir, f"metrics_log_{timestamp}.log")

# Configure logging to write to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Use plain message format to match original print output
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This ensures output is also printed to console
    ]
)

# Directories
predictions_dir = '/home/user/DPFL-Sasmita/FL-Baseline-Codes1/predictions40-50-168-T/'
metrics_dir = '/home/user/DPFL-Sasmita/FL-Baseline-Codes1/metrics40-50-168-T/'

# Expected client IDs
client_ids = range(101, 1409)  # CID range from 101 to 1408 (1308 clients)

# Model and strategies
model_name = "simple_cnn"
strategies = [ "AEpublic_k-means_2enc"]

# Process each strategy
for strategy in strategies:
    logging.info(f"\n=== Processing Strategy: {strategy} ===")
    
    # Lists to store issues
    missing_prediction_files = []
    empty_prediction_files = []
    nan_prediction_files = []
    missing_metric_files = []
    invalid_metric_files = []
    mae_list = []
    mse_list = []
    rmse_list = []
    mape_list = []
    smape_list = []
    nrmse_list = []

    # Analyze prediction and metric files for the current strategy
    for cid in client_ids:
        # Check prediction file
        pred_file = os.path.join(predictions_dir, f"{cid}_{model_name}_{strategy}.csv")
        pred_status = ""
        
        if not os.path.exists(pred_file):
            missing_prediction_files.append(cid)
            pred_status = "Missing"
        else:
            try:
                pred_df = pd.read_csv(pred_file)
                if pred_df.empty:
                    empty_prediction_files.append(cid)
                    pred_status = "Empty"
                elif pred_df[['true', 'pred']].isna().all().all():
                    nan_prediction_files.append(cid)
                    pred_status = "All NaN"
                else:
                    pred_status = "Valid"
            except Exception as e:
                empty_prediction_files.append(cid)
                pred_status = f"Error: {e}"
        
        # Check metric file
        metric_file = os.path.join(metrics_dir, f"cid{cid}_{model_name}_{strategy}_metrics.csv")
        metric_status = ""
        
        if not os.path.exists(metric_file):
            missing_metric_files.append(cid)
            metric_status = "Missing"
        else:
            try:
                metric_df = pd.read_csv(metric_file)
                if metric_df.empty or metric_df[['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'SMAPE (%)']].isna().all().all():
                    invalid_metric_files.append(cid)
                    metric_status = "Empty or all NaN"
                else:
                    # Extract metrics
                    metrics = metric_df.iloc[0]
                    mae = metrics.get('MAE', np.nan)
                    mse = metrics.get('MSE', np.nan)
                    rmse = metrics.get('RMSE', np.nan)
                    mape = metrics.get('MAPE (%)', np.nan)
                    smape = metrics.get('SMAPE (%)', np.nan)
                    
                    # Calculate NRMSE from prediction file
                    if pred_status == "Valid":
                        y_true = pred_df['true'].values
                        y_pred = pred_df['pred'].values
                        true_range = np.max(y_true) - np.min(y_true)
                        if true_range != 0:  # Avoid division by zero
                            nrmse = rmse / true_range
                        else:
                            nrmse = np.nan  # Handle case where true values are constant
                    else:
                        nrmse = np.nan
                    
                    # Append non-NaN metrics
                    if not pd.isna(mae):
                        mae_list.append(mae)
                    if not pd.isna(mse):
                        mse_list.append(mse)
                    if not pd.isna(rmse):
                        rmse_list.append(rmse)
                    if not pd.isna(mape):
                        mape_list.append(mape)
                    if not pd.isna(smape):
                        smape_list.append(smape)
                    if not pd.isna(nrmse):
                        nrmse_list.append(nrmse)
                    metric_status = "Valid"
            except Exception as e:
                invalid_metric_files.append(cid)
                metric_status = f"Error: {e}"
        
        # Detailed logging for problematic CIDs
        if cid in [1405, 1406, 1407, 1408] or pred_status != "Valid" or metric_status != "Valid":
            logging.info(f"CID {cid}: Prediction file: {pred_status}, Metric file: {metric_status}")

    # Calculate medians for valid metrics
    if mae_list:
        median_mae = pd.Series(mae_list).median()
        median_mse = pd.Series(mse_list).median()
        median_rmse = pd.Series(rmse_list).median()
        median_mape = pd.Series(mape_list).median()
        median_smape = pd.Series(smape_list).median()
        median_nrmse = pd.Series(nrmse_list).median() if nrmse_list else np.nan
        
        logging.info(f"\nResults for Strategy: {strategy}")
        logging.info(f"Number of processed metric files: {len(mae_list)}")
        logging.info(f"Median MAE: {median_mae:.4f}")
        logging.info(f"Median MSE: {median_mse:.4f}")
        logging.info(f"Median RMSE: {median_rmse:.4f}")
        logging.info(f"Median MAPE (%): {median_mape:.4f}")
        logging.info(f"Median SMAPE (%): {median_smape:.4f}")
        logging.info(f"Median NRMSE: {median_nrmse:.4f}")
    else:
        logging.info(f"\nNo valid metric files found for model {model_name} and strategy {strategy}.")

    # Combine unprocessed clients
    unprocessed_cids = set(missing_prediction_files + empty_prediction_files + nan_prediction_files + missing_metric_files + invalid_metric_files)

    # Report unprocessed clients
    if unprocessed_cids:
        logging.info(f"\nNumber of unprocessed clients: {len(unprocessed_cids)}")
        logging.info("Unprocessed client IDs:")
        logging.info(" ".join(map(str, sorted(unprocessed_cids))))
    else:
        logging.info("\nAll clients were processed successfully.")

    # Detailed summary of issues
    logging.info("\nSummary of Issues:")
    logging.info(f"Missing prediction files: {len(missing_prediction_files)}")
    logging.info(f"Empty prediction files: {len(empty_prediction_files)}")
    logging.info(f"NaN prediction files: {len(nan_prediction_files)}")
    logging.info(f"Missing metric files: {len(missing_metric_files)}")
    logging.info(f"Invalid (empty or NaN) metric files: {len(invalid_metric_files)}")