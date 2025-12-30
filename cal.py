import os
import pandas as pd
import numpy as np

# Directories
predictions_dir = '/home/user/DPFL-Sasmita/FL-Baseline-Codes/predictions40-50-168-T/'
metrics_dir = '/home/user/DPFL-Sasmita/FL-Baseline-Codes/metrics40-50-168-T/'

# Expected client IDs
client_ids = range(101, 1409)  # CID range from 101 to 1408 (1308 clients)

# Model and strategy
model_name = "das_no_cluster"
strategy = "simple_cnn"

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

# Analyze prediction and metric files
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
                metric_status = "Valid"
        except Exception as e:
            invalid_metric_files.append(cid)
            metric_status = f"Error: {e}"
    
    # Detailed logging for problematic CIDs
    if cid in [1405, 1406, 1407, 1408] or pred_status != "Valid" or metric_status != "Valid":
        print(f"CID {cid}: Prediction file: {pred_status}, Metric file: {metric_status}")

# Calculate medians for valid metrics
if mae_list:
    median_mae = pd.Series(mae_list).median()
    median_mse = pd.Series(mse_list).median()
    median_rmse = pd.Series(rmse_list).median()
    median_mape = pd.Series(mape_list).median()
    median_smape = pd.Series(smape_list).median()
    
    print(f"\nNumber of processed metric files: {len(mae_list)}")
    print(f"Median MAE: {median_mae:.4f}")
    print(f"Median MSE: {median_mse:.4f}")
    print(f"Median RMSE: {median_rmse:.4f}")
    print(f"Median MAPE (%): {median_mape:.4f}")
    print(f"Median SMAPE (%): {median_smape:.4f}")
else:
    print("\nNo valid metric files found for the specified model and strategy.")

# Combine unprocessed clients
unprocessed_cids = set(missing_prediction_files + empty_prediction_files + nan_prediction_files + missing_metric_files + invalid_metric_files)

# Report unprocessed clients
if unprocessed_cids:
    print(f"\nNumber of unprocessed clients: {len(unprocessed_cids)}")
    print("Unprocessed client IDs:")
    print(" ".join(map(str, sorted(unprocessed_cids))))
else:
    print("\nAll clients were processed successfully.")

# Detailed summary of issues
print("\nSummary of Issues:")
print(f"Missing prediction files: {len(missing_prediction_files)}")
print(f"Empty prediction files: {len(empty_prediction_files)}")
print(f"NaN prediction files: {len(nan_prediction_files)}")
print(f"Missing metric files: {len(missing_metric_files)}")
print(f"Invalid (empty or NaN) metric files: {len(invalid_metric_files)}")