import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def smape(y_true, y_pred):
    """Compute SMAPE (Symmetric Mean Absolute Percentage Error)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Avoid division by zero
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.where(denominator == 0, 1e-8, denominator)
    
    # Compute SMAPE
    smape_val = np.mean(np.abs(y_true - y_pred) / denominator) * 100 if len(y_true) > 0 else np.nan
    return smape_val

def mape(y_true, y_pred):
    """Compute MAPE (Mean Absolute Percentage Error)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Avoid division by zero
    y_true = np.where(y_true == 0, 1e-8, y_true)
    
    # Compute MAPE
    mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if len(y_true) > 0 else np.nan
    return mape_val

def evaluate_forecast_metrics_per_round(csv_path):
    """Compute metrics from a predictions CSV, handling NaN values."""
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}")
        return pd.DataFrame([{
            "round": "global",
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAPE (%)": np.nan,
            "SMAPE (%)": np.nan
        }]), "Missing"
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"[WARN] CSV is empty: {csv_path}")
            return pd.DataFrame([{
                "round": "global",
                "MAE": np.nan,
                "MSE": np.nan,
                "RMSE": np.nan,
                "MAPE (%)": np.nan,
                "SMAPE (%)": np.nan
            }]), "Empty"
        
        # Check for required columns
        if 'true' not in df.columns or 'pred' not in df.columns:
            print(f"[ERROR] Missing 'true' or 'pred' columns in: {csv_path}")
            return pd.DataFrame([{
                "round": "global",
                "MAE": np.nan,
                "MSE": np.nan,
                "RMSE": np.nan,
                "MAPE (%)": np.nan,
                "SMAPE (%)": np.nan
            }]), "Invalid columns"
        
        # Extract true and predicted values
        y_true = df["true"].values
        y_pred = df["pred"].values
        
        # Handle NaN values by filtering
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # Initialize metrics
        metrics = {
            "round": "global",
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAPE (%)": np.nan,
            "SMAPE (%)": np.nan
        }
        
        # Compute metrics if there are valid data points
        if len(y_true) > 0:
            metrics["MAE"] = mean_absolute_error(y_true, y_pred)
            metrics["MSE"] = mean_squared_error(y_true, y_pred)
            metrics["RMSE"] = np.sqrt(metrics["MSE"])
            metrics["MAPE (%)"] = mape(y_true, y_pred)
            metrics["SMAPE (%)"] = smape(y_true, y_pred)
            status = "Valid"
        else:
            print(f"[WARN] All values are NaN in: {csv_path}")
            status = "All NaN"
        
        return pd.DataFrame([metrics]), status
    
    except Exception as e:
        print(f"[ERROR] Failed to process {csv_path}: {e}")
        return pd.DataFrame([{
            "round": "global",
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAPE (%)": np.nan,
            "SMAPE (%)": np.nan
        }]), f"Error: {str(e)}"

def compute_metrics_for_predictions(
    model_name: str,
    strategy: str,
    cids: range,
    base_output_dir: str,
    metrics_dir: str
):
    """
    Compute metrics for all clients and save to CSV, with detailed error logging.

    Args:
        model_name (str): Name of the model (e.g., 'dual_cnn_gru_fcnn').
        strategy (str): Aggregation strategy (e.g., 'AEpublic_k-means_2enc').
        cids (range): Range of client IDs to process.
        base_output_dir (str): Directory containing prediction CSVs.
        metrics_dir (str): Directory to save metric CSVs.
    """
    os.makedirs(metrics_dir, exist_ok=True)
    
    all_metrics = []
    unprocessed_cids = []
    error_details = []
    
    for cid in cids:
        print(f"Processing Client ID: {cid}")
        csv_path = os.path.join(base_output_dir, f"{cid}_{model_name}_{strategy}.csv")
        metrics_csv = os.path.join(metrics_dir, f"cid{cid}_{model_name}_{strategy}_metrics.csv")
        
        # Compute metrics
        metrics_df, status = evaluate_forecast_metrics_per_round(csv_path)
        metrics_df["cid"] = cid
        
        if status != "Valid":
            unprocessed_cids.append(cid)
            error_details.append(f"CID {cid}: {status}")
        else:
            all_metrics.append(metrics_df)
            try:
                # Save individual metrics
                metrics_df.to_csv(metrics_csv, index=False)
                print(f"Metrics saved to {metrics_csv}")
            except Exception as e:
                print(f"[ERROR] Failed to save metrics to {metrics_csv}: {e}")
                unprocessed_cids.append(cid)
                error_details.append(f"CID {cid}: Failed to save metrics: {str(e)}")
    
    # Save aggregated metrics
    if all_metrics:
        aggregated_metrics = pd.concat(all_metrics, ignore_index=True)
        aggregated_csv = os.path.join(metrics_dir, f"{model_name}_{strategy}_all_metrics.csv")
        try:
            aggregated_metrics.to_csv(aggregated_csv, index=False)
            print(f"Aggregated metrics saved to {aggregated_csv}")
            
            # Calculate and print median metrics
            valid_metrics = aggregated_metrics.dropna(subset=['MAE'])
            if not valid_metrics.empty:
                print(f"\nNumber of processed metric files: {len(valid_metrics)}")
                print(f"Median MAE: {valid_metrics['MAE'].median():.4f}")
                print(f"Median MSE: {valid_metrics['MSE'].median():.4f}")
                print(f"Median RMSE: {valid_metrics['RMSE'].median():.4f}")
                print(f"Median MAPE (%): {valid_metrics['MAPE (%)'].median():.4f}")
                print(f"Median SMAPE (%): {valid_metrics['SMAPE (%)'].median():.4f}")
            else:
                print("\nNo valid metrics computed.")
        except Exception as e:
            print(f"[ERROR] Failed to save aggregated metrics to {aggregated_csv}: {e}")
    
    # Report unprocessed clients
    if unprocessed_cids:
        print(f"\nNumber of unprocessed clients: {len(unprocessed_cids)}")
        print("Unprocessed client IDs:")
        print(" ".join(map(str, sorted(unprocessed_cids))))
        print("\nError details:")
        for detail in error_details:
            print(detail)

if __name__ == "__main__":
    STRATEGIES = ["poc"]
    #MODELS = ["dual_cnn_gru_fcnn", "dual_gru_fcnn","dual_gru_gru"]
    MODELS = ["dual_cnn_gru_fcnn"]
    CID = range(101, 1409)  # 1308 clients
    BASE_OUTPUT_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/predictions40-50-168-T"
    METRICS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/metrics40-50-168-T"

    for model_name in MODELS:
        for strategy in STRATEGIES:
            compute_metrics_for_predictions(
                model_name=model_name,
                strategy=strategy,
                cids=CID,
                base_output_dir=BASE_OUTPUT_DIR,
                metrics_dir=METRICS_DIR
            )