import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from Models import MoELSTM, LSTMModel, model_fn
import numpy as np
from typing import List, Tuple

# Assuming these imports are available as per your original code
from Preprocess import (
    compute_metrics,
    convert_timeseries_to_numpy,
    create_dataloader,
    load_building_series,
    split_series_list,
)
from my_utils import train_model, load_energy_data_feather, get_weights, set_weights


@torch.no_grad()
def rolling_forecast_on_test(cid, model, filepath="train_final.feather", input_len=168, output_len=24):
    """
    Perform rolling window forecast on the test data using a trained model and return
    unscaled predictions and ground truths with actual timestamps.

    Args:
        cid (int): Client/building ID.
        model (nn.Module): Trained PyTorch model.
        filepath (str): Path to the Feather file.
        input_len (int): Input sequence length.
        output_len (int): Prediction horizon.

    Returns:
        Tuple[List[TimeSeries], List[TimeSeries]]: (predictions_ts_list, ground_truth_ts_list)
    """
    print(f"[DEBUG] rolling_forecast_on_test: CID={cid}")

    # Load and filter data
    df = pd.read_feather(filepath)
    df = df[df['building_id'] == cid]
    df['meter_reading'] = df['meter_reading'].fillna(0)

    if df.empty:
        raise ValueError(f"No data found for building_id {cid}")

    # Create TimeSeries and scale
    ts = TimeSeries.from_dataframe(
        df,
        time_col='timestamp',
        value_cols='meter_reading',
        fill_missing_dates=True,
        freq='h'
    )

    _, test_series = ts.split_before(0.75)

    # Scale
    scaler = MinMaxScaler(feature_range=(0.1, 1))
    transformer = Scaler(scaler)
    test_series_scaled = transformer.fit_transform(test_series)

    test_values_scaled = test_series_scaled.values().squeeze()
    test_timestamps = test_series_scaled.time_index

    predictions_ts_list = []
    ground_truth_ts_list = []

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    for i in range(0, len(test_values_scaled) - input_len - output_len + 1, output_len):
        input_seq = test_values_scaled[i:i+input_len]
        true_output = test_values_scaled[i+input_len:i+input_len+output_len]
        true_time = test_timestamps[i+input_len:i+input_len+output_len]

        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # [1, input_len, 1]

        pred = model(input_tensor)
        
        if pred.dim() == 3:
            pred = pred.squeeze(0).squeeze(-1)
        else:
            pred = pred.squeeze(0)

        # Convert prediction & ground truth to TimeSeries
        pred_ts = TimeSeries.from_times_and_values(true_time, pred.cpu().numpy())
        true_ts = TimeSeries.from_times_and_values(true_time, true_output)

        # Inverse transform
        pred_unscaled = transformer.inverse_transform(pred_ts)
        true_unscaled = transformer.inverse_transform(true_ts)

        predictions_ts_list.append(pred_unscaled)
        ground_truth_ts_list.append(true_unscaled)

    return predictions_ts_list, ground_truth_ts_list


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)) * 100


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true = np.where(y_true == 0, 1e-8, y_true)  # avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def nrmse_range(y_true, y_pred):
    """Normalized RMSE by range."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    true_range = np.max(y_true) - np.min(y_true)
    return rmse / true_range if true_range != 0 else np.nan


def nmae(y_true, y_pred):
    """Normalized Mean Absolute Error."""
    mae = mean_absolute_error(y_true, y_pred)
    mean_y = np.mean(y_true)
    return 100 * mae / mean_y if mean_y != 0 else np.nan


def nmbe(y_true, y_pred):
    """Normalized Mean Bias Error."""
    mean_bias = np.mean(y_true - y_pred)
    mean_y = np.mean(y_true)
    return 100 * mean_bias / mean_y if mean_y != 0 else np.nan


def nrmse_mean(y_true, y_pred):
    """Normalized RMSE by mean."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_y = np.mean(y_true)
    return 100 * rmse / mean_y if mean_y != 0 else np.nan


def evaluate_forecast_metrics_per_round(csv_path):
    """
    Reads forecast CSV and computes comprehensive metrics including 
    MAPE, MAE, SMAPE, RMSE, MSE, NRMSE_range, NMAE, NMBE, and NRMSE_mean.

    Args:
        csv_path (str): Path to the CSV with columns: timestamp, true, pred

    Returns:
        pd.DataFrame: Metrics summary
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty or invalid")

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


def get_model_predictions_csv(model_name: str, cid: int, epoch: int, model_dir: str, output_csv: str):
    """
    Load the model for a specific epoch, predict on test set for cid, and save predictions to CSV.

    Args:
        model_name (str): Name of the model (e.g., 'lstm', 'moe_lstm').
        cid (int): Client/building ID.
        epoch (int): Epoch number.
        model_dir (str): Directory containing model weights.
        output_csv (str): Path to save prediction CSV.
    """
    model_path = os.path.join(model_dir, f"{model_name}_central_epoch_{epoch}.pt")

    if not os.path.exists(model_path):
        print(f"[WARN] Model not found: {model_path}")
        return

    model = model_fn(model_name)
    # Use weights_only=False for compatibility with older checkpoints
    # This is safe if you trust the source of your model files
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint contains metadata (epoch, model_state_dict, avg_loss, etc.)
        state_dict = checkpoint['model_state_dict']
    else:
        # Checkpoint is just the state_dict
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    pred_ts_list, gt_ts_list = rolling_forecast_on_test(cid=cid, model=model)

    rows = []
    for pred_ts, true_ts in zip(pred_ts_list, gt_ts_list):
        df_pred = pd.DataFrame({"timestamp": pred_ts.time_index, "pred": pred_ts.values().squeeze()})
        df_true = pd.DataFrame({"timestamp": true_ts.time_index, "true": true_ts.values().squeeze()})
        df_merged = pd.merge(df_true, df_pred, on="timestamp", how="inner")
        rows.append(df_merged[["timestamp", "true", "pred"]])

    # Combine all rows
    final_df = pd.concat(rows, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"[INFO] Forecasts written to {output_csv}")


def get_model_predictions_metric_multi_epoch(
    MODELS,
    EPOCHS,
    BASE_RESULTS_DIR: str,
    BASE_OUTPUT_DIR: str,
    METRICS_DIR: str,
    CID: range
):
    """
    For each client in CID, model in MODELS, and epoch in EPOCHS,
    generates forecast predictions and computes metrics.

    Args:
        MODELS (List[str]): List of model names (e.g., ["lstm", "moe_lstm"]).
        EPOCHS (List[int]): List of epoch numbers (e.g., [10, 11, ..., 30]).
        BASE_RESULTS_DIR (str): Directory containing saved model weights.
        BASE_OUTPUT_DIR (str): Directory to save prediction CSVs.
        METRICS_DIR (str): Directory to save metric CSVs.
        CID (range): Range of client IDs (e.g., range(0, 1409)).
    """
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    for epoch in EPOCHS:
        print(f"\n{'='*80}")
        print(f"PROCESSING EPOCH: {epoch}")
        print(f"{'='*80}\n")
        
        for cid in tqdm(CID, desc=f"Epoch {epoch} - Processing Client IDs"):
            for model_name in MODELS:
                model_dir = os.path.join(BASE_RESULTS_DIR, model_name)
                output_csv = os.path.join(BASE_OUTPUT_DIR, f"epoch{epoch}_{cid}_{model_name}.csv")
                metrics_csv = os.path.join(METRICS_DIR, f"epoch{epoch}_cid{cid}_{model_name}_metrics.csv")

                try:
                    # Generate predictions and save to CSV
                    get_model_predictions_csv(
                        model_name=model_name,
                        cid=cid,
                        epoch=epoch,
                        model_dir=model_dir,
                        output_csv=output_csv
                    )

                    # Evaluate metrics and save to CSV
                    if os.path.exists(output_csv):
                        metrics_df = evaluate_forecast_metrics_per_round(output_csv)
                        metrics_df.insert(0, "epoch", epoch)
                        metrics_df.insert(1, "building_id", cid)
                        metrics_df.insert(2, "model", model_name)
                        metrics_df.to_csv(metrics_csv, index=False)
                    else:
                        print(f"[WARN] No predictions found for {output_csv}")

                except Exception as e:
                    print(f"[ERROR] epoch={epoch}, model={model_name}, cid={cid}: {e}")

        print(f"\nCompleted Epoch {epoch}\n")


def aggregate_all_metrics(METRICS_DIR: str, output_file: str = "all_epochs_metrics_summary.csv"):
    """
    Aggregates all individual metric CSV files into a single summary file.
    
    Args:
        METRICS_DIR (str): Directory containing all metric CSV files.
        output_file (str): Output filename for the aggregated metrics.
    """
    all_metrics = []
    
    for file in os.listdir(METRICS_DIR):
        if file.endswith("_metrics.csv"):
            filepath = os.path.join(METRICS_DIR, file)
            df = pd.read_csv(filepath)
            all_metrics.append(df)
    
    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)
        output_path = os.path.join(METRICS_DIR, output_file)
        combined_df.to_csv(output_path, index=False)
        print(f"\n[INFO] Aggregated metrics saved to: {output_path}")
        
        # Print summary statistics with all metrics
        print("\n" + "="*100)
        print("SUMMARY STATISTICS BY EPOCH")
        print("="*100)
        summary = combined_df.groupby(['epoch', 'model']).agg({
            'MAE': 'mean',
            'RMSE': 'mean',
            'MAPE (%)': 'mean',
            'SMAPE (%)': 'mean',
            'NRMSE_range': 'mean',
            'NMAE (%)': 'mean',
            'NMBE (%)': 'mean',
            'NRMSE_mean (%)': 'mean'
        }).reset_index()
        print(summary.to_string(index=False))
    else:
        print("[WARN] No metric files found to aggregate")


def compute_median_metrics_by_epoch(
    METRICS_DIR: str,
    MODELS: List[str],
    EPOCHS: List[int],
    output_file: str = "median_metrics_by_epoch.csv"
):
    """
    Computes median metrics across all buildings for each epoch and model.
    
    Args:
        METRICS_DIR (str): Directory containing metric CSV files.
        MODELS (List[str]): List of model names.
        EPOCHS (List[int]): List of epoch numbers.
        output_file (str): Output filename for median summary.
    """
    summary_rows = []
    
    for epoch in tqdm(EPOCHS, desc="Computing median metrics"):
        for model_name in MODELS:
            # Lists to collect metrics across buildings
            mae_l, mse_l, rmse_l = [], [], []
            mape_l, smape_l = [], []
            nrmse_range_l, nmae_l, nmbe_l, nrmse_mean_l = [], [], [], []
            
            valid_count = 0
            
            # Collect metrics from all buildings for this epoch and model
            for file in os.listdir(METRICS_DIR):
                if file.startswith(f"epoch{epoch}_") and f"_{model_name}_" in file and file.endswith("_metrics.csv"):
                    filepath = os.path.join(METRICS_DIR, file)
                    try:
                        df = pd.read_csv(filepath)
                        if df.empty:
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
                        
                        valid_count += 1
                    except Exception as e:
                        print(f"[Read error] {filepath}: {e}")
            
            # Compute medians
            def safe_median(lst): return np.median(lst) if lst else np.nan
            
            summary_rows.append({
                "Epoch": epoch,
                "Model": model_name,
                "Valid_Buildings": valid_count,
                "Median_MAE": safe_median(mae_l),
                "Median_MSE": safe_median(mse_l),
                "Median_RMSE": safe_median(rmse_l),
                "Median_MAPE": safe_median(mape_l),
                "Median_SMAPE": safe_median(smape_l),
                "Median_NRMSE_range": safe_median(nrmse_range_l),
                "Median_NMAE": safe_median(nmae_l),
                "Median_NMBE": safe_median(nmbe_l),
                "Median_NRMSE_mean": safe_median(nrmse_mean_l),
            })
    
    # Save CSV summary
    summary_df = pd.DataFrame(summary_rows)
    output_path = os.path.join(METRICS_DIR, output_file)
    summary_df.to_csv(output_path, index=False)
    
    print("\n" + "="*100)
    print("MEDIAN METRICS BY EPOCH AND MODEL")
    print("="*100)
    print(summary_df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nMedian metrics summary saved to: {output_path}")
    print("="*100)
    
    return summary_df


# ============================================================================
# MAIN CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    EPOCHS = list(range(10, 31))  # Epochs 10 to 30 inclusive
    MODELS = ["lstm", "gru", "simple_ann", "moe_lstm", "simple_cnn", "cnn_gru_no_cov"]
    BASE_RESULTS_DIR = "results"
    BASE_OUTPUT_DIR = "predictions_epochs_10_30"
    METRICS_DIR = "metrics_epochs_10_30"
    CID = range(0, 1409)  # All client IDs
    
    print(f"Starting evaluation for epochs: {EPOCHS[0]} to {EPOCHS[-1]}")
    print(f"Models: {MODELS}")
    print(f"Number of clients: {len(CID)}")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print(f"Metrics directory: {METRICS_DIR}")
    
    # Run the evaluation for all epochs
    get_model_predictions_metric_multi_epoch(
        MODELS=MODELS,
        EPOCHS=EPOCHS,
        BASE_RESULTS_DIR=BASE_RESULTS_DIR,
        BASE_OUTPUT_DIR=BASE_OUTPUT_DIR,
        METRICS_DIR=METRICS_DIR,
        CID=CID
    )
    
    # Aggregate all metrics into a single file
    aggregate_all_metrics(METRICS_DIR)
    
    # Compute median metrics by epoch for better comparison
    compute_median_metrics_by_epoch(
        METRICS_DIR=METRICS_DIR,
        MODELS=MODELS,
        EPOCHS=EPOCHS
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE FOR ALL EPOCHS!")
    print("="*80)