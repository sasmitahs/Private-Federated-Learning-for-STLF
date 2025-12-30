import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from darts import TimeSeries
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
from Models import model_fn
from darts.dataprocessing.transformers import Scaler

# ============================================================================
# Helper Functions
# ============================================================================

def safe_inverse_transform_scaler(scaler: MinMaxScaler, arr: np.ndarray, n_channels_expected: int = 4):
    """Inverse transform when we only have 1 channel to revert (the target)."""
    arr = np.asarray(arr).reshape(-1, 1)
    pad = np.zeros((arr.shape[0], max(0, n_channels_expected - 1)))
    stacked = np.concatenate([arr, pad], axis=1)
    return scaler.inverse_transform(stacked)[:, 0]

def _get_num_embeddings_for_model(model) -> int:
    """Extract num_embeddings from model's primary_use embedding layer."""
    for attr in ("primary_use_embedding", "primary_use_embed", "primary_use_embeddding"):
        emb = getattr(model, attr, None)
        if emb is not None and isinstance(emb, torch.nn.Embedding):
            return emb.num_embeddings
    return None

# ============================================================================
# Rolling Forecast Function
# ============================================================================

@torch.no_grad()
def rolling_forecast_on_test(
    cid: int,
    model: torch.nn.Module,
    filepath: str = "train_final.feather",
    input_len: int = 168,
    output_len: int = 24,
) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    """
    Predictions: Multi-channel (as in code 1)
    True values: Single-channel meter_reading, Darts scaler (as in code 2)
    """
    print(f"[DEBUG] rolling_forecast_on_test: CID={cid}")

    # Load data
    df = pd.read_feather(filepath)
    df = df[df['building_id'] == cid].copy()
    if df.empty:
        raise ValueError(f"No data found for building_id {cid}")

    # Fill missing values
    df['meter_reading'] = df['meter_reading'].fillna(0.0)
    df['air_temperature'] = df['air_temperature'].fillna(df['air_temperature'].mean())
    df['primary_use_idx'] = df.get('primary_use', 0)
    if 'primary_use' in df.columns:
        primary_map = {cat: idx for idx, cat in enumerate(df['primary_use'].unique())}
        df['primary_use_idx'] = df['primary_use'].map(primary_map).fillna(0).astype(int)

    # Build multi-channel input for predictions
    meter_values = df['meter_reading'].values
    try:
        stl = STL(meter_values, period=24)
        res = stl.fit()
        trend, seasonal, resid = res.trend, res.seasonal, res.resid
    except:
        trend, seasonal, resid = np.zeros_like(meter_values), np.zeros_like(meter_values), meter_values.copy()

    multi_channel = np.stack([meter_values, trend, seasonal, resid], axis=-1)
    air_temp_vals = df['air_temperature'].values
    primary_use_vals = df['primary_use_idx'].values

    split_idx = int(0.75 * len(multi_channel))
    test_values = multi_channel[split_idx:]
    test_air_temp = air_temp_vals[split_idx:]
    test_primary_use = primary_use_vals[split_idx:]
    test_time = pd.to_datetime(df['timestamp'].values[split_idx:])

    if len(test_values) < input_len + output_len:
        return [], []

    # Scale multi-channel inputs
    scaler_mc = MinMaxScaler(feature_range=(0.1, 1))
    test_values_scaled = scaler_mc.fit_transform(test_values)
    test_air_temp_scaled = MinMaxScaler(feature_range=(0.1, 1)).fit_transform(test_air_temp.reshape(-1, 1))

    # Build single-channel true values
    ts = TimeSeries.from_dataframe(df[split_idx:], time_col='timestamp', value_cols='meter_reading', fill_missing_dates=True, freq='h')
    transformer = Scaler(MinMaxScaler(feature_range=(0.1,1)))
    test_series_scaled = transformer.fit_transform(ts)
    test_values_true_scaled = test_series_scaled.values().squeeze()
    test_time_true = test_series_scaled.time_index

    predictions_ts_list = []
    ground_truth_ts_list = []

    model.eval()
    device = next(model.parameters()).device
    n_emb = _get_num_embeddings_for_model(model)
    has_embedding = n_emb is not None

    max_start = len(test_values_scaled) - input_len - output_len
    for i in range(0, max_start + 1, output_len):
        x_ts_np = test_values_scaled[i:i+input_len]
        x_air_np = test_air_temp_scaled[i:i+input_len]
        primary_np = test_primary_use[i:i+input_len]
        x_ts = torch.tensor(x_ts_np, dtype=torch.float32).unsqueeze(0).to(device)
        x_cov = torch.tensor(x_air_np, dtype=torch.float32).unsqueeze(0).to(device)
        primary_use_tensor = torch.tensor(np.clip(primary_np, 0, n_emb-1) if has_embedding else np.zeros(input_len), dtype=torch.long).unsqueeze(0).to(device)

        try:
            pred = model(x_ts, x_cov=x_cov, primary_use=primary_use_tensor)
        except TypeError:
            pred = model(x_ts)

        if pred.dim() == 3: pred = pred.squeeze(0).squeeze(-1)
        elif pred.dim() == 2: pred = pred.squeeze(0)
        pred_np = pred.detach().cpu().numpy()

        true_start = i + input_len
        true_end = true_start + output_len
        true_output = test_values_true_scaled[true_start:true_end]
        true_ts = TimeSeries.from_times_and_values(test_time_true[true_start:true_end], true_output)
        true_unscaled = transformer.inverse_transform(true_ts)

        pred_ts = TimeSeries.from_times_and_values(test_time_true[true_start:true_end], pred_np)
        pred_unscaled = scaler_mc.inverse_transform(np.pad(pred_np.reshape(-1,1), ((0,0),(0,3)), mode='constant'))[:,0]
        pred_ts = TimeSeries.from_times_and_values(test_time_true[true_start:true_end], pred_unscaled)

        predictions_ts_list.append(pred_ts)
        ground_truth_ts_list.append(true_unscaled)

    return predictions_ts_list, ground_truth_ts_list

# ============================================================================
# Metrics Computation
# ============================================================================

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)) * 100

def mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_forecast_metrics_per_round(csv_path):
    """Compute metrics from a predictions CSV."""
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}")
        return pd.DataFrame([{
            "round": "global",
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAPE (%)": np.nan,
            "SMAPE (%)": np.nan
        }])
    
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
        }])
    
    # Drop rows with NaN in 'true' or 'pred'
    df = df.dropna(subset=['true', 'pred'])
    
    if df.empty:
        print(f"[WARN] No valid data after dropping NaNs: {csv_path}")
        return pd.DataFrame([{
            "round": "global",
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAPE (%)": np.nan,
            "SMAPE (%)": np.nan
        }])
    
    y_true = df["true"].values
    y_pred = df["pred"].values
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape_val = mape(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    
    return pd.DataFrame([{
        "round": "global",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape_val,
        "SMAPE (%)": smape_val
    }])

# ============================================================================
# Model Predictions with Rounds
# ============================================================================

def get_model_predictions_csv(
    model_name: str,
    cid: int,
    rounds: List[int],
    model_dir: str,
    output_csv: str,
    aggr_strat: str,
    filepath: str = "train_final.feather"
):
    """Load model for specified rounds, predict, and save to CSV."""
    rows = []
    
    for round_num in rounds:
        # Construct model path with round and strategy (adjusted for .pt extension)
        model_path = os.path.join(model_dir, f"{model_name}_round_{round_num}_{aggr_strat}.pt")
        if not os.path.exists(model_path):
            print(f"[WARN] Model not found: {model_path}")
            continue

        # Load model
        model = model_fn(model_name)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # Handle DataParallel wrapper
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        # Get predictions
        try:
            pred_ts_list, ground_truth_ts_list = rolling_forecast_on_test(cid=cid, model=model, filepath=filepath)
        except Exception as e:
            print(f"[ERROR] Forecast failed for CID={cid}, round={round_num}: {e}")
            continue
        
        if len(pred_ts_list) == 0:
            print(f"[WARN] No predictions for CID={cid}, round={round_num}")
            continue
        
        # Save predictions for this round
        for pred_ts, true_ts in zip(pred_ts_list, ground_truth_ts_list):
            df_pred = pd.DataFrame({"timestamp": pred_ts.time_index, "pred": pred_ts.values().squeeze()})
            df_true = pd.DataFrame({"timestamp": true_ts.time_index, "true": true_ts.values().squeeze()})
            df_merged = pd.merge(df_true, df_pred, on="timestamp", how="inner")
            df_merged["round"] = round_num
            rows.append(df_merged)
    
    if rows:
        final_df = pd.concat(rows, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"[INFO] Saved predictions to {output_csv}")
    else:
        print(f"[WARN] No predictions generated for CID={cid}")

# ============================================================================
# Main Evaluation Loop
# ============================================================================

def get_model_predictions_metric(
    MODELS,
    STRATEGIES,
    ROUNDS,
    BASE_RESULTS_DIR: str,
    BASE_OUTPUT_DIR: str,
    METRICS_DIR: str,
    CID: range
):
    """
    For each client in CID, model in MODELS, and strategy in STRATEGIES,
    generates forecast predictions and computes metrics.

    Args:
        MODELS (List[str]): List of model names (e.g., ["dual_cnn_gru_fcnn", "dual_gru_fcnn"])
        STRATEGIES (List[str]): List of aggregation strategies (e.g., ["AEpublic_k-means_2enc"])
        ROUNDS (List[int]): Rounds to evaluate (e.g., [40])
        BASE_RESULTS_DIR (str): Directory containing saved model weights.
        BASE_OUTPUT_DIR (str): Directory to save prediction CSVs.
        METRICS_DIR (str): Directory to save metric CSVs.
        CID (range): Range of client IDs (e.g., range(101, 102))
    """
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    for cid in CID:
        print(f"\nProcessing Client ID: {cid}")

        for model_name in MODELS:
            for strategy in STRATEGIES:
                model_dir = os.path.join(BASE_RESULTS_DIR, model_name)
                output_csv = os.path.join(BASE_OUTPUT_DIR, f"{cid}_{model_name}_{strategy}.csv")
                metrics_csv = os.path.join(METRICS_DIR, f"cid{cid}_{model_name}_{strategy}_metrics.csv")

                print(f"\n Model: {model_name}, Strategy: {strategy}")

                try:
                    # Generate predictions and save to CSV
                    get_model_predictions_csv(
                        model_name=model_name,
                        cid=cid,
                        rounds=ROUNDS,
                        model_dir=model_dir,
                        output_csv=output_csv,
                        aggr_strat=strategy
                    )

                    # Evaluate metrics and save to CSV
                    metrics_df = evaluate_forecast_metrics_per_round(output_csv)
                    metrics_df.to_csv(metrics_csv, index=False)
                    print(f"Metrics saved to {metrics_csv}")

                except Exception as e:
                    print(f"[ERROR] model={model_name}, strategy={strategy}: {e}")

# ============================================================================
# Run Evaluation
# ============================================================================

if __name__ == "__main__":
    STRATEGIES = ["poc" ]
    MODELS = ["dual_cnn_gru_fcnn"]
    ROUNDS = [40]  # Add your rounds here
    CID = range(101, 1409)  # 1308 clients
    BASE_RESULTS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/results"  # Update with your actual path
    BASE_OUTPUT_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/predictions40-50-168-T"
    METRICS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/metrics40-50-168-T"

    get_model_predictions_metric(
        MODELS=MODELS,
        STRATEGIES=STRATEGIES,
        ROUNDS=ROUNDS,
        BASE_RESULTS_DIR=BASE_RESULTS_DIR,
        BASE_OUTPUT_DIR=BASE_OUTPUT_DIR,
        METRICS_DIR=METRICS_DIR,
        CID=CID
    )