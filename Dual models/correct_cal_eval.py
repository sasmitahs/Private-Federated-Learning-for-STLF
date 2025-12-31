import os
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import logging
from datetime import datetime
from numpy.core.multiarray import scalar
from numpy import dtype

# ============================================================================
# FIX: Allow safe NumPy globals for torch.load with weights_only=True
# ============================================================================
torch.serialization.add_safe_globals([scalar, dtype])
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# Add all NumPy dtype classes that might be in the saved models
try:
    torch.serialization.add_safe_globals([
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
        np.dtypes.Int64DType,
        np.dtypes.Int32DType,
        np.dtypes.UInt8DType,
        np.dtypes.BoolDType,
    ])
except AttributeError:
    pass

# Set up logging
log_dir = '/home/user/DPFL-Sasmita/FL-Baseline-Codes/logs/'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join(log_dir, f"central_per_client_metrics_log_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# ============================================================================
# Helper Functions
# ============================================================================
def safe_inverse_transform_scaler(scaler: MinMaxScaler, arr: np.ndarray, n_channels_expected: int = 4):
    arr = np.asarray(arr).reshape(-1, 1)
    pad = np.zeros((arr.shape[0], max(0, n_channels_expected - 1)))
    stacked = np.concatenate([arr, pad], axis=1)
    return scaler.inverse_transform(stacked)[:, 0]

def _get_num_embeddings_for_model(model) -> int:
    for attr in ("primary_use_embedding", "primary_use_embed", "primary_use_embeddding"):
        emb = getattr(model, attr, None)
        if emb is not None and isinstance(emb, torch.nn.Embedding):
            return emb.num_embeddings
    return None

def adapt_state_dict(state_dict, model):
    model_state = model.state_dict()
    adapted_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state:
            if value.shape == model_state[key].shape:
                adapted_state_dict[key] = value
            else:
                logging.info(f"[WARN] Size mismatch for {key}: checkpoint {value.shape} vs model {model_state[key].shape}")
        else:
            logging.info(f"[WARN] Key {key} not in model")
    return adapted_state_dict

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
    df = pd.read_feather(filepath)
    df = df[df['building_id'] == cid].copy()
    if df.empty:
        raise ValueError(f"No data found for building_id {cid}")

    df['meter_reading'] = df['meter_reading'].fillna(0.0)
    df['air_temperature'] = df['air_temperature'].fillna(df['air_temperature'].mean())
    df['primary_use_idx'] = df.get('primary_use', 0)
    if 'primary_use' in df.columns:
        primary_map = {cat: idx for idx, cat in enumerate(df['primary_use'].unique())}
        df['primary_use_idx'] = df['primary_use'].map(primary_map).fillna(0).astype(int)

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

    scaler_mc = MinMaxScaler(feature_range=(0.1, 1))
    test_values_scaled = scaler_mc.fit_transform(test_values)
    test_air_temp_scaled = MinMaxScaler(feature_range=(0.1, 1)).fit_transform(test_air_temp.reshape(-1, 1))

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

        if pred.dim() == 3:
            pred = pred.squeeze(0).squeeze(-1)
        elif pred.dim() == 2:
            pred = pred.squeeze(0)

        pred_np = pred.detach().cpu().numpy()

        true_start = i + input_len
        true_end = true_start + output_len
        true_output = test_values_true_scaled[true_start:true_end]
        true_ts = TimeSeries.from_times_and_values(test_time_true[true_start:true_end], true_output)
        true_unscaled = transformer.inverse_transform(true_ts)

        pred_unscaled = scaler_mc.inverse_transform(np.pad(pred_np.reshape(-1,1), ((0,0),(0,3)), mode='constant'))[:,0]
        pred_ts = TimeSeries.from_times_and_values(test_time_true[true_start:true_end], pred_unscaled)

        predictions_ts_list.append(pred_ts)
        ground_truth_ts_list.append(true_unscaled)

    return predictions_ts_list, ground_truth_ts_list

# ============================================================================
# Metrics Functions
# ============================================================================
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)) * 100

def mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def nrmse(y_true, y_pred, rmse):
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

def compute_metrics_for_client(y_true, y_pred):
    """Compute all metrics for a single client's predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape_val = mape(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    nrmse_val = nrmse(y_true, y_pred, rmse)
    nmae_val = nmae(y_true, y_pred)
    nmbe_val = nmbe(y_true, y_pred)
    nrmse_mean_val = nrmse_mean(y_true, y_pred)
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape_val,
        "SMAPE (%)": smape_val,
        "NRMSE": nrmse_val,
        "NMAE (%)": nmae_val,
        "NMBE (%)": nmbe_val,
        "NRMSE_mean (%)": nrmse_mean_val
    }

# ============================================================================
# Main: Evaluate Centralized Global Models - Per Client with Median Summary
# ============================================================================
def evaluate_central_models_per_client_with_median(
    MODELS: List[str],
    ROUNDS: List[int],
    BASE_RESULTS_DIR: str,
    BASE_OUTPUT_DIR: str,
    METRICS_DIR: str,
    CID: range,
    filepath: str = "train_final.feather"
):
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Track all per-client metrics for median calculation
    all_round_metrics = []

    for model_name in MODELS:
        model_dir = os.path.join(BASE_RESULTS_DIR, model_name)
        if not os.path.exists(model_dir):
            logging.info(f"[ERROR] Model directory not found: {model_dir}")
            continue

        logging.info(f"\n{'='*80}")
        logging.info(f"Processing Centralized Model: {model_name}")
        logging.info(f"{'='*80}")

        for round_num in ROUNDS:
            model_path = os.path.join(model_dir, f"{model_name}_central_epoch_{round_num}.pt")
            if not os.path.exists(model_path):
                logging.info(f"[WARN] Model not found: {model_path}")
                continue

            logging.info(f"\n--- Loading model for Round {round_num} ---")
            logging.info(f"Model path: {model_path}")

            try:
                from Models import model_fn
                model = model_fn(model_name)

                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                except Exception as e:
                    logging.info(f"[WARN] weights_only=True failed, trying weights_only=False: {e}")
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                # Extract state_dict from checkpoint if it's a dictionary with 'model_state_dict' key
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    logging.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    state_dict = checkpoint

                if any(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

                adapted_state_dict = adapt_state_dict(state_dict, model)
                model.load_state_dict(adapted_state_dict, strict=False)
                model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()

            except Exception as e:
                logging.info(f"[ERROR] Failed to load model {model_path}: {e}")
                continue

            # Store per-client metrics for this round
            per_client_metrics = []

            # Process each client individually
            for cid in tqdm(CID, desc=f"Round {round_num} - Processing clients"):
                try:
                    pred_ts_list, true_ts_list = rolling_forecast_on_test(cid=cid, model=model, filepath=filepath)
                    
                    if len(pred_ts_list) == 0:
                        continue

                    # Combine all predictions and ground truth for this client
                    all_preds = []
                    all_trues = []
                    
                    for pred_ts, true_ts in zip(pred_ts_list, true_ts_list):
                        all_preds.extend(pred_ts.values().squeeze())
                        all_trues.extend(true_ts.values().squeeze())
                    
                    y_true = np.array(all_trues)
                    y_pred = np.array(all_preds)
                    
                    # Handle any NaN values
                    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                    if mask.sum() == 0:
                        continue
                    
                    y_true = y_true[mask]
                    y_pred = y_pred[mask]
                    
                    # Compute metrics for this client
                    client_metrics = compute_metrics_for_client(y_true, y_pred)
                    client_metrics['cid'] = cid
                    client_metrics['round'] = round_num
                    client_metrics['model'] = model_name
                    
                    per_client_metrics.append(client_metrics)
                    
                except Exception as e:
                    logging.debug(f"[WARN] Failed for CID {cid}: {e}")
                    continue

            if len(per_client_metrics) == 0:
                logging.info(f"[INFO] No valid predictions for round {round_num}")
                continue

            # Save per-client metrics to CSV
            per_client_df = pd.DataFrame(per_client_metrics)
            per_client_csv = os.path.join(METRICS_DIR, f"per_client_{model_name}_epoch_{round_num}.csv")
            per_client_df.to_csv(per_client_csv, index=False)
            logging.info(f"\n✓ Saved per-client metrics: {per_client_csv}")
            logging.info(f"  Valid clients: {len(per_client_metrics)}")

            # Compute median metrics across all clients
            metric_cols = ["MAE", "MSE", "RMSE", "MAPE (%)", "SMAPE (%)", "NRMSE", "NMAE (%)", "NMBE (%)", "NRMSE_mean (%)"]
            median_metrics = {col: per_client_df[col].median() for col in metric_cols}
            median_metrics['round'] = round_num
            median_metrics['model'] = model_name
            median_metrics['num_clients'] = len(per_client_metrics)
            
            # Log median metrics
            logging.info(f"\n{'='*60}")
            logging.info(f"MEDIAN METRICS - {model_name} - Round {round_num}")
            logging.info(f"Number of clients: {len(per_client_metrics)}")
            logging.info(f"{'='*60}")
            logging.info(f"MAE:         {median_metrics['MAE']:.4f}")
            logging.info(f"RMSE:        {median_metrics['RMSE']:.4f}")
            logging.info(f"MAPE:        {median_metrics['MAPE (%)']:.2f}%")
            logging.info(f"SMAPE:       {median_metrics['SMAPE (%)']:.2f}%")
            logging.info(f"NMAE:        {median_metrics['NMAE (%)']:.2f}%")
            logging.info(f"NMBE:        {median_metrics['NMBE (%)']:.2f}%")
            logging.info(f"NRMSE_mean:  {median_metrics['NRMSE_mean (%)']:.2f}%")
            logging.info(f"{'='*60}\n")
            
            all_round_metrics.append(median_metrics)

    # Save summary of median metrics across all rounds
    if all_round_metrics:
        summary_df = pd.DataFrame(all_round_metrics)
        summary_csv = os.path.join(METRICS_DIR, f"median_summary_all_rounds_{timestamp}.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        logging.info(f"\n{'='*80}")
        logging.info("SUMMARY: Median Metrics Across All Rounds")
        logging.info(f"{'='*80}")
        logging.info("\n" + summary_df.to_string(index=False))
        logging.info(f"\n✓ Summary saved to: {summary_csv}")
    else:
        logging.info("[INFO] No results to summarize.")

# ============================================================================
# Run Evaluation
# ============================================================================
if __name__ == "__main__":
    MODELS = ["dual_simple_ann_fcnn"]  # Add more models if needed
    ROUNDS = list(range(10, 31))    # Rounds 10 to 30 inclusive
    CID = range(101, 1409)
    BASE_RESULTS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/results"
    BASE_OUTPUT_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/predictions_central"
    METRICS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/metrics_central_per_client"

    evaluate_central_models_per_client_with_median(
        MODELS=MODELS,
        ROUNDS=ROUNDS,
        BASE_RESULTS_DIR=BASE_RESULTS_DIR,
        BASE_OUTPUT_DIR=BASE_OUTPUT_DIR,
        METRICS_DIR=METRICS_DIR,
        CID=CID
    )

    print("\n" + "="*80)
    print("✓ Done! Centralized model evaluation with per-client metrics completed.")
    print("="*80)
    print(f"Per-client metrics saved in: {METRICS_DIR}")
    print(f"Log file: {log_file}")