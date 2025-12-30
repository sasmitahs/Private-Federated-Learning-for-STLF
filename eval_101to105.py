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
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"eval_difficulty_no_cluster_no_ae_{timestamp}.log")
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
                logging.debug(f"[WARN] Size mismatch for {key}: checkpoint {value.shape} vs model {model_state[key].shape}")
        else:
            logging.debug(f"[WARN] Key {key} not in model")
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
# Main: Evaluate Difficulty Models (No Cluster, No AE) - 4 Runs
# ============================================================================
def evaluate_difficulty_no_cluster_no_ae(
    model_name: str,
    rounds: List[int],
    runs: List[int],
    base_results_dir: str,
    output_dir: str,
    metrics_dir: str,
    clients: List[int],
    filepath: str = "train_final.feather"
):
    """
    Evaluate difficulty-based federated models (no cluster, no AE) across multiple runs.
    
    Args:
        model_name: Name of the model (e.g., 'dual_cnn_gru_fcnn')
        rounds: List of rounds to evaluate
        runs: List of runs to evaluate (e.g., [1, 2, 3, 4])
        base_results_dir: Base directory containing run folders
        output_dir: Directory to save prediction outputs
        metrics_dir: Directory to save metrics
        clients: List of client IDs to evaluate
        filepath: Path to training data feather file
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    all_results = []
    
    logging.info("="*80)
    logging.info(f"DIFFICULTY (NO CLUSTER, NO AE) EVALUATION - {model_name}")
    logging.info(f"Clients: {clients}")
    logging.info(f"Rounds: {min(rounds)} to {max(rounds)}")
    logging.info(f"Runs: {runs}")
    logging.info("="*80)
    
    for run_num in runs:
        logging.info(f"\n{'='*80}")
        logging.info(f"PROCESSING RUN {run_num}")
        logging.info(f"{'='*80}")
        
        run_dir = os.path.join(base_results_dir, f"run_{run_num}")
        if not os.path.exists(run_dir):
            logging.warning(f"[WARN] Run directory not found: {run_dir}")
            continue
        
        for round_num in rounds:
            model_path = os.path.join(
                run_dir, 
                f"{model_name}_round_{round_num}_difficulty_no_cluster_no_ae.pt"
            )
            
            if not os.path.exists(model_path):
                logging.debug(f"[SKIP] Model not found: {model_path}")
                continue
            
            logging.info(f"\n--- Run {run_num} | Round {round_num} ---")
            
            try:
                # Load model
                from Models import model_fn
                model = model_fn(model_name)
                
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                except Exception as e:
                    logging.debug(f"[INFO] Loading with weights_only=False: {e}")
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                # Extract state_dict
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present
                if any(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                
                adapted_state_dict = adapt_state_dict(state_dict, model)
                model.load_state_dict(adapted_state_dict, strict=False)
                model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()
                
            except Exception as e:
                logging.error(f"[ERROR] Failed to load model {model_path}: {e}")
                continue
            
            # Evaluate on each client
            for cid in clients:
                try:
                    pred_ts_list, true_ts_list = rolling_forecast_on_test(
                        cid=cid, 
                        model=model, 
                        filepath=filepath
                    )
                    
                    if len(pred_ts_list) == 0:
                        logging.debug(f"[SKIP] No predictions for client {cid}")
                        continue
                    
                    # Combine all predictions and ground truth for this client
                    all_preds = []
                    all_trues = []
                    
                    for pred_ts, true_ts in zip(pred_ts_list, true_ts_list):
                        all_preds.extend(pred_ts.values().squeeze())
                        all_trues.extend(true_ts.values().squeeze())
                    
                    y_true = np.array(all_trues)
                    y_pred = np.array(all_preds)
                    
                    # Handle NaN values
                    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                    if mask.sum() == 0:
                        logging.debug(f"[SKIP] All NaN values for client {cid}")
                        continue
                    
                    y_true = y_true[mask]
                    y_pred = y_pred[mask]
                    
                    # Compute metrics
                    metrics = compute_metrics_for_client(y_true, y_pred)
                    metrics['run'] = run_num
                    metrics['round'] = round_num
                    metrics['client_id'] = cid
                    metrics['model'] = model_name
                    metrics['num_predictions'] = len(y_pred)
                    
                    all_results.append(metrics)
                    
                    logging.info(f"  ✓ Client {cid}: MAE={metrics['MAE']:.2f}, "
                               f"RMSE={metrics['RMSE']:.2f}, "
                               f"MAPE={metrics['MAPE (%)']:.2f}%")
                    
                except Exception as e:
                    logging.debug(f"[ERROR] Failed for client {cid}: {e}")
                    continue
            
            # Clear model from memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save all results
    if len(all_results) == 0:
        logging.warning("[WARN] No results generated!")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    detailed_csv = os.path.join(metrics_dir, f"{model_name}_difficulty_no_cluster_no_ae_detailed_{timestamp}.csv")
    results_df.to_csv(detailed_csv, index=False)
    logging.info(f"\n✓ Saved detailed results: {detailed_csv}")
    
    # Compute summary statistics per run and round
    logging.info("\n" + "="*80)
    logging.info("SUMMARY STATISTICS")
    logging.info("="*80)
    
    metric_cols = ["MAE", "RMSE", "MAPE (%)", "SMAPE (%)", "NMAE (%)", "NMBE (%)", "NRMSE_mean (%)"]
    
    # Summary by run and round
    summary_by_run_round = results_df.groupby(['run', 'round'])[metric_cols].agg(['mean', 'median', 'std']).reset_index()
    summary_csv = os.path.join(metrics_dir, f"{model_name}_difficulty_no_cluster_no_ae_summary_{timestamp}.csv")
    summary_by_run_round.to_csv(summary_csv, index=False)
    logging.info(f"✓ Saved summary by run/round: {summary_csv}")
    
    # Overall summary by run
    summary_by_run = results_df.groupby('run')[metric_cols].agg(['mean', 'median', 'std']).reset_index()
    overall_summary_csv = os.path.join(metrics_dir, f"{model_name}_difficulty_no_cluster_no_ae_run_summary_{timestamp}.csv")
    summary_by_run.to_csv(overall_summary_csv, index=False)
    logging.info(f"✓ Saved summary by run: {overall_summary_csv}")
    
    # Print overall statistics
    logging.info("\n" + "="*80)
    logging.info("OVERALL STATISTICS (Mean across all runs and rounds)")
    logging.info("="*80)
    for col in metric_cols:
        mean_val = results_df[col].mean()
        median_val = results_df[col].median()
        std_val = results_df[col].std()
        logging.info(f"{col:20s}: Mean={mean_val:8.4f}, Median={median_val:8.4f}, Std={std_val:8.4f}")
    
    # Best performing round per run
    logging.info("\n" + "="*80)
    logging.info("BEST ROUND PER RUN (lowest median RMSE)")
    logging.info("="*80)
    for run_num in runs:
        run_data = results_df[results_df['run'] == run_num]
        if len(run_data) == 0:
            continue
        best_round_idx = run_data.groupby('round')['RMSE'].median().idxmin()
        best_round_data = run_data[run_data['round'] == best_round_idx]
        logging.info(f"\nRun {run_num} - Best Round: {best_round_idx}")
        logging.info(f"  Median RMSE: {best_round_data['RMSE'].median():.4f}")
        logging.info(f"  Median MAE:  {best_round_data['MAE'].median():.4f}")
        logging.info(f"  Median MAPE: {best_round_data['MAPE (%)'].median():.2f}%")
    
    # Client-wise performance summary
    logging.info("\n" + "="*80)
    logging.info("CLIENT-WISE PERFORMANCE (Average across all runs and rounds)")
    logging.info("="*80)
    client_summary = results_df.groupby('client_id')[metric_cols].agg(['mean', 'median', 'std'])
    logging.info("\n" + client_summary.to_string())
    
    client_summary_csv = os.path.join(metrics_dir, f"{model_name}_difficulty_no_cluster_no_ae_client_summary_{timestamp}.csv")
    client_summary.to_csv(client_summary_csv)
    logging.info(f"\n✓ Saved client-wise summary: {client_summary_csv}")

# ============================================================================
# Run Evaluation
# ============================================================================
if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "dual_cnn_gru_fcnn"
    ROUNDS = list(range(1, 41))  # Rounds 1 to 40
    RUNS = [1, 2, 3, 4]  # All 4 runs
    CLIENTS = [101, 102, 103, 104, 105]  # Clients 101-105
    
    BASE_RESULTS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/results_difficulty_no_cluster_no_ae"
    OUTPUT_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/predictions_difficulty_no_cluster_no_ae"
    METRICS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/metrics_difficulty_no_cluster_no_ae"
    FILEPATH = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/train_final.feather"
    
    # Run evaluation
    evaluate_difficulty_no_cluster_no_ae(
        model_name=MODEL_NAME,
        rounds=ROUNDS,
        runs=RUNS,
        base_results_dir=BASE_RESULTS_DIR,
        output_dir=OUTPUT_DIR,
        metrics_dir=METRICS_DIR,
        clients=CLIENTS,
        filepath=FILEPATH
    )
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE!")
    print("="*80)
    print(f"Results saved in: {METRICS_DIR}")
    print(f"Log file: {log_file}")
    print("="*80)