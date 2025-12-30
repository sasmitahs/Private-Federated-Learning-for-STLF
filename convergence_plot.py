import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import logging
from datetime import datetime
import glob

# Set up logging
log_dir = '/home/user/DPFL-Sasmita/FL-Baseline-Codes/logs/'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join(log_dir, f"convergence_log_{timestamp}.log")
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

def adapt_state_dict(state_dict, model):
    """Adapt state dictionary to match model architecture."""
    model_state = model.state_dict()
    adapted_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state:
            if value.shape == model_state[key].shape:
                adapted_state_dict[key] = value
            else:
                logging.info(f"[WARN] Size mismatch for {key}: checkpoint shape {value.shape}, model shape {model_state[key].shape}. Skipping.")
        else:
            logging.info(f"[WARN] Key {key} not found in model. Skipping.")
    return adapted_state_dict

# ============================================================================
# Rolling Forecast Function
# ============================================================================

@torch.no_grad()
def rolling_forecast_on_test(
    cid: int,
    model: torch.nn.Module,
    filepath: str = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/train_final.feather",
    input_len: int = 168,
    output_len: int = 24,
) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    logging.info(f"[DEBUG] rolling_forecast_on_test: CID={cid}")
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

def nrmse(y_true, y_pred, rmse):
    true_range = np.max(y_true) - np.min(y_true)
    return rmse / true_range if true_range != 0 else np.nan

def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute all metrics from prediction DataFrame."""
    if df.empty:
        return {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan, "SMAPE (%)": np.nan, "NRMSE": np.nan}
    
    df = df.dropna(subset=['true', 'pred'])
    if df.empty:
        return {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan, "SMAPE (%)": np.nan, "NRMSE": np.nan}
    
    y_true = df["true"].values
    y_pred = df["pred"].values
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape_val = mape(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    nrmse_val = nrmse(y_true, y_pred, rmse)
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape_val,
        "SMAPE (%)": smape_val,
        "NRMSE": nrmse_val
    }

# ============================================================================
# Get Predictions and Metrics for a Specific Round and Strategy
# ============================================================================

def get_predictions_and_metrics(
    model_name: str,
    cid: int,
    round_num: int,
    model_dir: str,
    aggr_strat: str,
    filepath: str = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/train_final.feather"
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Get predictions and compute metrics for a specific round and strategy for a given client.
    Returns a DataFrame with 'timestamp', 'true', 'pred', 'round' and a dict of metrics.
    """
    model_path = os.path.join(model_dir, f"{model_name}_round_{round_num}_{aggr_strat}.pt")
    if not os.path.exists(model_path):
        logging.info(f"[WARN] Model not found: {model_path}")
        return pd.DataFrame(), {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan, "SMAPE (%)": np.nan, "NRMSE": np.nan}

    try:
        from Models import model_fn
        model = model_fn(model_name)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        adapted_state_dict = adapt_state_dict(state_dict, model)
        model.load_state_dict(adapted_state_dict, strict=False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        pred_ts_list, ground_truth_ts_list = rolling_forecast_on_test(cid=cid, model=model, filepath=filepath)
        
        if len(pred_ts_list) == 0:
            logging.info(f"[WARN] No predictions for CID={cid}, round={round_num}")
            return pd.DataFrame(), {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan, "SMAPE (%)": np.nan, "NRMSE": np.nan}
        
        rows = []
        for pred_ts, true_ts in zip(pred_ts_list, ground_truth_ts_list):
            df_pred = pd.DataFrame({"timestamp": pred_ts.time_index, "pred": pred_ts.values().squeeze()})
            df_true = pd.DataFrame({"timestamp": true_ts.time_index, "true": true_ts.values().squeeze()})
            df_merged = pd.merge(df_true, df_pred, on="timestamp", how="inner")
            df_merged["round"] = round_num
            rows.append(df_merged)
        
        final_df = pd.concat(rows, ignore_index=True)
        metrics = compute_metrics(final_df)
        return final_df, metrics
        
    except Exception as e:
        logging.info(f"[ERROR] Prediction failed for CID={cid}, round={round_num}, strategy={aggr_strat}: {e}")
        return pd.DataFrame(), {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan, "SMAPE (%)": np.nan, "NRMSE": np.nan}

# ============================================================================
# Find Median Client Based on SMAPE for Round 40
# ============================================================================

def find_median_client_smape(model_name: str, metrics_dir: str, round_num: int = 40) -> int:
    """
    Find the client with SMAPE closest to the median for round 40 across all strategies.
    Returns a single client ID.
    """
    strategies = {
        "POC_with_clustering": "poc",
        "POC_without_clustering": "poc_nocluster",
        "DAS_with_clustering": "AEpublic_k-means_2enc",
        "DAS_without_clustering": "no-cluster_no-AE"
        
    }
    
    smape_data = []
    
    for strat_name, aggr_strat in strategies.items():
        pattern = os.path.join(metrics_dir, f"cid*_{model_name}_{aggr_strat}_metrics.csv")
        files = glob.glob(pattern)
        
        for file in files:
            filename = os.path.basename(file)
            parts = filename.replace('.csv', '').split('_')
            cid = int(parts[0].replace('cid', ''))
            df = pd.read_csv(file)
            
            # Filter for round 40
            df = df[df['round'] == round_num]
            if not df.empty:
                smape_val = df['SMAPE (%)'].iloc[0]
                if pd.notna(smape_val):
                    smape_data.append({'cid': cid, 'strategy': strat_name, 'smape': smape_val})
    
    # Convert to DataFrame
    smape_df = pd.DataFrame(smape_data)
    if smape_df.empty:
        logging.info(f"No SMAPE data found for round {round_num}.")
        return 101  # Fallback to a default client ID
    
    # Compute median SMAPE for round 40
    median_smape = smape_df['smape'].median()
    
    # Find client with SMAPE closest to median
    smape_df['smape_deviation'] = (smape_df['smape'] - median_smape).abs()
    median_client = smape_df.loc[smape_df['smape_deviation'].idxmin(), 'cid']
    
    logging.info(f"Median SMAPE for round {round_num}: {median_smape:.4f}")
    logging.info(f"Selected median client: {median_client}")
    return int(median_client)

# ============================================================================
# Generate Metrics for All Rounds for Median Client
# ============================================================================

def generate_metrics_for_median_client(
    median_client: int,
    model_name: str,
    strategies: Dict[str, str],
    rounds: List[int],
    model_dir: str,
    metrics_dir: str,
    filepath: str = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/train_final.feather"
) -> Dict[str, pd.DataFrame]:
    """
    Generate predictions and metrics for the median client for rounds 1 to 40.
    Returns a dict of strategy -> metrics DataFrame.
    """
    metrics_data = {strat: [] for strat in strategies}
    
    for strat_name, aggr_strat in strategies.items():
        for round_num in rounds:
            predictions, metrics = get_predictions_and_metrics(
                model_name=model_name,
                cid=median_client,
                round_num=round_num,
                model_dir=model_dir,
                aggr_strat=aggr_strat,
                filepath=filepath
            )
            metrics['round'] = round_num
            metrics_data[strat_name].append(metrics)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics_data[strat_name])
        metrics_csv = os.path.join(metrics_dir, f"cid{median_client}_{model_name}_{aggr_strat}_metrics_all_rounds.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        logging.info(f"Saved metrics for {strat_name} to {metrics_csv}")
    
    return {strat: pd.DataFrame(data) for strat, data in metrics_data.items()}

# ============================================================================
# Plot Convergence Graphs
# ============================================================================

def plot_convergence_graphs(
    median_client: int,
    model_name: str,
    strategies: Dict[str, str],
    rounds: List[int],
    metrics_data: Dict[str, pd.DataFrame],
    model_dir: str,
    filepath: str = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/train_final.feather",
    save_dir: str = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/convergence_plots"
):
    """
    Plot convergence graph for SMAPE (%) for the median client across all strategies.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot convergence for SMAPE only
    plt.figure(figsize=(10, 6))
    for strat_name in strategies:
        df = metrics_data[strat_name]
        smape_values = df["SMAPE (%)"].values
        plt.plot(rounds, smape_values, label=strat_name, marker='.')
    
    plt.xlabel('Round')
    plt.ylabel('SMAPE (%)')
    plt.title(f'Convergence of SMAPE (%) for Client {median_client}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, f"convergence_smape_pct_client_{median_client}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved SMAPE convergence plot: {plot_path}")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Configuration
    model_name = "dual_cnn_gru_fcnn"
    metrics_dir = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/metrics40-50-168-T"
    model_dir = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/results/dual_cnn_gru_fcnn"
    filepath = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/train_final.feather"
    rounds = list(range(1, 41))  # Rounds 1 to 40
    save_dir = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/convergence_plots"
    
    strategies = {
        "DAS_with_clustering": "AEpublic_k-means_2enc_correct",
        "DAS_without_clustering": "no-cluster_no-AE_correct",
        
        "POC_without_clustering": "poc_nocluster"
    }
    
    # Step 1: Find median client based on SMAPE for round 40
    median_client = find_median_client_smape(model_name, metrics_dir, round_num=40)
    
    # Step 2: Generate metrics for the median client for rounds 1 to 40
    metrics_data = generate_metrics_for_median_client(
        median_client=median_client,
        model_name=model_name,
        strategies=strategies,
        rounds=rounds,
        model_dir=model_dir,
        metrics_dir=metrics_dir,
        filepath=filepath
    )
    
    # Step 3: Plot convergence graphs
    plot_convergence_graphs(
        median_client=median_client,
        model_name=model_name,
        strategies=strategies,
        rounds=rounds,
        metrics_data=metrics_data,
        model_dir=model_dir,
        filepath=filepath,
        save_dir=save_dir
    )
    
    print(f"Convergence plots saved to {save_dir}")
    print("Plots include:")
    print("- Convergence plots for each metric (MAE, MSE, RMSE, MAPE, SMAPE, NRMSE) over rounds 1-40")
    print("- Time series plot comparing true vs predicted values for the last valid round")