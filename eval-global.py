import os
import torch
import torch.nn as nn
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

# ============================================================================
# Logging Setup
# ============================================================================
log_dir = '/home/user/DPFL-Sasmita/FL-Baseline-Codes/logs/'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join(log_dir, f"global_evaluation_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# ============================================================================
# Helper Functions (from first code)
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
                logging.info(f"[WARN] Size mismatch for {key}: checkpoint {value.shape}, model {model_state[key].shape}. Skipping.")
        else:
            logging.info(f"[WARN] Key {key} not found in model. Skipping.")
    return adapted_state_dict

# ============================================================================
# Rolling Forecast (unchanged)
# ============================================================================
@torch.no_grad()
def rolling_forecast_on_test(
    cid: int,
    model: torch.nn.Module,
    filepath: str = "train_final.feather",
    input_len: int = 168,
    output_len: int = 24,
) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    # ... (exact same as your first code - omitted for brevity but kept in full below)
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
        pred_unscaled = scaler_mc.inverse_transform(np.pad(pred_np.reshape(-1,1), ((0,0),(0,3)), mode='constant'))[:,0]
        pred_ts = TimeSeries.from_times_and_values(test_time_true[true_start:true_end], pred_unscaled)
        predictions_ts_list.append(pred_ts)
        ground_truth_ts_list.append(true_unscaled)
    return predictions_ts_list, ground_truth_ts_list

# ============================================================================
# Enhanced Metric Functions
# ============================================================================
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)) * 100

def mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def nrmse_range(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
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

def evaluate_forecast_metrics(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        logging.info(f"[WARN] CSV not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        logging.info(f"[WARN] CSV empty: {csv_path}")
        return pd.DataFrame()
    df = df.dropna(subset=['true', 'pred'])
    if df.empty:
        return pd.DataFrame()
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
    return pd.DataFrame([{
        "MAE": mae, "MSE": mse, "RMSE": rmse,
        "MAPE (%)": mape_val, "SMAPE (%)": smape_val,
        "NRMSE_range": nrmse_range_val,
        "NMAE (%)": nmae_val, "NMBE (%)": nmbe_val, "NRMSE_mean (%)": nrmse_mean_val
    }])

# ============================================================================
# Prediction Generation
# ============================================================================
def get_model_predictions_csv(
    model_name: str, cid: int, model_path: str, output_csv: str, filepath: str = "train_final.feather"
):
    if os.path.exists(output_csv):
        logging.info(f"[INFO] Prediction file already exists: {output_csv}")
        return
    rows = []
    try:
        from Models import model_fn
        model = model_fn(model_name)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        adapted_state_dict = adapt_state_dict(state_dict, model)
        model.load_state_dict(adapted_state_dict, strict=False)
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        pred_ts_list, ground_truth_ts_list = rolling_forecast_on_test(cid=cid, model=model, filepath=filepath)
        if len(pred_ts_list) == 0:
            logging.info(f"[WARN] No predictions for CID={cid}")
            return
        for pred_ts, true_ts in zip(pred_ts_list, ground_truth_ts_list):
            df_pred = pd.DataFrame({"timestamp": pred_ts.time_index, "pred": pred_ts.values().squeeze()})
            df_true = pd.DataFrame({"timestamp": true_ts.time_index, "true": true_ts.values().squeeze()})
            df_merged = pd.merge(df_true, df_pred, on="timestamp", how="inner")
            rows.append(df_merged)
        if rows:
            final_df = pd.concat(rows, ignore_index=True)
            final_df.to_csv(output_csv, index=False)
            logging.info(f"[INFO] Saved predictions to {output_csv}")
    except Exception as e:
        logging.info(f"[ERROR] Forecast failed for CID={cid}: {e}")

# ============================================================================
# Main Evaluation + Median Summary
# ============================================================================
def evaluate_global_model_and_compute_medians(
    model_name: str,
    model_path: str,
    base_output_dir: str,
    metrics_dir: str,
    cid_range: range,
    filepath: str = "train_final.feather"
):
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    logging.info(f"\n=== Evaluating Global Model: {model_name} ===")
    lists = {
        "MAE": [], "MSE": [], "RMSE": [], "MAPE (%)": [], "SMAPE (%)": [], "NRMSE_range": [],
        "NMAE (%)": [], "NMBE (%)": [], "NRMSE_mean (%)": []
    }
    valid_count = 0
    missing_pred, empty_pred, invalid_metric = [], [], []
    for cid in tqdm(cid_range, desc="Processing Buildings"):
        output_csv = os.path.join(base_output_dir, f"{cid}_{model_name}_global.csv")
        metrics_csv = os.path.join(metrics_dir, f"cid{cid}_{model_name}_global_metrics.csv")
        try:
            get_model_predictions_csv(model_name, cid, model_path, output_csv, filepath)
            if not os.path.exists(output_csv):
                missing_pred.append(cid)
                continue
            pred_df = pd.read_csv(output_csv)
            if pred_df.empty or pred_df[['true', 'pred']].isna().all().all():
                empty_pred.append(cid)
                continue
            metrics_df = evaluate_forecast_metrics(output_csv)
            if metrics_df.empty:
                invalid_metric.append(cid)
                continue
            metrics_df.to_csv(metrics_csv, index=False)
            metrics = metrics_df.iloc[0]
            for key in lists:
                val = metrics.get(key, np.nan)
                if pd.notna(val):
                    lists[key].append(val)
            valid_count += 1
        except Exception as e:
            logging.info(f"[ERROR] CID={cid}: {e}")
            invalid_metric.append(cid)
    # ==================== Summary ====================
    logging.info("\n" + "="*80)
    logging.info(f"SUMMARY FOR {model_name} (Global)")
    logging.info(f"Valid buildings processed: {valid_count}")
    logging.info(f"Missing predictions: {len(missing_pred)}")
    logging.info(f"Empty/invalid predictions: {len(empty_pred)}")
    logging.info(f"Invalid metrics: {len(invalid_metric)}")
    if valid_count > 0:
        medians = {k: np.median(v) for k, v in lists.items() if len(v) > 0}
        means = {k: np.mean(v) for k, v in lists.items() if len(v) > 0}
        vars_ = {k: np.var(v, ddof=1) for k, v in lists.items() if len(v) > 0}
        logging.info("\nMedians:")
        for k in medians:
            logging.info(f"  Median {k}: {medians[k]:.4f}")
        logging.info("\nMeans:")
        for k in means:
            logging.info(f"  Mean {k}: {means[k]:.4f}")
        logging.info("\nVariances:")
        for k in vars_:
            logging.info(f"  Var {k}: {vars_[k]:.4f}")
    else:
        logging.info("No valid metrics computed.")
    logging.info("="*80)

# ============================================================================
# Run
# ============================================================================
if __name__ == "__main__":
    MODEL_NAME = "dual_cnn_gru_fcnn"
    MODEL_PATH = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/results/dual_cnn_gru_fcnn/dual_cnn_gru_fcnn_epoch_10.pt"
    BASE_OUTPUT_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/predictions_global"
    METRICS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/metrics_global"
    CID_RANGE = range(101, 1409)  # adjust as needed

    evaluate_global_model_and_compute_medians(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        base_output_dir=BASE_OUTPUT_DIR,
        metrics_dir=METRICS_DIR,
        cid_range=CID_RANGE
    )