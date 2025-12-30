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

# Set up logging
log_dir = '/home/user/DPFL-Sasmita/FL-Baseline-Codes/logs/'
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
    filepath: str = "train_final.feather",
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

def evaluate_forecast_metrics_per_round(csv_path):
    if not os.path.exists(csv_path):
        logging.info(f"[WARN] CSV not found: {csv_path}")
        return pd.DataFrame([{
            "round": "global",
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAPE (%)": np.nan,
            "SMAPE (%)": np.nan,
            "NRMSE": np.nan
        }])
    
    df = pd.read_csv(csv_path)
    if df.empty:
        logging.info(f"[WARN] CSV is empty: {csv_path}")
        return pd.DataFrame([{
            "round": "global",
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAPE (%)": np.nan,
            "SMAPE (%)": np.nan,
            "NRMSE": np.nan
        }])
    
    df = df.dropna(subset=['true', 'pred'])
    
    if df.empty:
        logging.info(f"[WARN] No valid data after dropping NaNs: {csv_path}")
        return pd.DataFrame([{
            "round": "global",
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAPE (%)": np.nan,
            "SMAPE (%)": np.nan,
            "NRMSE": np.nan
        }])
    
    y_true = df["true"].values
    y_pred = df["pred"].values
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape_val = mape(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    nrmse_val = nrmse(y_true, y_pred, rmse)
    
    return pd.DataFrame([{
        "round": "global",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape_val,
        "SMAPE (%)": smape_val,
        "NRMSE": nrmse_val
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
    rows = []
    
    for round_num in rounds:
        model_path = os.path.join(model_dir, f"{model_name}_round_{round_num}_{aggr_strat}.pt")
        if not os.path.exists(model_path):
            logging.info(f"[WARN] Model not found: {model_path}")
            continue

        try:
            from Models import model_fn
            model = model_fn(model_name)
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            adapted_state_dict = adapt_state_dict(state_dict, model)
            model.load_state_dict(adapted_state_dict, strict=False)
            model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            
            pred_ts_list, ground_truth_ts_list = rolling_forecast_on_test(cid=cid, model=model, filepath=filepath)
            
            if len(pred_ts_list) == 0:
                logging.info(f"[WARN] No predictions for CID={cid}, round={round_num}")
                continue
            
            for pred_ts, true_ts in zip(pred_ts_list, ground_truth_ts_list):
                df_pred = pd.DataFrame({"timestamp": pred_ts.time_index, "pred": pred_ts.values().squeeze()})
                df_true = pd.DataFrame({"timestamp": true_ts.time_index, "true": true_ts.values().squeeze()})
                df_merged = pd.merge(df_true, df_pred, on="timestamp", how="inner")
                df_merged["round"] = round_num
                rows.append(df_merged)
        
        except Exception as e:
            logging.info(f"[ERROR] Forecast failed for CID={cid}, round={round_num}: {e}")
            continue
    
    if rows:
        final_df = pd.concat(rows, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        logging.info(f"[INFO] Saved predictions to {output_csv}")
    else:
        logging.info(f"[WARN] No predictions generated for CID={cid}")

# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_and_compute_mean_variance_metrics(
    MODELS: List[str],
    STRATEGIES: List[str],
    ROUNDS: List[int],
    BASE_RESULTS_DIR: str,
    BASE_OUTPUT_DIR: str,
    METRICS_DIR: str,
    CID: range
):
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    for model_name in MODELS:
        for strategy in STRATEGIES:
            logging.info(f"\n=== Processing Strategy: {strategy} ===")
            
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

            for cid in tqdm(CID, desc="Processing Clients"):
                logging.info(f"\nProcessing Client ID: {cid}")
                model_dir = os.path.join(BASE_RESULTS_DIR, model_name)
                output_csv = os.path.join(BASE_OUTPUT_DIR, f"{cid}_{model_name}_{strategy}.csv")
                metrics_csv = os.path.join(METRICS_DIR, f"cid{cid}_{model_name}_{strategy}_metrics.csv")

                logging.info(f"Model: {model_name}, Strategy: {strategy}")

                try:
                    get_model_predictions_csv(
                        model_name=model_name,
                        cid=cid,
                        rounds=ROUNDS,
                        model_dir=model_dir,
                        output_csv=output_csv,
                        aggr_strat=strategy
                    )

                    metrics_df = evaluate_forecast_metrics_per_round(output_csv)
                    metrics_df.to_csv(metrics_csv, index=False)
                    logging.info(f"Metrics saved to {metrics_csv}")

                    if not metrics_df.empty and not metrics_df[['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'SMAPE (%)', 'NRMSE']].isna().all().all():
                        metrics = metrics_df.iloc[0]
                        mae = metrics.get('MAE', np.nan)
                        mse = metrics.get('MSE', np.nan)
                        rmse = metrics.get('RMSE', np.nan)
                        mape = metrics.get('MAPE (%)', np.nan)
                        smape = metrics.get('SMAPE (%)', np.nan)
                        nrmse = metrics.get('NRMSE', np.nan)

                        if not pd.isna(mae): mae_list.append(mae)
                        if not pd.isna(mse): mse_list.append(mse)
                        if not pd.isna(rmse): rmse_list.append(rmse)
                        if not pd.isna(mape): mape_list.append(mape)
                        if not pd.isna(smape): smape_list.append(smape)
                        if not pd.isna(nrmse): nrmse_list.append(nrmse)
                    else:
                        invalid_metric_files.append(cid)
                        logging.info(f"CID {cid}: Metric file: Empty or all NaN")

                except Exception as e:
                    logging.info(f"[ERROR] model={model_name}, strategy={strategy}, CID={cid}: {e}")
                    invalid_metric_files.append(cid)

                if not os.path.exists(output_csv):
                    missing_prediction_files.append(cid)
                    logging.info(f"CID {cid}: Prediction file: Missing")
                else:
                    try:
                        pred_df = pd.read_csv(output_csv)
                        if pred_df.empty:
                            empty_prediction_files.append(cid)
                            logging.info(f"CID {cid}: Prediction file: Empty")
                        elif pred_df[['true', 'pred']].isna().all().all():
                            nan_prediction_files.append(cid)
                            logging.info(f"CID {cid}: Prediction file: All NaN")
                    except Exception as e:
                        empty_prediction_files.append(cid)
                        logging.info(f"CID {cid}: Prediction file: Error: {e}")

            if mae_list:
                mean_mae = np.mean(mae_list)
                std_mae = np.std(mae_list)
                mean_mse = np.mean(mse_list)
                std_mse = np.std(mse_list)
                mean_rmse = np.mean(rmse_list)
                std_rmse = np.std(rmse_list)
                mean_mape = np.mean(mape_list)
                std_mape = np.std(mape_list)
                mean_smape = np.mean(smape_list)
                std_smape = np.std(smape_list)
                mean_nrmse = np.mean(nrmse_list) if nrmse_list else np.nan
                std_nrmse = np.std(nrmse_list) if nrmse_list else np.nan
                
                logging.info(f"\nResults for Strategy: {strategy}")
                logging.info(f"Number of processed metric files: {len(mae_list)}")
                logging.info(f"MAE (mean ± std): {mean_mae:.4f} ± {std_mae:.4f}")
                logging.info(f"MSE (mean ± std): {mean_mse:.4f} ± {std_mse:.4f}")
                logging.info(f"RMSE (mean ± std): {mean_rmse:.4f} ± {std_rmse:.4f}")
                logging.info(f"MAPE (%) (mean ± std): {mean_mape:.4f} ± {std_mape:.4f}")
                logging.info(f"SMAPE (%) (mean ± std): {mean_smape:.4f} ± {std_smape:.4f}")
                logging.info(f"NRMSE (mean ± std): {mean_nrmse:.4f} ± {std_nrmse:.4f}")
            else:
                logging.info(f"\nNo valid metric files found for model {model_name} and strategy {strategy}.")

            unprocessed_cids = set(missing_prediction_files + empty_prediction_files + nan_prediction_files + missing_metric_files + invalid_metric_files)

            if unprocessed_cids:
                logging.info(f"\nNumber of unprocessed clients: {len(unprocessed_cids)}")
                logging.info("Unprocessed client IDs:")
                logging.info(" ".join(map(str, sorted(unprocessed_cids))))
            else:
                logging.info("\nAll clients were processed successfully.")

            logging.info("\nSummary of Issues:")
            logging.info(f"Missing prediction files: {len(missing_prediction_files)}")
            logging.info(f"Empty prediction files: {len(empty_prediction_files)}")
            logging.info(f"NaN prediction files: {len(nan_prediction_files)}")
            logging.info(f"Missing metric files: {len(missing_metric_files)}")
            logging.info(f"Invalid (empty or NaN) metric files: {len(invalid_metric_files)}")

# ============================================================================
# Run Evaluation
# ============================================================================

if __name__ == "__main__":
    STRATEGIES = ["poc","poc_no-clutering","das_no_cluster","AEpublic_k-means_2enc"]  # "global_model", "fedAvg_diff", "fedProx_diff", "diff-diff", "scaffold_diff", "fedAvg_diff0", "fedProx", "fedAvg_lr", "scaffold_lr", "diff_lr2", "das11", "das2", "fedAvg_lr", "fedAvg_diffsample_dhc"
    MODELS = ["simple_cnn"]
    ROUNDS = [40]
    CID = range(101, 1409)
    BASE_RESULTS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/results"
    BASE_OUTPUT_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/predictions40-50-168-T"
    METRICS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/metrics40-50-168-T"

    evaluate_and_compute_mean_variance_metrics(
        MODELS=MODELS,
        STRATEGIES=STRATEGIES,
        ROUNDS=ROUNDS,
        BASE_RESULTS_DIR=BASE_RESULTS_DIR,
        BASE_OUTPUT_DIR=BASE_OUTPUT_DIR,
        METRICS_DIR=METRICS_DIR,
        CID=CID
    )