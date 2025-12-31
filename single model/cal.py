import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from tqdm import tqdm

# ==================== LOGGING SETUP ====================
log_dir = '/home/user/DPFL-Sasmita/FL-Baseline-Codes1/logs/'
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

# ==================== METRIC FUNCTIONS ====================
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)) * 100

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def nrmse(y_true, y_pred):
    """Normalized Root Mean Squared Error (range normalization)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    true_range = np.max(y_true) - np.min(y_true)
    if true_range == 0:
        return np.nan
    return rmse / true_range

# ==================== ROLLING FORECAST FUNCTION ====================
@torch.no_grad()
def rolling_forecast_on_test(cid, model, filepath="train_final.feather", input_len=168, output_len=24):
    """
    Perform rolling window forecast on test data.
    """
    print(f"[DEBUG] rolling_forecast_on_test: CID={cid}")

    df = pd.read_feather(filepath)
    df = df[df['building_id'] == cid]
    df['meter_reading'] = df['meter_reading'].fillna(0)

    if df.empty:
        raise ValueError(f"No data found for building_id {cid}")

    ts = TimeSeries.from_dataframe(
        df,
        time_col='timestamp',
        value_cols='meter_reading',
        fill_missing_dates=True,
        freq='h'
    )

    _, test_series = ts.split_before(0.75)

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

        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

        pred = model(input_tensor)
        
        if pred.dim() == 3:
            pred = pred.squeeze(0).squeeze(-1)
        else:
            pred = pred.squeeze(0)

        pred_ts = TimeSeries.from_times_and_values(true_time, pred.cpu().numpy())
        true_ts = TimeSeries.from_times_and_values(true_time, true_output)

        pred_unscaled = transformer.inverse_transform(pred_ts)
        true_unscaled = transformer.inverse_transform(true_ts)

        predictions_ts_list.append(pred_unscaled)
        ground_truth_ts_list.append(true_unscaled)

    return predictions_ts_list, ground_truth_ts_list

# ==================== METRICS EVALUATION ====================
def evaluate_forecast_metrics_per_round(csv_path):
    """
    Compute MAE, MSE, RMSE, MAPE, SMAPE, and NRMSE per round.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty or invalid")

    metrics_list = []

    for rnd in sorted(df['round'].unique()):
        df_rnd = df[df['round'] == rnd]
        df_rnd = df_rnd.fillna(0.005)
        y_true = df_rnd["true"].values
        y_pred = df_rnd["pred"].values

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape_val = mape(y_true, y_pred)
        smape_val = smape(y_true, y_pred)
        nrmse_val = nrmse(y_true, y_pred)

        metrics_list.append({
            "round": rnd,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE (%)": mape_val,
            "SMAPE (%)": smape_val,
            "NRMSE": nrmse_val
        })

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df

# ==================== PREDICTION GENERATION ====================
def get_model_predictions_csv(model_name: str, cid: int, aggr_strat: str, rounds: list, 
                               model_dir: str, output_csv: str, model_fn):
    """
    Generate predictions for each round and save to CSV.
    """
    rows = []

    for rnd in tqdm(rounds, desc=f"Rounds for CID {cid}"):
        model_path = os.path.join(model_dir, f"{model_name}_round_{rnd}_{aggr_strat}.pt")

        if not os.path.exists(model_path):
            print(f"[WARN] Model not found: {model_path}")
            continue

        model = model_fn(model_name)
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()

        pred_ts_list, gt_ts_list = rolling_forecast_on_test(cid=cid, model=model)

        for pred_ts, true_ts in zip(pred_ts_list, gt_ts_list):
            df_pred = pd.DataFrame({"timestamp": pred_ts.time_index, "pred": pred_ts.values().squeeze()})
            df_true = pd.DataFrame({"timestamp": true_ts.time_index, "true": true_ts.values().squeeze()})

            df_merged = pd.merge(df_true, df_pred, on="timestamp", how="inner")
            df_merged["round"] = rnd

            rows.append(df_merged[["timestamp", "true", "pred", "round"]])

    final_df = pd.concat(rows, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"[INFO] Forecasts written to {output_csv}")

# ==================== MAIN PIPELINE ====================
def get_model_predictions_metric(MODELS, STRATEGIES, ROUNDS, BASE_RESULTS_DIR: str,
                                 BASE_OUTPUT_DIR: str, METRICS_DIR: str, CID: range, model_fn):
    """
    Generate predictions and compute metrics for all clients, models, and strategies.
    """
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    for cid in tqdm(CID, desc="Processing Clients"):
        print(f"\nProcessing Client ID: {cid}")

        for model_name in MODELS:
            for strategy in STRATEGIES:
                model_dir = os.path.join(BASE_RESULTS_DIR, model_name)
                output_csv = os.path.join(BASE_OUTPUT_DIR, f"{cid}_{model_name}_{strategy}.csv")
                metrics_csv = os.path.join(METRICS_DIR, f"cid{cid}_{model_name}_{strategy}_metrics.csv")

                print(f"Model: {model_name}, Strategy: {strategy}")

                try:
                    get_model_predictions_csv(
                        model_name=model_name,
                        cid=cid,
                        rounds=ROUNDS,
                        model_dir=model_dir,
                        output_csv=output_csv,
                        aggr_strat=strategy,
                        model_fn=model_fn
                    )

                    metrics_df = evaluate_forecast_metrics_per_round(output_csv)
                    metrics_df.to_csv(metrics_csv, index=False)
                    print(f"Metrics saved to {metrics_csv}")

                except Exception as e:
                    print(f"[ERROR] model={model_name}, strategy={strategy}: {e}")

# ==================== MEDIAN CALCULATION ====================
def calculate_median_metrics(predictions_dir: str, metrics_dir: str, strategies: list, 
                             model_name: str, client_ids: range):
    """
    Calculate median of all metrics across clients for each strategy.
    """
    for strategy in strategies:
        logging.info(f"\n=== Processing Strategy: {strategy} ===")
        
        mae_list = []
        mse_list = []
        rmse_list = []
        mape_list = []
        smape_list = []
        nrmse_list = []
        
        missing_files = []
        invalid_files = []

        for cid in client_ids:
            metric_file = os.path.join(metrics_dir, f"cid{cid}_{model_name}_{strategy}_metrics.csv")
            
            if not os.path.exists(metric_file):
                missing_files.append(cid)
                continue
            
            try:
                metric_df = pd.read_csv(metric_file)
                if metric_df.empty:
                    invalid_files.append(cid)
                    continue
                
                metrics = metric_df.iloc[0]
                
                if not pd.isna(metrics.get('MAE')):
                    mae_list.append(metrics['MAE'])
                if not pd.isna(metrics.get('MSE')):
                    mse_list.append(metrics['MSE'])
                if not pd.isna(metrics.get('RMSE')):
                    rmse_list.append(metrics['RMSE'])
                if not pd.isna(metrics.get('MAPE (%)')):
                    mape_list.append(metrics['MAPE (%)'])
                if not pd.isna(metrics.get('SMAPE (%)')):
                    smape_list.append(metrics['SMAPE (%)'])
                if not pd.isna(metrics.get('NRMSE')):
                    nrmse_list.append(metrics['NRMSE'])
                    
            except Exception as e:
                invalid_files.append(cid)
                logging.info(f"Error processing CID {cid}: {e}")

        # Calculate and log medians
        if mae_list:
            logging.info(f"\nResults for Strategy: {strategy}")
            logging.info(f"Number of processed metric files: {len(mae_list)}")
            logging.info(f"Median MAE: {pd.Series(mae_list).median():.4f}")
            logging.info(f"Median MSE: {pd.Series(mse_list).median():.4f}")
            logging.info(f"Median RMSE: {pd.Series(rmse_list).median():.4f}")
            logging.info(f"Median MAPE (%): {pd.Series(mape_list).median():.4f}")
            logging.info(f"Median SMAPE (%): {pd.Series(smape_list).median():.4f}")
            logging.info(f"Median NRMSE: {pd.Series(nrmse_list).median():.4f}")
        else:
            logging.info(f"\nNo valid metric files found for {model_name} and {strategy}.")

        if missing_files:
            logging.info(f"\nMissing metric files: {len(missing_files)}")
        if invalid_files:
            logging.info(f"Invalid metric files: {len(invalid_files)}")

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Import your model function
    from Models import model_fn  # Adjust import as needed
    
    STRATEGIES = ["random_no_cluster","random_sampling"]
    MODELS = ["simple_cnn","cnn_gru_no_cov","cnn_gru", "simple_ann", "gru","lstm","moe_lstm"]
    ROUNDS = [40]
    BASE_RESULTS_DIR = "results"
    BASE_OUTPUT_DIR = "predictions40-50-168-T"
    METRICS_DIR = "metrics40-50-168-T"
    CID = range(101, 1409)
    
    # Step 1: Generate predictions and metrics
    get_model_predictions_metric(
        MODELS=MODELS,
        STRATEGIES=STRATEGIES,
        ROUNDS=ROUNDS,
        BASE_RESULTS_DIR=BASE_RESULTS_DIR,
        BASE_OUTPUT_DIR=BASE_OUTPUT_DIR,
        METRICS_DIR=METRICS_DIR,
        CID=CID,
        model_fn=model_fn
    )
    
    # Step 2: Calculate median metrics
    calculate_median_metrics(
        predictions_dir=BASE_OUTPUT_DIR,
        metrics_dir=METRICS_DIR,
        strategies=STRATEGIES,
        model_name=MODELS[0],
        client_ids=CID
    )