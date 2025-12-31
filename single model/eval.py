import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from Models import MoELSTM
import os
from collections import OrderedDict
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import random
from Models import MoELSTM, LSTMModel, train_model
from Preprocess import (
    compute_metrics,
    convert_timeseries_to_numpy,
    create_dataloader,
    load_building_series,
    split_series_list,
)
from Models import model_fn
from tqdm import tqdm
from my_utils import train_model, load_energy_data_feather, get_weights, set_weights
# from energy_ts_diffusion.task import convert_timeseries_to_numpy  # adjust as per your project
from tqdm import tqdm

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
    device = next(model.parameters()).device
    device = 'cuda'
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
import os
import pandas as pd

def model_strategy_csv(Base_dir, Target_dir, round_num, METRIC, sortBy="Model_Strategy"):
    """
    Computes boxplot statistics for each model_strategy CSV
    and saves the pivot table in the target directory.

    The metric column is dynamically named with METRIC.
    """
    os.makedirs(Target_dir, exist_ok=True)

    all_rows = []

    for csv_file in os.listdir(Base_dir):
        if not csv_file.endswith(".csv"):
            continue

        csv_path = os.path.join(Base_dir, csv_file)
        df = pd.read_csv(csv_path)

        # Filter for the specified round
        df_round = df[df["round"] == round_num]

        if df_round.empty:
            print(f"[SKIP] No round {round_num} in: {csv_file}")
            continue

        metric_values = df_round[METRIC].dropna()

        # Compute stats
        count = metric_values.count()
        min_val = metric_values.min()
        q1 = metric_values.quantile(0.25)
        median = metric_values.median()
        q3 = metric_values.quantile(0.75)
        max_val = metric_values.max()
        iqr = q3 - q1

        # Whiskers
        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr

        # Non-outlier min/max
        non_outliers = metric_values[(metric_values >= lower_whisker) & (metric_values <= upper_whisker)]
        min_non_outlier = non_outliers.min()
        max_non_outlier = non_outliers.max()

        # Extract model_strategy and strategy
        model_strategy = csv_file.replace("_metrics.csv", "")
        try:
            model, strategy = model_strategy.split("_", 1)
        except ValueError:
            strategy = "unknown"

        # ✅ The metric name is now a column with the median value.
        row = {
            "Model_Strategy": model_strategy,
            # "Strategy": strategy,
            "Round": round_num,
            "METRIC": METRIC,  # dynamic metric column with value
            "Count": count,
            "Min": min_val,
            "Q1": q1,
            "Median": median,
            "Q3": q3,
            "Max": max_val,
            "IQR": iqr,
            # "Lower Whisker": lower_whisker,
            # "Upper Whisker": upper_whisker,
            "True Min (Non-outlier)": min_non_outlier,
            "True Max (Non-outlier)": max_non_outlier
        }

        all_rows.append(row)

    # Combine to DataFrame
    pivot_df = pd.DataFrame(all_rows)
    pivot_df.sort_values(by=sortBy, inplace=True)

    # Clean output name
    safe_metric = METRIC.replace(" ", "_").replace("%", "percent")
    pivot_output = os.path.join(
        Target_dir,
        f"all_stats_round{round_num}_metric_{safe_metric}_sort_by_{sortBy}.csv"
    )

    pivot_df.to_csv(pivot_output, index=False)

    print(f"[OK] One-row-per-model_strategy pivot saved: {pivot_output}")
# evaluate forecasts - working correctly 1411 buildings count 
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)) * 100

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true = np.where(y_true == 0, 1e-8, y_true)  # avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_forecast_metrics_per_round(csv_path):
    """
    Reads forecast CSV and computes MAPE, MAE, SMAPE, RMSE, and MSE per round.

    Args:
        csv_path (str): Path to the CSV with columns: timestamp, true, pred, round

    Returns:
        pd.DataFrame: Metrics summary per round
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

        metrics_list.append({
            "round": rnd,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE (%)": mape_val,
            "SMAPE (%)": smape_val
        })

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df
# try in function
def get_model_metric_grouped_by_model_strategy(
    MODELS: List[str],
    STRATEGIES: List[str],
    ROUNDS: List[int],
    BASE_PREDICTIONS_DIR: str,
    BASE_METRICS_DIR: str,
    METRIC: str = "MAE"
) -> None:
    """
    Computes metrics for each model and strategy, grouped by client ID and round.
    
    Args:
        MODELS (List[str]): List of model names.
        STRATEGIES (List[str]): List of strategy names.
        ROUNDS (List[int]): List of round numbers.
        BASE_PREDICTIONS_DIR (str): Directory containing prediction CSVs.
        BASE_METRICS_DIR (str): Directory to save metrics CSVs.
        METRIC (str): The metric to compute, e.g., "MAE", "MSE", etc.
    """
    os.makedirs(BASE_METRICS_DIR, exist_ok=True)

    for CID in tqdm(range(1411), desc="Processing Client IDs"):
        for model_name in MODELS:
            for strategy in STRATEGIES:
                input_csv = os.path.join(BASE_PREDICTIONS_DIR, f"{CID}_{model_name}_{strategy}.csv")
                output_csv = os.path.join(BASE_METRICS_DIR, f"{model_name}_{strategy}_metrics.csv")

                if not os.path.isfile(input_csv):
                    print(f"[SKIP] Missing: {input_csv}")
                    continue

                try:
                    # Compute metrics for the given CSV
                    metrics_df = evaluate_forecast_metrics_per_round(input_csv)
                    metrics_df.insert(0, "building_id", CID)  # add client ID
                    metrics_df.insert(1, "model", model_name)
                    metrics_df.insert(2, "strategy", strategy)

                    if os.path.isfile(output_csv):
                        metrics_df.to_csv(output_csv, index=False)
                    else:
                        metrics_df.to_csv(output_csv, index=False)

                    print(f"[OK] {output_csv} <- CID {CID}")

                except Exception as e:
                    print(f"[ERROR] CID={CID} | {model_name}{strategy} | {e}")
# Get model predictions and save to CSV
# This function will load the model for each round, perform predictions, and save results to a CSV file. - call rolling_forecast_on_test
def get_model_predictions_csv(model_name: str, cid: int,aggr_strat: str ,rounds: list, model_dir: str, output_csv: str):
    """
    For each round, load the model, predict on test set for cid, and save all preds in a single CSV.
    """
    rows = []

    for rnd in tqdm(rounds):
        model_path = os.path.join(model_dir, f"{model_name}_round_{rnd}_{aggr_strat}.pt")

        if not os.path.exists(model_path):
            print(f"[WARN] Model not found: {model_path}")
            continue

        model = model_fn(model_name)
        # model.load_state_dict(torch.load(model_path), weights_only=True)  # Ensure weights_only=True if using a custom model
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to('cuda')
        model.eval()

        pred_ts_list, gt_ts_list = rolling_forecast_on_test(cid=cid, model=model)

        for pred_ts, true_ts in zip(pred_ts_list, gt_ts_list):
            df_pred = pd.DataFrame({"timestamp": pred_ts.time_index, "pred": pred_ts.values().squeeze()})
            df_true = pd.DataFrame({"timestamp": true_ts.time_index, "true": true_ts.values().squeeze()})

            # df_merged = df_true.join(df_pred, how="inner")
            df_merged = pd.merge(df_true, df_pred, on="timestamp", how="inner")
            df_merged["round"] = rnd


            rows.append(df_merged[["timestamp", "true", "pred", "round"]])

    # Combine all rows
    final_df = pd.concat(rows, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"[INFO] Forecasts written to {output_csv}")

# Model List , strategy List, round List, base_results_dir, base_output_dir, metrics_dir,cid range

# # Model List , strategy List, round List, base_results_dir, base_output_dir, metrics_dir,cid range

import os

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
        MODELS (List[str]): List of model names (e.g., ["gru", "lstm"])
        STRATEGIES (List[str]): List of aggregation strategies (e.g., ["_scaffold", "_diff"])
        ROUNDS (List[int]): Rounds to evaluate (e.g., list(range(9, 11)))
        BASE_RESULTS_DIR (str): Directory containing saved model weights.
        BASE_OUTPUT_DIR (str): Directory to save prediction CSVs.
        METRICS_DIR (str): Directory to save metric CSVs.
        CID (range): Range of client IDs (e.g., range(1411))
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


STRATEGIES = ["AEpublic_CNN+GRU","AEpublic_CNN"]# _diff",,"fedProx_diff "diff-diff", scaffold_diff,"fedAvg_diff0","fedProx",,"fedAvg_lr","scaffold_lr" "diff_lr2","das11","das2", "fedAvg_lr","fedAvg_diffsample_dhc"
MODELS = ["gru"]#,"gru"
# CID = 45
ROUNDS = [40]
BASE_RESULTS_DIR = "results"
BASE_OUTPUT_DIR = "predictions-baseline"
METRICS_DIR = "metrics-baseline" 
CID = range(0,1409) 

get_model_predictions_metric(
    MODELS=MODELS,
    STRATEGIES=STRATEGIES,
    ROUNDS=ROUNDS,
    BASE_RESULTS_DIR=BASE_RESULTS_DIR,
    BASE_OUTPUT_DIR=BASE_OUTPUT_DIR,
    METRICS_DIR=METRICS_DIR,
    CID=CID
)