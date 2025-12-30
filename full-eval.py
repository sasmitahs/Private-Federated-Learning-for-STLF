import os
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm

# ============================================================================
# IMPORTANT: Import your model constructor from Models.py
# ============================================================================
from Models import model_fn  # This should work if model_fn is defined in Models.py

# If the above fails, uncomment and adjust below:
# from Models import DualCNNGRUFCNN  # Replace with your actual class name
# def model_fn(model_name: str):
#     if model_name == "dual_cnn_gru_fcnn":
#         return DualCNNGRUFCNN()  # Add arguments if constructor requires them
#     raise ValueError(f"Unknown model: {model_name}")

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
    print(f"[DEBUG] Starting forecast for CID={cid}")

    df = pd.read_feather(filepath)
    df = df[df['building_id'] == cid].copy()
    if df.empty:
        print(f"[WARN] No data found for building_id {cid}")
        return [], []

    df['meter_reading'] = df['meter_reading'].fillna(0.0)
    df['air_temperature'] = df['air_temperature'].fillna(df['air_temperature'].mean())
    df['primary_use_idx'] = 0
    if 'primary_use' in df.columns:
        primary_map = {cat: idx for idx, cat in enumerate(df['primary_use'].unique())}
        df['primary_use_idx'] = df['primary_use'].map(primary_map).fillna(0).astype(int)

    meter_values = df['meter_reading'].values
    try:
        stl = STL(meter_values, period=24)
        res = stl.fit()
        trend, seasonal, resid = res.trend, res.seasonal, res.resid
    except:
        trend = np.zeros_like(meter_values)
        seasonal = np.zeros_like(meter_values)
        resid = meter_values.copy()

    multi_channel = np.stack([meter_values, trend, seasonal, resid], axis=-1)
    air_temp_vals = df['air_temperature'].values
    primary_use_vals = df['primary_use_idx'].values

    split_idx = int(0.75 * len(multi_channel))
    test_values = multi_channel[split_idx:]
    test_air_temp = air_temp_vals[split_idx:]
    test_primary_use = primary_use_vals[split_idx:]
    test_time = pd.to_datetime(df['timestamp'].values[split_idx:])

    if len(test_values) < input_len + output_len:
        print(f"[WARN] Test sequence too short for CID {cid}")
        return [], []

    scaler_mc = MinMaxScaler(feature_range=(0.1, 1))
    test_values_scaled = scaler_mc.fit_transform(test_values)
    test_air_temp_scaled = MinMaxScaler(feature_range=(0.1, 1)).fit_transform(test_air_temp.reshape(-1, 1))

    ts = TimeSeries.from_dataframe(
        df[split_idx:], time_col='timestamp', value_cols='meter_reading',
        fill_missing_dates=True, freq='h'
    )
    transformer = Scaler(MinMaxScaler(feature_range=(0.1, 1)))
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
    step = output_len

    for i in range(0, max_start + 1, step):
        x_ts_np = test_values_scaled[i:i + input_len]
        x_air_np = test_air_temp_scaled[i:i + input_len]
        primary_np = test_primary_use[i:i + input_len]

        x_ts = torch.tensor(x_ts_np, dtype=torch.float32).unsqueeze(0).to(device)
        x_cov = torch.tensor(x_air_np, dtype=torch.float32).unsqueeze(0).to(device)
        primary_use_tensor = torch.tensor(
            np.clip(primary_np, 0, n_emb - 1) if has_embedding else np.zeros(input_len),
            dtype=torch.long
        ).unsqueeze(0).to(device)

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

        padded_pred = np.pad(pred_np.reshape(-1, 1), ((0, 0), (0, 3)), mode='constant')
        pred_unscaled_vals = scaler_mc.inverse_transform(padded_pred)[:, 0]
        pred_ts = TimeSeries.from_times_and_values(test_time_true[true_start:true_end], pred_unscaled_vals)

        predictions_ts_list.append(pred_ts)
        ground_truth_ts_list.append(true_unscaled)

    return predictions_ts_list, ground_truth_ts_list

# ============================================================================
# Main: Generate Predictions for All 4 Runs (Direct in results/ folder)
# ============================================================================
def generate_predictions_for_all_runs(
    model_name: str = "dual_cnn_gru_fcnn",
    base_strategy: str = "poc",
    runs: List[int] = None,
    model_dir: str = "results",  # All .pt files directly in this folder
    output_dir: str = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/predictions40-50-168-T",
    cid_range: range = range(101, 1409),
    data_filepath: str = "train_final.feather"
):
    if runs is None:
        runs = [1, 2, 3, 4]

    os.makedirs(output_dir, exist_ok=True)
    print(f"Prediction output directory: {output_dir}")

    for run in runs:
        strategy = f"{base_strategy}_run{run}"
        model_path = os.path.join(model_dir, f"{model_name}_{strategy}.pt")

        if not os.path.exists(model_path):
            print(f"[WARN] Model not found: {model_path} → Skipping run {run}")
            continue

        print(f"\n{'='*70}")
        print(f"LOADING MODEL FOR RUN {run}")
        print(f"Path: {model_path}")
        print(f"{'='*70}")

        # Load model
        model = model_fn(model_name)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()

        print(f"Generating predictions for {len(cid_range)} clients (Run {run})...")

        for cid in tqdm(cid_range, desc=f"Run {run} - Clients"):
            output_csv = os.path.join(output_dir, f"{cid}_{model_name}_{strategy}.csv")

            if os.path.exists(output_csv):
                continue  # Skip if already generated

            try:
                pred_list, true_list = rolling_forecast_on_test(
                    cid=cid, model=model, filepath=data_filepath
                )

                if len(pred_list) == 0:
                    continue

                rows = []
                for pred_ts in zip(pred_list, true_list):
                    pred_ts, true_ts = pred_ts
                    df_pred = pd.DataFrame({"timestamp": pred_ts.time_index, "pred": pred_ts.values().squeeze()})
                    df_true = pd.DataFrame({"timestamp": true_ts.time_index, "true": true_ts.values().squeeze()})
                    df_merged = pd.merge(df_true, df_pred, on="timestamp", how="inner")
                    df_merged["round"] = 40
                    rows.append(df_merged)

                if rows:
                    final_df = pd.concat(rows, ignore_index=True)
                    final_df = final_df[["timestamp", "true", "pred", "round"]]
                    final_df.to_csv(output_csv, index=False)

            except Exception as e:
                print(f"[ERROR] CID {cid} | Run {run}: {e}")

        print(f"Completed Run {run} → Predictions saved with suffix _{strategy}")

# ============================================================================
# Execute
# ============================================================================
if __name__ == "__main__":
    generate_predictions_for_all_runs(
        model_name="dual_cnn_gru_fcnn",
        base_strategy="poc",
        runs=[1, 2, 3, 4],
        model_dir="results",  # Folder containing dual_cnn_gru_fcnn_poc_run1.pt etc.
        output_dir="/home/user/DPFL-Sasmita/FL-Baseline-Codes/predictions40-50-168-T",
        cid_range=range(101, 1409),
        data_filepath="train_final.feather"  # Update path if needed
    )

    print("\n" + "="*70)
    print("ALL 4 RUNS COMPLETED SUCCESSFULLY!")
    print("Generated prediction CSVs: *_poc_run1.csv to *_poc_run4.csv")
    print("You can now run your evaluation script on these files.")
    print("="*70)