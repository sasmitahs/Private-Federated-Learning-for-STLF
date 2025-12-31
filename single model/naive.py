import pandas as pd
import numpy as np
from darts import TimeSeries
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings
import os
import logging
from datetime import datetime
warnings.filterwarnings('ignore')


# ============================================================
# ENHANCED METRIC FUNCTIONS (7 total)
# ============================================================

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.where(denominator == 0, 0,
                            np.abs(y_true - y_pred) / denominator)) * 100

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def nrmse_range(y_true, y_pred):
    """Normalized RMSE by range."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    true_range = np.max(y_true) - np.min(y_true)
    return rmse / true_range if true_range != 0 else np.nan

def nmae(y_true, y_pred):
    """Normalized MAE (ASHRAE style)."""
    mae = mean_absolute_error(y_true, y_pred)
    mean_y = np.mean(y_true)
    return 100 * mae / mean_y if mean_y != 0 else np.nan

def nmbe(y_true, y_pred):
    """Normalized Mean Bias Error."""
    mean_bias = np.mean(y_true - y_pred)
    mean_y = np.mean(y_true)
    return 100 * mean_bias / mean_y if mean_y != 0 else np.nan

def nrmse_mean(y_true, y_pred):
    """Normalized RMSE by mean (ASHRAE)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_y = np.mean(y_true)
    return 100 * rmse / mean_y if mean_y != 0 else np.nan

def calculate_enhanced_metrics(y_true, y_pred):
    """Calculate all 7 metrics."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape_val = mape(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    nrmse_range_val = nrmse_range(y_true, y_pred)
    nmae_val = nmae(y_true, y_pred)
    nmbe_val = nmbe(y_true, y_pred)
    nrmse_mean_val = nrmse_mean(y_true, y_pred)

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape_val,
        'smape': smape_val,
        'nrmse_range': nrmse_range_val,
        'nmae': nmae_val,
        'nmbe': nmbe_val,
        'nrmse_mean': nrmse_mean_val
    }


# ============================================================
# NAIVE MODEL PREDICTION (UNCHANGED LOGIC)
# ============================================================

def naive_model_prediction(cid, filepath="train_final.feather", 
                         lookback_hours=168, forecast_horizon=24, verbose=False, max_examples=3):
    try:
        df = pd.read_feather(filepath)
        df = df[df['building_id'] == cid]
        
        if df.empty:
            raise ValueError(f"No data found for building_id {cid}")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['meter_reading'] = df['meter_reading'].fillna(0)
        
        ts = TimeSeries.from_dataframe(
            df,
            time_col='timestamp',
            value_cols='meter_reading',
            fill_missing_dates=True,
            freq='h'
        )
        
        split_time = pd.Timestamp('2016-11-12 17:00:00')
        train_ts, test_ts = ts.split_before(split_time)
        
        if len(test_ts) < lookback_hours + forecast_horizon:
            raise ValueError(f"Insufficient test data: need {lookback_hours + forecast_horizon}, got {len(test_ts)}")
        if test_ts.end_time() < pd.Timestamp('2016-12-31 16:00:00'):
            raise ValueError(f"Test ends at {test_ts.end_time()}, must go to 2016-12-31 16:00:00")

        values = np.nan_to_num(ts.values(), nan=0.0)
        timestamps = ts.time_index
        test_start_idx = ts.get_index_at_point(split_time)

        predictions = []
        actuals = []
        pred_timestamps = []
        example_count = 0

        for i in range(test_start_idx, len(values) - forecast_horizon + 1):
            if i >= lookback_hours:
                pred_window = values[i - lookback_hours : i - lookback_hours + forecast_horizon]
                actual_window = values[i : i + forecast_horizon]
                
                if len(pred_window) == forecast_horizon:
                    predictions.append(pred_window)
                    actuals.append(actual_window)
                    pred_timestamps.extend(timestamps[i : i + forecast_horizon])
                    
                    if example_count < max_examples and verbose:
                        print(f"\nExample {example_count + 1} for CID {cid}:")
                        print(f"Pred: {timestamps[i]} → {timestamps[i + forecast_horizon - 1]}")
                        print(f"Lookback: {timestamps[i - lookback_hours]}")
                        for h in range(min(3, forecast_horizon)):
                            print(f"  {timestamps[i+h].strftime('%m-%d %H:%M')} | True: {actual_window[h]:.2f} | Pred: {pred_window[h]:.2f}")
                        example_count += 1

        if len(predictions) == 0:
            raise ValueError("No predictions generated.")

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        pred_timestamps = np.array(pred_timestamps)

        metrics = calculate_enhanced_metrics(actuals, predictions)

        if verbose:
            print(f"\nMetrics for CID {cid}:")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  SMAPE: {metrics['smape']:.2f}%")
            print(f"  NRMSE_range: {metrics['nrmse_range']:.4f}")
            print(f"  NMAE: {metrics['nmae']:.2f}%")
            print(f"  NMBE: {metrics['nmbe']:.2f}%")
            print(f"  NRMSE_mean: {metrics['nrmse_mean']:.2f}%")

        return {
            'predictions': predictions,
            'actuals': actuals,
            'timestamps': pred_timestamps,
            'metrics': metrics,
            'train_size': len(train_ts),
            'test_size': len(test_ts),
            'num_predictions': len(predictions)
        }
    
    except Exception as e:
        raise ValueError(f"Error in CID {cid}: {str(e)}")


# ============================================================
# RUN ALL BUILDINGS + SAVE + SUMMARY
# ============================================================

def run_all_buildings(filepath="train_final.feather", start_building=0, end_building=1409, 
                     save_individual=True, output_folder="predictions_naive_weekly", 
                     metrics_folder="metrics_naive_weekly"):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)
    print(f"Predictions → {output_folder}/")
    print(f"Metrics → {metrics_folder}/")

    all_metrics = []
    failed_buildings = []
    successful_buildings = []

    print(f"Processing buildings {start_building} to {end_building}...")
    print("=" * 80)

    for cid in tqdm(range(start_building, end_building + 1), desc="Buildings"):
        try:
            result = naive_model_prediction(
                cid=cid,
                filepath=filepath,
                verbose=False,
                max_examples=0
            )

            # Save prediction CSV
            if save_individual:
                pred_df = pd.DataFrame({
                    'timestamp': result['timestamps'],
                    'true': result['actuals'].flatten(),
                    'pred': result['predictions'].flatten()
                })
                pred_path = os.path.join(output_folder, f"predictions_building_{cid}.csv")
                pred_df.to_csv(pred_path, index=False)

            # Save enhanced metrics CSV
            metrics_row = {
                'building_id': cid,
                'mae': result['metrics']['mae'],
                'rmse': result['metrics']['rmse'],
                'mape': result['metrics']['mape'],
                'smape': result['metrics']['smape'],
                'nrmse_range': result['metrics']['nrmse_range'],
                'nmae': result['metrics']['nmae'],
                'nmbe': result['metrics']['nmbe'],
                'nrmse_mean': result['metrics']['nrmse_mean'],
                'train_size': result['train_size'],
                'test_size': result['test_size'],
                'num_predictions': result['num_predictions']
            }
            all_metrics.append(metrics_row)
            successful_buildings.append(cid)

            # Save individual metric CSV
            metric_df = pd.DataFrame([metrics_row])
            metric_path = os.path.join(metrics_folder, f"cid{cid}_naive_weekly_metrics.csv")
            metric_df.to_csv(metric_path, index=False)

        except Exception as e:
            failed_buildings.append({'building_id': cid, 'error': str(e)})
            print(f"Failed CID {cid}: {e}")

    # ============================================================
    # SUMMARY STATISTICS
    # ============================================================

    if not all_metrics:
        print("No successful predictions.")
        return

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(metrics_folder, "all_buildings_full_metrics.csv"), index=False)

    # Compute median/mean
    med = df[['mae', 'rmse', 'mape', 'smape', 'nrmse_range', 'nmae', 'nmbe', 'nrmse_mean']].median()
    mean = df[['mae', 'rmse', 'mape', 'smape', 'nrmse_range', 'nmae', 'nmbe', 'nrmse_mean']].mean()
    var = df[['nmae', 'nmbe', 'nrmse_mean']].var(ddof=1)

    print("\n" + "="*80)
    print("AGGREGATE RESULTS (Naive Weekly)")
    print("="*80)
    print(f"Successful: {len(successful_buildings)} | Failed: {len(failed_buildings)}")
    print("\nMEDIAN METRICS:")
    print(f"  MAE:        {med['mae']:.6f}")
    print(f"  RMSE:       {med['rmse']:.6f}")
    print(f"  MAPE:       {med['mape']:.2f}%")
    print(f"  SMAPE:      {med['smape']:.2f}%")
    print(f"  NRMSE_range:{med['nrmse_range']:.4f}")
    print(f"  NMAE:       {med['nmae']:.2f}%")
    print(f"  NMBE:       {med['nmbe']:.2f}%")
    print(f"  NRMSE_mean: {med['nrmse_mean']:.2f}%")

    print("\nMEAN METRICS:")
    print(f"  NMAE: {mean['nmae']:.2f}% | Var: {var['nmae']:.2f}")
    print(f"  NMBE: {mean['nmbe']:.2f}% | Var: {var['nmbe']:.2f}")
    print(f"  NRMSE_mean: {mean['nrmse_mean']:.2f}% | Var: {var['nrmse_mean']:.2f}")
    print("="*80)

    # Save summary log
    log_path = os.path.join(metrics_folder, f"naive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_path, 'w') as f:
        f.write("NAIVE WEEKLY MODEL - FULL METRICS SUMMARY\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Buildings: {start_building}–{end_building} | Success: {len(successful_buildings)}\n\n")
        f.write("MEDIAN:\n" + med.to_string() + "\n\n")
        f.write("MEAN (ASHRAE):\n" + mean[['nmae','nmbe','nrmse_mean']].to_string() + "\n\n")
        f.write("VARIANCE (ASHRAE):\n" + var.to_string())
    print(f"Summary log: {log_path}")

    if failed_buildings:
        pd.DataFrame(failed_buildings).to_csv(os.path.join(metrics_folder, "failed_buildings.csv"), index=False)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results_df = run_all_buildings(
        filepath="train_final.feather",
        start_building=100,
        end_building=1409,
        save_individual=True,
        output_folder="predictions_naive_weekly",
        metrics_folder="metrics_naive_weekly"
    )

    print(f"\nDone! Check folders:\n  → {os.path.abspath('predictions_naive_weekly')}\n  → {os.path.abspath('metrics_naive_weekly')}")