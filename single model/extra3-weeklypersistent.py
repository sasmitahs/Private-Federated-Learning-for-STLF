import pandas as pd
import numpy as np
from darts import TimeSeries
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================
# ORIGINAL METRIC FUNCTION + 3 ASHRAE METRICS ADDED
# ============================================================

def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, SMAPE (%), NRMSE, + NMAE, NMBE, NRMSE_mean."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    
    true_range = np.max(y_true) - np.min(y_true) if np.max(y_true) != np.min(y_true) else 1e-8
    nrmse = rmse / true_range

    # --- NEW: ASHRAE METRICS ---
    mean_y = np.mean(y_true)
    nmae = 100 * mae / mean_y if mean_y != 0 else np.nan
    nmbe = 100 * np.mean(y_true - y_pred) / mean_y if mean_y != 0 else np.nan
    nrmse_mean = 100 * rmse / mean_y if mean_y != 0 else np.nan

    return {
        'mae': mae,
        'rmse': rmse,
        'smape': smape,
        'nrmse': nrmse,
        'nmae': nmae,           # <--- NEW
        'nmbe': nmbe,           # <--- NEW
        'nrmse_mean': nrmse_mean  # <--- NEW
    }

# ============================================================
# ORIGINAL PREDICTION FUNCTION (UNCHANGED)
# ============================================================

def naive_model_prediction(cid, filepath="meter_0_data_cleaned.feather", 
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
            raise ValueError(f"Insufficient test data for building_id {cid}. Need at least {lookback_hours + forecast_horizon} samples, got {len(test_ts)}.")
        
        if test_ts.end_time() < pd.Timestamp('2016-12-31 16:00:00'):
            raise ValueError(f"Test data for building_id {cid} ends at {test_ts.end_time()}, does not cover required period up to 2016-12-31 16:00:00.")
        
        full_ts = ts
        values = full_ts.values()
        timestamps = full_ts.time_index
        values = np.nan_to_num(values, nan=0.0)
        
        predictions = []
        actuals = []
        pred_timestamps = []
        
        test_start_idx = ts.get_index_at_point(split_time)
        test_end_idx = len(values)
        
        example_count = 0
        
        for i in range(test_start_idx, test_end_idx - forecast_horizon + 1):
            if i >= lookback_hours:
                pred_window = values[i - lookback_hours : i - lookback_hours + forecast_horizon]
                actual_window = values[i : i + forecast_horizon]
                
                if len(pred_window) == forecast_horizon and len(actual_window) == forecast_horizon:
                    predictions.append(pred_window)
                    actuals.append(actual_window)
                    pred_timestamps.extend(timestamps[i : i + forecast_horizon])
                    
                    if example_count < max_examples and verbose:
                        print(f"\nExample {example_count + 1} for Building ID {cid}:")
                        print(f"Prediction Window (t to t+{forecast_horizon-1}): {timestamps[i]} to {timestamps[i + forecast_horizon - 1]}")
                        print(f"Lookback Window (t-{lookback_hours} to t-{lookback_hours + forecast_horizon - 1}): {timestamps[i - lookback_hours]} to {timestamps[i - lookback_hours + forecast_horizon - 1]}")
                        print(f"Day of Week (Prediction): {timestamps[i].strftime('%A')}")
                        print(f"Day of Week (Lookback): {timestamps[i - lookback_hours].strftime('%A')}")
                        print("Hourly Load Values:")
                        print("Hour | Current Load | Predicted Load (1 week ago)")
                        print("-" * 50)
                        for h in range(min(5, forecast_horizon)):
                            current_time = timestamps[i + h]
                            lookback_time = timestamps[i - lookback_hours + h]
                            current_load = float(actual_window[h])
                            predicted_load = float(pred_window[h])
                            print(f"{current_time.strftime('%Y-%m-%d %H:%M')} | {current_load:.2f} | {predicted_load:.2f} (from {lookback_time.strftime('%Y-%m-%d %H:%M')})")
                        example_count += 1
        
        if len(predictions) == 0:
            raise ValueError(f"No predictions generated for building_id {cid}. Check if test set has enough data.")
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        pred_timestamps = np.array(pred_timestamps)
        
        predictions = np.nan_to_num(predictions, nan=0.0)
        actuals = np.nan_to_num(actuals, nan=0.0)
        
        metrics = calculate_metrics(actuals, predictions)
        
        if verbose:
            print(f"\nMetrics for Building ID {cid}:")
            print(f"MAE: {metrics['mae']:.6f}")
            print(f"RMSE: {metrics['rmse']:.6f}")
            print(f"SMAPE: {metrics['smape']:.4f}%")
            print(f"NRMSE: {metrics['nrmse']:.6f}")
            print(f"NMAE: {metrics['nmae']:.4f}%")        # <--- NEW
            print(f"NMBE: {metrics['nmbe']:.4f}%")        # <--- NEW
            print(f"NRMSE_mean: {metrics['nrmse_mean']:.4f}%")  # <--- NEW
        
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
        raise ValueError(f"Error processing building_id {cid}: {str(e)}")

# ============================================================
# ORIGINAL RUN ALL BUILDINGS (UPDATED TO INCLUDE NEW METRICS)
# ============================================================

def run_all_buildings(filepath="train_final.feather", start_building=0, end_building=1409, 
                     save_individual=True, output_folder="predictions"):
    if save_individual:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Predictions will be saved to folder: {output_folder}/")
    
    all_metrics = []
    failed_buildings = []
    successful_buildings = []
    
    print(f"Processing buildings {start_building} to {end_building}...")
    print("=" * 60)
    
    for building_id in tqdm(range(start_building, end_building + 1), desc="Processing buildings"):
        try:
            results = naive_model_prediction(
                cid=building_id,
                filepath=filepath,
                verbose=True,
                max_examples=3
            )
            
            # --- UPDATED: Include 3 new ASHRAE metrics ---
            all_metrics.append({
                'building_id': building_id,
                'mae': results['metrics']['mae'],
                'rmse': results['metrics']['rmse'],
                'smape': results['metrics']['smape'],
                'nrmse': results['metrics']['nrmse'],
                'nmae': results['metrics']['nmae'],        # <--- NEW
                'nmbe': results['metrics']['nmbe'],        # <--- NEW
                'nrmse_mean': results['metrics']['nrmse_mean'],  # <--- NEW
                'train_size': results['train_size'],
                'test_size': results['test_size'],
                'num_predictions': results['num_predictions']
            })
            
            successful_buildings.append(building_id)
            
            if save_individual:
                pred_df = pd.DataFrame({
                    'timestamp': results['timestamps'],
                    'true': results['actuals'].flatten(),
                    'pred': results['predictions'].flatten()
                })
                output_path = os.path.join(output_folder, f"predictions_building_{building_id}.csv")
                pred_df.to_csv(output_path, index=False)
                
        except Exception as e:
            failed_buildings.append({
                'building_id': building_id,
                'error': str(e)
            })
            print(f"Failed building {building_id}: {str(e)}")
    
    if not all_metrics:
        print("\nWarning: No buildings were successfully processed.")
        return
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # --- UPDATED: Median/Mean for all 7 metrics ---
    median_metrics = {
        'median_mae': metrics_df['mae'].median(),
        'median_rmse': metrics_df['rmse'].median(),
        'median_smape': metrics_df['smape'].median(),
        'median_nrmse': metrics_df['nrmse'].median(),
        'median_nmae': metrics_df['nmae'].median(),        # <--- NEW
        'median_nmbe': metrics_df['nmbe'].median(),        # <--- NEW
        'median_nrmse_mean': metrics_df['nrmse_mean'].median()  # <--- NEW
    }
    
    mean_metrics = {
        'mean_mae': metrics_df['mae'].mean(),
        'mean_rmse': metrics_df['rmse'].mean(),
        'mean_smape': metrics_df['smape'].mean(),
        'mean_nrmse': metrics_df['nrmse'].mean(),
        'mean_nmae': metrics_df['nmae'].mean(),            # <--- NEW
        'mean_nmbe': metrics_df['nmbe'].mean(),            # <--- NEW
        'mean_nrmse_mean': metrics_df['nrmse_mean'].mean()  # <--- NEW
    }
    
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"Total buildings processed: {end_building - start_building + 1}")
    print(f"Successful: {len(successful_buildings)}")
    print(f"Failed: {len(failed_buildings)}")
    print("\n" + "-" * 60)
    print("MEDIAN METRICS (across all buildings):")
    print("-" * 60)
    print(f"Median MAE:        {median_metrics['median_mae']:.6f}")
    print(f"Median RMSE:       {median_metrics['median_rmse']:.6f}")
    print(f"Median SMAPE:      {median_metrics['median_smape']:.4f}%")
    print(f"Median NRMSE:      {median_metrics['median_nrmse']:.6f}")
    print(f"Median NMAE:       {median_metrics['median_nmae']:.4f}%")        # <--- NEW
    print(f"Median NMBE:       {median_metrics['median_nmbe']:.4f}%")        # <--- NEW
    print(f"Median NRMSE_mean: {median_metrics['median_nrmse_mean']:.4f}%")  # <--- NEW
    print("\n" + "-" * 60)
    print("MEAN METRICS (across all buildings):")
    print("-" * 60)
    print(f"Mean MAE:          {mean_metrics['mean_mae']:.6f}")
    print(f"Mean RMSE:         {mean_metrics['mean_rmse']:.6f}")
    print(f"Mean SMAPE:        {mean_metrics['mean_smape']:.4f}%")
    print(f"Mean NRMSE:        {mean_metrics['mean_nrmse']:.6f}")
    print(f"Mean NMAE:         {mean_metrics['mean_nmae']:.4f}%")            # <--- NEW
    print(f"Mean NMBE:         {mean_metrics['mean_nmbe']:.4f}%")            # <--- NEW
    print(f"Mean NRMSE_mean:   {mean_metrics['mean_nrmse_mean']:.4f}%")      # <--- NEW
    print("=" * 60)
    
    metrics_df.to_csv("all_buildings_metrics_summary.csv", index=False)
    print("\nDetailed metrics saved to: all_buildings_metrics_summary.csv")
    
    if save_individual:
        print(f"All individual predictions saved to folder: {output_folder}/")
    
    if failed_buildings:
        failed_df = pd.DataFrame(failed_buildings)
        failed_df.to_csv("failed_buildings.csv", index=False)
        print(f"Failed buildings saved to: failed_buildings.csv")
        print(f"\nFirst 5 failed buildings:")
        print(failed_df.head())
    
    return {
        'metrics_df': metrics_df,
        'median_metrics': median_metrics,
        'mean_metrics': mean_metrics,
        'failed_buildings': failed_buildings,
        'successful_buildings': successful_buildings
    }

# ============================================================
# MAIN (UNCHANGED)
# ============================================================

def main():
    results = run_all_buildings(
        filepath="train_final.feather",
        start_building=100,
        end_building=1409,
        save_individual=True,
        output_folder="predictions"
    )
    
    print("\n" + "=" * 60)
    print("ADDITIONAL STATISTICS")
    print("=" * 60)
    if not results['metrics_df'].empty:
        df = results['metrics_df']
        print(f"MAE - Min: {df['mae'].min():.6f}, Max: {df['mae'].max():.6f}, Std: {df['mae'].std():.6f}")
        print(f"RMSE - Min: {df['rmse'].min():.6f}, Max: {df['rmse'].max():.6f}, Std: {df['rmse'].std():.6f}")
        print(f"SMAPE - Min: {df['smape'].min():.4f}%, Max: {df['smape'].max():.4f}%, Std: {df['smape'].std():.4f}%")
        print(f"NRMSE - Min: {df['nrmse'].min():.6f}, Max: {df['nrmse'].max():.6f}, Std: {df['nrmse'].std():.6f}")
        print(f"NMAE - Min: {df['nmae'].min():.4f}%, Max: {df['nmae'].max():.4f}%, Std: {df['nmae'].std():.4f}%")        # <--- NEW
        print(f"NMBE - Min: {df['nmbe'].min():.4f}%, Max: {df['nmbe'].max():.4f}%, Std: {df['nmbe'].std():.4f}%")        # <--- NEW
        print(f"NRMSE_mean - Min: {df['nrmse_mean'].min():.4f}%, Max: {df['nrmse_mean'].max():.4f}%, Std: {df['nrmse_mean'].std():.4f}%")  # <--- NEW
    else:
        print("No statistics computed due to empty metrics DataFrame.")
    print("=" * 60)

if __name__ == "__main__":
    main()