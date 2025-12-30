import pandas as pd
import numpy as np
from pmdarima import auto_arima
from darts import TimeSeries
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, SMAPE (%), and NRMSE for predictions."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    
    true_range = np.max(y_true) - np.min(y_true) if np.max(y_true) != np.min(y_true) else 1e-8
    nrmse = rmse / true_range
    
    return {
        'mae': mae,
        'rmse': rmse,
        'smape': smape,
        'nrmse': nrmse
    }

# ============================================================
# AUTOARIMA MODEL PREDICTION FUNCTION
# ============================================================

def autoarima_model_prediction(cid, filepath="meter_0_data_cleaned.feather", 
                               forecast_horizon=24, verbose=False):
    """
    AutoARIMA Forecasting for a single building using the same preprocessing steps.
    Uses weekly seasonality (m=168 for hourly data).
    Trains AutoARIMA on 75% of data and predicts the next 25%.
    """
    try:
        # Load data
        df = pd.read_feather(filepath)
        df = df[df['building_id'] == cid]
        
        if df.empty:
            raise ValueError(f"No data found for building_id {cid}")
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Fill NaN values with 0 (keep preprocessing consistent)
        df['meter_reading'] = df['meter_reading'].fillna(0)
        
        if verbose:
            print(f"NaNs in meter_reading after fillna: {df['meter_reading'].isna().sum()}")
        
        # Convert to TimeSeries (ensure consistent frequency and fill missing timestamps)
        ts = TimeSeries.from_dataframe(
            df,
            time_col='timestamp',
            value_cols='meter_reading',
            fill_missing_dates=True,
            freq='h'
        )
        
        if verbose:
            print(f"NaNs in TimeSeries: {np.any(np.isnan(ts.values()))}")
        
        # Split data 75% train, 25% test
        train_ts, test_ts = ts.split_before(0.75)
        
        if verbose:
            print(f"Total samples: {len(ts)}")
            print(f"Train samples: {len(train_ts)}")
            print(f"Test samples: {len(test_ts)}")
        
        # Get numpy arrays for AutoARIMA
        train_values = np.nan_to_num(train_ts.values().flatten(), nan=0.0)
        test_values = np.nan_to_num(test_ts.values().flatten(), nan=0.0)
        test_timestamps = test_ts.time_index
        
        if verbose:
            print(f"NaNs in train_values: {np.any(np.isnan(train_values))}")
            print(f"NaNs in test_values: {np.any(np.isnan(test_values))}")
        
        # ====================================================
        # Fit AutoARIMA model
        # ====================================================
        model = auto_arima(
            train_values,
            seasonal=False,
            m=168,  # weekly seasonality for hourly data
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        
        # Forecast
        forecast = model.predict(n_periods=len(test_values))
        
        # Replace residual NaNs (if any)
        forecast = np.nan_to_num(forecast, nan=0.0)
        
        # ====================================================
        # Metrics
        # ====================================================
        metrics = calculate_metrics(test_values, forecast)
        
        if verbose:
            print(f"\nMetrics for Building ID {cid}:")
            print(f"MAE: {metrics['mae']:.6f}")
            print(f"RMSE: {metrics['rmse']:.6f}")
            print(f"SMAPE: {metrics['smape']:.4f}%")
            print(f"NRMSE: {metrics['nrmse']:.6f}")
        
        return {
            'predictions': forecast,
            'actuals': test_values,
            'timestamps': test_timestamps,
            'metrics': metrics,
            'train_size': len(train_ts),
            'test_size': len(test_ts)
        }
    
    except Exception as e:
        raise ValueError(f"Error processing building_id {cid}: {str(e)}")

# ============================================================
# LOOP THROUGH ALL BUILDINGS
# ============================================================

def run_all_buildings(filepath="train_final.feather", start_building=0, end_building=1409, 
                     save_individual=True, output_folder="predictions_autoarima_weekly"):
    """
    Run AutoARIMA (weekly seasonal) for all buildings using the same preprocessing steps.
    """
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
            results = autoarima_model_prediction(
                cid=building_id,
                filepath=filepath,
                verbose=False
            )
            
            all_metrics.append({
                'building_id': building_id,
                'mae': results['metrics']['mae'],
                'rmse': results['metrics']['rmse'],
                'smape': results['metrics']['smape'],
                'nrmse': results['metrics']['nrmse'],
                'train_size': results['train_size'],
                'test_size': results['test_size']
            })
            
            successful_buildings.append(building_id)
            
            if save_individual:
                pred_df = pd.DataFrame({
                    'timestamp': results['timestamps'],
                    'true': results['actuals'],
                    'pred': results['predictions']
                })
                output_path = os.path.join(output_folder, f"predictions_building_{building_id}.csv")
                pred_df.to_csv(output_path, index=False)
                
        except Exception as e:
            failed_buildings.append({
                'building_id': building_id,
                'error': str(e)
            })
            print(f"Failed building {building_id}: {str(e)}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    if not all_metrics:
        print("\nWarning: No buildings were successfully processed.")
        return
    
    metrics_df = pd.DataFrame(all_metrics)
    
    median_metrics = {
        'median_mae': metrics_df['mae'].median(),
        'median_rmse': metrics_df['rmse'].median(),
        'median_smape': metrics_df['smape'].median(),
        'median_nrmse': metrics_df['nrmse'].median()
    }
    
    mean_metrics = {
        'mean_mae': metrics_df['mae'].mean(),
        'mean_rmse': metrics_df['rmse'].mean(),
        'mean_smape': metrics_df['smape'].mean(),
        'mean_nrmse': metrics_df['nrmse'].mean()
    }
    
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS (AutoARIMA - Weekly Seasonal)")
    print("=" * 60)
    print(f"Total buildings processed: {end_building - start_building + 1}")
    print(f"Successful: {len(successful_buildings)}")
    print(f"Failed: {len(failed_buildings)}")
    print("\n" + "-" * 60)
    print("MEDIAN METRICS (across all buildings):")
    print("-" * 60)
    print(f"Median MAE:    {median_metrics['median_mae']:.6f}")
    print(f"Median RMSE:   {median_metrics['median_rmse']:.6f}")
    print(f"Median SMAPE:  {median_metrics['median_smape']:.4f}%")
    print(f"Median NRMSE:  {median_metrics['median_nrmse']:.6f}")
    print("\n" + "-" * 60)
    print("MEAN METRICS (across all buildings):")
    print("-" * 60)
    print(f"Mean MAE:      {mean_metrics['mean_mae']:.6f}")
    print(f"Mean RMSE:     {mean_metrics['mean_rmse']:.6f}")
    print(f"Mean SMAPE:    {mean_metrics['mean_smape']:.4f}%")
    print(f"Mean NRMSE:    {mean_metrics['mean_nrmse']:.6f}")
    print("=" * 60)
    
    # Save summary
    metrics_df.to_csv("autoarima_weekly_metrics.csv", index=False)
    if failed_buildings:
        pd.DataFrame(failed_buildings).to_csv("autoarima_failed_buildings.csv", index=False)
    
    return {
        'metrics_df': metrics_df,
        'median_metrics': median_metrics,
        'mean_metrics': mean_metrics,
        'failed_buildings': failed_buildings,
        'successful_buildings': successful_buildings
    }

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main execution function."""
    results = run_all_buildings(
        filepath="train_final.feather",
        start_building=100,
        end_building=1409,  # Adjust this for full run
        save_individual=True,
        output_folder="predictions_autoarima_weekly"
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
    else:
        print("No statistics computed due to empty metrics DataFrame.")
    print("=" * 60)

if __name__ == "__main__":
    main()
