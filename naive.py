import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

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

def regenerate_metrics_for_building(building_id, predictions_folder="predictions"):
    """
    Regenerate metrics for a single building from stored predictions.
    
    Args:
        building_id (int): Building ID to process.
        predictions_folder (str): Folder containing prediction CSV files.
    
    Returns:
        dict: Metrics and metadata for the building, or None if failed.
    """
    try:
        # Load predictions CSV
        pred_file = os.path.join(predictions_folder, f"predictions_building_{building_id}.csv")
        if not os.path.exists(pred_file):
            raise ValueError(f"Prediction file not found: {pred_file}")
        
        pred_df = pd.read_csv(pred_file)
        
        # Ensure required columns exist
        if not all(col in pred_df.columns for col in ['true', 'pred']):
            raise ValueError(f"Invalid columns in {pred_file}. Expected 'true' and 'pred'.")
        
        # Extract true and predicted values
        y_true = pred_df['true'].values
        y_pred = pred_df['pred'].values
        
        # Replace any NaNs
        y_true = np.nan_to_num(y_true, nan=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Count number of predictions
        num_predictions = len(y_true) // 24  # Assuming 24-hour forecast horizon
        
        return {
            'building_id': building_id,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'smape': metrics['smape'],
            'nrmse': metrics['nrmse'],
            'num_predictions': num_predictions
        }
    
    except Exception as e:
        return {
            'building_id': building_id,
            'error': str(e)
        }

def regenerate_all_metrics(start_building=0, end_building=15, predictions_folder="predictions"):
    """
    Regenerate metrics for all buildings from stored predictions and compute aggregate statistics.
    
    Args:
        start_building (int): First building ID.
        end_building (int): Last building ID.
        predictions_folder (str): Folder containing prediction CSV files.
    
    Returns:
        dict: Contains metrics DataFrame, median/mean metrics, and lists of successful/failed buildings.
    """
    all_metrics = []
    failed_buildings = []
    successful_buildings = []
    
    print(f"Regenerating metrics for buildings {start_building} to {end_building}...")
    print("=" * 60)
    
    for building_id in tqdm(range(start_building, end_building + 1), desc="Processing buildings"):
        result = regenerate_metrics_for_building(building_id, predictions_folder)
        
        if 'error' in result:
            failed_buildings.append({
                'building_id': building_id,
                'error': result['error']
            })
            print(f"Failed building {building_id}: {result['error']}")
        else:
            all_metrics.append(result)
            successful_buildings.append(building_id)
    
    # Check if any metrics were collected
    if not all_metrics:
        print("\nWarning: No buildings were successfully processed. Check predictions folder or building IDs.")
        return {
            'metrics_df': pd.DataFrame(),
            'median_metrics': {},
            'mean_metrics': {},
            'failed_buildings': failed_buildings,
            'successful_buildings': successful_buildings
        }
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Calculate median and mean metrics
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
    
    # Print results
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
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
    
    # Save summary to CSV
    metrics_df.to_csv("regenerated_metrics_summary.csv", index=False)
    print("\nDetailed metrics saved to: regenerated_metrics_summary.csv")
    
    if failed_buildings:
        failed_df = pd.DataFrame(failed_buildings)
        failed_df.to_csv("failed_buildings_regenerated.csv", index=False)
        print(f"Failed buildings saved to: failed_buildings_regenerated.csv")
        print(f"\nFirst 5 failed buildings:")
        print(failed_df.head())
    
    return {
        'metrics_df': metrics_df,
        'median_metrics': median_metrics,
        'mean_metrics': mean_metrics,
        'failed_buildings': failed_buildings,
        'successful_buildings': successful_buildings
    }

def main():
    """Main execution function."""
    results = regenerate_all_metrics(
        start_building=0,
        end_building=1409,  # Limited for testing, adjust as needed
        predictions_folder="predictions"
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