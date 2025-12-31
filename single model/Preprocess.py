import os
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from darts import TimeSeries
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import torch


def compute_metrics(y_true, y_pred):
    """
    Compute MAE, MSE, and RMSE between true and predicted values.
    
    Parameters:
    - y_true (np.ndarray): True values
    - y_pred (np.ndarray): Predicted values
    
    Returns:
    - dict: {'MAE': ..., 'MSE': ..., 'RMSE': ...}
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def load_building_series(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    series_list = []

    for file in all_files:
        df = pd.read_csv(file, parse_dates=['timestamp'])
        df = df.sort_values('timestamp')
        ts = TimeSeries.from_dataframe(df, 'timestamp', 'kWh')
        series_list.append(ts)

    return series_list


def split_series_list(series_list, train_ratio=0.75):
    train_series = []
    test_series = []
    for ts in series_list:
        train, test = ts.split_before(train_ratio)
        train_series.append(train)
        test_series.append(test)
    return train_series, test_series


from typing import List, Tuple, Optional

def convert_timeseries_to_numpy(
    series: TimeSeries,
    input_len: int,
    output_len: int,
    drop_incomplete: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a single Darts TimeSeries into input-output pairs for supervised learning.

    Args:
        series (TimeSeries): The input time series.
        input_len (int): Length of input window.
        output_len (int): Length of output window.
        drop_incomplete (bool): If True, drops windows that can't fit input+output.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays (X, y) where:
            - X shape: (num_samples, input_len, num_features)
            - y shape: (num_samples, output_len, num_features)
    """
    values = series.values()  # shape: (T, D)
    T = len(values)
    max_i = T - input_len - output_len + 1

    if drop_incomplete and max_i <= 0:
        return np.empty((0, input_len, values.shape[1])), np.empty((0, output_len, values.shape[1]))

    X_all, y_all = [], []

    for i in range(max_i):
        x_i = values[i : i + input_len]
        y_i = values[i + input_len : i + input_len + output_len]
        X_all.append(x_i)
        y_all.append(y_i)

    return np.array(X_all), np.array(y_all)


def create_dataloader(X, y, batch_size=32):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
