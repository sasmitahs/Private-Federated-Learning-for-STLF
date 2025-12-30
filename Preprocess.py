import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from darts import TimeSeries

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

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

def convert_timeseries_to_numpy(series, input_len, output_len):
    """Convert a Darts TimeSeries to numpy arrays for input and output sequences."""
    values = series.values()
    if len(values) < input_len + output_len:
        return np.array([]), np.array([])

    X, y = [], []
    for i in range(len(values) - input_len - output_len + 1):
        X.append(values[i : i + input_len])
        y.append(values[i + input_len : i + input_len + output_len, 0])  # Predict meter_reading
    return np.array(X), np.array(y)

def create_dataloader(X_ts, X_air_temp, X_primary_use, y, batch_size=32):
    """Create a DataLoader from numpy arrays."""
    # Debug tensor shapes
    print(f"Creating DataLoader with shapes:")
    print(f"X_ts shape: {X_ts.shape}")
    print(f"X_air_temp shape: {X_air_temp.shape}")
    print(f"X_primary_use shape: {X_primary_use.shape}")
    print(f"y shape: {y.shape}")

    dataset = TensorDataset(
        torch.tensor(X_ts, dtype=torch.float32),
        torch.tensor(X_air_temp, dtype=torch.float32),
        torch.tensor(X_primary_use, dtype=torch.int64),  # Expect [batch_size, seq_len]
        torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)