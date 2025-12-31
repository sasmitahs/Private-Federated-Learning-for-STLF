import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from statsmodels.tsa.seasonal import STL
from Preprocess import compute_metrics, convert_timeseries_to_numpy, create_dataloader

def load_energy_data_feather(cid, filepath="train_final.feather", input_len=168, output_len=24):
    """Load, preprocess, and return train/test dataloaders for a client."""
    df = pd.read_feather(filepath)
    df = df[df['building_id'] == cid]
    df['meter_reading'] = df['meter_reading'].fillna(0)
    df['air_temperature'] = df['air_temperature'].fillna(df['air_temperature'].mean())

    if df.empty:
        raise ValueError(f"No data found for building_id {cid}")

    # Convert primary_use to integer indices
    primary_use_categories = df['primary_use'].unique()
    primary_use_map = {cat: idx for idx, cat in enumerate(primary_use_categories)}
    df['primary_use_idx'] = df['primary_use'].map(primary_use_map)

    # Construct time series for meter_reading, air_temperature, and primary_use
    try:
        ts = TimeSeries.from_dataframe(
            df,
            time_col='timestamp',
            value_cols=['meter_reading', 'air_temperature'],
            fill_missing_dates=True,
            freq='h'
        )
        primary_use_ts = TimeSeries.from_dataframe(
            df,
            time_col='timestamp',
            value_cols=['primary_use_idx'],
            fill_missing_dates=True,
            freq='h'
        )
    except Exception as e:
        raise ValueError(f"Failed to construct TimeSeries: {e}")

    # Ensure consistent time index
    time_index = ts.time_index
    primary_use_ts = primary_use_ts.slice(time_index[0], time_index[-1])
    if len(primary_use_ts) != len(ts):
        raise ValueError(f"Primary use time series length {len(primary_use_ts)} does not match ts length {len(ts)}")

    # Ensure meter_reading has no NaN values for STL
    meter_reading_values = ts['meter_reading'].values().flatten()  # Flatten to 1D
    meter_reading_values = np.nan_to_num(meter_reading_values, nan=0.0)

    # Apply STL decomposition to meter_reading
    stl = STL(meter_reading_values, period=24)
    result = stl.fit()
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    # Align STL components to time_index length
    target_length = len(time_index)
    trend = trend[:target_length]
    seasonal = seasonal[:target_length]
    residual = residual[:target_length]

    # Pad if necessary (should be rare due to fill_missing_dates)
    if len(trend) < target_length:
        trend = np.pad(trend, (0, target_length - len(trend)), mode='constant', constant_values=0)
    if len(seasonal) < target_length:
        seasonal = np.pad(seasonal, (0, target_length - len(seasonal)), mode='constant', constant_values=0)
    if len(residual) < target_length:
        residual = np.pad(residual, (0, target_length - len(residual)), mode='constant', constant_values=0)

    # Debug shapes
    print(f"Client {cid}:")
    print(f"meter_reading_values shape: {meter_reading_values.shape}")
    print(f"trend shape: {trend.shape}")
    print(f"seasonal shape: {seasonal.shape}")
    print(f"residual shape: {residual.shape}")

    # Combine meter_reading and decomposition components into a DataFrame
    stacked_values = np.stack([
        meter_reading_values,
        trend,
        seasonal,
        residual
    ], axis=-1)
    ts_df = pd.DataFrame(
        stacked_values,
        columns=['meter_reading', 'trend', 'seasonal', 'residual'],
        index=time_index
    )
    ts_combined = TimeSeries.from_dataframe(ts_df)

    # Verify length
    if len(ts_combined) != len(time_index):
        raise ValueError(f"ts_combined length {len(ts_combined)} does not match time_index length {len(time_index)}")

    # Split into train and test
    train_series, test_series = ts_combined.split_before(0.75)
    # Align train_air_temp and train_primary_use with train_series time index
    train_air_temp = ts['air_temperature'].slice_intersect(train_series)
    train_primary_use = primary_use_ts.slice_intersect(train_series)

    if len(train_series) < input_len + output_len:
        raise ValueError(f"Insufficient data for building_id {cid}. Train length: {len(train_series)}")

    # Debug lengths and shapes before scaling
    print(f"train_series length: {len(train_series)}, values shape: {train_series.values().shape}")
    print(f"train_air_temp length: {len(train_air_temp)}, values shape: {train_air_temp.values().shape}")
    print(f"train_primary_use length: {len(train_primary_use)}, values shape: {train_primary_use.values().shape}")

    # Scale time series and air_temperature with separate scalers
    series_scaler = MinMaxScaler(feature_range=(0.1, 1))
    air_temp_scaler = MinMaxScaler(feature_range=(0.1, 1))
    series_transformer = Scaler(series_scaler)
    air_temp_transformer = Scaler(air_temp_scaler)
    transformed_train_series = series_transformer.fit_transform(train_series)
    transformed_train_air_temp = air_temp_transformer.fit_transform(train_air_temp)

    X_train_ts, y_train = convert_timeseries_to_numpy(transformed_train_series, input_len, output_len)
    X_train_air_temp, _ = convert_timeseries_to_numpy(transformed_train_air_temp, input_len, output_len)
    X_train_primary_use, _ = convert_timeseries_to_numpy(train_primary_use, input_len, output_len)

    # Squeeze X_primary_use to remove extra dimension
    X_train_primary_use = X_train_primary_use.squeeze(-1)  # [num_samples, seq_len]

    # Debug tensor shapes
    print(f"X_train_ts shape: {X_train_ts.shape}")
    print(f"X_train_air_temp shape: {X_train_air_temp.shape}")
    print(f"X_train_primary_use shape: {X_train_primary_use.shape}")
    print(f"y_train shape: {y_train.shape}")

    X_train_ts = np.nan_to_num(X_train_ts, nan=0.0)
    X_train_air_temp = np.nan_to_num(X_train_air_temp, nan=0.0)
    X_train_primary_use = np.nan_to_num(X_train_primary_use, nan=0.0).astype(np.int64)
    y_train = np.nan_to_num(y_train, nan=0.0)

    if len(X_train_ts) == 0:
        raise ValueError(f"Client {cid} has no data after preprocessing.")

    train_loader = create_dataloader(
        X_train_ts, X_train_air_temp, X_train_primary_use, y_train, batch_size=1024
    )

    return train_loader, None

def load_energy_data_hour_feather(cid, filepath="train_final.feather", input_len=168, output_len=24):
    """Load time series with hour feature for a given building ID."""
    df = pd.read_feather(filepath)
    df = df[df['building_id'] == cid]
    df['meter_reading'] = df['meter_reading'].fillna(0)
    df['air_temperature'] = df['air_temperature'].fillna(df['air_temperature'].mean())

    if df.empty:
        raise ValueError(f"No data found for building_id {cid}")

    # Convert timestamp and extract hour
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour.astype(float) / 23.0

    # Convert primary_use to integer indices
    primary_use_categories = df['primary_use'].unique()
    primary_use_map = {cat: idx for idx, cat in enumerate(primary_use_categories)}
    df['primary_use_idx'] = df['primary_use'].map(primary_use_map)

    # Construct time series
    try:
        ts = TimeSeries.from_dataframe(
            df,
            time_col='timestamp',
            value_cols=['meter_reading', 'hour', 'air_temperature'],
            fill_missing_dates=True,
            freq='h'
        )
        primary_use_ts = TimeSeries.from_dataframe(
            df,
            time_col='timestamp',
            value_cols=['primary_use_idx'],
            fill_missing_dates=True,
            freq='h'
        )
    except Exception as e:
        raise ValueError(f"Failed to construct TimeSeries: {e}")

    # Ensure consistent time index
    time_index = ts.time_index
    primary_use_ts = primary_use_ts.slice(time_index[0], time_index[-1])
    if len(primary_use_ts) != len(ts):
        raise ValueError(f"Primary use time series length {len(primary_use_ts)} does not match ts length {len(ts)}")

    # Ensure meter_reading has no NaN values for STL
    meter_reading_values = ts['meter_reading'].values().flatten()  # Flatten to 1D
    meter_reading_values = np.nan_to_num(meter_reading_values, nan=0.0)

    # Apply STL decomposition to meter_reading
    stl = STL(meter_reading_values, period=24)
    result = stl.fit()
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    # Align STL components to time_index length
    target_length = len(time_index)
    trend = trend[:target_length]
    seasonal = seasonal[:target_length]
    residual = residual[:target_length]

    # Pad if necessary
    if len(trend) < target_length:
        trend = np.pad(trend, (0, target_length - len(trend)), mode='constant', constant_values=0)
    if len(seasonal) < target_length:
        seasonal = np.pad(seasonal, (0, target_length - len(seasonal)), mode='constant', constant_values=0)
    if len(residual) < target_length:
        residual = np.pad(residual, (0, target_length - len(residual)), mode='constant', constant_values=0)

    # Debug shapes
    print(f"Client {cid}:")
    print(f"meter_reading_values shape: {meter_reading_values.shape}")
    print(f"trend shape: {trend.shape}")
    print(f"seasonal shape: {seasonal.shape}")
    print(f"residual shape: {residual.shape}")
    print(f"hour shape: {ts['hour'].values().flatten().shape}")

    # Combine meter_reading, decomposition components, and hour into a DataFrame
    stacked_values = np.stack([
        meter_reading_values,
        trend,
        seasonal,
        residual,
        ts['hour'].values().flatten()  # Flatten to 1D
    ], axis=-1)
    ts_df = pd.DataFrame(
        stacked_values,
        columns=['meter_reading', 'trend', 'seasonal', 'residual', 'hour'],
        index=time_index
    )
    ts_combined = TimeSeries.from_dataframe(ts_df)

    # Verify length
    if len(ts_combined) != len(time_index):
        raise ValueError(f"ts_combined length {len(ts_combined)} does not match time_index length {len(time_index)}")

    # Split into train and test
    train_series, test_series = ts_combined.split_before(0.75)
    # Align train_air_temp and train_primary_use with train_series time index
    train_air_temp = ts['air_temperature'].slice_intersect(train_series)
    train_primary_use = primary_use_ts.slice_intersect(train_series)

    if len(train_series) < input_len + output_len:
        raise ValueError(f"Insufficient data for building_id {cid}. Train length: {len(train_series)}")

    # Debug lengths and shapes before scaling
    print(f"train_series length: {len(train_series)}, values shape: {train_series.values().shape}")
    print(f"train_air_temp length: {len(train_air_temp)}, values shape: {train_air_temp.values().shape}")
    print(f"train_primary_use length: {len(train_primary_use)}, values shape: {train_primary_use.values().shape}")

    # Scale time series and air_temperature with separate scalers
    series_scaler = MinMaxScaler(feature_range=(0.1, 1))
    air_temp_scaler = MinMaxScaler(feature_range=(0.1, 1))
    series_transformer = Scaler(series_scaler)
    air_temp_transformer = Scaler(air_temp_scaler)
    transformed_train_series = series_transformer.fit_transform(train_series)
    transformed_train_air_temp = air_temp_transformer.fit_transform(train_air_temp)

    X_train_ts, y_train = convert_timeseries_to_numpy(transformed_train_series, input_len, output_len)
    X_train_air_temp, _ = convert_timeseries_to_numpy(transformed_train_air_temp, input_len, output_len)
    X_train_primary_use, _ = convert_timeseries_to_numpy(train_primary_use, input_len, output_len)

    # Squeeze X_primary_use to remove extra dimension
    X_train_primary_use = X_train_primary_use.squeeze(-1)  # [num_samples, seq_len]

    # Debug tensor shapes
    print(f"X_train_ts shape: {X_train_ts.shape}")
    print(f"X_train_air_temp shape: {X_train_air_temp.shape}")
    print(f"X_train_primary_use shape: {X_train_primary_use.shape}")
    print(f"y_train shape: {y_train.shape}")

    X_train_ts = np.nan_to_num(X_train_ts, nan=0.0)
    X_train_air_temp = np.nan_to_num(X_train_air_temp, nan=0.0)
    X_train_primary_use = np.nan_to_num(X_train_primary_use, nan=0.0).astype(np.int64)
    y_train = np.nan_to_num(y_train, nan=0.0)

    if len(X_train_ts) == 0:
        raise ValueError(f"Client {cid} has no data after preprocessing.")

    train_loader = create_dataloader(
        X_train_ts, X_train_air_temp, X_train_primary_use, y_train, batch_size=1024
    )

    return train_loader, None

def train_model(model, train_loader, device=None, learning_rate=0.001, loss_fn=None, optimizer_class=optim.Adam, epochs=50):
    """Train the model and return the average loss."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    loss_fn = loss_fn or nn.MSELoss()
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    loss_history = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            if len(batch) == 4:
                X_ts, X_air_temp, X_primary_use, y_batch = batch
                X_ts, X_air_temp, X_primary_use, y_batch = (
                    X_ts.to(device),
                    X_air_temp.to(device),
                    X_primary_use.to(device),
                    y_batch.to(device)
                )
                optimizer.zero_grad()
                output = model(X_ts, X_air_temp, X_primary_use)
                loss = loss_fn(output, y_batch)
            else:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if y_batch.dim() == 3 and y_batch.shape[-1] == 1:
                    y_batch = y_batch.squeeze(-1)
                optimizer.zero_grad()
                output = model(X_batch)
                try:
                    loss = loss_fn(output, y_batch, X_batch)
                except TypeError:
                    loss = loss_fn(output, y_batch)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(train_loader))

    return get_weights(model), loss_history

def train_model_hour(model, train_loader, device=None, learning_rate=0.001, loss_fn=None, optimizer_class=optim.Adam, epochs=50):
    """Train the model with hour feature and return the average loss."""
    return train_model(model, train_loader, device, learning_rate, loss_fn, optimizer_class, epochs)

def train_model_transformer(model, train_loader, device=None, learning_rate=0.001, loss_fn=None, optimizer_class=optim.Adam, epochs=50):
    """Train the transformer model and return the average loss."""
    return train_model(model, train_loader, device, learning_rate, loss_fn, optimizer_class, epochs)

def evaluate_model(model, dataloader, device):
    """Evaluate the model and return MSE and RMSE."""
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                X_ts, X_air_temp, X_primary_use, y_batch = batch
                X_ts, X_air_temp, X_primary_use, y_batch = (
                    X_ts.to(device),
                    X_air_temp.to(device),
                    X_primary_use.to(device),
                    y_batch.to(device)
                )
                output = model(X_ts, X_air_temp, X_primary_use)
            else:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if y_batch.dim() == 3 and y_batch.shape[-1] == 1:
                    y_batch = y_batch.squeeze(-1)
                output = model(X_batch)

            preds.append(output.cpu().numpy())
            trues.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return get_weights(model), mse, rmse

def get_weights(model):
    return [p.detach().cpu().numpy() for p in model.parameters()]

def set_weights(model, weights):
    params = list(model.parameters())
    if len(weights) != len(params):
        raise ValueError(f"Mismatch in weights ({len(weights)}) and model parameters ({len(params)})")

    for p, w in zip(params, weights):
        p.data = torch.tensor(w, dtype=p.dtype, device=p.device)

def normalize_to_unit_range(weights):
    min_w = min(weights)
    max_w = max(weights)
    if max_w - min_w < 1e-8:
        return [1.0 for _ in weights]
    return [(w - min_w) / (max_w - min_w) for w in weights]

def softmax(weights):
    weights = np.array(weights)
    max_w = np.max(weights)
    exp_weights = np.exp(weights - max_w)
    return (exp_weights / (np.sum(exp_weights) + 1e-8)).tolist()