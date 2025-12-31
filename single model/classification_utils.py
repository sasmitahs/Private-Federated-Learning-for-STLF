"""Energy-TS-Diffusion: Training, evaluation, and data loading utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from Models import MoELSTM
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import random
from Models import MoELSTM, LSTMModel

from Preprocess import (
    compute_metrics,
    # convert_timeseries_to_numpy,
    create_dataloader,
    load_building_series,
    split_series_list,
)
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader



import torch.nn.functional as F


def convert_timeseries_to_numpy(ts, input_len=24, output_len=8):
    X, y = [], []
    for i in range(len(ts) - input_len - output_len):
        input_chunk = ts[i : i + input_len].values()  # [24, 2]
        target_chunk = ts[i + input_len : i + input_len + output_len].values()  # [8, 2]
        
        # Use only meter_reading (index 0) as target
        y.append(target_chunk[:, 0])  # [8]
        X.append(input_chunk)        # [24, 2]

    return np.array(X), np.array(y)



def load_energy_data_feather(cid, filepath="meter_0_data_cleaned.feather"):
    """Load, preprocess, and return train/test dataloaders for a client."""
    df = pd.read_feather(filepath)
    df = df[df['building_id'] == cid]
    df['meter_reading'] = df['meter_reading'].fillna(0)

    if df.empty:
        raise ValueError(f"No data found for building_id {cid}")

    try:
        ts = TimeSeries.from_dataframe(
            df,
            time_col='timestamp',
            value_cols='meter_reading',
            fill_missing_dates=True,
            freq='h'
        )
    except Exception as e:
        raise ValueError(f"Failed to construct TimeSeries: {e}")

    train_series, test_series = ts.split_before(0.75)

    if len(train_series) == 0 or len(test_series) == 0:
        raise ValueError(f"Empty time series for building_id {cid}. Train: {len(train_series)}, Test: {len(test_series)}")

    scaler = MinMaxScaler(feature_range=(0.1, 1))
    transformer = Scaler(scaler)
    transformed_train_series = transformer.fit_transform(train_series)
    transformed_test_series = transformer.transform(test_series)

    X_train, y_train = convert_timeseries_to_numpy(transformed_train_series, input_len=24, output_len=8)
    X_test, y_test = convert_timeseries_to_numpy(transformed_test_series, input_len=24, output_len=8)

    X_train = np.nan_to_num(X_train, nan=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0)


    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(f"Client {cid} has no data after preprocessing.")

    train_loader = create_dataloader(X_train, y_train, batch_size=1024)
    test_loader = create_dataloader(X_test, y_test, batch_size=256)

    return train_loader, test_loader


def load_anomaly_dataloader(
    feather_path: str,
    building_id: int,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """
    Load a PyTorch DataLoader from a feather file filtered by building_id.

    Args:
        feather_path (str): Path to the feather file.
        building_id (int): Building ID to filter data.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: PyTorch DataLoader with features and labels.
    """
    # Read feather file
    df = pd.read_feather(feather_path)

    # Filter by building_id
    df = df[df["building_id"] == building_id]

    # Drop non-feature columns (assumes 'anomaly' and 'building_id' are not features)
    feature_cols = [col for col in df.columns if col not in ["anomaly", "building_id"]]
    features = df[feature_cols].values.astype("float32")
    labels = df["anomaly"].values.astype("float32")  # or long if classification

    # Convert to PyTorch tensors
    X = torch.tensor(features)
    y = torch.tensor(labels)

    # Wrap in TensorDataset and DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def get_class_weights(y_tensor):
    """
    Compute weights for the positive and negative class.
    Args:
        y_tensor (torch.Tensor): Tensor of shape (N,) or (N, 1) with binary labels (0 or 1).
    Returns:
        pos_weight: A scalar tensor to use with BCEWithLogitsLoss(pos_weight=...)
    """
    y_tensor = y_tensor.view(-1)
    num_pos = (y_tensor == 1).sum().float()
    num_neg = (y_tensor == 0).sum().float()
    
    # Avoid divide-by-zero
    if num_pos == 0 or num_neg == 0:
        return torch.tensor(1.0)
    
    pos_weight = num_neg / num_pos
    return pos_weight

def train_model_binary(
    model,
    train_loader,
    device=None,
    learning_rate=0.001,
    loss_fn=None,
    optimizer_class=optim.Adam,
    epochs=50
):
    """Train a binary classification model and return the average loss per epoch."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    y_all = []
    for _, y_batch in train_loader:
        y_all.append(y_batch)
    y_all = torch.cat(y_all)
    pos_weight = get_class_weights(y_all).to(device)*6

    loss_fn = loss_fn or nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Binary classification loss
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    loss_history = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()  # Ensure float dtype

            # Handle label shape (should be [batch_size, 1])
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)
            elif y_batch.dim() == 2 and y_batch.shape[1] != 1:
                raise ValueError("Labels should be shape [batch_size] or [batch_size, 1]")

            optimizer.zero_grad()
            output = model(X_batch)  # Should output logits (no sigmoid)

            if output.shape != y_batch.shape:
                output = output.view_as(y_batch)

            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        loss_history.append(epoch_loss / len(train_loader))

    return get_weights(model), loss_history




def evaluate_model(model, dataloader, device):
    """Evaluate the model and return MSE and RMSE."""
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
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

# def get_model():
    # return

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
        # All weights are nearly equal; return uniform
        return [1.0 for _ in weights]
    return [(w - min_w) / (max_w - min_w) for w in weights]

def softmax(weights):
    weights = np.array(weights)
    max_w = np.max(weights)  # for numerical stability
    exp_weights = np.exp(weights - max_w)
    return (exp_weights / (np.sum(exp_weights) + 1e-8)).tolist()