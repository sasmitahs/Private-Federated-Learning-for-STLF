from darts import TimeSeries
from darts.models import ARIMA
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict

# Define DEVICE for TimeSeriesDifficultyWeight
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modified SimpleCNN Model
class SimpleCNN(nn.Module):
    def __init__(self, input_len: int = 168, input_size: int = 1, output_len: int = 24):
        """
        Simple CNN model with two 1D convolutional layers for time-series input.
        
        Args:
            input_len (int): Length of input sequence (e.g., 168 hours)
            input_size (int): Number of input features (channels, e.g., 1 for meter reading)
            output_len (int): Length of output sequence (e.g., 24 hours)
        """
        super(SimpleCNN, self).__init__()
        
        self.input_len = input_len
        self.input_size = input_size
        self.output_len = output_len

        # First 1D Convolutional Layer: input_size -> 32 channels, kernel_size=5, padding=2
        self.conv1 = nn.Conv1d(
            in_channels=input_size,  # Changed to input_size (default 1)
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second 1D Convolutional Layer: 32 -> 32 channels, kernel_size=5, padding=2
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Calculate the output size after convolutions and pooling
        reduced_len = input_len // 4
        self.fc_input_size = 32 * reduced_len

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, output_len)

    def forward(self, x):
        """
        Forward pass for SimpleCNN.
        
        Args:
            x: Input tensor of shape [batch, seq_len, input_size]
        
        Returns:
            Output tensor of shape [batch, output_len]
        """
        x = x.transpose(1, 2)  # [batch, input_size, seq_len]
        x = self.conv1(x)      # [batch, 32, seq_len]
        x = self.relu1(x)
        x = self.pool1(x)      # [batch, 32, seq_len//2]
        x = self.conv2(x)      # [batch, 32, seq_len//2]
        x = self.relu2(x)
        x = self.pool2(x)      # [batch, 32, seq_len//4]
        x = x.view(x.size(0), -1)  # [batch, 32 * (seq_len//4)]
        x = F.relu(self.fc1(x))    # [batch, 128]
        x = self.fc2(x)            # [batch, output_len]
        return x

# Existing Models (unchanged)
class DualCNNGRUPosEncFCNNDecoder(nn.Module):
    def __init__(
        self,
        input_len: int = 168,
        output_len: int = 24,
        ts_input_size: int = 4,
        num_covariates: int = 1,
        num_primary_use: int = 16,
        embed_dim: int = 8,
    ):
        super(DualCNNGRUPosEncFCNNDecoder, self).__init__()

        self.input_len = input_len
        self.output_len = output_len
        self.ts_input_size = ts_input_size
        self.num_covariates = num_covariates
        self.embed_dim = embed_dim

        self.primary_use_embed = nn.Embedding(num_primary_use, embed_dim)

        self.conv1 = nn.Conv1d(in_channels=ts_input_size, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)

        cov_in_channels = num_covariates + embed_dim
        self.cov_conv1 = nn.Conv1d(in_channels=cov_in_channels, out_channels=32, kernel_size=5, padding=2)
        self.cov_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        self.gru_input_size = 128 + 64
        self.gru_hidden = 128
        self.gru = nn.GRU(self.gru_input_size, self.gru_hidden, batch_first=True)

        self.fc1 = nn.Linear(self.gru_hidden, 256)
        self.fc2 = nn.Linear(256, output_len)

    def forward(self, x_ts, x_cov, primary_use):
        x_ts = x_ts.transpose(1, 2)
        x_ts = F.relu(self.conv1(x_ts))
        x_ts = self.pool(x_ts)
        x_ts = F.relu(self.conv2(x_ts))
        x_ts = self.pool(x_ts)
        x_ts = x_ts.transpose(1, 2)

        primary_use_embed = self.primary_use_embed(primary_use)
        cov_input = torch.cat([primary_use_embed, x_cov], dim=-1)
        cov_input = cov_input.transpose(1, 2)
        cov_input = F.relu(self.cov_conv1(cov_input))
        cov_input = self.pool(cov_input)
        cov_input = F.relu(self.cov_conv2(cov_input))
        cov_input = self.pool(cov_input)
        cov_input = cov_input.transpose(1, 2)

        combined = torch.cat([x_ts, cov_input], dim=-1)
        gru_out, _ = self.gru(combined)
        last = gru_out[:, -1, :]
        x = F.relu(self.fc1(last))
        out = self.fc2(x)
        return out

class TrainableHourWeightedMSELoss(nn.Module):
    def __init__(self):
        super(TrainableHourWeightedMSELoss, self).__init__()
        self.hour_weights = nn.Parameter(torch.ones(24))

    def forward(self, predictions, targets, input_batch):
        if input_batch.ndim != 3 or input_batch.shape[2] < 2:
            raise ValueError(f"Expected input shape [B, T, 2], got {input_batch.shape}")
        raw_hours = input_batch[:, -1, 1] * 23.0
        hour_indices = raw_hours.round().long().clamp(0, 23)
        weights = self.hour_weights[hour_indices]
        mse = F.mse_loss(predictions, targets, reduction='none')
        weighted_mse = mse * weights.unsqueeze(-1)
        return weighted_mse.mean()

class ExponentialWeightedMSELoss(nn.Module):
    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        squared_error = (predictions - targets) ** 2
        forecast_len = predictions.shape[1]
        device = predictions.device
        weights = self.k ** torch.arange(forecast_len, device=device, dtype=predictions.dtype)
        weights = weights.view(1, forecast_len)
        weighted_squared_error = squared_error * weights
        return weighted_squared_error.mean()

class SimpleANN(nn.Module):
    def __init__(self, input_len=168, input_dim=2, output_len=24, hidden_dim=64):
        super(SimpleANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_len * input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_len)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_arima_model(train_series):
    model = ARIMA(5, 1, 0)
    model.fit(train_series)
    return model

def train_model(model, dataloader, epochs=50, learning_rate=0.001, loss_fn=None, optimizer_class=optim.Adam):
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    loss_history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in tqdm(dataloader):
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch.squeeze(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    return model, loss_history

class TimeSeriesDifficultyWeight:
    def __init__(self, num_clients, accumulate_iters=20):
        self.num_clients = num_clients
        self.last_loss = torch.ones(num_clients).float().to(DEVICE)
        self.learn_score = torch.zeros(num_clients).float().to(DEVICE)
        self.unlearn_score = torch.zeros(num_clients).float().to(DEVICE)
        self.ema_difficulty = torch.ones(num_clients).float().to(DEVICE)
        self.accumulate_iters = accumulate_iters

    def update(self, cid: int, loss_history: List[float]) -> float:
        current_loss = torch.tensor(loss_history[-1], dtype=torch.float32).to(DEVICE)
        previous_loss = self.last_loss[cid]
        delta = current_loss - previous_loss
        ratio = torch.log((current_loss + 1e-8) / (previous_loss + 1e-8))
        learn = torch.where(delta < 0, -delta * ratio, torch.tensor(0.0, device=current_loss.device))
        unlearn = torch.where(delta >= 0, delta * ratio, torch.tensor(0.0, device=current_loss.device))
        momentum = (self.accumulate_iters - 1) / self.accumulate_iters
        self.learn_score[cid] = momentum * self.learn_score[cid] + (1 - momentum) * learn
        self.unlearn_score[cid] = momentum * self.unlearn_score[cid] + (1 - momentum) * unlearn
        diff_ratio = (self.unlearn_score[cid] + 1e-8) / (self.learn_score[cid] + 1e-8)
        difficulty = diff_ratio
        self.ema_difficulty[cid] = momentum * self.ema_difficulty[cid] + (1 - momentum) * difficulty
        self.last_loss[cid] = current_loss
        return self.ema_difficulty[cid].item()

    def get_normalized_weights(self, client_ids: List[int]) -> List[float]:
        weights = [self.ema_difficulty[cid].item() for cid in client_ids]
        total = sum(weights)
        if total == 0:
            return [1.0 / len(client_ids)] * len(client_ids)
        return [w / total for w in weights]

def moving_average_forecast(series, input_window: int, output_window: int) -> np.ndarray:
    history = series[-input_window:].values().flatten()
    avg = np.mean(history)
    return np.full(shape=(output_window,), fill_value=avg)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class BinaryClassificationANN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[18, 8], dropout=0.1):
        super(BinaryClassificationANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dims[1], 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x

class VarEncoderDecoder(nn.Module):
    def __init__(self, hidden_layers, hidden_size, latent_dim=32, seq_length=512):
        super().__init__()
        hidden_sizes = [seq_length]
        for i in range(hidden_layers):
            hidden_sizes.append(hidden_size)
            hidden_size //= 2
        self.encoder = nn.ModuleList()
        for i in range(1, len(hidden_sizes)):
            linear = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
            activation = nn.ReLU()
            norm = nn.BatchNorm1d(hidden_sizes[i])
            self.encoder.append(linear)
            self.encoder.append(norm)
            self.encoder.append(activation)
        self.encoder = nn.Sequential(*self.encoder)
        self.enc_fc_mu = nn.Linear(hidden_sizes[-1], latent_dim)
        self.enc_fc_var = nn.Linear(hidden_sizes[-1], latent_dim)
        self.decoder_in = nn.Linear(latent_dim, hidden_sizes[-1])
        self.decoder = nn.ModuleList()
        hidden_sizes = list(reversed(hidden_sizes))
        for i in range(1, len(hidden_sizes)):
            linear = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
            activation = nn.LeakyReLU()
            norm = nn.BatchNorm1d(hidden_sizes[i])
            self.decoder.append(linear)
            self.decoder.append(norm)
            self.decoder.append(activation)
        self.decoder = nn.Sequential(*self.decoder)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, X):
        X = self.encoder(X)
        mu = self.enc_fc_mu(X)
        log_var = self.enc_fc_var(X)
        X = self.reparameterize(mu, log_var)
        X = self.decoder_in(X)
        X = self.decoder(X)
        return X

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out

class MoELSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts, ffn_hidden_size, output_size):
        super(MoELSTM, self).__init__()
        self.lower_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.experts = nn.ModuleList([nn.Linear(hidden_size, ffn_hidden_size) for _ in range(num_experts)])
        self.expert_activation = nn.ReLU()
        self.gate = nn.Linear(hidden_size, num_experts)
        self.gate_activation = nn.Softmax(dim=-1)
        self.upper_lstm = nn.LSTM(input_size=ffn_hidden_size, hidden_size=hidden_size, batch_first=True)
        self.final_ffn = nn.Linear(hidden_size, ffn_hidden_size)
        self.final_activation = nn.ReLU()
        self.output_layer = nn.Linear(ffn_hidden_size, output_size)
        self.num_experts = num_experts

    def forward(self, x):
        lower_lstm_out, _ = self.lower_lstm(x)
        expert_outputs = [self.expert_activation(expert(lower_lstm_out)) for expert in self.experts]
        gate_logits = self.gate(lower_lstm_out)
        gate_weights = self.gate_activation(gate_logits)
        batch_size, seq_len, _ = lower_lstm_out.shape
        device = lower_lstm_out.device
        weighted_expert_outputs = torch.zeros(batch_size, seq_len, expert_outputs[0].shape[-1], device=device)
        for i, expert_out in enumerate(expert_outputs):
            expert_weight = gate_weights[:, :, i:i+1]
            weighted_expert_outputs += expert_out * expert_weight
        upper_lstm_out, _ = self.upper_lstm(weighted_expert_outputs)
        last_output = upper_lstm_out[:, -1, :]
        ffn_out = self.final_ffn(last_output)
        ffn_out = self.final_activation(ffn_out)
        output = self.output_layer(ffn_out)
        return output

class CNNLSTMForecast(nn.Module):
    def __init__(self, input_len=168, forecast_len=24, conv_channels=32, lstm_hidden=64):
        super(CNNLSTMForecast, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, forecast_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class MoEGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts, ffn_hidden_size, output_size):
        super(MoEGRU, self).__init__()
        self.lower_gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.experts = nn.ModuleList([nn.Linear(hidden_size, ffn_hidden_size) for _ in range(num_experts)])
        self.expert_activation = nn.ReLU()
        self.gate = nn.Linear(hidden_size, num_experts)
        self.gate_activation = nn.Softmax(dim=-1)
        self.upper_gru = nn.GRU(input_size=ffn_hidden_size, hidden_size=hidden_size, batch_first=True)
        self.final_ffn = nn.Linear(hidden_size, ffn_hidden_size)
        self.final_activation = nn.ReLU()
        self.output_layer = nn.Linear(ffn_hidden_size, output_size)
        self.num_experts = num_experts

    def forward(self, x):
        lower_gru_out, _ = self.lower_gru(x)
        expert_outputs = [self.expert_activation(expert(lower_gru_out)) for expert in self.experts]
        gate_logits = self.gate(lower_gru_out)
        gate_weights = self.gate_activation(gate_logits)
        batch_size, seq_len, _ = lower_gru_out.shape
        device = x.device
        weighted_output = torch.zeros(batch_size, seq_len, expert_outputs[0].shape[-1], device=device)
        for i, expert_out in enumerate(expert_outputs):
            weight = gate_weights[:, :, i:i+1]
            weighted_output += expert_out * weight
        upper_gru_out, _ = self.upper_gru(weighted_output)
        last_output = upper_gru_out[:, -1, :]
        ffn_out = self.final_activation(self.final_ffn(last_output))
        return self.output_layer(ffn_out)

class CNNGRUForecast(nn.Module):
    def __init__(self, input_size=1, input_len=168, forecast_len=24, conv_channels=32, gru_hidden=64):
        super(CNNGRUForecast, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gru = nn.GRU(input_size=conv_channels, hidden_size=gru_hidden, batch_first=True)
        self.fc = nn.Linear(gru_hidden, forecast_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 168,
        forecast_horizon: int = 24,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.input_embedding = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=context_length + forecast_horizon)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.forecast_tokens = nn.Parameter(torch.randn(forecast_horizon, 1, d_model))

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = src.size(0)
        src_embedded = self.input_embedding(src)
        src_embedded = self.pos_encoding(src_embedded)
        forecast_tokens = self.forecast_tokens.expand(batch_size, -1, -1)
        full_sequence = torch.cat([src_embedded, forecast_tokens], dim=1)
        seq_len = full_sequence.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(src.device)
        transformer_output = self.transformer_encoder(full_sequence, mask=causal_mask)
        forecast_output = transformer_output[:, -self.forecast_horizon:, :]
        forecast_output = self.dropout(forecast_output)
        predictions = self.output_projection(forecast_output)
        return predictions

class PositionalEncoding2(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer2(nn.Module):
    def __init__(
        self,
        context_length: int = 168,
        forecast_horizon: int = 24,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.input_embedding = nn.Linear(1, d_model)
        self.forecast_tokens = nn.Parameter(torch.randn(1, forecast_horizon, d_model))
        self.pos_encoding = PositionalEncoding2(d_model, max_len=context_length + forecast_horizon)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.input_embedding(x)
        forecast_tokens = self.forecast_tokens.expand(batch_size, -1, -1)
        full_input = torch.cat([x, forecast_tokens], dim=1)
        full_input = self.pos_encoding(full_input)
        seq_len = full_input.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        transformer_output = self.encoder(full_input, mask=causal_mask)
        forecast_output = transformer_output[:, -self.forecast_horizon:, :]
        output = self.output_projection(self.dropout(forecast_output))
        return output

class CNNGRUNoCov(nn.Module):
    def __init__(
        self,
        input_len: int = 168,
        output_len: int = 24,
        input_size: int = 1,
    ):
        super(CNNGRUNoCov, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.input_size = input_size
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gru_input_size = 128
        self.gru_hidden = 128
        self.gru = nn.GRU(self.gru_input_size, self.gru_hidden, batch_first=True)
        self.fc1 = nn.Linear(self.gru_hidden, 256)
        self.fc2 = nn.Linear(256, output_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        last = gru_out[:, -1, :]
        out = F.relu(self.fc1(last))
        out = self.fc2(out)
        return out

# Updated model_fn to include SimpleCNN with dynamic input_size
def model_fn(model_name: str):
    if model_name.endswith("_hour"):
        input_size = 2  # meter_reading + hour
        base_name = model_name.replace("_hour", "")
    else:
        input_size = 1
        base_name = model_name

    match base_name:
        case "simple_cnn":
            return SimpleCNN(input_len=168, input_size=input_size, output_len=24)  # Use dynamic input_size
        case "moe_gru":
            return MoEGRU(
                input_size=input_size,
                hidden_size=64,
                output_size=8,
                num_experts=5,
                ffn_hidden_size=32,
            )
        case "moe_lstm":
            return MoELSTM(
                input_size=input_size,
                hidden_size=64,
                output_size=24,
                num_experts=6,
                ffn_hidden_size=32,
            )
        case "lstm":
            return LSTMModel(
                input_size=input_size,
                hidden_size=82,
                output_size=24
            )
        case "gru":
            return GRUModel(
                input_size=input_size,
                hidden_size=82,
                output_size=24
            )
        case "simple_ann":
            return SimpleANN(input_len=168, input_dim=input_size, output_len=24)
        case "anomaly_ann":
            return BinaryClassificationANN(input_dim=36)
        case "vae":
            return VarEncoderDecoder(input_size)
        case "cnn_lstm":
            return CNNLSTMForecast()
        case "transformer":
            return TimeSeriesTransformer2()
        case "cnn_gru":
            return CNNGRUForecast(input_size=input_size)
        case "cnn_gru_no_cov":
            return CNNGRUNoCov(input_size=input_size)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")