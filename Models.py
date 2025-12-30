import math
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from darts import TimeSeries
from darts.models import ARIMA

# Define DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Existing classes (unchanged)
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
    def __init__(self, input_len=24, input_dim=2, output_len=8, hidden_dim=64):
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

# New Model 1: DualSimpleANNPosEncFCNNDecoder
class DualSimpleANNPosEncFCNNDecoder(nn.Module):
    def __init__(
        self,
        input_len: int = 168,
        output_len: int = 24,
        ts_input_size: int = 4,       # meter_reading + decomposition (3)
        num_covariates: int = 1,      # e.g., air_temperature
        num_primary_use: int = 16,    # number of primary_use categories
        embed_dim: int = 16,          # embedding dimension for primary_use
        hidden_dim: int = 64,         # hidden dimension for SimpleANN
        dropout: float = 0.2
    ):
        super(DualSimpleANNPosEncFCNNDecoder, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.ts_input_size = ts_input_size
        self.num_covariates = num_covariates
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Embedding for primary_use
        self.primary_use_embed = nn.Embedding(num_primary_use, embed_dim)

        # Project time-series input to hidden_dim for positional encoding
        self.ts_input_proj = nn.Linear(ts_input_size, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding2(hidden_dim, max_len=input_len)

        # SimpleANN for time-series branch
        self.ts_ann = SimpleANN(
            input_len=input_len,
            input_dim=hidden_dim,  # After projection + positional encoding
            output_len=hidden_dim,  # Output a hidden representation
            hidden_dim=hidden_dim
        )

        # SimpleANN for covariate branch
        self.cov_ann = SimpleANN(
            input_len=input_len,
            input_dim=num_covariates + embed_dim,  # primary_use embedding + covariates
            output_len=hidden_dim,  # Output a hidden representation
            hidden_dim=hidden_dim
        )

        # FCNN Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenated ts_ann + cov_ann outputs
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_len)
        )

    def forward(self, x_ts, x_cov, primary_use):
        """
        x_ts: [batch, seq_len, ts_input_size]
        x_cov: [batch, seq_len, num_covariates]
        primary_use: [batch, seq_len] (long/int indices)
        """
        # Process time-series input
        ts_proj = self.ts_input_proj(x_ts)  # [batch, seq_len, hidden_dim]
        ts_input = self.pos_encoding(ts_proj)  # [batch, seq_len, hidden_dim]
        ts_out = self.ts_ann(ts_input)  # [batch, hidden_dim]

        # Process covariates
        primary_use_embed = self.primary_use_embed(primary_use)  # [batch, seq_len, embed_dim]
        cov_input = torch.cat([primary_use_embed, x_cov], dim=-1)  # [batch, seq_len, embed_dim + num_covariates]
        cov_out = self.cov_ann(cov_input)  # [batch, hidden_dim]

        # Combine and decode
        combined = torch.cat([ts_out, cov_out], dim=-1)  # [batch, hidden_dim * 2]
        out = self.decoder(combined)  # [batch, output_len]
        return out

# New Model 2: DualCNNANNPosEncFCNNDecoder
class DualCNNANNPosEncFCNNDecoder(nn.Module):
    def __init__(
        self,
        input_len: int = 168,
        output_len: int = 24,
        ts_input_size: int = 4,       # meter_reading + decomposition (3)
        num_covariates: int = 1,      # e.g., air_temperature
        num_primary_use: int = 16,    # number of primary_use categories
        embed_dim: int = 16,          # embedding dimension for primary_use
        hidden_dim: int = 64,         # hidden dimension for SimpleANN
        dropout: float = 0.2
    ):
        super(DualCNNANNPosEncFCNNDecoder, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.ts_input_size = ts_input_size
        self.num_covariates = num_covariates
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Embedding for primary_use
        self.primary_use_embed = nn.Embedding(num_primary_use, embed_dim)

        # Project time-series input to hidden_dim for positional encoding
        self.ts_input_proj = nn.Linear(ts_input_size, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding2(hidden_dim, max_len=input_len)

        # CNN for time-series branch
        self.ts_conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=64, kernel_size=5, padding=2)
        self.ts_conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # CNN for covariate branch
        cov_in_channels = num_covariates + embed_dim
        self.cov_conv1 = nn.Conv1d(in_channels=cov_in_channels, out_channels=32, kernel_size=5, padding=2)
        self.cov_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        # Compute CNN output length after pooling
        cnn_out_len = input_len // 4  # Two pooling layers with kernel_size=2
        cnn_out_features = 128 + 64  # ts_conv2 (128) + cov_conv2 (64)

        # SimpleANN to process concatenated CNN outputs
        self.ann = SimpleANN(
            input_len=cnn_out_len,
            input_dim=cnn_out_features,
            output_len=hidden_dim,
            hidden_dim=hidden_dim
        )

        # FCNN Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_len)
        )

    def forward(self, x_ts, x_cov, primary_use):
        """
        x_ts: [batch, seq_len, ts_input_size]
        x_cov: [batch, seq_len, num_covariates]
        primary_use: [batch, seq_len] (long/int indices)
        """
        # Process time-series input
        ts_proj = self.ts_input_proj(x_ts)  # [batch, seq_len, hidden_dim]
        ts_input = self.pos_encoding(ts_proj)  # [batch, seq_len, hidden_dim]
        ts_input = ts_input.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        ts_out = F.relu(self.ts_conv1(ts_input))  # [batch, 64, seq_len]
        ts_out = self.pool(ts_out)  # [batch, 64, seq_len/2]
        ts_out = F.relu(self.ts_conv2(ts_out))  # [batch, 128, seq_len/2]
        ts_out = self.pool(ts_out)  # [batch, 128, seq_len/4]
        ts_out = ts_out.transpose(1, 2)  # [batch, seq_len/4, 128]

        # Process covariates
        primary_use_embed = self.primary_use_embed(primary_use)  # [batch, seq_len, embed_dim]
        cov_input = torch.cat([primary_use_embed, x_cov], dim=-1)  # [batch, seq_len, embed_dim + num_covariates]
        cov_input = cov_input.transpose(1, 2)  # [batch, embed_dim + num_covariates, seq_len]
        cov_out = F.relu(self.cov_conv1(cov_input))  # [batch, 32, seq_len]
        cov_out = self.pool(cov_out)  # [batch, 32, seq_len/2]
        cov_out = F.relu(self.cov_conv2(cov_out))  # [batch, 64, seq_len/2]
        cov_out = self.pool(cov_out)  # [batch, 64, seq_len/4]
        cov_out = cov_out.transpose(1, 2)  # [batch, seq_len/4, 64]

        # Combine CNN outputs
        combined = torch.cat([ts_out, cov_out], dim=-1)  # [batch, seq_len/4, 128 + 64]

        # Process with SimpleANN
        ann_out = self.ann(combined)  # [batch, hidden_dim]

        # Decode
        out = self.decoder(ann_out)  # [batch, output_len]
        return out

# Existing models (unchanged, for completeness in model_fn)
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
    def __init__(self, input_len=168, forecast_len=24, conv_channels=32, gru_hidden=64):
        super(CNNGRUForecast, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=3, padding=1)
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

class TCNModel(nn.Module):
    def __init__(self, input_size=1, output_size=24, num_channels=[32, 32, 32], kernel_size=3, dropout=0.2, seq_len=168):
        super(TCNModel, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.layers = []
        for i, channels in enumerate(num_channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            conv = nn.Conv1d(
                in_channels=input_size if i == 0 else num_channels[i-1],
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding
            )
            self.layers.append(conv)
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        self.tcn = nn.Sequential(*self.layers)
        self.fc = nn.Linear(num_channels[-1] * seq_len, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

class Informer(nn.Module):
    def __init__(
        self,
        input_size=1,
        context_length=168,
        forecast_horizon=24,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1
    ):
        super(Informer, self).__init__()
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding2(d_model, max_len=context_length + forecast_horizon)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder_input = nn.Parameter(torch.randn(1, forecast_horizon, d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_embedding(x)
        x = self.pos_encoding(x)
        memory = self.encoder(x)
        decoder_input = self.decoder_input.expand(batch_size, -1, -1)
        decoder_input = self.pos_encoding(decoder_input)
        seq_len = x.size(1) + decoder_input.size(1)
        tgt_mask = torch.triu(torch.ones(decoder_input.size(1), decoder_input.size(1)), diagonal=1).bool().to(x.device)
        output = self.decoder(decoder_input, memory, tgt_mask=tgt_mask)
        output = self.output_projection(self.dropout(output))
        return output.squeeze(-1)

class DilatedRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=24, num_layers=3, dropout=0.2):
        super(DilatedRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnns = nn.ModuleList([
            nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dilations = [2 ** i for i in range(num_layers)]
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hiddens = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :]
            for i in range(self.num_layers):
                if t % self.dilations[i] == 0:
                    hiddens[i] = self.rnns[i](input_t, hiddens[i])
                    input_t = self.dropout(hiddens[i])
            if t == seq_len - 1:
                outputs.append(hiddens[-1])
        out = self.fc(outputs[-1])
        return out

class DualGRUPosEncGRUDecoder(nn.Module):
    def __init__(
        self,
        ts_input_size=4,
        cov_input_size=17,
        hidden_size=64,
        output_size=24,
        num_layers=2,
        dropout=0.2,
        seq_len=168,
        num_primary_use=16
    ):
        super(DualGRUPosEncGRUDecoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.primary_use_embedding = nn.Embedding(num_primary_use, 16)
        self.ts_input_proj = nn.Linear(ts_input_size, hidden_size)
        self.pos_encoding = PositionalEncoding2(hidden_size, max_len=seq_len)
        self.ts_encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.cov_encoder = nn.GRU(
            input_size=cov_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.decoder = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_ts, x_cov, primary_use):
        primary_use_embed = self.primary_use_embedding(primary_use)
        cov_input = torch.cat([primary_use_embed, x_cov], dim=-1)
        ts_proj = self.ts_input_proj(x_ts)
        ts_input = self.pos_encoding(ts_proj)
        ts_out, _ = self.ts_encoder(ts_input)
        cov_out, _ = self.cov_encoder(cov_input)
        combined = torch.cat([ts_out, cov_out], dim=-1)
        dec_out, _ = self.decoder(combined)
        out = self.fc(dec_out[:, -1, :])
        return out

class DualGRUPosEncFCNNDecoder(nn.Module):
    def __init__(
        self,
        ts_input_size=4,
        cov_input_size=17,
        hidden_size=64,
        output_size=24,
        num_layers=2,
        dropout=0.2,
        seq_len=168,
        num_primary_use=16
    ):
        super(DualGRUPosEncFCNNDecoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.primary_use_embedding = nn.Embedding(num_primary_use, 16)
        self.ts_input_proj = nn.Linear(ts_input_size, hidden_size)
        self.pos_encoding = PositionalEncoding2(hidden_size, max_len=seq_len)
        self.ts_encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.cov_encoder = nn.GRU(
            input_size=cov_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x_ts, x_cov, primary_use):
        primary_use_embed = self.primary_use_embedding(primary_use)
        cov_input = torch.cat([primary_use_embed, x_cov], dim=-1)
        ts_proj = self.ts_input_proj(x_ts)
        ts_input = self.pos_encoding(ts_proj)
        ts_out, _ = self.ts_encoder(ts_input)
        cov_out, _ = self.cov_encoder(cov_input)
        combined = torch.cat([ts_out[:, -1, :], cov_out[:, -1, :]], dim=-1)
        out = self.decoder(combined)
        return out

class DualCNNGRUPosEncFCNNDecoder(nn.Module):
    def __init__(
        self,
        input_len: int = 168,
        output_len: int = 24,
        ts_input_size: int = 4,
        num_covariates: int = 1,
        num_primary_use: int = 16,
        embed_dim: int = 16,
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

# Training functions (unchanged)
def train_arima_model(train_series):
    model = ARIMA(5, 1, 0)
    model.fit(train_series)
    return model

def train_model(model, dataloader, epochs=50, learning_rate=0.001, loss_fn=None, optimizer_class=optim.Adam):
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
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
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    return model

def moving_average_forecast(series, input_window: int, output_window: int) -> np.ndarray:
    history = series[-input_window:].values().flatten()
    avg = np.mean(history)
    return np.full(shape=(output_window,), fill_value=avg)

# Updated model_fn to include new models
def model_fn(model_name: str):
    if model_name.endswith("_hour"):
        ts_input_size = 5  # meter_reading + decomposition (3) + hour
        base_name = model_name.replace("_hour", "")
    else:
        ts_input_size = 4  # meter_reading + decomposition (3)
        base_name = model_name

    match base_name:
        case "moe_gru":
            return MoEGRU(
                input_size=ts_input_size,
                hidden_size=64,
                output_size=8,
                num_experts=5,
                ffn_hidden_size=32,
            )
        case "moe_lstm":
            return MoELSTM(
                input_size=ts_input_size,
                hidden_size=64,
                output_size=24,
                num_experts=6,
                ffn_hidden_size=32,
            )
        case "lstm":
            return LSTMModel(
                input_size=ts_input_size,
                hidden_size=82,
                output_size=24
            )
        case "gru":
            return GRUModel(
                input_size=ts_input_size,
                hidden_size=82,
                output_size=24
            )
        case "simple_ann":
            return SimpleANN(input_len=24, input_dim=ts_input_size, output_len=8)
        case "anomaly_ann":
            return BinaryClassificationANN(input_dim=36)
        case "vae":
            return VarEncoderDecoder(hidden_layers=3, hidden_size=128, seq_length=168)
        case "cnn-lstm":
            return CNNLSTMForecast()
        case "transformer":
            return TimeSeriesTransformer2()
        case "tcn":
            return TCNModel(input_size=ts_input_size, output_size=24, seq_len=168)
        case "informer":
            return Informer(input_size=ts_input_size)
        case "dilated_rnn":
            return DilatedRNN(input_size=ts_input_size, output_size=24)
        case "dual_gru_gru":
            return DualGRUPosEncGRUDecoder(ts_input_size=ts_input_size)
        case "dual_gru_fcnn":
            return DualGRUPosEncFCNNDecoder(ts_input_size=ts_input_size)
        case "dual_cnn_gru_fcnn":
            return DualCNNGRUPosEncFCNNDecoder(ts_input_size=ts_input_size)
        case "dual_simple_ann_fcnn":
            return DualSimpleANNPosEncFCNNDecoder(ts_input_size=ts_input_size)
        case "dual_cnn_ann_fcnn":
            return DualCNNANNPosEncFCNNDecoder(ts_input_size=ts_input_size)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")