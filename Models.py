


from darts import TimeSeries
from darts.models import ARIMA
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch

# import torch
# import torch.nn as nn
import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainableHourWeightedMSELoss(nn.Module):
    def __init__(self):
        super(TrainableHourWeightedMSELoss, self).__init__()
        # Learnable scalar for each hour (0 to 23)
        self.hour_weights = nn.Parameter(torch.ones(24))  # Initialized to 1.0

    def forward(self, predictions, targets, input_batch):
        """
        Args:
            predictions: [B, H]
            targets:     [B, H]
            input_batch: [B, T, 2] where 2 features = [meter, hour]
        """
        if input_batch.ndim != 3 or input_batch.shape[2] < 2:
            raise ValueError(f"Expected input shape [B, T, 2], got {input_batch.shape}")

        # Extract hour from the last input timestep
        raw_hours = input_batch[:, -1, 1] * 23.0  # [B], denormalized hour
        hour_indices = raw_hours.round().long().clamp(0, 23)  # [B]
        weights = self.hour_weights[hour_indices]  # [B]

        

        # Compute MSE per timestep
        mse = F.mse_loss(predictions, targets, reduction='none')  # [B, H]
        
        # Broadcast weights: [B] → [B, 1]
        weighted_mse = mse * weights.unsqueeze(-1)  # [B, H]
        return weighted_mse.mean()





class ExponentialWeightedMSELoss(nn.Module):
    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            predictions: Tensor of shape (batch_size, forecast_len)
            targets: Tensor of shape (batch_size, forecast_len)
        Returns:
            Scalar loss value (weighted MSE)
        """
        squared_error = (predictions - targets) ** 2  # shape: (B, n)

        forecast_len = predictions.shape[1]

        # Generate exponential weights: [1, k, k^2, ..., k^(n-1)]
        device = predictions.device
        weights = self.k ** torch.arange(forecast_len, device=device, dtype=predictions.dtype)  # shape: (n,)
        weights = weights.view(1, forecast_len)  # reshape for broadcasting: (1, n)

        weighted_squared_error = squared_error * weights  # broadcast to (B, n)

        return weighted_squared_error.mean()


class SimpleANN(nn.Module):
    def __init__(self, input_len=24, input_dim=2, output_len=8, hidden_dim=64):
        """
        input_len: number of time steps (e.g., 24)
        input_dim: features per timestep (e.g., meter_reading + hour = 2)
        output_len: prediction horizon (e.g., 8)
        """
        super(SimpleANN, self).__init__()
        self.flatten = nn.Flatten()  # input shape: [B, 24, 2] → [B, 48]
        self.fc1 = nn.Linear(input_len * input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_len)

    def forward(self, x):
        x = self.flatten(x)        # [B, 48]
        x = F.relu(self.fc1(x))    # [B, hidden]
        x = F.relu(self.fc2(x))    # [B, hidden]
        x = self.fc3(x)            # [B, 8]
        return x


def train_arima_model(train_series):
    # Create ARIMA model
    model = ARIMA(5, 1, 0)  # You can adjust the order if needed
    model.fit(train_series)  # Train the ARIMA model
    return model

def train_model(model, dataloader, epochs=50, learning_rate=0.001, loss_fn=None, optimizer_class=optim.Adam):
    """
    General training loop for any PyTorch model.

    Parameters:
    - model (nn.Module): The PyTorch model to train
    - dataloader (DataLoader): DataLoader for training data
    - epochs (int): Number of training epochs
    - learning_rate (float): Learning rate for optimizer
    - loss_fn (callable): Loss function (default is nn.MSELoss)
    - optimizer_class: Optimizer class from torch.optim (default: Adam)

    Returns:
    - model (nn.Module): The trained model
    """
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


class TimeSeriesDifficultyWeight:
    def __init__(self, num_clients, accumulate_iters=20):
        self.num_clients = num_clients
        self.last_loss = torch.ones(num_clients).float().to(DEVICE)
        self.learn_score = torch.zeros(num_clients).float().to(DEVICE)
        self.unlearn_score = torch.zeros(num_clients).float().to(DEVICE)
        self.ema_difficulty = torch.ones(num_clients).float().to(DEVICE)
        self.accumulate_iters = accumulate_iters

    def update(self, cid: int, loss_history: List[float]) -> float:
        """
        Update difficulty based on loss trend for a client.
        Expects a list of per-epoch losses.
        """
        current_loss = torch.tensor(loss_history[-1], dtype=torch.float32).to(DEVICE)
        previous_loss = self.last_loss[cid]
        delta = current_loss - previous_loss
        ratio = torch.log((current_loss + 1e-8) / (previous_loss + 1e-8))

        learn = torch.where(delta < 0, -delta * ratio, torch.tensor(0.0, device=current_loss.device))
        unlearn = torch.where(delta >= 0, delta * ratio, torch.tensor(0.0, device=current_loss.device))

        # EMA update
        momentum = (self.accumulate_iters - 1) / self.accumulate_iters
        self.learn_score[cid] = momentum * self.learn_score[cid] + (1 - momentum) * learn
        self.unlearn_score[cid] = momentum * self.unlearn_score[cid] + (1 - momentum) * unlearn

        # Difficulty score
        diff_ratio = (self.unlearn_score[cid] + 1e-8) / (self.learn_score[cid] + 1e-8)
        difficulty = diff_ratio #torch.pow(diff_ratio, 1 / 5)

        # Smooth difficulty over rounds
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
    """
    Forecast future values using naive moving average on a single TimeSeries.

    Parameters:
    - series (TimeSeries): A darts TimeSeries object (assumes univariate)
    - input_window (int): Number of past steps to average
    - output_window (int): Number of future steps to predict

    Returns:
    - np.ndarray: Predicted values (output_window,)
    """
    history = series[-input_window:].values().flatten()
    avg = np.mean(history)
    return np.full(shape=(output_window,), fill_value=avg)



# class LSTMModel(nn.Module):
#     def __init__(self, input_size=1, hidden_size=64, output_size=1):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # x: [batch_size, seq_len, input_size]
#         lstm_out, _ = self.lstm(x)
#         out = self.fc(lstm_out[:, -1, :])  # We take the output of the last time step
#         return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout = 0.2, output_size=1):
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
        # x: shape (batch, time, input_features)
        out, _ = self.lstm(x)            # out: (batch, time, hidden_size)
        out = out[:, -1, :]              # Take last time step output: (batch, hidden_size)
        out = self.fc(out)               # Final output: (batch, output_size)
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
        
        self.output_layer = nn.Linear(hidden_dims[1], 1)  # Output: single logit

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.output_layer(x)  # No sigmoid
        return x





class VarEncoderDecoder(nn.Module):
    def __init__(self,hidden_layers,hidden_size,latent_dim=32,seq_length=512):
        super().__init__()
        
        hidden_sizes=[]
        hidden_sizes.append(seq_length)
        for i in range(hidden_layers):
            hidden_sizes.append(hidden_size)
            hidden_size//=2
        
        self.encoder=nn.ModuleList()
        for i in range(1,len(hidden_sizes)):
            linear=nn.Linear(hidden_sizes[i-1],hidden_sizes[i])
            activation=nn.ReLU()
            norm=nn.BatchNorm1d(hidden_sizes[i])
            self.encoder.append(linear)
            self.encoder.append(norm)
            self.encoder.append(activation)

        self.encoder=nn.Sequential(*self.encoder)
        
        #mu,var
        self.enc_fc_mu=nn.Linear(hidden_sizes[-1],latent_dim)
        self.enc_fc_var=nn.Linear(hidden_sizes[-1],latent_dim)

        self.decoder_in=nn.Linear(latent_dim,hidden_sizes[-1])
        self.decoder=nn.ModuleList()
        hidden_sizes=list(reversed(hidden_sizes))
        for i in range(1,len(hidden_sizes)):
            linear=nn.Linear(hidden_sizes[i-1],hidden_sizes[i])
            activation=nn.LeakyReLU()
            norm=nn.BatchNorm1d(hidden_sizes[i])
            self.decoder.append(linear)
            self.decoder.append(norm)
            self.decoder.append(activation)        
        self.decoder=nn.Sequential(*self.decoder)
        
    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self,X):
        X=self.encoder(X)
        # print(X.shape)
        mu=self.enc_fc_mu(X)
        log_var=self.enc_fc_var(X)
        X=self.reparameterize(mu,log_var)
        X=self.decoder_in(X)
        # print(X.shape)
        X=self.decoder(X)
        # print(X.shape)
        return X








class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=2, dropout = 0.2):# did change here dropout this time
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        gru_out, _ = self.gru(x)  # gru_out shape: [batch_size, seq_len, hidden_size]
        out = self.fc(gru_out[:, -1, :])  # Use last timestep's output
        return out  # shape: [batch_size, output_size]


class MoELSTM(nn.Module):
    """
    Mixture-of-Experts LSTM model.
    
    The architecture consists of:
    1. Input sequence
    2. Lower LSTM layer
    3. Multiple FFN experts + Gate network
    4. Weighted combination of expert outputs
    5. Upper LSTM layer
    6. Final FFN layer
    7. Output layer
    """

    def __init__(self, input_size, hidden_size, num_experts, ffn_hidden_size, output_size):
        """
        Initialize the MoE-LSTM model.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of LSTM hidden states
            num_experts: Number of expert networks
            ffn_hidden_size: Hidden size of feed-forward expert networks
            output_size: Size of output
        """
        super(MoELSTM, self).__init__()
        
        # Lower LSTM layer
        self.lower_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Expert FFNs
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, ffn_hidden_size) for _ in range(num_experts)
        ])
        
        # Expert activation
        self.expert_activation = nn.ReLU()
        
        # Gate network
        self.gate = nn.Linear(hidden_size, num_experts)
        self.gate_activation = nn.Softmax(dim=-1)
        
        # Upper LSTM layer
        self.upper_lstm = nn.LSTM(
            input_size=ffn_hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Final FFN (Dense) layer
        self.final_ffn = nn.Linear(hidden_size, ffn_hidden_size)
        self.final_activation = nn.ReLU()
        
        # Output layer
        self.output_layer = nn.Linear(ffn_hidden_size, output_size)
        
        # Save parameter for later use
        self.num_experts = num_experts

    def forward(self, x):
        """
        Forward pass of the MoE-LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Lower LSTM layer
        lower_lstm_out, _ = self.lower_lstm(x)
        # lower_lstm_out shape: (batch_size, seq_len, hidden_size)
        
        # Apply each expert to the lower LSTM output
        expert_outputs = []
        for expert in self.experts:
            # Apply linear layer to each timestep
            expert_out = expert(lower_lstm_out)
            # Apply activation
            expert_out = self.expert_activation(expert_out)
            # Add to list of expert outputs
            expert_outputs.append(expert_out)

        # Get gate outputs (weights for each expert)
        gate_logits = self.gate(lower_lstm_out)
        gate_weights = self.gate_activation(gate_logits)
        # gate_weights shape: (batch_size, seq_len, num_experts)
        
        # Initialize weighted expert outputs
        batch_size, seq_len, _ = lower_lstm_out.shape
        device = lower_lstm_out.device
        weighted_expert_outputs = torch.zeros(batch_size, seq_len, expert_outputs[0].shape[-1], device=device)

        # For each expert, multiply its output by corresponding gate weight
        for i, expert_out in enumerate(expert_outputs):
            # Extract weight for this expert and add dimension for broadcasting
            # Shape: (batch_size, seq_len, 1)
            expert_weight = gate_weights[:, :, i:i+1]
            
            # Weight the expert output
            weighted_expert = expert_out * expert_weight
            
            # Add to weighted sum
            weighted_expert_outputs += weighted_expert

        # Upper LSTM layer
        upper_lstm_out, _ = self.upper_lstm(weighted_expert_outputs)
        # Take the output from the last time step
        last_output = upper_lstm_out[:, -1, :]
        
        # Final FFN layer
        ffn_out = self.final_ffn(last_output)
        ffn_out = self.final_activation(ffn_out)
        
        # Output layer
        output = self.output_layer(ffn_out)
        
        return output


class CNNLSTMForecast(nn.Module):
    def __init__(self, input_len=168, forecast_len=24, conv_channels=32, lstm_hidden=64):
        super(CNNLSTMForecast, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)  # Reduces sequence length by half

        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, batch_first=True)

        self.fc = nn.Linear(lstm_hidden, forecast_len)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        # x: [batch_size, 1, 168]
        x = self.conv1(x)       # [batch_size, conv_channels, 168]
        x = self.relu(x)
        x = self.pool(x)        # [batch_size, conv_channels, 84]

        # LSTM expects [batch_size, seq_len, input_size]
        x = x.permute(0, 2, 1)  # [batch_size, 84, conv_channels]
        out, _ = self.lstm(x)   # [batch_size, 84, lstm_hidden]
        out = out[:, -1, :]     # take last time step

        out = self.fc(out)      # [batch_size, forecast_len]
        return out


class MoEGRU(nn.Module):
    """
    Mixture-of-Experts GRU model.
    
    The architecture consists of:
    1. Input sequence
    2. Lower GRU layer
    3. Multiple FFN experts + Gate network
    4. Weighted combination of expert outputs
    5. Upper GRU layer
    6. Final FFN layer
    7. Output layer
    """

    def __init__(self, input_size, hidden_size, num_experts, ffn_hidden_size, output_size):
        super(MoEGRU, self).__init__()

        # Lower GRU layer
        self.lower_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Expert FFNs
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, ffn_hidden_size) for _ in range(num_experts)
        ])

        self.expert_activation = nn.ReLU()

        # Gate network
        self.gate = nn.Linear(hidden_size, num_experts)
        self.gate_activation = nn.Softmax(dim=-1)

        # Upper GRU layer
        self.upper_gru = nn.GRU(
            input_size=ffn_hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Final FFN layer
        self.final_ffn = nn.Linear(hidden_size, ffn_hidden_size)
        self.final_activation = nn.ReLU()

        # Output layer
        self.output_layer = nn.Linear(ffn_hidden_size, output_size)

        self.num_experts = num_experts

    def forward(self, x):
        """
        Forward pass of the MoE-GRU model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Lower GRU
        lower_gru_out, _ = self.lower_gru(x)  # shape: (B, T, H)

        # Expert outputs
        expert_outputs = []
        for expert in self.experts:
            out = self.expert_activation(expert(lower_gru_out))
            expert_outputs.append(out)

        # Gating weights
        gate_logits = self.gate(lower_gru_out)  # shape: (B, T, E)
        gate_weights = self.gate_activation(gate_logits)

        # Weighted sum of expert outputs
        batch_size, seq_len, _ = lower_gru_out.shape
        device = x.device
        weighted_output = torch.zeros(batch_size, seq_len, expert_outputs[0].shape[-1], device=device)

        for i, expert_out in enumerate(expert_outputs):
            weight = gate_weights[:, :, i:i+1]
            weighted_output += expert_out * weight

        # Upper GRU
        upper_gru_out, _ = self.upper_gru(weighted_output)

        # Final time step
        last_output = upper_gru_out[:, -1, :]

        # Final FFN
        ffn_out = self.final_activation(self.final_ffn(last_output))
        return self.output_layer(ffn_out)




class CNNGRUForecast(nn.Module):
    def __init__(self, input_len=168, forecast_len=24, conv_channels=32, gru_hidden=64):
        super(CNNGRUForecast, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)  # Reduces temporal length

        self.gru = nn.GRU(input_size=conv_channels, hidden_size=gru_hidden, batch_first=True)

        self.fc = nn.Linear(gru_hidden, forecast_len)

    def forward(self, x):

        x = x.permute(0, 2, 1) 
        # x: [batch_size, 1, 168]
        x = self.conv1(x)       # [batch_size, conv_channels, 168]
        x = self.relu(x)
        x = self.pool(x)        # [batch_size, conv_channels, 84]

        x = x.permute(0, 2, 1)  # [batch_size, 84, conv_channels]
        out, _ = self.gru(x)    # [batch_size, 84, gru_hidden]
        out = out[:, -1, :]     # last GRU output

        out = self.fc(out)      # [batch_size, forecast_len]
        return out



class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
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
    """Transformer model for univariate time series forecasting"""
    
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
        
        # Input embedding and positional encoding
        self.input_embedding = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=context_length + forecast_horizon)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable forecast tokens
        self.forecast_tokens = nn.Parameter(torch.randn(forecast_horizon, 1, d_model))
        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = src.size(0)

        # Embed input
        src_embedded = self.input_embedding(src)  # (batch_size, context_length, d_model)
        src_embedded = self.pos_encoding(src_embedded)  # Same shape

        # Forecast tokens: (batch_size, forecast_horizon, d_model)
        forecast_tokens = self.forecast_tokens.expand(batch_size, -1, -1)

        # Concatenate context and forecast tokens
        full_sequence = torch.cat([src_embedded, forecast_tokens], dim=1)

        # Causal mask (batch_first=True needs 2D mask of shape (seq_len, seq_len))
        seq_len = full_sequence.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(src.device)

        # Apply transformer
        transformer_output = self.transformer_encoder(full_sequence, mask=causal_mask)

        # Extract forecast part
        forecast_output = transformer_output[:, -self.forecast_horizon:, :]
        forecast_output = self.dropout(forecast_output)
        predictions = self.output_projection(forecast_output)  # (batch_size, forecast_horizon, 1)

        return predictions


import math
class PositionalEncoding2(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
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

        # 1. Input projection: from 1 → d_model
        self.input_embedding = nn.Linear(1, d_model)

        # 2. Forecast tokens (learnable embeddings)
        self.forecast_tokens = nn.Parameter(torch.randn(1, forecast_horizon, d_model))

        # 3. Positional encoding for (context + forecast) steps
        self.pos_encoding = PositionalEncoding2(d_model, max_len=context_length + forecast_horizon)

        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True  # Makes all inputs/outputs (batch, seq, dim)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 5. Output projection: d_model → 1
        self.output_projection = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, context_length, 1)

        Returns:
            Tensor of shape (batch_size, forecast_horizon, 1)
        """
        batch_size = x.size(0)

        # 1. Project input from (1) to (d_model)
        x = self.input_embedding(x)  # (batch_size, context_length, d_model)

        # 2. Expand forecast tokens
        forecast_tokens = self.forecast_tokens.expand(batch_size, -1, -1)  # (batch_size, forecast_horizon, d_model)

        # 3. Concatenate context + forecast tokens
        full_input = torch.cat([x, forecast_tokens], dim=1)  # (batch_size, context + horizon, d_model)

        # 4. Add positional encoding
        full_input = self.pos_encoding(full_input)

        # 5. Causal mask (prevents forecast tokens from attending to future)
        seq_len = full_input.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)

        # 6. Transformer encoder
        transformer_output = self.encoder(full_input, mask=causal_mask)  # (batch_size, seq_len, d_model)

        # 7. Slice only forecast part
        forecast_output = transformer_output[:, -self.forecast_horizon:, :]  # (batch_size, forecast_horizon, d_model)

        # 8. Project back to 1 value per timestep
        output = self.output_projection(self.dropout(forecast_output))  # (batch_size, forecast_horizon, 1)
        return output









def model_fn(model_name: str):
    if model_name.endswith("_hour"):
        input_size = 2  # meter_reading + hour
        base_name = model_name.replace("_hour", "")
    else:
        input_size = 1
        base_name = model_name

    match base_name:
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
            return SimpleANN(input_len=24, input_dim=input_size, output_len=8)
        case "anomaly_ann":
            return BinaryClassificationANN(input_dim=36)
        case "vae": 
            return VarEncoderDecoder(input_size)
        case "cnn-lstm":
            return CNNLSTMForecast()
        case "transformer":
            return TimeSeriesTransformer2()
        case _:
            raise ValueError(f"Unknown model name: {model_name}")
