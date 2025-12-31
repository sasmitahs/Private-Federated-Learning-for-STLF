# Private-Federated-Learning-for-STLF

## Repository Overview

This repository implements privacy-preserving federated learning approaches for **Short-Term Load Forecasting (STLF)** in electrical power systems. STLF involves predicting electricity demand (load) in the near future (e.g., hours to days ahead).

The project compares two main approaches:
- **Single model**: standard models for STLF.
- **Dual models**: models which can take 2 inputs (time series data and covariates)

## File Descriptions

### `Dual models/` Directory

This directory implements the **private federated learning** pipeline using dual or adversarial models. Common in privacy-focused FL, "dual models" often refer to setups with a primary forecasting model and an auxiliary/adversarial model that enforces privacy (e.g., via differential privacy or inference attack prevention).

Typical files you may find here (based on standard implementations and commit history suggesting Python scripts were organized here):

- `federated_training.py` or `dual_model.py`  
  Main script for training the dual-model federated system.  
  - Sets up federated clients (e.g., different regions or buildings with private load data).  
  - Defines the primary STLF model (e.g., LSTM, Transformer, or CNN-based).  
  - Defines the secondary/privacy model (e.g., discriminator or noise-adding mechanism).  
  - Implements federated averaging (FedAvg) or variants with privacy (e.g., DP-SGD).  
  - Handles client selection, local training, and global aggregation.

- `privacy_utils.py` or `dp_mechanisms.py`  
  Utility functions for adding differential privacy noise, clipping gradients, or measuring privacy loss.

- `data_preprocessing.py`  
  Loads and preprocesses electricity load datasets (e.g., from public sources like ERCOT, PJM, or household-level datasets).  
  Splits data across federated clients (non-IID distribution common in real load data).

- `evaluation.py`  
  Evaluates forecasting accuracy (MAE, RMSE, MAPE) and privacy metrics (e.g., membership inference attack success rate).

- `config.yaml` or `hyperparameters.py`  
  Configuration for learning rates, epochs, privacy budgets (epsilon), number of clients, etc.

### `single model/` Directory

This directory contains the **baseline non-federated** approach — a single centralized model trained on pooled data.

Typical files:

- `centralized_training.py` or `single_model.py`  
  Main script for training a single global STLF model.  
  - Loads all data centrally.  
  - Defines the same architecture as the primary model in the dual setup (for fair comparison).  
  - Trains using standard optimizers (Adam, etc.) without federation or privacy noise.

- `data_preprocessing.py`  
  Similar to the dual version but loads all data into one dataset (no client splitting).

- `evaluation.py`  
  Computes forecasting metrics on test sets.  
  Used to compare against the federated/private version (typically showing privacy-accuracy trade-off).

- `model.py`  
  Shared model architecture definition (e.g., LSTM or GRU network for time-series forecasting).

## Usage

1. **Install dependencies** (likely required):
   ```bash
   pip install torch numpy pandas scikit-learn matplotlib
