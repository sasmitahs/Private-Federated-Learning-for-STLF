# Federated Learning for Short-Term Load Forecasting (STLF)

This repository implements federated learning approaches for **Short-Term Load Forecasting (STLF)** in electrical power systems. STLF involves predicting near-future electricity demand (load) at the building level.

The project explores two main modeling paradigms:

- **Single-model approaches**: Standard deep learning models that take only historical meter readings as input (no additional covariates).
- **Dual-model approaches**: Advanced models that accept dual inputs — historical time series data plus covariates (e.g., air temperature, primary use embedding, STL-decomposed components such as trend, seasonal, and residual).

Federated learning strategies include **FedAvg**, **FedProx**, **SCAFFOLD**, clustering-based variants (e.g., DAS, Power-of-Choice), and random sampling baselines, with and without client clustering.

The codebase supports training, evaluation, baseline comparisons (e.g., weekly persistence, AutoARIMA), and comprehensive **ASHRAE-compliant** metric computation (MAE, RMSE, MAPE, SMAPE, NMAE, NMBE, NRMSE).

Data is assumed to be in Feather format (e.g., `train_final.feather` or similar), containing columns such as `building_id`, `timestamp`, `meter_reading`, `air_temperature`, etc.

## Repository Structure

- **`Dual models/`** — Code for dual-input models (multi-channel with covariates + STL decomposition).
- **`single model/`** — Code for single-input models (historical meter readings only).
- **`README.md`** — This file.

## Files in `Dual models/` Folder

**my_utils.py**  
Provides data loading and preprocessing for energy time series with STL decomposition, scaling, and feature construction (multi-channel: meter + trend + seasonal + residual), plus generic training, evaluation, and weight-handling utilities for multi-input models.

**correct_cal_eval.py**  
Evaluates centralized global models on individual buildings using rolling 24-hour forecasts, computes per-client metrics (MAE, RMSE, MAPE, SMAPE, NMAE, NMBE, etc.), and summarizes median performance across clients and rounds.

**cal.py**  
Scans prediction and metric files for a specified model/strategy, checks for missing/empty/NaN files, collects valid metrics, and reports median MAE, MSE, RMSE, MAPE, SMAPE plus lists of unprocessed clients.

**cal_NRMSE.py**  
Generates rolling forecasts for federated models across multiple rounds and strategies, saves per-client predictions, computes extended metrics including NRMSE, and outputs median results with detailed processing diagnostics.

**eval-global.py**  
Evaluates a single global model checkpoint on all buildings, generates rolling forecasts, saves predictions/metrics per client, and provides comprehensive summary statistics (medians, means, variances) for MAE, RMSE, MAPE, SMAPE, NMAE, NMBE, NRMSE.

**eval.py**  
Processes existing prediction CSVs, computes core forecasting metrics (MAE, MSE, RMSE, MAPE, SMAPE) per client while handling NaNs robustly, saves individual and aggregated metrics, and prints median performance across valid clients.

**eval1.py**  
Loads federated models from specified rounds/strategies, performs rolling forecasts per client using multi-channel STL inputs, saves predictions to CSV, computes basic metrics (MAE, RMSE, MAPE, SMAPE), and stores results for analysis.

**extr13-autoarima.py**  
Recomputes ASHRAE-standard metrics (MAE, RMSE, MAPE, SMAPE, NMAE, NMBE, NRMSE) from existing AutoARIMA prediction CSVs per building, saves individual metrics, and generates a detailed median/mean/variance summary focused on NMAE/NMBE/NRMSE_mean.

**extra3-centralised.py**  
Processes existing centralized/global model prediction CSVs across multiple models and strategies, recomputes comprehensive ASHRAE metrics per client, saves per-client results, and produces a logged median/mean/variance summary table for NMAE, NMBE, and NRMSE_mean performance comparison.

**extra3.py**  
Recomputes comprehensive forecasting metrics (MAE, RMSE, MAPE, SMAPE, NRMSE_range, plus ASHRAE NMAE/NMBE/NRMSE_mean) from existing prediction CSVs across multiple models/strategies/rounds, saves per-client metrics, and generates detailed median/mean/variance summaries.

**Models.py**  
Defines dual-input neural network architectures (e.g., CNN-GRU, ANN-FCNN variants with covariate and embedding support).

**Preprocess.py**  
Handles data loading, STL decomposition, multi-channel feature creation, and dataloader construction for dual models.

**global.py**  
Trains and saves centralized (global) dual models.

**Various strategy scripts**  
(e.g., `Newmodel-clustering-FedProx.py`, `NoClustering-FedProx.py`, `clustering-das.py`, `poc_fedprox_cluster.py`, `random_nocluster_fedprox.py`, etc.)  
Implement federated learning simulations with different aggregation strategies (FedProx, SCAFFOLD, DAS clustering, Power-of-Choice, random sampling) on dual models, with/without client clustering.

**cluster_analysis.py**  
Utilities for client clustering analysis (likely for heterogeneity assessment).

## Files in `single model/` Folder

**my_utils.py**  
Provides data loading utilities for energy time-series forecasting: loads Feather data per building, constructs Darts TimeSeries, scales with MinMaxScaler, converts to supervised (X,y) format, and offers training/evaluation functions for PyTorch models including weight handling.

**extra3.py**  
Recomputes comprehensive forecasting metrics (MAE, RMSE, MAPE, SMAPE, NRMSE_range, plus ASHRAE NMAE/NMBE/NRMSE_mean) from existing prediction CSVs across multiple models/strategies/rounds, saves per-client metrics, and generates detailed median/mean/variance summaries.

**extra3-weeklypersistent.py**  
Implements a weekly-persistent naive baseline (copies load from 168 hours ago) for all buildings, generates rolling 24-hour forecasts, saves individual predictions, computes extended metrics including ASHRAE NMAE/NMBE/NRMSE_mean, and reports median/mean performance across buildings.

**extra3-centralised.py**  
Recomputes full ASHRAE-compliant metrics from existing centralized model prediction CSVs across multiple models/strategies, saves per-client results, and produces an extended median summary covering MAE, MSE, RMSE, MAPE, SMAPE, NRMSE_range, NMAE, NMBE, and NRMSE_mean.

**eval.py**  
Generates rolling forecasts using trained models for specified clients/models/strategies/rounds, saves predictions to CSV, computes basic metrics (MAE, MSE, RMSE, MAPE, SMAPE) per round, and supports grouped metric aggregation and boxplot statistics extraction.

**eval-global.py**  
Evaluates centralized/global models across multiple epochs: performs rolling forecasts per building, saves predictions/metrics, aggregates results, and computes median MAE/RMSE/MAPE/SMAPE/NRMSE_range/NMAE/NMBE/NRMSE_mean by epoch and model for comprehensive comparison.

**correct_cal_eval.py**  
Complete pipeline for federated/centralized model evaluation: generates rolling forecasts from saved weights, computes full ASHRAE metrics per client/round, saves predictions and metrics, and produces a final summary table with median/mean/variance across all valid clients.

**cal.py**  
Diagnostic script that scans prediction/metric files for specified models/strategies, identifies missing/empty/invalid cases, computes median MAE/MSE/RMSE/MAPE/SMAPE/NRMSE across valid clients, and logs detailed processing statistics and unprocessed client IDs.

**cal-NRMSE.py**  
Analyzes existing prediction/metric files for a given model/strategy, recomputes NRMSE (range-normalized) where needed, reports median MAE/RMSE/MAPE/SMAPE/NRMSE, and provides comprehensive diagnostics on missing, empty, or invalid files across clients.

**Models.py**  
Defines single-input neural network architectures (e.g., GRU, LSTM, Simple CNN, MoE-LSTM).

**Preprocess.py**  
Data loading and preprocessing tailored for single-input models.

**Strategy.py**  
Base utilities for federated learning strategies in single-model setting.

**classification_utils.py**  
Utilities for client classification or clustering in single-model experiments.

**global.py**  
Trains centralized single models.

**naive.py**  
Simple naive baselines (e.g., persistence).

**Various strategy scripts**  
(e.g., `das-clustering.py`, `das-nocluster.py`, `das_cluster_fedprox.py`, `poc-clustering.py`, `randoms_nocluster.py`, etc.)  
Implement federated learning with FedAvg, FedProx, clustering (DAS, random), Power-of-Choice, and other variants on single-input models.

## Usage

1. Prepare data in Feather format with required columns (`building_id`, `timestamp`, `meter_reading`, etc.).
2. Run training scripts (e.g., `global.py` or any strategy-specific `.py` file) to produce model checkpoints in `results/`.
3. Use evaluation scripts (`eval.py`, `correct_cal_eval.py`) to generate predictions and per-client metrics.
4. Run summary scripts (`extra3.py`, `cal.py`) for aggregated ASHRAE-compliant results and diagnostics.

## Metrics

All evaluation scripts compute both standard and ASHRAE-recommended metrics:

- MAE, MSE, RMSE
- MAPE (%), SMAPE (%)
- NRMSE (range-normalized)
- NMAE (%), NMBE (%), NRMSE_mean (%) (ASHRAE-normalized)

Results are typically aggregated using the **median** across buildings for robustness against outliers.

## Contributors

- sasmitahs
- samy101 (Pandarasamy Arjunan)

Feel free to open issues, submit pull requests, or contribute improvements!
