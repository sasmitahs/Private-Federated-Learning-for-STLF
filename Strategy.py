from flwr.server.strategy import FedAvg
from energy_ts_diffusion.task import get_weights, set_weights
import torch
import numpy as np
import csv
import os
from flwr.server.strategy import FedAvgM

import os
import pickle


from typing import List, Tuple, Optional, Dict
import flwr as fl
import numpy as np
from flwr.common import FitRes, Parameters, NDArrays, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters


def calc_phase(delta_t: np.ndarray, delta_w_arr: List[np.ndarray]) -> List[float]:
    theta_arr = []
    for delta_w in delta_w_arr:
        similarity = np.inner(delta_w, delta_t)
        norm_t = np.linalg.norm(delta_t)
        norm_w = np.linalg.norm(delta_w)
        theta_i = similarity / (norm_t * norm_w + 1e-8)
        theta_arr.append(theta_i)
    return theta_arr

def calc_sync_weights(phase_arr: List[float]) -> List[float]:
    theta_mean = np.mean(phase_arr)
    base_sin = np.sum([np.sin(theta_mean - theta) for theta in phase_arr])
    sync_weights = [np.sin(theta_mean - theta) / (base_sin + 1e-8) for theta in phase_arr]
    return sync_weights


class SyncFedAvgBeta(FedAvg):
    def __init__(self, model_fn, model_dir="checkpoints", **kwargs):
        super().__init__(**kwargs)
        self.model_fn = model_fn
        self.model_dir = model_dir
        self.parameters = None 
        self.k = 0.05
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            print("[DEBUG] Got none result")
            return None, {}

        weights_list = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        print(f"[DEBUG] weights: {weights_list[0].shape}")
        
        if not weights_list:
            return None, {}
        
        # Layer-wise aggregation instead of flattening
        num_layers = len(weights_list[0])
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            layer_weights = [weights[layer_idx] for weights in weights_list]
            
            # Calculate sync weights for this layer
            flat_layer_weights = [w.flatten() for w in layer_weights]
            
            if len(flat_layer_weights) > 1:
                delta_t = np.mean(flat_layer_weights, axis=0) - flat_layer_weights[0]
                delta_w_arr = [w - flat_layer_weights[0] for w in flat_layer_weights]
                
                phase_arr = calc_phase(delta_t, delta_w_arr)
                sync_weights = calc_sync_weights(phase_arr)
                sync_weights = np.array(sync_weights) * self.k
                # sync_weights = sync_weights / (sync_weights.sum() + 1e-8)

                for i, (_, fit_res) in enumerate(results):
                    print(f"[SYNC WEIGHT] Client {i}: {sync_weights[i]:.4f}")
            else:
                sync_weights = [1.0]
            
            # Aggregate this layer
            weighted_layer = sum(layer_weights[i] * sync_weights[i] for i in range(len(layer_weights)))
            aggregated_weights.append(weighted_layer)
        
        # parameters = fl.common.ndarrays_to_parameters(aggregated_weights)

        # print(f"[DEBUG] Aggregating round {rnd}: len(weights_list[0]) = {len(weights_list[0])}")

        
        # Save model
        if aggregated_weights:
            model = self.model_fn()
            set_weights(model, aggregated_weights)
            torch.save(model.state_dict(), f"{self.model_dir}/global_model_round_{rnd}.pth")
            print(f"[SAVE] Saved global model at round {rnd}")
            parameters = fl.common.ndarrays_to_parameters(aggregated_weights)
            self.parameters = parameters  # Save for fallback
            return self.parameters, {}
        else:
            print("[WARNING] Empty aggregated weights — returning previous.")
            return self.parameters, {} if self.parameters else (None, {})




class SyncFedAvg(FedAvg):
    def __init__(self, model_fn, model_dir="checkpoints", **kwargs):
        super().__init__(**kwargs)
        self.model_fn = model_fn
        self.model_dir = model_dir
        self.parameters = ndarrays_to_parameters(get_weights(model_fn())) #None  # Cached valid weights

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            print("[WARNING] No results received. Using cached parameters.")
            return self.parameters, {} if self.parameters else (None, {})

        weights_list = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        if not weights_list or any(len(w) == 0 for w in weights_list):
            print("[ERROR] Received empty weights from one or more clients.")
            return self.parameters, {} if self.parameters else (None, {})

        num_layers = len(weights_list[0])
        aggregated_weights = []

        for layer_idx in range(num_layers):
            layer_weights = [weights[layer_idx] for weights in weights_list]
            flat_layer_weights = [w.flatten() for w in layer_weights]

            if len(flat_layer_weights) > 1:
                delta_t = np.mean(flat_layer_weights, axis=0) - flat_layer_weights[0]
                delta_w_arr = [w - flat_layer_weights[0] for w in flat_layer_weights]

                phase_arr = calc_phase(delta_t, delta_w_arr)
                sync_weights = calc_sync_weights(phase_arr)
                sync_weights = np.array(sync_weights)
                # sync_weights = sync_weights / (sync_weights.sum() + 1e-8)

                for i in range(len(sync_weights)):
                    print(f"[SYNC WEIGHT] Client {i}: {sync_weights[i]:.4f}")
            else:
                sync_weights = [1.0]

            # Aggregate this layer
            # layer_weights[i] + sync_weights[i]*
            weighted_layer = sum(layer_weights[i] * sync_weights[i] for i in range(len(layer_weights)))
            aggregated_weights.append(weighted_layer)

        # Convert and store parameters
        parameters = fl.common.ndarrays_to_parameters(aggregated_weights)
        # print([f"DEBUG:{parameters} "])
        self.parameters = parameters  # Cache for fallback

        # Save model
        if aggregated_weights:
            model = self.model_fn()
            set_weights(model, aggregated_weights)
            os.makedirs(self.model_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{self.model_dir}/global_model_round_{rnd}.pth")
            print(f"[SAVE] Saved global model at round {rnd}")

        return parameters, {}
    