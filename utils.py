# utils.py
import numpy as np

def normalize_to_unit_range(weights):
    min_w = min(weights)
    max_w = max(weights)
    if max_w - min_w < 1e-8:
        return [1.0 for _ in weights]
    return [(w - min_w) / (max_w - min_w) for w in weights]

def softmax(weights):
    weights = np.array(weights)
    max_w = np.max(weights)  # for numerical stability
    exp_weights = np.exp(weights - max_w)
    return (exp_weights / (np.sum(exp_weights) + 1e-8)).tolist()