import numpy as np
from my_utils import normalize_to_unit_range, softmax

# from typing import Tuple
from typing import List, Tuple, Optional, Dict
def average_weights(weights_list, client_weights=None):
    """
    Compute the (optionally weighted) average of model weights.

    Args:
        weights_list (List[List[np.ndarray]]): List of weights from each client.
        client_weights (List[float], optional): Weights for each client. Should sum to 1.
                                                If None, equal weights are used.

    Returns:
        List[np.ndarray]: Weighted average of weights.
    """
    num_clients = len(weights_list)

    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients

    if len(weights_list) != len(client_weights):
        raise ValueError("Length of weights_list and client_weights must match")

    avg_weights = []
    for layer_weights in zip(*weights_list):
        layer_stack = np.stack(layer_weights)
        weighted_layer = np.tensordot(client_weights, layer_stack, axes=1)
        avg_weights.append(weighted_layer)
    return avg_weights


def fedavgm_update(
    server_weights: List[np.ndarray],
    client_weights_list: List[List[np.ndarray]],
    velocity: List[np.ndarray],
    client_weights: List[float] = None,
    server_lr: float = 1.0,
    momentum: float = 0.9,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Perform FedAvgM server update.

    Args:
        server_weights (List[np.ndarray]): Current server weights.
        client_weights_list (List[List[np.ndarray]]): Client model weights.
        velocity (List[np.ndarray]): Server momentum buffer (velocity).
        client_weights (List[float], optional): Client contribution weights.
        server_lr (float): Server learning rate (scaling for the update step).
        momentum (float): Momentum term.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: (Updated server weights, updated velocity)
    """
    # Step 1: Compute weighted average of client weights (FedAvg)
    avg_weights = average_weights(client_weights_list, client_weights)

    # Step 2: Compute update direction (delta)
    deltas = [s_w - avg_w for s_w, avg_w in zip(server_weights, avg_weights)]

    # Step 3: Update velocity (momentum buffer)
    new_velocity = [
        momentum * v + d for v, d in zip(velocity, deltas)
    ]

    # Step 4: Apply update
    new_server_weights = [
        s_w - server_lr * v for s_w, v in zip(server_weights, new_velocity)
    ]

    return new_server_weights, new_velocity



def fed_adam_update(
    global_weights,
    local_weights_list,
    m_t,
    v_t,
    beta1=0.9,
    beta2=0.999,
    eta=0.001,
    eps=1e-8,
):
    """
    Performs FedAdam aggregation step.

    Args:
        global_weights (List[np.ndarray]): Current global model weights.
        local_weights_list (List[List[np.ndarray]]): List of model weights from each client.
        m_t (List[np.ndarray]): First moment vector (momentum).
        v_t (List[np.ndarray]): Second moment vector (RMS).
        beta1 (float): Exponential decay rate for the first moment.
        beta2 (float): Exponential decay rate for the second moment.
        eta (float): Server learning rate.
        eps (float): Small constant for numerical stability.

    Returns:
        updated_global_weights: New global weights after applying FedAdam.
        m_t: Updated first moment.
        v_t: Updated second moment.
    """

    # Step 1: Compute average model delta
    delta_weights = []
    for layer_idx in range(len(global_weights)):
        avg_layer = np.mean(
            [local[layer_idx] - global_weights[layer_idx] for local in local_weights_list],
            axis=0
        )
        delta_weights.append(avg_layer)

    # Step 2: Update m_t and v_t using Adam rules
    new_m_t = []
    new_v_t = []
    updated_weights = []

    for g, m, v, d in zip(global_weights, m_t, v_t, delta_weights):
        m_new = beta1 * m + (1 - beta1) * d
        v_new = beta2 * v + (1 - beta2) * (d ** 2)

        g_new = g - eta * m_new / (np.sqrt(v_new) + eps)

        updated_weights.append(g_new)
        new_m_t.append(m_new)
        new_v_t.append(v_new)

    return updated_weights, new_m_t, new_v_t


import numpy as np
from typing import List

def calc_phase(delta_avg: np.ndarray, delta_list: List[np.ndarray]) -> List[float]:
    """Calculate cosine similarity (phase) between average direction and each client's update."""
    phase_arr = []
    norm_avg = np.linalg.norm(delta_avg) + 1e-8
    for delta in delta_list:
        norm_delta = np.linalg.norm(delta) + 1e-8
        similarity = np.inner(delta, delta_avg)
        theta = similarity / (norm_avg * norm_delta)
        phase_arr.append(theta)
    return phase_arr

def calc_sync_weights(phase_arr: List[float]) -> List[float]:
    """Compute synchronization weights based on Kuramoto-style phase difference."""
    theta_mean = np.mean(phase_arr)
    base_sin = sum(np.sin(theta_mean - theta) for theta in phase_arr) + 1e-8
    sync_weights = [np.sin(theta_mean - theta) / base_sin for theta in phase_arr]
    return sync_weights

def sync_aggregate(
    base_weights: List[np.ndarray],
    client_weights_list: List[List[np.ndarray]]
) -> List[np.ndarray]:
    """
    Synchronization-based aggregation of client models.

    Args:
        base_weights (List[np.ndarray]): Server's current model weights.
        client_weights_list (List[List[np.ndarray]]): List of weights from each client.

    Returns:
        List[np.ndarray]: Updated server model weights.
    """
    num_clients = len(client_weights_list)
    num_layers = len(base_weights)

    # Step 1: Flatten deltas for similarity computation
    flattened_deltas = []
    for weights in client_weights_list:
        delta = [w - b for w, b in zip(weights, base_weights)]
        delta_flat = np.concatenate([d.flatten() for d in delta])
        flattened_deltas.append(delta_flat)

    # Step 2: Compute average delta direction
    avg_delta_flat = np.mean(flattened_deltas, axis=0)

    # Step 3: Compute sync weights
    phase_arr = calc_phase(avg_delta_flat, flattened_deltas)
    sync_weights = calc_sync_weights(phase_arr)
    sync_weights = np.array(sync_weights)

    sync_weights = sync_weights* 0.05

    print(f"SYNC Weights:{sync_weights} ")

    # Step 4: Aggregate updates layer by layer
    updated_weights = []
    for layer_idx in range(num_layers):
        layer_stack = np.stack([client[layer_idx] for client in client_weights_list])
        weighted_layer = np.tensordot(sync_weights, layer_stack, axes=1)
        new_layer = base_weights[layer_idx] + weighted_layer
        updated_weights.append(new_layer)

    return updated_weights




def sync_aggregate_norm(
    base_weights: List[np.ndarray],
    client_weights_list: List[List[np.ndarray]]
) -> List[np.ndarray]:
    """
    Synchronization-based aggregation of client models.

    Args:
        base_weights (List[np.ndarray]): Server's current model weights.
        client_weights_list (List[List[np.ndarray]]): List of weights from each client.

    Returns:
        List[np.ndarray]: Updated server model weights.
    """
    num_clients = len(client_weights_list)
    num_layers = len(base_weights)

    # Step 1: Flatten deltas for similarity computation
    flattened_deltas = []
    for weights in client_weights_list:
        delta = [w - b for w, b in zip(weights, base_weights)]
        delta_flat = np.concatenate([d.flatten() for d in delta])
        flattened_deltas.append(delta_flat)

    # Step 2: Compute average delta direction
    avg_delta_flat = np.mean(flattened_deltas, axis=0)

    # Step 3: Compute sync weights
    phase_arr = calc_phase(avg_delta_flat, flattened_deltas)
    sync_weights = calc_sync_weights(phase_arr)
    normalized_weights = normalize_to_unit_range(sync_weights)


    print(f"SYNC Weights:{normalized_weights} ")

    # Step 4: Aggregate updates layer by layer
    updated_weights = []
    for layer_idx in range(num_layers):
        layer_stack = np.stack([client[layer_idx] for client in client_weights_list])
        weighted_layer = np.tensordot(normalized_weights, layer_stack, axes=1)
        new_layer = base_weights[layer_idx] + weighted_layer
        updated_weights.append(new_layer)

    return updated_weights




def sync_aggregate_softmax(
    base_weights: List[np.ndarray],
    client_weights_list: List[List[np.ndarray]]
) -> List[np.ndarray]:
    """
    Synchronization-based aggregation of client models.

    Args:
        base_weights (List[np.ndarray]): Server's current model weights.
        client_weights_list (List[List[np.ndarray]]): List of weights from each client.

    Returns:
        List[np.ndarray]: Updated server model weights.
    """
    num_clients = len(client_weights_list)
    num_layers = len(base_weights)

    # Step 1: Flatten deltas for similarity computation
    flattened_deltas = []
    for weights in client_weights_list:
        delta = [w - b for w, b in zip(weights, base_weights)]
        delta_flat = np.concatenate([d.flatten() for d in delta])
        flattened_deltas.append(delta_flat)

    # Step 2: Compute average delta direction
    avg_delta_flat = np.mean(flattened_deltas, axis=0)

    # Step 3: Compute sync weights
    phase_arr = calc_phase(avg_delta_flat, flattened_deltas)
    sync_weights = calc_sync_weights(phase_arr)
    normalized_weights = softmax(sync_weights)


    print(f"SYNC Weights:{normalized_weights} ")

    # Step 4: Aggregate updates layer by layer
    updated_weights = []
    for layer_idx in range(num_layers):
        layer_stack = np.stack([client[layer_idx] for client in client_weights_list])
        weighted_layer = np.tensordot(normalized_weights, layer_stack, axes=1)
        new_layer = base_weights[layer_idx] + weighted_layer
        updated_weights.append(new_layer)

    return updated_weights
