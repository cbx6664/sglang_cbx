"""
BLDM (Balanced Load Distribution Method) algorithm implementation for EPLB.
"""

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: Optional[int],
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BLDM algorithm for expert parallelism load balancing.
    
    Args:
        tokens_per_expert: [num_steps, num_layers, num_logical_experts] or [num_layers, num_logical_experts]
        num_physical_experts: Total number of physical experts (should equal num_logical_experts for BLDM)
        num_local_physical_experts: Number of physical experts per GPU
        num_groups: Number of expert groups (not used in BLDM)
        num_nodes: Number of nodes (not used in BLDM)
    
    Returns:
        physical_to_logical_map: [num_layers, num_physical_experts]
        logical_to_physical_map: [num_layers, num_logical_experts, 1] 
        expert_count: [num_layers, num_logical_experts] (all ones for BLDM)
    """
    logger.info(f"[EPLB] Rebalancing experts with BLDM algorithm")
    # Handle both 2D and 3D input tensors
    if len(tokens_per_expert.shape) == 3:
        # Sum across steps dimension to get average expert hotness
        weights = tokens_per_expert.sum(dim=0)  # [num_layers, num_logical_experts]
    else:
        weights = tokens_per_expert  # [num_layers, num_logical_experts]
    
    num_layers, num_logical_experts = weights.shape
    num_gpus = num_physical_experts // num_local_physical_experts
    
    # BLDM is a reordering algorithm, so num_physical_experts should equal num_logical_experts
    assert num_physical_experts == num_logical_experts, \
        f"BLDM requires num_physical_experts ({num_physical_experts}) == num_logical_experts ({num_logical_experts})"
    
    assert num_logical_experts % num_gpus == 0, \
        f"Experts ({num_logical_experts}) cannot be evenly divided among GPUs ({num_gpus})"
    
    # Apply BLDM algorithm layer by layer
    all_phy2log = []
    
    for layer_idx in range(num_layers):
        hotness = weights[layer_idx]  # [num_logical_experts]
        
        # Run BLDM balancing for this layer
        sequential_solution = _balance_layer(hotness, num_gpus)
        all_phy2log.append(sequential_solution)
    
    # Stack all layers into phy2log tensor
    phy2log = torch.stack(all_phy2log, dim=0)  # [num_layers, num_logical_experts]
    
    # Create logical_to_physical_map (inverse mapping)
    log2phy = torch.full(
        (num_layers, num_logical_experts, 1),
        -1,
        dtype=torch.int64,
        device=phy2log.device,
    )
    
    # Fill in the inverse mapping
    for layer_idx in range(num_layers):
        for phy_idx, log_idx in enumerate(phy2log[layer_idx]):
            log2phy[layer_idx, log_idx, 0] = phy_idx
    
    # Expert count is always 1 for BLDM (no replication)
    expert_count = torch.ones(
        num_layers, num_logical_experts, dtype=torch.int64, device=phy2log.device
    )
    
    return phy2log, log2phy, expert_count


def _balance_layer(hotness: torch.Tensor, num_gpus: int) -> torch.Tensor:
    """
    Apply BLDM algorithm to balance one layer.
    
    Args:
        hotness: Expert hotness for one layer [num_experts]
        num_gpus: Number of GPUs
        
    Returns:
        Sequential solution (phy2log mapping for this layer)
    """
    num_experts = hotness.shape[0]
    
    # Convert to list with expert indices
    hotnesses = hotness.tolist()
    hotnesses = [(hotness_val, expert_idx) for expert_idx, hotness_val in enumerate(hotnesses)]
    hotnesses.sort(key=lambda x: x[0])  # Sort by hotness value
    
    # Initialize buckets
    buckets = []
    m = len(hotnesses) // num_gpus
    k = 0
    for i in range(m):
        local_bucket = []
        for j in range(num_gpus):
            local_bucket.append([hotnesses[k]])
            k = k + 1
        buckets.append(local_bucket)
    
    # Apply BLDM algorithm
    while len(buckets) > 1:
        _bldm_step(buckets)
    
    # Retrieve solution
    solution = buckets[0]
    sequential_solution = []
    for partition in solution:
        for _, expert in partition:
            sequential_solution.append(expert)
    
    return torch.tensor(sequential_solution, dtype=torch.int64)


def _argsort(seq):
    """Efficient argsort implementation."""
    return sorted(range(len(seq)), key=seq.__getitem__)


def _bldm_step(buckets):
    """
    One step of the BLDM algorithm.
    
    Args:
        buckets: List of partitions, each containing sublists of (hotness, expert_id) tuples
    """
    d_diff = []
    sums = []

    for partition in buckets:
        local_sums = [sum(elem[0] for elem in subset) for subset in partition]
        sorted_sums = _argsort(local_sums)
        sums.append(sorted_sums)
        d_diff.append(local_sums[sorted_sums[-1]] - local_sums[sorted_sums[0]])
    
    sorted_d_diff = _argsort(d_diff)
    p1_ind = sorted_d_diff[-1]
    p2_ind = sorted_d_diff[-2]
    p1 = buckets[p1_ind]
    p2 = buckets[p2_ind]
    
    new_partition = [
        p1[sums[p1_ind][j]] + p2[sums[p2_ind][len(sums[p2_ind]) - j - 1]] 
        for j in range(len(sums[p1_ind]))
    ]

    del buckets[p1_ind]
    if p1_ind < p2_ind:
        p2_ind = p2_ind - 1
    del buckets[p2_ind]
    buckets.append(new_partition)
