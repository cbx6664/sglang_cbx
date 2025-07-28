from typing import Tuple
import torch


def _bldm_greedy_balance_vectorized(
    expert_loads: torch.Tensor, num_gpus: int
) -> torch.Tensor:
    """Vectorized BLDM greedy balancing for all layers simultaneously."""
    num_layers, num_experts = expert_loads.shape
    device = expert_loads.device
    
    # Sort experts by load (descending) for all layers
    sorted_loads, sorted_indices = torch.sort(expert_loads, dim=-1, descending=True)
    
    # Initialize assignments and GPU loads
    assignment = torch.zeros(num_layers, num_experts, dtype=torch.int64, device=device)
    gpu_loads = torch.zeros(num_layers, num_gpus, device=device)
    
    # Process all experts across all layers in parallel
    for expert_pos in range(num_experts):
        expert_indices = sorted_indices[:, expert_pos]
        expert_load_values = sorted_loads[:, expert_pos]
        
        # Find GPU with minimum load for each layer
        min_gpu_indices = torch.argmin(gpu_loads, dim=-1)
        
        # Assign experts to GPUs using advanced indexing
        layer_indices = torch.arange(num_layers, device=device)
        assignment[layer_indices, expert_indices] = min_gpu_indices
        
        # Update GPU loads
        gpu_loads[layer_indices, min_gpu_indices] += expert_load_values
    
    return assignment


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BLDM-based expert rebalancing algorithm.
    
    Args:
        tokens_per_expert: [num_layers, num_logical_experts] - token counts per expert
        num_physical_experts: total number of physical experts
        num_gpus: number of GPUs
    
    Returns:
        physical_to_logical_map: [num_layers, num_physical_experts] - mapping from physical to logical experts
        logical_to_physical_map: [num_layers, num_logical_experts, 1] - mapping from logical to physical experts  
        expert_count: [num_layers, num_logical_experts] - number of replicas per logical expert (always 1)
    """
    device = tokens_per_expert.device
    
    # Input validation and normalization
    if tokens_per_expert.dim() == 1:
        weight = tokens_per_expert.unsqueeze(0)
        num_layers = 1
        num_logical_experts = tokens_per_expert.size(0)
    elif tokens_per_expert.dim() == 2:
        weight = tokens_per_expert
        num_layers, num_logical_experts = weight.shape
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {tokens_per_expert.dim()}D tensor")
    
    # Validation
    if num_physical_experts != num_logical_experts:
        raise NotImplementedError(
            f"BLDM assumes no expert replication. "
            f"Got num_physical_experts={num_physical_experts}, num_logical_experts={num_logical_experts}"
        )
    
    if num_logical_experts % num_gpus != 0:
        raise ValueError(
            f"Number of experts ({num_logical_experts}) must be divisible by number of GPUs ({num_gpus})"
        )
    
    # Vectorized GPU assignment
    gpu_assignment = _bldm_greedy_balance_vectorized(weight, num_gpus)
    
    # Create physical-to-logical mapping
    _, physical_to_logical_map = torch.sort(gpu_assignment, dim=-1)
    
    # Create logical-to-physical mapping
    logical_to_physical_map = torch.full(
        (num_layers, num_logical_experts, 1), -1, dtype=torch.int64, device=device
    )
    
    # Fill logical-to-physical mapping using advanced indexing
    layer_indices = torch.arange(num_layers, device=device).unsqueeze(1)
    physical_indices = torch.arange(num_logical_experts, device=device).unsqueeze(0)
    logical_to_physical_map[layer_indices, physical_to_logical_map, 0] = physical_indices
    
    # Expert count (always 1 since no replication)
    expert_count = torch.ones((num_layers, num_logical_experts), dtype=torch.int64, device=device)
    
    return physical_to_logical_map, logical_to_physical_map, expert_count


__all__ = ["rebalance_experts"]
