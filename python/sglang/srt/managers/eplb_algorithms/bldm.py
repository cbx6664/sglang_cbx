from typing import Tuple
import torch


def _bldm_greedy_balance(
    expert_loads: torch.Tensor, num_gpus: int
) -> torch.Tensor:
    """
    BLDM greedy balancing algorithm.
    
    Args:
        expert_loads: [num_experts] - total load for each expert across all layers
        num_gpus: number of GPUs
    
    Returns:
        assignment: [num_experts] - GPU assignment for each expert
    """
    num_experts = expert_loads.size(0)
    device = expert_loads.device
    
    # Sort experts by load (descending)
    sorted_loads, sorted_indices = torch.sort(expert_loads, descending=True)
    
    # Initialize GPU loads and assignments
    gpu_loads = torch.zeros(num_gpus, device=device)
    assignment = torch.zeros(num_experts, dtype=torch.int64, device=device)
    
    # Greedy assignment: assign each expert to the GPU with minimum current load
    for i, expert_idx in enumerate(sorted_indices):
        expert_load = sorted_loads[i]
        
        # Find GPU with minimum load
        min_gpu = torch.argmin(gpu_loads).item()
        
        # Assign expert to this GPU
        assignment[expert_idx] = min_gpu
        gpu_loads[min_gpu] += expert_load
    
    return assignment


def _create_physical_to_logical_map(
    gpu_assignment: torch.Tensor, num_gpus: int
) -> torch.Tensor:
    """
    Create physical-to-logical mapping from GPU assignments.
    
    Args:
        gpu_assignment: [num_experts] - GPU ID for each expert
        num_gpus: number of GPUs
    
    Returns:
        phy2log: [num_experts] - physical-to-logical mapping
    """
    num_experts = gpu_assignment.size(0)
    device = gpu_assignment.device
    
    # Group experts by GPU
    experts_per_gpu = num_experts // num_gpus
    phy2log = torch.zeros(num_experts, dtype=torch.int64, device=device)
    
    physical_idx = 0
    for gpu_id in range(num_gpus):
        # Find experts assigned to this GPU
        gpu_experts = torch.where(gpu_assignment == gpu_id)[0]
        
        # Assign physical indices
        for expert_idx in gpu_experts:
            phy2log[physical_idx] = expert_idx
            physical_idx += 1
    
    return phy2log


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BLDM-based expert rebalancing algorithm (simplified, no replication).
    
    Args:
        tokens_per_expert: [num_layers, num_logical_experts] - token counts per expert
        num_physical_experts: total number of physical experts (must equal num_logical_experts)
        num_gpus: number of GPUs
    
    Returns:
        physical_to_logical_map: [num_layers, num_physical_experts] - mapping from physical to logical experts
        logical_to_physical_map: [num_layers, num_logical_experts, 1] - mapping from logical to physical experts  
        expert_count: [num_layers, num_logical_experts] - number of replicas per logical expert (always 1)
    """
    device = tokens_per_expert.device
    
    # Handle input validation
    if tokens_per_expert.dim() == 1:
        weight = tokens_per_expert.unsqueeze(0)
        num_layers = 1
        num_logical_experts = tokens_per_expert.size(0)
    elif tokens_per_expert.dim() == 2:
        weight = tokens_per_expert
        num_layers, num_logical_experts = weight.shape
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {tokens_per_expert.dim()}D tensor")
    
    # Validate no replication assumption
    if num_physical_experts != num_logical_experts:
        raise NotImplementedError(
            f"BLDM assumes no expert replication. "
            f"Got num_physical_experts={num_physical_experts}, num_logical_experts={num_logical_experts}"
        )
    
    # Validate divisibility
    if num_logical_experts % num_gpus != 0:
        raise ValueError(
            f"Number of experts ({num_logical_experts}) must be divisible by number of GPUs ({num_gpus})"
        )
    
    # Calculate total load across all layers for each expert
    total_expert_loads = weight.sum(dim=0)  # [num_logical_experts]
    
    # Apply BLDM greedy balancing
    gpu_assignment = _bldm_greedy_balance(total_expert_loads, num_gpus)
    
    # Create physical-to-logical mapping
    phy2log = _create_physical_to_logical_map(gpu_assignment, num_gpus)
    
    # Expand to all layers (same mapping for all layers)
    physical_to_logical_map = phy2log.unsqueeze(0).expand(num_layers, -1)
    
    # Create logical-to-physical mapping (no replication, so each logical maps to exactly one physical)
    logical_to_physical_map = torch.full(
        (num_layers, num_logical_experts, 1),
        -1,
        dtype=torch.int64,
        device=device
    )
    
    # Fill logical-to-physical mapping
    for physical_idx, logical_idx in enumerate(phy2log):
        logical_to_physical_map[:, logical_idx, 0] = physical_idx
    
    # Expert count (always 1 since no replication)
    expert_count = torch.ones((num_layers, num_logical_experts), dtype=torch.int64, device=device)
    
    return physical_to_logical_map, logical_to_physical_map, expert_count


__all__ = ["rebalance_experts"]
