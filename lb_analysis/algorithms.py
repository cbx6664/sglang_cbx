"""
Load balancing algorithms for expert parallelism.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import torch


class LoadBalanceAlgorithm(ABC):
    """Base class for load balancing algorithms."""
    
    @abstractmethod
    def run(self, weights: torch.Tensor, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the load balancing algorithm.
        
        Args:
            weights: Original expert weights [num_layers, num_experts]
            config: Algorithm configuration
            
        Returns:
            Dictionary containing algorithm results including phy2log mapping
        """
        pass


class EPLBAlgorithm(LoadBalanceAlgorithm):
    """Expert Parallelism Load Balancer (EPLB) algorithm."""
    
    def run(self, weights: torch.Tensor, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run EPLB algorithm."""
        phy2log, log2phy, logcnt = rebalance_experts(
            weight=weights,
            num_replicas=config["num_replicas"],
            num_groups=config.get("num_groups", 1),
            num_nodes=config.get("num_nodes", 1),
            num_gpus=config["num_gpus"],
        )
        
        return {
            'phy2log': phy2log,
            'log2phy': log2phy,
            'logcnt': logcnt
        }


class BLDMAlgorithm(LoadBalanceAlgorithm):
    """BLDM (Balanced Load Distribution Method) algorithm implementation."""
    
    def run(self, weights: torch.Tensor, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run BLDM algorithm."""
        num_layers, num_experts = weights.shape
        num_gpus = config["num_gpus"]
        
        # Apply BLDM algorithm layer by layer
        all_phy2log = []
        
        for layer_idx in range(num_layers):
            hotness = weights[layer_idx]  # [num_experts]
            
            # Run BLDM balancing for this layer
            sequential_solution = self._balance_layer(hotness, num_gpus)
            all_phy2log.append(sequential_solution)
        
        # Stack all layers into phy2log tensor
        phy2log = torch.stack(all_phy2log, dim=0)  # [num_layers, num_experts]
        
        return {
            'phy2log': phy2log,
            'log2phy': None,
            'logcnt': None
        }
    
    def _balance_layer(self, hotness: torch.Tensor, num_gpus: int) -> torch.Tensor:
        """
        Apply BLDM algorithm to balance one layer.
        
        Args:
            hotness: Expert hotness for one layer [num_experts]
            num_gpus: Number of GPUs
            
        Returns:
            Sequential solution (phy2log mapping for this layer)
        """
        num_experts = hotness.shape[0]
        assert num_experts % num_gpus == 0, "Experts cannot be evenly divided among GPUs"
        
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
            self._bldm_step(buckets)
        
        # Retrieve solution
        solution = buckets[0]
        sequential_solution = []
        for partition in solution:
            for _, expert in partition:
                sequential_solution.append(expert)
        
        return torch.tensor(sequential_solution, dtype=torch.int64)
    
    @staticmethod
    def _argsort(seq):
        """Efficient argsort implementation."""
        return sorted(range(len(seq)), key=seq.__getitem__)
    
    def _bldm_step(self, buckets):
        """
        One step of the BLDM algorithm.
        
        Args:
            buckets: List of partitions, each containing sublists of (hotness, expert_id) tuples
        """
        d_diff = []
        sums = []

        for partition in buckets:
            local_sums = [sum(elem[0] for elem in subset) for subset in partition]
            sorted_sums = self._argsort(local_sums)
            sums.append(sorted_sums)
            d_diff.append(local_sums[sorted_sums[-1]] - local_sums[sorted_sums[0]])
        
        sorted_d_diff = self._argsort(d_diff)
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


class SimpleReplicationAlgorithm(LoadBalanceAlgorithm):
    """Simple Replication algorithm - replicate hottest experts to balance load."""
    
    def run(self, weights: torch.Tensor, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run Simple Replication algorithm."""
        num_layers, num_logical_experts = weights.shape
        num_gpus = config["num_gpus"]
        extra_experts = config.get("extra_experts", 0)
        
        # Apply simple replication layer by layer
        all_phy2log = []
        
        for layer_idx in range(num_layers):
            layer_weights = weights[layer_idx]  # [num_logical_experts]
            
            # Apply replication balancing for this layer following notebook implementation
            phy2log = self._balance_layer(layer_weights, num_logical_experts, num_gpus, extra_experts)
            all_phy2log.append(phy2log)
        
        # Stack all layers into phy2log tensor
        phy2log = torch.stack(all_phy2log, dim=0)  # [num_layers, num_logical_experts]
        
        return {
            'phy2log': phy2log,
            'log2phy': None,
            'logcnt': None
        }
    
    def _balance_layer(self, hotness: torch.Tensor, num_experts: int, num_gpus: int, additional_experts_per_gpu: int) -> torch.Tensor:
        """
        Apply Simple Replication algorithm to balance one layer.
        This follows the exact logic from notebook's ReplicationBalancer.
        
        Args:
            hotness: Expert hotness for one layer [num_experts]
            num_experts: Total number of experts
            num_gpus: Number of GPUs
            additional_experts_per_gpu: Number of hottest experts to replicate
            
        Returns:
            Sequential solution (phy2log mapping for this layer)
        """
        # Calculate experts per GPU
        nb_experts_per_gpu = num_experts // num_gpus
        
        # Find hottest experts to replicate (following notebook implementation)
        hottest_experts = torch.argsort(hotness).flip(dims=[0])[:additional_experts_per_gpu]
        
        # Create assignment matrix: [num_gpus, num_experts]
        ones = torch.ones(nb_experts_per_gpu)
        assignment = torch.zeros(size=(num_gpus, num_experts), dtype=torch.int64)
        
        # Step 1: Assign original experts to their home GPUs
        for gpu in range(num_gpus):
            assignment[gpu][
                gpu * nb_experts_per_gpu : (gpu + 1) * nb_experts_per_gpu
            ] = ones
        
        # Step 2: Replicate hottest experts to all other GPUs (following notebook logic)
        for hottest_expert in hottest_experts:
            for gpu in range(num_gpus):
                # We are not on the original GPU the expert has been assigned to
                if gpu != hottest_expert // nb_experts_per_gpu:
                    assignment[gpu][hottest_expert] = 1
        
        # Step 3: Convert assignment matrix to sequential solution
        sequential_solution = []
        for gpu in range(num_gpus):
            # Get all experts assigned to this GPU
            gpu_experts = torch.where(assignment[gpu] == 1)[0]
            sequential_solution.extend(gpu_experts.tolist())
        
        return torch.tensor(sequential_solution, dtype=torch.int64)


# EPLB algorithm implementation (extracted from your original code)
def balanced_packing(
    weight: torch.Tensor, num_packs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()  # O(n log n)
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (i for i in range(num_packs) if pack_items[i] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
                perm.shape
            ),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
        ).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy
    )  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts, must be a multiple of `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(
            num_layers, -1
        ),
    )
    return phy2log, log2phy, logcnt 