import json
import numpy as np
from ortools.sat.python import cp_model
import argparse
import random
import copy
import time
import os
import torch
from typing import Dict, Optional, Tuple
import json
import time
import copy

def load_initial_expert_mapping(initial_expert_mapping_file):
    with open(initial_expert_mapping_file, "r") as f:
        expert_mapping = json.load(f)
    return expert_mapping["mapping"]

def _load_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)

def parse_config(config_file, num_nodes, num_gpus, num_replicas, layer_expert_dist):

    config = _load_config(config_file)

    num_experts = config["num_experts"]
    per_token_cost = config["per_token_cost"]
    expert_move_oh = config["expert_move_oh"]
    expert_move_cost = config.get("expert_move_cost", -1)

    if expert_move_cost < 0:
        print("Using explicit expert size and network BW")
        expert_size = config.get("expert_size", -1)
        network_bw = config.get("network_bw", -1)
        assert expert_size > 0 and network_bw > 0, "Invalid expert size or network bandwidth"
        expert_move_cost = expert_size / network_bw

    iteration_oh = config.get("iteration_oh", 0)

    # Use the provided layer_expert_dist (as in your class version)
    egress_demand_values = layer_expert_dist

    # Use your loader for the initial mapping
    initial_expert_mapping = load_initial_expert_mapping(config["expert_mapping_file"])

    heuristic_max_moves = config.get("heuristic_max_moves", 10)
    minimize_moves = config["minimize_moves"]

    return {
        "num_experts": num_experts,
        "per_token_cost": per_token_cost,
        "expert_move_oh": expert_move_oh,
        "expert_move_cost": expert_move_cost,
        "iteration_oh": iteration_oh,
        "egress_demand_values": egress_demand_values,
        "initial_expert_mapping": initial_expert_mapping,
        "heuristic_max_moves": heuristic_max_moves,
        "minimize_moves": minimize_moves,
    }


def _compute_flat_expert_load(egress_demand_values, num_experts):
    """Flatten expert demand across sources into a single load vector."""
    flat = [0] * num_experts
    for s in egress_demand_values:
        for k in range(len(s)):
            flat[k] += s[k]
    return flat, sum(flat)


def best_swap_heuristic(config_file,
                        layer_expert_dist,
                        num_replicas,
                        num_nodes,
                        num_gpus,
                        num_iterations):
    
    egress_demand_values = layer_expert_dist

    # Parse config and pull params
    cfg = parse_config(config_file, num_nodes, num_gpus, num_replicas, layer_expert_dist)
    num_experts = cfg["num_experts"]
    per_token_cost = cfg["per_token_cost"]
    expert_move_cost = cfg["expert_move_cost"]
    expert_move_oh = cfg["expert_move_oh"]
    iteration_oh = cfg["iteration_oh"]
    heuristic_max_moves = cfg["heuristic_max_moves"]
    minimize_moves = cfg["minimize_moves"]
    initial_expert_mapping = cfg["initial_expert_mapping"]

    num_sources = num_nodes
    num_expert_nodes = num_gpus
    max_num_experts_per_node = (num_replicas) / (num_gpus * num_nodes)
    num_redundant_experts = (num_replicas - num_experts) / (num_gpus * num_nodes)
    
    
    # num_sources is always set to be 1
    if num_sources == 1:
        egress_demand_values = egress_demand_values.reshape(num_sources, num_experts)
    else:
        raise NotImplementedError(
            f"num sources always need to be 1")

    # Compute flat expert loads
    flat_expert_load, total_expert_load = _compute_flat_expert_load(egress_demand_values, num_experts)

    # --- your helper: kept signature, implemented as a closure to avoid globals ---
    def compute_gpu_load_for_mapping(expert_mapping):
        res = [0] * num_expert_nodes
        for w in range(num_expert_nodes):
            for k in range(num_experts):
                res[w] += flat_expert_load[k] if expert_mapping[w][k] == 1 else 0
        return res

    # inner swap helper
    def _max_balance_swap(v, w, expert_mapping, target_load):
        current_load_v = sum(flat_expert_load[k] for k in range(num_experts)
                             if expert_mapping[v][k] == 1) / total_expert_load
        current_load_w = sum(flat_expert_load[k] for k in range(num_experts)
                             if expert_mapping[w][k] == 1) / total_expert_load

        experts_v = [(k, flat_expert_load[k] / total_expert_load)
                     for k in range(num_experts) if expert_mapping[v][k] == 1]
        experts_w = [(k, flat_expert_load[k] / total_expert_load)
                     for k in range(num_experts) if expert_mapping[w][k] == 1]

        best_error, best_exchange = float("inf"), None

        for e_v, load_v_norm in experts_v:
            for e_w, load_w_norm in experts_w:
                new_load_v = current_load_v - load_v_norm + load_w_norm
                new_load_w = current_load_w - load_w_norm + load_v_norm
                error = (abs(new_load_v - target_load) + abs(new_load_w - target_load)) / 2
                if error < best_error:
                    best_error = error
                    best_exchange = (e_v, e_w)

        if best_exchange:
            e_v, e_w = best_exchange
            # swap the expert assignment bits
            expert_mapping[v][e_v], expert_mapping[w][e_v] = 0, 1
            expert_mapping[w][e_w], expert_mapping[v][e_w] = 0, 1

        return best_exchange is not None

    # === Main loop ===
    target_load = 1.0 / num_expert_nodes
    gpu_load = compute_gpu_load_for_mapping(initial_expert_mapping)

    res = {
        "initial_cost": max(gpu_load) * per_token_cost + num_iterations * iteration_oh,
        "final_cost": max(gpu_load) * per_token_cost + num_iterations * iteration_oh,
        "expert_cost": 0,
        "number_of_moves": 0,
        "max_moves": 0,
        "normalized_load_mean": float(np.mean(gpu_load) / (max(gpu_load) + 1e-6)),
        "normalized_load_std": float(np.std(gpu_load) / (np.mean(gpu_load) + 1e-6)),
        "final_gpu_loads": gpu_load,
    }

    total_moves = 0
    new_expert_mapping = copy.deepcopy(initial_expert_mapping)
    start_time = time.time()

    for moves in range(1, heuristic_max_moves + 1):
        norm_gpu_load = [it / total_expert_load for it in compute_gpu_load_for_mapping(new_expert_mapping)]
        sorted_excess = sorted(
            [(load - target_load, idx) for idx, load in enumerate(norm_gpu_load)],
            key=lambda e: e[0], reverse=True
        )

        # balance most-loaded with least-loaded
        for i in range(len(sorted_excess) // 2):
            v, w = sorted_excess[i][1], sorted_excess[-(i + 1)][1]
            if _max_balance_swap(v, w, new_expert_mapping, target_load):
                total_moves += 2

        # evaluate cost
        if minimize_moves:
            cost = (moves * expert_move_cost + expert_move_oh +
                    per_token_cost * max(compute_gpu_load_for_mapping(new_expert_mapping)) +
                    num_iterations * iteration_oh)
        else:
            cost = (per_token_cost * max(compute_gpu_load_for_mapping(new_expert_mapping)) +
                    num_iterations * iteration_oh)

        if cost < res["final_cost"]:
            gpu_load = compute_gpu_load_for_mapping(new_expert_mapping)
            res.update({
                "final_cost": float(cost),
                "expert_cost": expert_move_cost * moves + expert_move_oh,
                "number_of_moves": total_moves,
                "max_moves": moves,
                "normalized_load_mean": float(np.mean(gpu_load) / max(gpu_load)),
                "normalized_load_std": float(np.std(gpu_load) / np.mean(gpu_load)),
                "final_gpu_loads": gpu_load
            })
        else:
            break

    # distance metrics
    total_distance, max_distance_per_node = 0, 0
    for w in range(num_expert_nodes):
        node_distance = 0
        for k in range(num_experts):
            if new_expert_mapping[w][k] == 1 and initial_expert_mapping[w][k] == 0:
                node_distance += 1
            if new_expert_mapping[w][k] != initial_expert_mapping[w][k]:
                total_distance += 1
        max_distance_per_node = max(max_distance_per_node, node_distance)

    res.update({
        "number_of_moves": total_distance,
        "max_moves": max_distance_per_node,
        "execution_time": time.time() - start_time
    })

    # Convert mapping representation from boolean to indices
    init_expert_mapping = [[j for j, val in enumerate(row) if val] for row in initial_expert_mapping]
    new_expert_mapping = [[j for j, val in enumerate(row) if val] for row in new_expert_mapping]
    
    return new_expert_mapping


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
    num_iterations: int,
    config_file: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


    """
    Entry point for ILP load balancer.
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

    weight = weight.float().cpu()
    
    # Input validation and normalization
    if weight.dim() == 1:
        weight = weight.unsqueeze(0)
        num_layers = 1
        num_logical_experts = weight.size(0)
    elif weight.dim() == 2:
        weight = weight
        num_layers, num_logical_experts = weight.shape
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {weight.dim()}D tensor")

    if enable_hierarchical:
        raise NotImplementedError(
            f"ILP based hierarchical load balancing is not implemented. "
        )
    if num_groups > 1:
        raise NotImplementedError(
            f"ILP is not a group-limited routing algorithm."
        )
    else:
        new_placement_all_layers = []
        for layer_idx in range(num_layers):  # iterate over num_layers
            layer_expert_dist = weight[layer_idx]
            new_placement = best_swap_heuristic(config_file, layer_expert_dist, num_replicas, num_nodes, num_gpus, num_iterations)
            new_placement_all_layers.append(new_placement)

        phy2log = np.array(new_placement_all_layers)
        phy2log = phy2log.reshape(phy2log.shape[0], phy2log.shape[1]*phy2log.shape[2])
        phy2log = torch.tensor(phy2log)
        # since replication is not supported. simply creating a tensor with zeros
        phyrank = torch.zeros_like(phy2log)
        
        logcnt = torch.tensor(
            [np.unique(row, return_counts=True)[1] for row in phy2log],
            dtype=torch.int64
        )

        maxlogcnt = logcnt.max().item()

        log2phy: torch.Tensor = torch.full(
            (num_layers, num_logical_experts, maxlogcnt),
            -1,
            dtype=torch.int64,
            device=logcnt.device,
        )
        
        expert_ids, log2phy = torch.sort(phy2log, dim=-1)
        log2phy = log2phy.unsqueeze(-1)

        return phy2log, log2phy, logcnt