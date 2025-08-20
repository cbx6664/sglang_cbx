import json
import numpy as np
from ortools.sat.python import cp_model
import argparse
import random
import copy
import time
import os
import torch


class ILPSolver:
    def __init__(self, config_file):
        self.solver = cp_model.CpSolver()
        self.model = cp_model.CpModel()
        self.parse_config(config_file)

        self.flat_expert_load = [0] * self.num_experts
        for s in self.egress_demand_values:
            for k in range(len(s)):
                self.flat_expert_load[k] += s[k]

        self.total_expert_load = sum(self.flat_expert_load)

    def parse_config(self, config_file):
        def load_config(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
            return config
        
        config = load_config(config_file)
        self.num_sources = config['num_sources']
        self.num_expert_nodes = config['num_expert_nodes']
        self.num_experts = config['num_experts']
        self.max_num_experts_per_node = config['max_num_experts_per_node']
        self.num_redundant_experts = config['num_redundant_experts']

        self.per_token_cost = config['per_token_cost']
        self.expert_size = config['expert_size']
        self.expert_move_oh = config['expert_move_oh']
        self.network_bw = config['network_bw']

        self.expert_move_cost = self.expert_size / self.network_bw

        self.iteration_oh = config.get('iteration_oh', 0)
        self.num_iterations = config.get('num_iterations', 1)
        self.egress_demand_values = self.load_token_distribution(config["token_distribution_file"], config["expert_layer"], config['token_discount'])
        self.initial_expert_mapping = self.load_initial_expert_mapping(config["expert_mapping_file"])

        self.heuristic_max_moves = config.get('heuristic_max_moves', 10)

        assert self.num_expert_nodes * self.max_num_experts_per_node >= self.num_experts
        assert len(self.initial_expert_mapping) == self.num_expert_nodes and len(self.initial_expert_mapping[0]) == self.num_experts, f"Mismatch between num experts {self.num_experts} and initial expert mapping dimensions: {len(self.initial_expert_mapping[0])}"
        assert len(self.egress_demand_values) == self.num_sources and len(self.egress_demand_values[0]) == self.num_experts

        # Set solver parameters
        try:
            if config["solver_cutoff_time"] is not None:
                self.solver.parameters.max_time_in_seconds = config["solver_cutoff_time"]
            self.solver.parameters.num_workers = config["solver_num_threads"]
            self.solver.parameters.log_search_progress = config["solver_log_progress"]
        except Exception:
            pass

    def load_token_distribution(self, file_path, layer, token_discount):
        token_distribution = [0] * self.num_experts
        # Check file extension to determine format
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            with open(file_path, 'r') as f:
                for line in f:
                    current_tokens = line.strip().split(',')
                    for i in range(len(current_tokens)):
                        token_distribution[i] += int(current_tokens[i])
        elif file_extension == '.pt':
            data = torch.load(file_path, map_location=torch.device('cpu'))
            data = data['logical_count']
            layer_data = data[:, layer, :]
            print(layer_data.shape[1])
            for line in range(layer_data.shape[0]):
                for i in range(layer_data.shape[1]):
                    token_distribution[i] += int(layer_data[line, i])
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}. Must be .csv or .pt")
        
        for i in range(len(token_distribution)):
            token_distribution[i] = int(token_distribution[i] * token_discount)
        
        return [token_distribution]

    def load_initial_expert_mapping(self, initial_expert_mapping_file):
        with open(initial_expert_mapping_file, "r") as f:
            expert_mapping = json.load(f)
        return expert_mapping['mapping']

    def get_number_moves(self, old_placement, new_placement):
        number_of_moves_per_node = [
            sum([(1 - old_placement[(w, k)]) * int(self.solver.Value(new_placement[(w, k)]))
                for k in range(self.num_experts)])
            for w in range(self.num_expert_nodes)
        ]
        return sum(number_of_moves_per_node)

    def get_max_moves(self, old_placement, new_placement):
        max_moves_per_node = [
            sum([(1 - old_placement[(w, k)]) * int(self.solver.Value(new_placement[(w, k)]))
                for k in range(self.num_experts)])
            for w in range(self.num_expert_nodes)
        ]
        return max(max_moves_per_node)

    def compute_gpu_load(self, ingress_demand):
        final_gpu_loads = [
            sum([self.solver.Value(ingress_demand[(w, k)])
                for k in range(self.num_experts)])
            for w in range(self.num_expert_nodes)
        ]
        return final_gpu_loads
    
    def compute_gpu_load_for_mapping(self, expert_mapping):
        res = [0] * self.num_expert_nodes
        for w in range(self.num_expert_nodes):
            for k in range(self.num_experts):
                res[w] += self.flat_expert_load[k] if expert_mapping[w][k] == 1 else 0
        return res
    
    def randomize_initial_expert_mapping(self):
        new_expert_mapping = [
            [0 for _ in range(self.num_experts)] for _ in range(self.num_expert_nodes)]

        # Guarantee that all experts are assigned to at least one node
        available_nodes = [i for i in range(self.num_expert_nodes)]
        for e in range(self.num_experts):
            node = random.choice(available_nodes)
            new_expert_mapping[node][e] = 1
            if sum(new_expert_mapping[node]) == self.max_num_experts_per_node:
                available_nodes.remove(node)

        # Assign missing experts
        for node in range(self.num_expert_nodes):
            zeros_indices = [idx for idx in range(
                self.num_experts) if new_expert_mapping[node][idx] == 0]
            missing_number_of_experts = sum(
                new_expert_mapping[node]) - self.max_num_experts_per_node

            for i in range(missing_number_of_experts):
                new_expert_mapping[node][zeros_indices[i]] = 1

        self.initial_expert_mapping = new_expert_mapping

    def solve(self):
        total_load = sum(self.egress_demand_values[i][j] for i in range(
            self.num_sources) for j in range(self.num_experts))
        total_demand_per_k = [0] * self.num_experts
        for k in range(self.num_experts):
            total_demand_per_k[k] = sum(
                self.egress_demand_values[i][k] for i in range(self.num_sources))

        # Input variables:
        # Egress demand from all sources to all experts
        egress_demand = {(i, j): self.egress_demand_values[i][j] for i in range(
            self.num_sources) for j in range(self.num_experts)}
        # Old expert placement
        old_placement = {(i, j): self.initial_expert_mapping[i][j] for i in range(
            self.num_expert_nodes) for j in range(self.num_experts)}

        # Problem variables
        # New expert placement
        new_placement = {}
        for i in range(self.num_expert_nodes):
            for j in range(self.num_experts):
                new_placement[(i, j)] = self.model.NewBoolVar(f'p_{i}_{j}')

        # Ingress demand at all destinations for all experts
        ingress_demand = {}
        for i in range(self.num_expert_nodes):
            for k in range(self.num_experts):
                ingress_demand[(i, k)] = self.model.NewIntVar(
                    0, int(total_demand_per_k[k]), f't_{i}_{k}')

        # Auxiliary variable for load-placement linearization
        U = {
            (w, k): int(sum(egress_demand[(i, k)] for i in range(self.num_sources)))
            for w in range(self.num_expert_nodes)
            for k in range(self.num_experts)
        }

        # Flow conservation constraints
        for k in range(self.num_experts):
            self.model.Add(
                sum(int(egress_demand[(i, k)]) for i in range(self.num_sources)) ==
                sum(ingress_demand[(w, k)]
                    for w in range(self.num_expert_nodes))
            )

        for w in range(self.num_expert_nodes):
            for k in range(self.num_experts):
                self.model.Add(
                    ingress_demand[(w, k)] <= new_placement[(w, k)] * U[(w, k)])

        # Maximum experts per node constraint
        for w in range(self.num_expert_nodes):
            self.model.Add(sum(new_placement[(w, k)] for k in range(
                self.num_experts)) <= self.max_num_experts_per_node)

        # Total reallocation cost: 0->1 Hamming distance
        num_reallocations = [[] for _ in range(self.num_expert_nodes)]
        for w in range(self.num_expert_nodes):
            for k in range(self.num_experts):
                if old_placement[(w, k)] == 0:
                    num_reallocations[w].append(
                        new_placement[(w, k)]*(1 - old_placement[(w, k)]))
        max_reallocations = [sum(num_reallocations[w])
                             for w in range(self.num_expert_nodes)]
        max_reallocation = self.model.NewIntVar(
            0, int(self.max_num_experts_per_node), 'max_reallocation')
        for reallocation in max_reallocations:
            self.model.Add(max_reallocation >= reallocation)

        # Token serving cost: processing time at the most congested GPU
        load_vars = []
        for w in range(self.num_expert_nodes):
            load_var = self.model.NewIntVar(0, int(total_load), f'load_{w}')
            self.model.Add(load_var == sum(
                ingress_demand[(w, k)] for k in range(self.num_experts)))
            load_vars.append(load_var)

        # Auxiliary variable for max load
        max_load = self.model.NewIntVar(0, int(total_load), 'max_load')
        for load in load_vars:
            self.model.Add(max_load >= load)

        # Create a boolean variable that is 1 if and only if max_reallocation is 0
        max_reallocation_is_zero = self.model.NewBoolVar('max_reallocation_is_zero')
        self.model.Add(max_reallocation == 0).OnlyEnforceIf(max_reallocation_is_zero)
        self.model.Add(max_reallocation > 0).OnlyEnforceIf(max_reallocation_is_zero.Not())

        # Expert move overhead only applied if there are any moves
        expert_move_oh = self.model.NewIntVar(0, 1, 'expert_move_oh')
        self.model.Add(expert_move_oh == 0).OnlyEnforceIf(max_reallocation_is_zero)
        self.model.Add(expert_move_oh == 1).OnlyEnforceIf(max_reallocation_is_zero.Not())

        # Objective: minimize the token cost + reallocation cost
        self.model.Minimize(self.num_iterations * self.iteration_oh + self.per_token_cost * max_load
                            + self.expert_move_cost * max_reallocation + expert_move_oh * self.expert_move_oh)

        # Solve
        start_time = time.time()
        self.solver.Solve(self.model)

        # Process results
        total_demand = sum([self.egress_demand_values[i][j] for i in range(
            self.num_sources) for j in range(self.num_experts)])
        normalized_final_loads = [
            load / total_demand for load in self.compute_gpu_load(ingress_demand)]

        # Print new placement information
        print("New Expert Placement:")
        for w in range(self.num_expert_nodes):
            node_placement = []
            for k in range(self.num_experts):
                node_placement.append(int(self.solver.Value(new_placement[(w, k)])))

        return {
            "initial_cost": max(self.compute_gpu_load_for_mapping(self.initial_expert_mapping)) * self.per_token_cost + self.num_iterations * self.iteration_oh,
            "final_cost": self.solver.ObjectiveValue(),
            "expert_cost": self.expert_move_cost * self.solver.Value(max_reallocation) + self.solver.Value(expert_move_oh) * self.expert_move_oh,
            "number_of_moves": self.get_number_moves(old_placement, new_placement),
            "max_moves": self.get_max_moves(old_placement, new_placement),
            "normalized_load_mean": float(np.mean(normalized_final_loads) / max(normalized_final_loads)),
            "normalized_load_std": float(np.std(normalized_final_loads) / np.mean(normalized_final_loads)),
            "final_gpu_loads": self.compute_gpu_load(ingress_demand),
            "execution_time": time.time() - start_time
        }


class BestSwapHeuristic(ILPSolver):
    def __init__(self, config_file):
        self.parse_config(config_file)
        
        self.flat_expert_load = [0] * self.num_experts
        for s in self.egress_demand_values:
            for k in range(len(s)):
                self.flat_expert_load[k] += s[k]
        
        self.total_expert_load = sum(self.flat_expert_load)


    def solve(self):
        def _max_balance_swap(v, w, expert_mapping):
            # Calculate current loads before swapping
            current_load_v = sum(self.flat_expert_load[k] for k in range(self.num_experts)
                                 if expert_mapping[v][k] == 1) / self.total_expert_load
            current_load_w = sum(self.flat_expert_load[k] for k in range(self.num_experts)
                                 if expert_mapping[w][k] == 1) / self.total_expert_load

            # Get experts on each node
            experts_v = [(k, self.flat_expert_load[k]/self.total_expert_load)
                         for k in range(self.num_experts) if expert_mapping[v][k] == 1]
            experts_w = [(k, self.flat_expert_load[k]/self.total_expert_load)
                         for k in range(self.num_experts) if expert_mapping[w][k] == 1]

            best_error, best_exchange = float('inf'), None
            
            # TODO: O(N^2) can do better?
            # Can sort the arrays and use binary search to bring down to O(nlog(n))?
            for e_v, load_v_norm in experts_v:
                for e_w, load_w_norm in experts_w:
                    # Calculate what the loads would be after swapping
                    new_load_v = current_load_v - load_v_norm + load_w_norm
                    new_load_w = current_load_w - load_w_norm + load_v_norm

                    # Calculate the balance error as average error in respect to the target load.
                    # TODO: Can use a better metric?
                    error = (abs(new_load_v - target_load) +
                             abs(new_load_w - target_load)) / 2
                    if error < best_error:
                        best_error = error
                        best_exchange = (e_v, e_w)

            # Perform the best swap found
            if best_exchange:
                e_v, e_w = best_exchange
                expert_mapping[v][e_v], expert_mapping[w][e_v] = 0, 1
                expert_mapping[w][e_w], expert_mapping[v][e_w] = 0, 1

            return best_exchange is not None

        target_load = 1.0 / self.num_expert_nodes

        gpu_load = self.compute_gpu_load_for_mapping(self.initial_expert_mapping)
        res = {
            "initial_cost": max(self.compute_gpu_load_for_mapping(self.initial_expert_mapping)) * self.per_token_cost + self.num_iterations * self.iteration_oh,
            "final_cost": max(self.compute_gpu_load_for_mapping(self.initial_expert_mapping)) * self.per_token_cost + self.num_iterations * self.iteration_oh,
            "expert_cost": 0,
            "number_of_moves": 0,
            "max_moves": 0,
            "normalized_load_mean": float(np.mean(gpu_load) / max(gpu_load)),
            "normalized_load_std": float(np.std(gpu_load) / np.mean(gpu_load)),
            "final_gpu_loads": gpu_load
        }

        total_moves = 0
        new_expert_mapping = copy.deepcopy(self.initial_expert_mapping)
        start_time = time.time()
        for moves in range(1, self.heuristic_max_moves + 1):
            # Compute the excess/deficit demand for each GPU and sort it in a decreasing order
            norm_gpu_load = [
                it / self.total_expert_load for it in self.compute_gpu_load_for_mapping(new_expert_mapping)]
            sorted_excess_demands = sorted(
                [(load - target_load, idx) for idx, load in enumerate(norm_gpu_load)], key=lambda e: e[0], reverse=True)

            # Try to balance the two extremes of the sorted_excess_demands. I.e., balance the most with the least loaded GPU
            for i in range(len(sorted_excess_demands) // 2):
                v, w = sorted_excess_demands[i][1], sorted_excess_demands[-(
                    i+1)][1]
                if _max_balance_swap(v, w, new_expert_mapping):
                    total_moves += 2
            # Update cost and check if a better solution has been found
            cost = moves * self.expert_move_cost + self.expert_move_oh + self.per_token_cost * max(self.compute_gpu_load_for_mapping(new_expert_mapping)) + self.num_iterations * self.iteration_oh
            if cost < res["final_cost"]:
                gpu_load = self.compute_gpu_load_for_mapping(new_expert_mapping)
                res["final_cost"] = float(cost)
                res["expert_cost"] = self.expert_move_cost * moves + self.expert_move_oh
                res["number_of_moves"] = total_moves
                res["max_moves"] = moves
                res["normalized_load_mean"] = float(
                    np.mean(gpu_load) / max(gpu_load))
                res["normalized_load_std"] = float(
                    np.std(gpu_load) / np.mean(gpu_load))
                res["final_gpu_loads"] = gpu_load
            else:
                # No improvement anymore: exit earlier
                break


        total_distance = 0
        max_distance_per_node = 0
        
        for w in range(self.num_expert_nodes):
            node_distance = 0
            for k in range(self.num_experts):
                if new_expert_mapping[w][k] == 1 and self.initial_expert_mapping[w][k] == 0:
                    node_distance += 1
                if new_expert_mapping[w][k] != self.initial_expert_mapping[w][k]:
                    total_distance += 1
            max_distance_per_node = max(max_distance_per_node, node_distance)
        
        res["number_of_moves"] = total_distance
        res["max_moves"] = max_distance_per_node
        res['execution_time'] = time.time() - start_time

        return res

  
def parse_args():
    parser = argparse.ArgumentParser(description='Expert Placement Solver')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--randomize-placement', action='store_true',
                        help='Randomize initial expert placement if specified')
    parser.add_argument('--algorithm', default='heuristic',
                        help='Algorithm to use for the solution')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.algorithm == 'heuristic':
        solver = BestSwapHeuristic(args.config)
    elif args.algorithm == 'ilp':
        solver = ILPSolver(args.config)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")


    if args.randomize_placement:
        solver.randomize_initial_expert_mapping()

    result = solver.solve()
    print(result)
