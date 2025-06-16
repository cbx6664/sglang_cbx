"""
Load Balance Analyzer for Expert Parallelism
============================================

This module provides a comprehensive framework for analyzing load balancing
algorithms (EPLB, BLDM, etc.) for expert parallelism in MoE models.
"""

import os
import json
import time
import statistics
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path

import torch
import pandas as pd
import numpy as np

# Standard package imports - use relative imports within package
from .utils import (
    load_csv_to_tensor,
    load_json_to_tensor,
    natural_sort_key,
    measure_execution_time
)
from .algorithms import EPLBAlgorithm, BLDMAlgorithm, SimpleReplicationAlgorithm
from .visualization import LoadBalanceVisualizer


class LoadBalanceAnalyzer:
    """
    Main analyzer class for expert parallelism load balancing evaluation.
    
    This class provides a unified interface for:
    - Loading original token distributions
    - Running different load balancing algorithms  
    - Analyzing the results and computing metrics
    - Generating visualizations and reports
    """
    
    def __init__(self, 
                 data_path: str,
                 num_gpus: int,
                 output_dir: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize the analyzer.
        
        Args:
            data_path: Path to the original token distribution data
            num_gpus: Number of GPUs for load balancing
            output_dir: Directory to save analysis results
            device: Device to run computations on
        """
        self.data_path = Path(data_path)
        self.num_gpus = num_gpus
        self.output_dir = Path(output_dir) if output_dir else self.data_path / "analysis"
        self.device = device
        
        # Initialize components
        self.visualizer = LoadBalanceVisualizer(self.output_dir)
        self.algorithms = {
            'eplb': EPLBAlgorithm(),
            'bldm': BLDMAlgorithm(),
            'simple_replication': SimpleReplicationAlgorithm(),
        }
        
        # Data storage
        self.original_weights: Optional[torch.Tensor] = None
        self.results: Dict[str, Dict[str, Any]] = {}
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self, data_format: str = "csv") -> torch.Tensor:
        """
        Load original token distribution data.
        
        Args:
            data_format: Format of the data ("csv" or "json")
            
        Returns:
            Loaded weights tensor of shape [num_layers, num_experts]
        """
        print(f"Loading data from {self.data_path}...")
        
        if data_format == "csv":
            self.original_weights = load_csv_to_tensor(str(self.data_path))
        elif data_format == "json":
            self.original_weights = load_json_to_tensor(str(self.data_path))
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
            
        if self.original_weights is None:
            raise ValueError(f"Failed to load data from {self.data_path}")
            
        print(f"Loaded tensor with shape: {self.original_weights.shape}")
        return self.original_weights
    
    def analyze_vanilla_distribution(self) -> Dict[str, Any]:
        """
        Analyze the original (vanilla) token distribution.
        
        Returns:
            Dictionary containing vanilla analysis results
        """
        if self.original_weights is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        print("Analyzing vanilla distribution...")
        
        # Calculate GPU loads for vanilla distribution
        vanilla_gpu_loads = self._calculate_gpu_loads(self.original_weights)
        
        # Calculate metrics
        metrics = self._calculate_metrics(vanilla_gpu_loads)
        
        # Save results directly in the output directory (no extra vanilla subfolder)
        self.output_dir.mkdir(exist_ok=True)
        
        self._save_gpu_loads(vanilla_gpu_loads, self.output_dir / "gpu_loads.csv")
        
        # Generate visualizations
        self.visualizer.plot_gpu_loads_analysis(
            vanilla_gpu_loads,
            self.output_dir / "heatmap.png",
            self.output_dir / "boxplot.png",
            title_prefix="Vanilla"
        )
        
        # Store results
        result = {
            'gpu_loads': vanilla_gpu_loads,
            'metrics': metrics,
            'expert_moves': {'total_moves': 0, 'avg_moves_per_layer': 0.0},  # Vanilla has no moves
            'output_dir': self.output_dir
        }
        self.results['vanilla'] = result
        
        return result
    
    def run_algorithm(self, 
                     algorithm_name: str,
                     config: Dict[str, Any],
                     measure_time: bool = True) -> Dict[str, Any]:
        """
        Run a specific load balancing algorithm.
        
        Args:
            algorithm_name: Name of the algorithm ('eplb', 'bldm', etc.)
            config: Algorithm configuration parameters
            measure_time: Whether to measure execution time
            
        Returns:
            Dictionary containing algorithm results
        """
        if self.original_weights is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
        print(f"Running {algorithm_name.upper()} algorithm...")
        
        algorithm = self.algorithms[algorithm_name]
        
        # Run algorithm with timing if requested
        if measure_time:
            def run_func():
                return algorithm.run(self.original_weights, config)
            
            timing_stats = measure_execution_time(
                run_func, 
                warmup=config.get('warmup_runs', 2),
                repeat=config.get('repeat_runs', 5)
            )
            algorithm_result = timing_stats['result']
            execution_stats = {k: v for k, v in timing_stats.items() if k != 'result'}
        else:
            algorithm_result = algorithm.run(self.original_weights, config)
            execution_stats = {}
        
        # Remap weights according to algorithm result
        remapped_weights = self._remap_weights(
            algorithm_result['phy2log'], 
            self.original_weights
        )
        
        # Calculate GPU loads
        gpu_loads = self._calculate_gpu_loads(remapped_weights)
        
        # Calculate metrics
        metrics = self._calculate_metrics(gpu_loads)
        
        # Calculate expert moves if applicable
        expert_moves = self._calculate_expert_moves(algorithm_result, config)
        
        # Save results directly in the output directory (no extra algorithm subfolder)
        self.output_dir.mkdir(exist_ok=True)
        
        self._save_gpu_loads(gpu_loads, self.output_dir / "gpu_loads.csv")
        self._save_algorithm_result(algorithm_result, self.output_dir / "algorithm_result.json")
        
        # Generate visualizations
        self.visualizer.plot_gpu_loads_analysis(
            gpu_loads,
            self.output_dir / "heatmap.png", 
            self.output_dir / "boxplot.png",
            title_prefix=algorithm_name.upper()
        )
        
        # Store comprehensive results
        result = {
            'algorithm_result': algorithm_result,
            'remapped_weights': remapped_weights,
            'gpu_loads': gpu_loads,
            'metrics': metrics,
            'expert_moves': expert_moves,
            'execution_stats': execution_stats,
            'config': config,
            'output_dir': self.output_dir
        }
        self.results[algorithm_name] = result
        
        # Print summary
        self._print_algorithm_summary(algorithm_name, result)
        
        return result
    
    def compare_algorithms(self) -> Dict[str, Any]:
        """
        Compare results from different algorithms.
        
        Returns:
            Dictionary containing comparison metrics and visualizations
        """
        if len(self.results) < 2:
            raise ValueError("Need at least 2 algorithm results to compare")
            
        print("Comparing algorithm results...")
        print(f"Available results: {list(self.results.keys())}")
        
        # Extract metrics for comparison
        comparison_data = {}
        for name, result in self.results.items():
            if 'metrics' in result:
                comparison_data[name] = result['metrics']
                print(f"Added {name} metrics to comparison")
            else:
                print(f"Skipping {name}: no metrics found")
        
        if len(comparison_data) < 2:
            raise ValueError(f"Need at least 2 results with metrics to compare. Found: {list(comparison_data.keys())}")
            
        # Use current output_dir directly instead of creating comparison subfolder
        comparison_dir = self.output_dir
        comparison_dir.mkdir(exist_ok=True)
        
        try:
            # Pass results to visualizer for expert moves data
            self.visualizer._current_analyzer_results = self.results
            
            self.visualizer.plot_algorithm_comparison(
                comparison_data,
                comparison_dir / "metrics_comparison.png"
            )
            
            # Clean up the temporary reference
            if hasattr(self.visualizer, '_current_analyzer_results'):
                delattr(self.visualizer, '_current_analyzer_results')
        except Exception as e:
            print(f"Failed to generate comparison visualization: {str(e)}")
            # Clean up even if there's an error
            if hasattr(self.visualizer, '_current_analyzer_results'):
                delattr(self.visualizer, '_current_analyzer_results')
        
        # Generate detailed comparison report
        try:
            report = self._generate_comparison_report(comparison_data)
            with open(comparison_dir / "comparison_report.txt", 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Comparison report saved")
        except Exception as e:
            print(f"Failed to generate comparison report: {str(e)}")
            report = f"Error generating report: {str(e)}"
            
        print(f"Comparison results saved to {comparison_dir}")
        
        return {
            'comparison_data': comparison_data,
            'report': report,
            'output_dir': comparison_dir
        }
    
    def _calculate_gpu_loads(self, weights: torch.Tensor) -> torch.Tensor:
        """Calculate GPU loads from expert weights."""
        num_layers, num_experts = weights.shape
        
        # For extra experts scenarios, we need to ensure experts can be evenly divided
        # If not evenly divisible, pad the last GPU with fewer experts
        experts_per_gpu = num_experts // self.num_gpus
        remaining_experts = num_experts % self.num_gpus
        
        gpu_loads = torch.zeros((num_layers, self.num_gpus), dtype=weights.dtype)
        
        for layer_idx in range(num_layers):
            current_expert_idx = 0
            
            for gpu_idx in range(self.num_gpus):
                # Calculate how many experts this GPU gets
                if gpu_idx < remaining_experts:
                    # First 'remaining_experts' GPUs get one extra expert
                    experts_for_this_gpu = experts_per_gpu + 1
                else:
                    experts_for_this_gpu = experts_per_gpu
                
                # Sum the weights for experts assigned to this GPU
                end_idx = current_expert_idx + experts_for_this_gpu
                if end_idx <= num_experts:
                    gpu_loads[layer_idx, gpu_idx] = weights[layer_idx, current_expert_idx:end_idx].sum()
                else:
                    # Handle edge case where we run out of experts
                    gpu_loads[layer_idx, gpu_idx] = weights[layer_idx, current_expert_idx:].sum()
                
                current_expert_idx = end_idx
                
        return gpu_loads
    
    def _remap_weights(self, phy2log: torch.Tensor, original_weights: torch.Tensor) -> torch.Tensor:
        """Remap weights according to phy2log mapping."""
        num_layers, num_physical_experts = phy2log.shape
        new_weights = torch.zeros_like(phy2log, dtype=original_weights.dtype)
        
        for layer_idx in range(num_layers):
            logical_ids = phy2log[layer_idx]
            new_weights[layer_idx] = original_weights[layer_idx][logical_ids]
            
        return new_weights
    
    def _calculate_metrics(self, gpu_loads: torch.Tensor) -> Dict[str, float]:
        """Calculate load balancing metrics - focusing on most important indicators."""
        # Normalize loads per layer
        layer_sums = gpu_loads.sum(dim=1, keepdim=True)
        normalized_loads = gpu_loads / layer_sums
        
        # Calculate key load balancing metrics
        flattened = normalized_loads.flatten().numpy()
        ideal_load = 1.0 / self.num_gpus
        
        # Focus on the most important metrics for load balancing analysis
        metrics = {
            'std_dev': float(np.std(flattened)),
            'load_range': float(np.max(flattened) - np.min(flattened)),
            'coefficient_of_variation': float(np.std(flattened) / np.mean(flattened)),
            'max_deviation_from_ideal': float(np.max(np.abs(flattened - ideal_load)))
        }
        
        return metrics
    
    def _calculate_expert_moves(self, algorithm_result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate expert movement statistics following the correct repository implementation."""
        phy2log = algorithm_result.get('phy2log')
        if phy2log is None:
            return {'total_moves': 0, 'avg_moves_per_layer': 0.0}
        
        num_layers, num_experts = phy2log.shape
        num_gpus = config.get("num_gpus", self.num_gpus)
        
        # Check if this is Simple Replication algorithm by looking at config or detecting duplicates
        is_replication_algo = (
            'extra_experts' in config or 
            'additional_experts_per_gpu' in config or
            self._has_duplicates(phy2log)
        )
        
        if is_replication_algo:
            return self._calculate_replication_stats(phy2log, num_layers)
        else:
            return self._calculate_reordering_moves_correct(phy2log, num_layers, num_experts, num_gpus)
    
    def _calculate_reordering_moves_correct(self, phy2log: torch.Tensor, num_layers: int, num_experts: int, num_gpus: int) -> Dict[str, Any]:
        """
        Calculate expert movement statistics using the correct repository method.
        
        Logic from repository:
        1. Convert phy2log to assignment matrix [num_gpus, num_experts]
        2. For each GPU, count experts that don't belong to its "home" range
        3. Default assignment: expert i should be on GPU (i // experts_per_gpu)
        """
        experts_per_gpu = num_experts // num_gpus
        total_moves = 0
        
        for layer_idx in range(num_layers):
            # Convert phy2log to assignment matrix
            assignment = self._phy2log_to_assignment_matrix(phy2log[layer_idx], num_gpus, num_experts)
            
            # Count expert moves for this layer using repository logic
            moves = self._count_expert_moves_repository_style(assignment, experts_per_gpu)
            total_moves += moves
        
        return {
            'total_moves': total_moves,
            'avg_moves_per_layer': total_moves / num_layers if num_layers > 0 else 0.0,
            'algorithm_type': 'reordering'
        }
    
    def _phy2log_to_assignment_matrix(self, phy2log_layer: torch.Tensor, num_gpus: int, num_experts: int) -> torch.Tensor:
        """
        Convert phy2log mapping to assignment matrix.
        
        Args:
            phy2log_layer: [num_experts] - logical expert ID for each physical position
            num_gpus: Number of GPUs
            num_experts: Number of experts
            
        Returns:
            assignment: [num_gpus, num_experts] - assignment[gpu][expert] = 1 if expert is on gpu
        """
        experts_per_gpu = num_experts // num_gpus
        assignment = torch.zeros((num_gpus, num_experts), dtype=torch.int64)
        
        # Fill assignment matrix based on phy2log
        for physical_pos in range(num_experts):
            logical_expert_id = phy2log_layer[physical_pos].item()
            gpu_id = physical_pos // experts_per_gpu
            
            # Ensure we don't go out of bounds
            if gpu_id < num_gpus and logical_expert_id < num_experts:
                assignment[gpu_id][logical_expert_id] = 1
        
        return assignment
    
    def _count_expert_moves_repository_style(self, assignment: torch.Tensor, experts_per_gpu: int) -> int:
        """
        Count expert moves using the exact logic from the repository.
        
        This is the Python equivalent of:
        ```cpp
        for gpu in range(num_gpus):
            for expert in range(num_experts):
                if assignment[gpu][expert] == 1 and (expert // experts_per_gpu) != gpu:
                    moves += 1
        ```
        """
        num_gpus, num_experts = assignment.shape
        moves = 0
        
        for gpu in range(num_gpus):
            for expert in range(num_experts):
                # If this expert is assigned to this GPU
                if assignment[gpu][expert].item() == 1:
                    # But this expert's "home" GPU is different
                    home_gpu = expert // experts_per_gpu
                    if home_gpu != gpu:
                        moves += 1
        
        return moves
    
    def _has_duplicates(self, phy2log: torch.Tensor) -> bool:
        """Check if the phy2log mapping contains duplicate experts (indicating replication)."""
        for layer_idx in range(phy2log.shape[0]):
            unique_experts = torch.unique(phy2log[layer_idx])
            if len(unique_experts) < len(phy2log[layer_idx]):
                return True
        return False
    
    def _calculate_replication_stats(self, phy2log: torch.Tensor, num_layers: int) -> Dict[str, Any]:
        """Calculate replication statistics for Simple Replication algorithm."""
        total_replications = 0
        total_replicated_experts = 0
        
        for layer_idx in range(num_layers):
            # Count how many times each expert appears
            layer_experts = phy2log[layer_idx]
            expert_counts = torch.bincount(layer_experts)
            
            # Count experts that are replicated (appear more than once)
            replicated_experts = (expert_counts > 1).sum().item()
            
            # Count total number of replications (extra copies)
            replications = (expert_counts - 1).sum().item()
            
            total_replicated_experts += replicated_experts
            total_replications += replications
        
        return {
            'total_moves': total_replications,  # Keep same key name for compatibility
            'avg_moves_per_layer': total_replications / num_layers if num_layers > 0 else 0.0,
            'total_replications': total_replications,
            'avg_replications_per_layer': total_replications / num_layers if num_layers > 0 else 0.0,
            'total_replicated_experts': total_replicated_experts,
            'avg_replicated_experts_per_layer': total_replicated_experts / num_layers if num_layers > 0 else 0.0,
            'algorithm_type': 'replication'
        }
    
    def _save_gpu_loads(self, gpu_loads: torch.Tensor, path: Path):
        """Save GPU loads to CSV."""
        gpu_loads_df = pd.DataFrame(
            gpu_loads.numpy(),
            columns=[f"GPU{i}" for i in range(self.num_gpus)],
            index=list(range(gpu_loads.shape[0]))
        )
        gpu_loads_df.to_csv(path, index_label="Layer")
    
    def _save_algorithm_result(self, result: Dict[str, Any], path: Path):
        """Save algorithm result to JSON."""
        # Convert tensors to lists for JSON serialization
        json_result = {}
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                json_result[key] = value.tolist()
            else:
                json_result[key] = value
                
        with open(path, 'w') as f:
            json.dump(json_result, f, indent=2)
    
    def _print_algorithm_summary(self, algorithm_name: str, result: Dict[str, Any]):
        """Print a summary of algorithm results."""
        metrics = result['metrics']
        expert_moves = result['expert_moves']
        
        print(f"\n{algorithm_name.upper()} Algorithm Summary:")
        print("-" * 40)
        
        # Print load balancing metrics
        print("Load Balancing Metrics:")
        print(f"  • Standard Deviation: {metrics['std_dev']:.6f}")
        print(f"  • Load Range: {metrics['load_range']:.6f}")
        print(f"  • Coefficient of Variation: {metrics['coefficient_of_variation']:.6f}")
        print(f"  • Max Deviation from Ideal: {metrics['max_deviation_from_ideal']:.6f}")
        
        # Print expert movement/replication statistics
        if expert_moves:
            algorithm_type = expert_moves.get('algorithm_type', 'unknown')
            
            if algorithm_type == 'replication':
                print("\nExpert Replication Statistics:")
                print(f"  • Total Replications: {expert_moves.get('total_replications', 0)}")
                print(f"  • Avg Replications per Layer: {expert_moves.get('avg_replications_per_layer', 0.0):.1f}")
                print(f"  • Total Replicated Experts: {expert_moves.get('total_replicated_experts', 0)}")
                print(f"  • Avg Replicated Experts per Layer: {expert_moves.get('avg_replicated_experts_per_layer', 0.0):.1f}")
            else:
                print("\nExpert Movement Statistics:")
                print(f"  • Total Expert Moves: {expert_moves.get('total_moves', 0)}")
                print(f"  • Avg Moves per Layer: {expert_moves.get('avg_moves_per_layer', 0.0):.1f}")
        
        print(f"\nResults saved to: {result['output_dir']}")
    
    def _generate_comparison_report(self, comparison_data: Dict[str, Dict[str, float]]) -> str:
        """Generate a detailed comparison report focusing on key metrics."""
        report_lines = ["Load Balancing Algorithm Comparison Report", "=" * 50, ""]
        
        # Define the key metrics we care about for load balancing
        key_metrics = ['std_dev', 'load_range', 'coefficient_of_variation', 'max_deviation_from_ideal']
        
        for metric in key_metrics:
            if metric in next(iter(comparison_data.values())):
                report_lines.append(f"{metric.replace('_', ' ').title()}:")
                values = {name: data[metric] for name, data in comparison_data.items()}
                
                # Lower is better for load balancing metrics
                best_algo = min(values.items(), key=lambda x: x[1])
                worst_algo = max(values.items(), key=lambda x: x[1])
                
                for name, value in sorted(values.items(), key=lambda x: x[1]):
                    marker = " (BEST)" if name == best_algo[0] else " (WORST)" if name == worst_algo[0] else ""
                    report_lines.append(f"  {name}: {value:.6f}{marker}")
                report_lines.append("")
        
        # Add expert moves summary if available
        if any('expert_moves' in self.results.get(name, {}) for name in comparison_data.keys()):
            report_lines.append("Expert Movement Analysis:")
            for name in comparison_data.keys():
                expert_moves = self.results.get(name, {}).get('expert_moves', {})
                if expert_moves:
                    total_moves = expert_moves.get('total_moves', 0)
                    avg_moves = expert_moves.get('avg_moves_per_layer', 0.0)
                    report_lines.append(f"  {name}: {total_moves} total moves ({avg_moves:.2f} avg/layer)")
            report_lines.append("")
        
        # Add execution time summary if available
        if any('execution_stats' in self.results.get(name, {}) for name in comparison_data.keys()):
            report_lines.append("Execution Time Analysis:")
            for name in comparison_data.keys():
                execution_stats = self.results.get(name, {}).get('execution_stats', {})
                if execution_stats:
                    avg_time = execution_stats.get('avg', 0)
                    std_time = execution_stats.get('std', 0)
                    min_time = execution_stats.get('min', 0)
                    max_time = execution_stats.get('max', 0)
                    report_lines.append(f"  {name}: {avg_time:.2f} ± {std_time:.2f} ms (range: {min_time:.2f}-{max_time:.2f} ms)")
                else:
                    report_lines.append(f"  {name}: N/A (vanilla baseline)")
            report_lines.append("")
        
        return "\n".join(report_lines) 