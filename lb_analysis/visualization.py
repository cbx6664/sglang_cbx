"""
Visualization utilities for load balance analysis.
"""

from pathlib import Path
from typing import Dict, Any

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class LoadBalanceVisualizer:
    """Handles visualization of load balancing results."""
    
    def __init__(self, output_dir: Path):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        # Set style with fallback
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_gpu_loads_analysis(self, 
                               gpu_loads: torch.Tensor, 
                               heatmap_path: Path, 
                               boxplot_path: Path,
                               title_prefix: str = ""):
        """
        Plot normalized heatmap and boxplot of GPU loads.

        Args:
            gpu_loads: Tensor of shape [num_layers, num_gpus]
            heatmap_path: File path to save heatmap
            boxplot_path: File path to save boxplot
            title_prefix: Prefix for plot titles
        """
        # print(f"[DEBUG] Starting plot_gpu_loads_analysis with shape: {gpu_loads.shape}")
        # print(f"[DEBUG] GPU loads range: [{gpu_loads.min():.6f}, {gpu_loads.max():.6f}]")
        # print(f"[DEBUG] GPU loads sum: {gpu_loads.sum():.6f}")
        
        # Normalize the gpu loads per layer
        layer_sums = gpu_loads.sum(dim=1, keepdim=True)
        # print(f"[DEBUG] Layer sums range: [{layer_sums.min():.6f}, {layer_sums.max():.6f}]")
        
        # Check for zero layer sums
        zero_layers = (layer_sums == 0).squeeze()
        if zero_layers.any():
            # print(f"[WARNING] Found {zero_layers.sum()} layers with zero sum, this will cause NaN in normalization")
            # Handle zero layer sums by setting them to 1 to avoid division by zero
            layer_sums = torch.where(layer_sums == 0, torch.ones_like(layer_sums), layer_sums)
        
        normalized_gpu_loads = gpu_loads / layer_sums
        # print(f"[DEBUG] Normalized GPU loads range: [{normalized_gpu_loads.min():.6f}, {normalized_gpu_loads.max():.6f}]")
        # print(f"[DEBUG] Contains NaN: {torch.isnan(normalized_gpu_loads).any()}")
        # print(f"[DEBUG] Contains Inf: {torch.isinf(normalized_gpu_loads).any()}")

        # Heatmap: Normalized Load per layer per GPU
        plt.figure(figsize=(12, 10))
        try:
            sns.heatmap(
                normalized_gpu_loads.numpy(),
                cmap="Reds",
                annot=True,
                fmt=".3f",
                linewidths=0.5,
                cbar_kws={"label": "Normalized Load"},
                yticklabels=list(range(gpu_loads.shape[0])),
                xticklabels=[f"GPU{i}" for i in range(gpu_loads.shape[1])],
                annot_kws={"size": 8},
            )
            plt.xlabel("GPU Index")
            plt.ylabel("Layer ID")
            plt.title(f"{title_prefix} Normalized GPU Loads per Layer")
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {heatmap_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save heatmap: {e}")
        finally:
            plt.close()

        # Boxplot: Distribution of normalized GPU loads across layers
        plt.figure(figsize=(12, 8))
        
        try:
            # Create boxplot data
            data = normalized_gpu_loads.numpy()  # shape: [num_layers, num_gpus]
            num_layers, num_gpus = data.shape
            print(f"[DEBUG] Boxplot data shape: {data.shape}")
            
            # Remove NaN and Inf values for boxplot
            if np.isnan(data).any() or np.isinf(data).any():
                print(f"[WARNING] Removing NaN/Inf values from boxplot data")
                data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Prepare data for boxplot - each GPU's data across all layers
            gpu_data = [data[:, i] for i in range(num_gpus)]  # List of arrays, each with data for one GPU
            gpu_labels = [f"GPU{i}" for i in range(num_gpus)]
            
            # print(f"[DEBUG] GPU data lengths: {[len(gpu_data[i]) for i in range(min(3, num_gpus))]}")
            # print(f"[DEBUG] Sample GPU0 data: {gpu_data[0][:5] if len(gpu_data[0]) > 0 else 'empty'}")
            
            # Check if all data is the same (which would make boxplot invisible)
            data_std = data.std()
            # print(f"[DEBUG] Data standard deviation: {data_std:.6f}")
            
            if data_std < 1e-10:
                print(f"[WARNING] Data has very low variance (std={data_std:.6f}), boxplot may not be visible")
            
            # Create boxplot
            box_plot = plt.boxplot(gpu_data, labels=gpu_labels, patch_artist=True, showmeans=True)
            print(f"[DEBUG] Boxplot created successfully")
            
            # Color the boxes
            colors = sns.color_palette("husl", num_gpus)
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add ideal line
            ideal_load = 1.0 / num_gpus
            plt.axhline(y=ideal_load, color="red", linestyle="--", linewidth=2, 
                       label=f"Ideal Load ({ideal_load:.3f})")
            
            # Calculate and display overall statistics
            overall_std = data.flatten().std()
            overall_cv = overall_std / data.flatten().mean() if data.flatten().mean() != 0 else 0
            
            plt.text(0.02, 0.98, 
                    f"Overall Std Dev: σ = {overall_std:.6f}\nCoeff. of Variation: {overall_cv:.6f}",
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", 
                             facecolor="white", alpha=0.8))
            
            plt.xlabel("GPU Index")
            plt.ylabel("Normalized Load")
            plt.title(f"{title_prefix} GPU Load Distribution Across Layers")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Set y-axis limits to ensure visibility
            y_min, y_max = data.min(), data.max()
            if y_max - y_min > 0:
                margin = (y_max - y_min) * 0.1
                plt.ylim(y_min - margin, y_max + margin)
            
            plt.tight_layout()
            plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
            print(f"Boxplot saved to: {boxplot_path}")
            
            # Check file size
            if boxplot_path.exists():
                file_size = boxplot_path.stat().st_size
                print(f"[DEBUG] Boxplot file size: {file_size} bytes")
                if file_size == 0:
                    print(f"[ERROR] Boxplot file is empty!")
            
        except Exception as e:
            print(f"[ERROR] Failed to create boxplot: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
        finally:
            plt.close()

    def plot_algorithm_comparison(self, 
                                 comparison_data: Dict[str, Dict[str, float]], 
                                 output_path: Path):
        """
        Plot comparison of different algorithms - show std dev and expert moves for all algorithms.
        
        Args:
            comparison_data: Dict mapping algorithm names to their metrics
            output_path: Path to save the comparison plot
        """
        algorithms = list(comparison_data.keys())
        print(f"Plotting comparison for algorithms: {algorithms}")
        
        # Only show std_dev in metrics comparison
        if 'std_dev' not in next(iter(comparison_data.values())):
            print("Warning: No std_dev metric found in comparison data")
            return
        
        # Determine layout based on number of algorithms
        num_algorithms = len(algorithms)
        if num_algorithms <= 5:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        elif num_algorithms <= 10:
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        
        # Define colors
        colors = sns.color_palette("Set3", len(algorithms))
        algorithm_colors = {algo: color for algo, color in zip(algorithms, colors)}
        
        # Plot std_dev
        ax = axes[0]
        values = [comparison_data[algo]['std_dev'] for algo in algorithms]
        
        bars = ax.bar(algorithms, values, 
                     color=[algorithm_colors[algo] for algo in algorithms], 
                     alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Annotate bars with values
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Highlight the best (lowest) value
        best_idx = values.index(min(values))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        ax.set_title("Standard Deviation\n(Lower is Better)", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel("Standard Deviation", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Rotate labels if too many algorithms
        if len(algorithms) > 6 or len(max(algorithms, key=len)) > 8:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add expert moves comparison
        expert_moves_data = self._get_expert_moves_data(algorithms)
        print(f"Expert moves data found for: {list(expert_moves_data.keys())}")
        
        # Total moves comparison
        ax_total_moves = axes[1]
        total_moves = []
        for algo in algorithms:
            if algo in expert_moves_data:
                total_moves.append(expert_moves_data[algo]['total_moves'])
            else:
                total_moves.append(0)  # algorithms without moves (like vanilla) have 0
        
        bars = ax_total_moves.bar(algorithms, total_moves,
                              color=[algorithm_colors[algo] for algo in algorithms],
                              alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Annotate bars
        for bar, value in zip(bars, total_moves):
            height = bar.get_height()
            ax_total_moves.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value}',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Highlight the best (lowest) value for non-zero moves
        non_zero_moves = [m for m in total_moves if m > 0]
        if non_zero_moves:
            min_non_zero = min(non_zero_moves)
            if min_non_zero in total_moves:
                best_idx = total_moves.index(min_non_zero)
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
        
        ax_total_moves.set_title("Total Expert Moves\n(Lower is Better)", 
                             fontsize=14, fontweight='bold', pad=20)
        ax_total_moves.set_ylabel("Number of Moves", fontsize=12)
        ax_total_moves.grid(True, alpha=0.3, linestyle='--')
        ax_total_moves.set_ylim(bottom=0)
        
        if len(algorithms) > 6 or len(max(algorithms, key=len)) > 8:
            plt.setp(ax_total_moves.get_xticklabels(), rotation=45, ha='right')
        
        # Average moves per layer comparison
        ax_avg_moves = axes[2]
        avg_moves = []
        for algo in algorithms:
            if algo in expert_moves_data:
                avg_moves.append(expert_moves_data[algo]['avg_moves_per_layer'])
            else:
                avg_moves.append(0.0)  # algorithms without moves have 0 avg moves
        
        bars = ax_avg_moves.bar(algorithms, avg_moves,
                              color=[algorithm_colors[algo] for algo in algorithms],
                              alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Annotate bars
        for bar, value in zip(bars, avg_moves):
            height = bar.get_height()
            ax_avg_moves.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.1f}',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Highlight the best (lowest) value for non-zero moves
        non_zero_avg_moves = [m for m in avg_moves if m > 0]
        if non_zero_avg_moves:
            min_non_zero_avg = min(non_zero_avg_moves)
            if min_non_zero_avg in avg_moves:
                best_idx = avg_moves.index(min_non_zero_avg)
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
        
        ax_avg_moves.set_title("Average Moves per Layer\n(Lower is Better)", 
                             fontsize=14, fontweight='bold', pad=20)
        ax_avg_moves.set_ylabel("Moves per Layer", fontsize=12)
        ax_avg_moves.grid(True, alpha=0.3, linestyle='--')
        ax_avg_moves.set_ylim(bottom=0)
        
        if len(algorithms) > 6 or len(max(algorithms, key=len)) > 8:
            plt.setp(ax_avg_moves.get_xticklabels(), rotation=45, ha='right')
        
        # Add overall title
        plt.suptitle(f"Load Balancing Algorithm Comparison ({len(algorithms)} Algorithms)\nStandard Deviation & Expert Movement Analysis", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Algorithm comparison plot saved to: {output_path}")
        plt.close()
    
    def _get_expert_moves_data(self, algorithms):
        """Extract expert moves data for algorithms that have it."""
        expert_moves_data = {}
        
        # Check if we have access to the analyzer's results
        if hasattr(self, '_current_analyzer_results'):
            for algo in algorithms:
                # Try exact match first
                if algo in self._current_analyzer_results:
                    result_data = self._current_analyzer_results[algo]
                    if 'expert_moves' in result_data:
                        expert_moves = result_data['expert_moves']
                        if expert_moves and expert_moves.get('total_moves', 0) > 0:
                            # Adjust label based on algorithm type
                            algorithm_type = expert_moves.get('algorithm_type', 'unknown')
                            if algorithm_type == 'replication':
                                # For replication algorithms, show replication stats
                                expert_moves_data[algo] = {
                                    'total_moves': expert_moves.get('total_replications', 0),
                                    'avg_moves_per_layer': expert_moves.get('avg_replications_per_layer', 0.0),
                                    'algorithm_type': 'replication',
                                    'display_name': 'Replications'
                                }
                            else:
                                # For reordering algorithms, show move stats
                                expert_moves_data[algo] = {
                                    'total_moves': expert_moves.get('total_moves', 0),
                                    'avg_moves_per_layer': expert_moves.get('avg_moves_per_layer', 0.0),
                                    'algorithm_type': 'reordering',
                                    'display_name': 'Moves'
                                }
                else:
                    # Try partial matching for different naming schemes
                    matching_result = None
                    for result_key, result_data in self._current_analyzer_results.items():
                        # Check various matching patterns
                        if (algo in result_key or 
                            result_key in algo or
                            (algo == "vanilla" and "vanilla" in result_key) or
                            (algo.replace("_", "") in result_key.replace("_", ""))):
                            matching_result = result_data
                            break
                    
                    if matching_result and 'expert_moves' in matching_result:
                        expert_moves = matching_result['expert_moves']
                        if expert_moves and expert_moves.get('total_moves', 0) > 0:
                            algorithm_type = expert_moves.get('algorithm_type', 'unknown')
                            if algorithm_type == 'replication':
                                expert_moves_data[algo] = {
                                    'total_moves': expert_moves.get('total_replications', 0),
                                    'avg_moves_per_layer': expert_moves.get('avg_replications_per_layer', 0.0),
                                    'algorithm_type': 'replication',
                                    'display_name': 'Replications'
                                }
                            else:
                                expert_moves_data[algo] = {
                                    'total_moves': expert_moves.get('total_moves', 0),
                                    'avg_moves_per_layer': expert_moves.get('avg_moves_per_layer', 0.0),
                                    'algorithm_type': 'reordering',
                                    'display_name': 'Moves'
                                }
        
        return expert_moves_data

    def plot_load_distribution_evolution(self, 
                                        results_dict: Dict[str, torch.Tensor],
                                        output_path: Path):
        """
        Plot how load distribution evolves across different algorithms.
        
        Args:
            results_dict: Dict mapping algorithm names to their GPU loads
            output_path: Path to save the evolution plot
        """
        fig, axes = plt.subplots(1, len(results_dict), figsize=(6*len(results_dict), 5))
        if len(results_dict) == 1:
            axes = [axes]
        
        for i, (algo_name, gpu_loads) in enumerate(results_dict.items()):
            # Normalize loads
            layer_sums = gpu_loads.sum(dim=1, keepdim=True)
            normalized_loads = gpu_loads / layer_sums
            
            # Plot violin plot
            data = normalized_loads.numpy()
            parts = axes[i].violinplot([data[:, j] for j in range(data.shape[1])],
                                     positions=range(data.shape[1]),
                                     showmeans=True, showmedians=True)
            
            # Color the violins
            colors = sns.color_palette("husl", data.shape[1])
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # Add ideal line
            ideal_load = 1.0 / data.shape[1]
            axes[i].axhline(y=ideal_load, color="red", linestyle="--", 
                          linewidth=2, alpha=0.8)
            
            axes[i].set_title(f"{algo_name.upper()}")
            axes[i].set_xlabel("GPU Index")
            axes[i].set_ylabel("Normalized Load")
            axes[i].set_xticks(range(data.shape[1]))
            axes[i].set_xticklabels([f"GPU{j}" for j in range(data.shape[1])])
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle("Load Distribution Evolution Across Algorithms", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Load distribution evolution plot saved to: {output_path}")
        plt.close()

    def plot_layer_wise_variance(self, 
                               results_dict: Dict[str, torch.Tensor],
                               output_path: Path):
        """
        Plot layer-wise variance for different algorithms.
        
        Args:
            results_dict: Dict mapping algorithm names to their GPU loads
            output_path: Path to save the variance plot
        """
        plt.figure(figsize=(15, 8))
        
        for algo_name, gpu_loads in results_dict.items():
            # Normalize loads
            layer_sums = gpu_loads.sum(dim=1, keepdim=True)
            normalized_loads = gpu_loads / layer_sums
            
            # Calculate layer-wise variance
            layer_variances = normalized_loads.var(dim=1).numpy()
            layer_indices = range(len(layer_variances))
            
            plt.plot(layer_indices, layer_variances, marker='o', 
                    label=f"{algo_name.upper()}", linewidth=2, markersize=4)
        
        plt.xlabel("Layer Index")
        plt.ylabel("Load Variance")
        plt.title("Layer-wise Load Variance Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Layer-wise variance plot saved to: {output_path}")
        plt.close() 