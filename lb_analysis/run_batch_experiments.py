#!/usr/bin/env python3
"""
Batch Load Balance Experiments

This script runs multiple load balancing experiments with different configurations:
- Vanilla (original distribution)
- EPLB (8, 16, 32 GPUs)
- BLDM (8, 16, 32 GPUs)

Usage:
    python run_batch_experiments.py
"""

import os
import sys
import time
from pathlib import Path

# Fix OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import modules
try:
    from load_balance_analyzer import LoadBalanceAnalyzer
except ImportError:
    # Fix module imports if needed
    print("Fixing module imports...")

    analyzer_file = current_dir / "load_balance_analyzer.py"
    if analyzer_file.exists():
        with open(analyzer_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace relative imports
        content = content.replace("from .utils import", "from utils import")
        content = content.replace("from .algorithms import",
                                  "from algorithms import")
        content = content.replace("from .visualization import",
                                  "from visualization import")

        # Ensure SimpleReplicationAlgorithm is included
        if "SimpleReplicationAlgorithm" not in content:
            content = content.replace(
                "from algorithms import EPLBAlgorithm, BLDMAlgorithm",
                "from algorithms import EPLBAlgorithm, BLDMAlgorithm, SimpleReplicationAlgorithm"
            )

        with open(analyzer_file, 'w', encoding='utf-8') as f:
            f.write(content)

    from load_balance_analyzer import LoadBalanceAnalyzer


def main():
    """Run batch experiments with multiple configurations."""

    # Configuration
    DATA_PATH = r"C:\Users\bingxche\data\log\deepseek-v3_tp8_mixtral_dataset_5000_prompts_vanilla\moe_token_dist"
    OUTPUT_ROOT = r"C:\Users\bingxche\data\log\batch_load_balance_analysis"

    # Experiment configurations
    experiments = [
        # Vanilla
        {
            "name": "vanilla_8gpu",
            "algorithm": None,
            "num_gpus": 8,
            "config": {}
        },
        # Vanilla
        {
            "name": "vanilla_16gpu",
            "algorithm": None,
            "num_gpus": 16,
            "config": {}
        },
        # Vanilla
        {
            "name": "vanilla_32gpu",
            "algorithm": None,
            "num_gpus": 32,
            "config": {}
        },

        # EPLB experiments
        {
            "name": "eplb_8gpu",
            "algorithm": "eplb",
            "num_gpus": 8,
            "config": {
                "num_replicas": 256,
                "num_groups": 1,
                "num_nodes": 1,
                "num_gpus": 8,
                "warmup_runs": 2,
                "repeat_runs": 5
            }
        },
        {
            "name": "eplb_16gpu",
            "algorithm": "eplb",
            "num_gpus": 16,
            "config": {
                "num_replicas": 256,
                "num_groups": 1,
                "num_nodes": 1,
                "num_gpus": 16,
                "warmup_runs": 2,
                "repeat_runs": 5
            }
        },
        {
            "name": "eplb_32gpu",
            "algorithm": "eplb",
            "num_gpus": 32,
            "config": {
                "num_replicas": 256,
                "num_groups": 1,
                "num_nodes": 1,
                "num_gpus": 32,
                "warmup_runs": 2,
                "repeat_runs": 5
            }
        },

        # BLDM experiments
        {
            "name": "bldm_8gpu",
            "algorithm": "bldm",
            "num_gpus": 8,
            "config": {
                "num_replicas": 256,
                "num_gpus": 8,
                "warmup_runs": 2,
                "repeat_runs": 5
            }
        },
        {
            "name": "bldm_16gpu",
            "algorithm": "bldm",
            "num_gpus": 16,
            "config": {
                "num_replicas": 256,
                "num_gpus": 16,
                "warmup_runs": 2,
                "repeat_runs": 5
            }
        },
        {
            "name": "bldm_32gpu",
            "algorithm": "bldm",
            "num_gpus": 32,
            "config": {
                "num_replicas": 256,
                "num_gpus": 32,
                "warmup_runs": 2,
                "repeat_runs": 5
            }
        },

        # Simple Replication experiments with 8 extra experts
        {
            "name": "simple_replication_8extra_8gpu",
            "algorithm": "simple_replication",
            "num_gpus": 8,
            "config": {
                "num_gpus": 8,
                "extra_experts": 8,
                "warmup_runs": 2,
                "repeat_runs": 5
            }
        },

        # Simple Replication experiments with 16 extra experts
        {
            "name": "simple_replication_16extra_8gpu",
            "algorithm": "simple_replication",
            "num_gpus": 8,
            "config": {
                "num_gpus": 8,
                "extra_experts": 16,
                "warmup_runs": 2,
                "repeat_runs": 5
            }
        },
        {
            "name": "simple_replication_8extra_16gpu",
            "algorithm": "simple_replication",
            "num_gpus": 16,
            "config": {
                "num_gpus": 16,
                "extra_experts": 8,
                "warmup_runs": 2,
                "repeat_runs": 5
            }
        },
        {
            "name": "simple_replication_4extra_32gpu",
            "algorithm": "simple_replication",
            "num_gpus": 32,
            "config": {
                "num_gpus": 32,
                "extra_experts": 4,
                "warmup_runs": 2,
                "repeat_runs": 5
            }
        },
    ]

    # Run experiments
    print("Starting Batch Load Balance Experiments")
    print("=" * 60)
    print(f"Data Path: {DATA_PATH}")
    print(f"Output Root: {OUTPUT_ROOT}")
    print(f"Total Experiments: {len(experiments)}")
    print("=" * 60)

    # Create output directory
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    # Store results
    all_results = {}
    experiment_times = {}

    for i, exp in enumerate(experiments, 1):
        print(f"\nRunning Experiment {i}/{len(experiments)}: {exp['name']}")
        print("-" * 50)

        exp_start_time = time.time()

        try:
            # Create analyzer for each experiment
            analyzer = LoadBalanceAnalyzer(data_path=DATA_PATH,
                                           num_gpus=exp["num_gpus"],
                                           output_dir=output_root /
                                           exp["name"])

            # Load data
            print("Loading data...")
            analyzer.load_data(data_format="csv")

            if exp["algorithm"] is None:
                # Vanilla analysis
                print("Analyzing vanilla distribution...")
                results = analyzer.analyze_vanilla_distribution()
            else:
                # Algorithm analysis
                print(f"Running {exp['algorithm'].upper()} algorithm...")
                results = analyzer.run_algorithm(
                    algorithm_name=exp["algorithm"],
                    config=exp["config"],
                    measure_time=True)

            # Store results
            all_results[exp["name"]] = {
                "analyzer": analyzer,
                "results": results,
                "config": exp
            }

            exp_time = time.time() - exp_start_time
            experiment_times[exp["name"]] = exp_time

            print(f"Experiment {exp['name']} completed in {exp_time:.2f}s")

            # Print key metrics
            if "metrics" in results:
                metrics = results["metrics"]
                execution_stats = results.get("execution_stats", {})
                expert_moves = results.get("expert_moves", {})

                print(f"   Std Dev: {metrics['std_dev']:.6f}")
                print(f"   Load Range: {metrics['load_range']:.6f}")

                if execution_stats:
                    print(
                        f"   Execution Time: {execution_stats['avg']:.2f} ± {execution_stats['std']:.2f} ms"
                    )

                if expert_moves and expert_moves.get('total_moves', 0) > 0:
                    print(
                        f"   Expert Moves: {expert_moves['total_moves']} ({expert_moves['avg_moves_per_layer']:.1f}/layer)"
                    )
                else:
                    print(f"   Expert Moves: 0 (vanilla distribution)")

        except Exception as e:
            print(f"Experiment {exp['name']} failed: {str(e)}")
            experiment_times[exp["name"]] = -1
            continue

    # Generate comprehensive comparison
    print("\n" + "=" * 60)
    print("GENERATING COMPREHENSIVE COMPARISON")
    print("=" * 60)

    # Create comparison analyzer
    comparison_analyzer = LoadBalanceAnalyzer(data_path=DATA_PATH,
                                              num_gpus=8,
                                              output_dir=output_root /
                                              "comprehensive_comparison")

    # Add all experiment results to comparison
    print("Preparing comparison for all experiments...")
    for exp_name, exp_data in all_results.items():
        if "metrics" in exp_data["results"]:
            comparison_key = exp_name.replace("_gpu", "").replace("gpu", "")
            comparison_analyzer.results[comparison_key] = exp_data["results"]
            print(f"Added {comparison_key} to comparison")

    # Generate comparison report
    if len(comparison_analyzer.results) >= 2:
        try:
            comparison_results = comparison_analyzer.compare_algorithms()
            print(
                f"Comprehensive comparison saved to: {comparison_results['output_dir']}"
            )
        except Exception as e:
            print(f"Failed to generate comparison: {str(e)}")

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH EXPERIMENTS SUMMARY")
    print("=" * 60)

    # Group results by GPU count
    gpu_groups = {}
    for exp_name, exp_data in all_results.items():
        num_gpus = exp_data["config"]["num_gpus"]
        if num_gpus not in gpu_groups:
            gpu_groups[num_gpus] = []
        gpu_groups[num_gpus].append((exp_name, exp_data))

    for num_gpus in sorted(gpu_groups.keys()):
        print(f"\n{num_gpus} GPU Results:")
        print("-" * 30)

        for exp_name, exp_data in gpu_groups[num_gpus]:
            if "metrics" in exp_data["results"]:
                metrics = exp_data["results"]["metrics"]
                execution_stats = exp_data["results"].get(
                    "execution_stats", {})
                expert_moves = exp_data["results"].get("expert_moves", {})
                exec_time = experiment_times.get(exp_name, 0)

                algo_name = exp_data["config"]["algorithm"] or "vanilla"

                # Build output string
                output_parts = [
                    f"std_dev={metrics['std_dev']:.6f}",
                    f"range={metrics['load_range']:.6f}"
                ]

                if execution_stats:
                    avg_time = execution_stats.get('avg', 0)
                    std_time = execution_stats.get('std', 0)
                    output_parts.append(
                        f"exec_time={avg_time:.2f}±{std_time:.2f}ms")

                if expert_moves and expert_moves.get('total_moves', 0) > 0:
                    total_moves = expert_moves['total_moves']
                    output_parts.append(f"moves={total_moves}")

                output_parts.append(f"total_time={exec_time:.1f}s")

                print(f"  {algo_name:>15}: {', '.join(output_parts)}")

    # Performance ranking
    print(f"\nPERFORMANCE RANKING:")
    print("-" * 50)

    if all_results:
        # Sort by standard deviation
        valid_results = [(name, data) for name, data in all_results.items()
                         if "metrics" in data["results"]]

        if valid_results:
            print("\nLoad Balancing Performance (by Standard Deviation):")
            sorted_by_std = sorted(
                valid_results,
                key=lambda x: x[1]["results"]["metrics"]["std_dev"])
            for i, (name, data) in enumerate(sorted_by_std, 1):
                std_dev = data["results"]["metrics"]["std_dev"]
                execution_stats = data["results"].get("execution_stats", {})
                algo_name = data["config"]["algorithm"] or "vanilla"

                time_info = ""
                if execution_stats:
                    avg_time = execution_stats.get('avg', 0)
                    time_info = f" (exec: {avg_time:.2f}ms)"

                print(
                    f"  {i:2d}. {name:<30} - std_dev: {std_dev:.6f}{time_info}"
                )

    # Time statistics
    total_time = sum(t for t in experiment_times.values() if t > 0)
    successful_experiments = len(
        [t for t in experiment_times.values() if t > 0])
    avg_experiment_time = total_time / successful_experiments if successful_experiments > 0 else 0

    print(f"\nEXPERIMENT TIME STATISTICS:")
    print("-" * 40)
    print(
        f"  Total Experiment Time: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )
    print(
        f"  Successful Experiments: {successful_experiments}/{len(experiments)}"
    )
    print(f"  Average per Experiment: {avg_experiment_time:.2f}s")

    print(f"\nAll results saved to: {OUTPUT_ROOT}")
    print("\nBatch experiments completed successfully!")


if __name__ == "__main__":
    main()
