"""
Utility functions for load balance analysis.
"""

import os
import re
import json
import time
import statistics
from typing import List, Optional, Callable

import torch
import pandas as pd


def natural_sort_key(s: str) -> List:
    """Generate a natural sort key for strings.

    Args:
        s: Input string to be sorted

    Returns:
        A list that can be used for natural sorting
    """
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def load_csv_to_tensor(input_folder: str) -> Optional[torch.Tensor]:
    """Load CSV files from a folder and convert them to a tensor.

    Each CSV file represents one layer, and is converted to one row in the output tensor.

    Args:
        input_folder: Path to folder containing CSV files

    Returns:
        torch.Tensor of shape [num_layers, num_experts] or None if an error occurs
    """
    # Get all csv files and sort them naturally
    csv_files = sorted(
        [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.endswith(".csv")
        ],
        key=natural_sort_key,
    )
    num_layers = len(csv_files)

    if num_layers == 0:
        print(f"No CSV files found in {input_folder}")
        return None

    # Initialize a list to store rows
    rows = []

    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file, header=None)
            # Sum up all rows in the CSV file
            row_sums = df.sum(axis=0).values
            # Append to the list as a row
            rows.append(row_sums)
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
            return None

    # Stack all rows into a single tensor of shape [num_layers, num_experts]
    return torch.tensor(rows, dtype=torch.float32)


def load_json_to_tensor(json_path: str) -> torch.Tensor:
    """
    Load a [num_layers, num_experts] token count tensor from a JSON file.

    The JSON is expected to have the structure:
    {
        "logical_count": [
            [token_count_expert_0, ..., token_count_expert_n],
            ...
        ]
    }

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        torch.Tensor: A tensor of shape [num_layers, num_experts] (e.g., [61, 256]).
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File does not exist: {json_path}")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

    if "logical_count" not in data:
        raise KeyError("Missing 'logical_count' key in the JSON file")

    logical_count = data["logical_count"]

    if not isinstance(logical_count, list) or not all(
        isinstance(row, list) for row in logical_count
    ):
        raise TypeError("'logical_count' must be a 2D list")

    num_layers = len(logical_count)
    expert_counts = {len(row) for row in logical_count}

    if len(expert_counts) != 1:
        raise ValueError("Inconsistent number of experts per layer")

    try:
        tensor = torch.tensor(
            [[int(val) for val in row] for row in logical_count], dtype=torch.float32
        )
    except Exception as e:
        raise ValueError(f"Failed to convert values to int tensor: {e}")

    print(f"Loaded tensor with shape: {tensor.shape}")
    return tensor


def measure_execution_time(func: Callable[[], dict], warmup: int, repeat: int):
    """Measure execution time of a function with warmup and repeat runs."""
    for _ in range(warmup):
        func()

    times = []
    final_result = None
    for _ in range(repeat):
        start = time.perf_counter()
        final_result = func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "result": final_result,
        "avg": statistics.mean(times),
        "std": statistics.stdev(times) if repeat > 1 else 0,
        "min": min(times),
        "max": max(times),
        "all_times": times,
    }


def count_expert_moves(assignment: torch.Tensor) -> int:
    """Count expert moves from vanilla assignment.
    
    Args:
        assignment: One-hot assignment tensor [num_gpus, num_experts]
        
    Returns:
        Number of expert moves from original positions
    """
    nb_experts_per_gpu = assignment.shape[1] // assignment.shape[0]
    return sum(
        [
            sum(
                [
                    int(assignment[gpu][expert].item())
                    for expert in range(assignment.shape[1])
                    if (expert // nb_experts_per_gpu) != gpu
                ]
            )
            for gpu in range(assignment.shape[0])
        ]
    )


def get_naive_assignment(nb_gpus: int, nb_experts: int) -> torch.Tensor:
    """Return vanilla one-hot assignment of experts to GPUs."""
    nb_experts_per_gpu = nb_experts // nb_gpus
    return torch.Tensor(
        [
            [
                (
                    1
                    if i < (j + 1) * nb_experts_per_gpu and i >= j * nb_experts_per_gpu
                    else 0
                )
                for i in range(nb_experts)
            ]
            for j in range(nb_gpus)
        ]
    ).long()


def phy_to_assignment(
    phy2log: torch.Tensor, nb_gpus: int, nb_experts: int
) -> torch.Tensor:
    """
    Convert phy2log to one-hot assignment.
    
    If assignment[0][32] = 1, then in this layer, logical_expert 32 is located on gpu0.
    """
    assert nb_experts % nb_gpus == 0, "Experts cannot be evenly divided among GPUs."
    return torch.sum(
        torch.nn.functional.one_hot(phy2log).view(nb_gpus, nb_experts // nb_gpus, -1),
        dim=1,
    ) 