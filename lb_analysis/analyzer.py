"""
Analyzes token distribution and load balancing
"""

import os
import re
import json
from typing import Tuple, List, Optional

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


def calculate_gpu_loads(vanilla_hotness: torch.Tensor, num_gpus: int):
    num_layers, num_experts = vanilla_hotness.shape
    experts_per_gpu = num_experts // num_gpus

    assert num_experts % num_gpus == 0, "Experts cannot be evenly divided among GPUs."

    gpu_loads = torch.zeros((num_layers, num_gpus), dtype=vanilla_hotness.dtype)

    for layer_idx in range(num_layers):
        for gpu_idx in range(num_gpus):
            start_idx = gpu_idx * experts_per_gpu
            end_idx = (gpu_idx + 1) * experts_per_gpu
            gpu_loads[layer_idx, gpu_idx] = vanilla_hotness[
                layer_idx, start_idx:end_idx
            ].sum()

    return gpu_loads


def save_gpu_loads_to_csv(
    gpu_loads: torch.Tensor, output_folder: str, file_name: str, num_gpus: int
):
    gpu_loads_df = pd.DataFrame(
        gpu_loads.numpy(),
        columns=[f"GPU{i}" for i in range(num_gpus)],
    )
    csv_save_path = os.path.join(output_folder, file_name)
    gpu_loads_df.to_csv(csv_save_path)


def plot_gpu_loads_analysis(
    gpu_loads: torch.Tensor, heatmap_save_path: str, boxplot_save_path: str
):
    """
    Plot normalized heatmap and boxplot of GPU loads.

    Args:
        gpu_loads (torch.Tensor): Tensor of shape [58, 8] (layers x GPUs).
        heatmap_save_path (str): File path to save heatmap.
        boxplot_save_path (str): File path to save boxplot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Normalize the gpu loads per layer
    layer_sums = gpu_loads.sum(dim=1, keepdim=True)  # Sum across GPUs for each layer
    normalized_gpu_loads = gpu_loads / layer_sums  # Each row sums to 1

    # 1. Heatmap: Normalized Load per layer per GPU
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        normalized_gpu_loads.numpy(),
        cmap="Reds",
        annot=True,
        fmt=".2f",  # Show 2 decimal places for better precision
        linewidths=0.5,
        cbar_kws={"label": "Normalized Load"},
        yticklabels=list(range(gpu_loads.shape[0])),
        annot_kws={"size": 6},
    )
    plt.xlabel("GPU ID")
    plt.ylabel("Layer ID")
    plt.title("Normalized GPU Loads")
    plt.tight_layout()
    plt.savefig(heatmap_save_path)
    print(f"Normalized heatmap saved to: {heatmap_save_path}")
    plt.close()

    # 2. Boxplot: Distribution of normalized GPU loads across layers
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=normalized_gpu_loads.numpy())
    plt.axhline(
        y=1.0 / normalized_gpu_loads.shape[1], color="blue", linestyle="-"
    )  # Ideal line
    plt.xlabel("GPU ID")
    plt.ylabel("Normalized Load")
    plt.title("Normalized GPU Loads")
    plt.tight_layout()
    plt.savefig(boxplot_save_path)
    print(f"Normalized boxplot saved to: {boxplot_save_path}")
    plt.close()


def plot_hotness_heatmap(
    hotness: torch.Tensor,
    save_path: str,
    title: str = "Normalized Expert Hotness Heatmap",
):
    """
    Plot normalized heatmap of expert hotness.

    Args:
        hotness (torch.Tensor): Tensor of shape like [58, 256] (layers x experts).
        save_path (str): File path to save heatmap.
        title (str): Title for the plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Normalize per layer (row-wise)
    hotness = hotness.cpu()
    layer_sums = hotness.sum(dim=1, keepdim=True)
    normalized_hotness = hotness / layer_sums  # shape [58, 256]

    plt.figure(figsize=(16, 8))
    sns.heatmap(
        normalized_hotness.numpy(),
        cmap="Reds",
        annot=False,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "Normalized Hotness"},
        yticklabels=list(range(hotness.shape[0])),
        xticklabels=False  
    )
    plt.xlabel("Logical Expert ID")
    plt.ylabel("Layer ID")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Expert hotness heatmap saved to: {save_path}")
    plt.close()



def remap_weights(
    phy2log: torch.Tensor, original_weights: torch.Tensor
) -> torch.Tensor:
    """
    Remap the original logical expert weights according to the phy2log mapping.

    Args:
        phy2log (torch.Tensor): A tensor of shape [58, 256], where each value is a logical_expert_id.
        original_weights (torch.Tensor): A tensor of shape [58, 256], where the index is the logical_expert_id and the value is the corresponding weight.

    Returns:
        torch.Tensor: A new tensor of shape [58, 256], where each physical expert has its remapped weight.
    """
    num_layers, num_physical_experts = phy2log.shape
    new_weights = torch.zeros_like(phy2log, dtype=original_weights.dtype)

    for layer_idx in range(num_layers):
        # Get the mapping from physical experts to logical experts for this layer
        logical_ids = phy2log[layer_idx]  # Shape: [256]
        # Fetch the corresponding weights from original_weights using the logical expert IDs
        new_weights[layer_idx] = original_weights[layer_idx][logical_ids]

    return new_weights


def load_expert_allocation(file_path: str) -> Optional[torch.Tensor]:
    """Load expert allocation from a JSON or CSV file and return as a torch.Tensor."""
    if not file_path or not os.path.exists(file_path):
        print(f"File not found at {file_path}")
        return None

    try:
        if file_path.lower().endswith(".json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Assuming the JSON contains a list of lists
                return torch.tensor(data)
        elif file_path.lower().endswith(".csv"):
            # Assuming the CSV file has no header and contains numeric data
            df = pd.read_csv(file_path, header=None)
            return torch.tensor(df.values)
        else:
            print(f"Unsupported file format for {file_path}. Please use .json or .csv.")
            return None
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to load expert allocation from {file_path}: {e}")
        return None


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    input_folder = r"C:\Users\bingxche\data\log\deepseek-v3_tp8_mixtral_dataset_5000_prompts_vanilla\moe_token_dist"
    output_folder = os.path.join(input_folder,"bldm")
    os.makedirs(output_folder, exist_ok=True)
    
    vanilla_hotness = load_csv_to_tensor(input_folder=input_folder)
    hotness = remap_weights(
        phy2log=load_expert_allocation(r"C:\Users\bingxche\data\log\deepseek-v3_tp8_mixtral_dataset_5000_prompts_vanilla\moe_token_dist\analysis\bldm_phy2log.csv"),
        original_weights=vanilla_hotness
    )
    hotness_heatmap_path = os.path.join(output_folder, "bldm_hotness_heatmap.png")
    plot_hotness_heatmap(hotness=hotness, save_path=hotness_heatmap_path)
    
    gpu_loads = calculate_gpu_loads(vanilla_hotness=hotness, num_gpus=8)
    heatmap_save_path = os.path.join(output_folder, "bldm_gpu_loads_heatmap.png")
    boxplot_save_path = os.path.join(output_folder, "bldm_gpu_loads_boxplot.png")
    plot_gpu_loads_analysis(gpu_loads=gpu_loads, heatmap_save_path=heatmap_save_path, boxplot_save_path=boxplot_save_path)
    
