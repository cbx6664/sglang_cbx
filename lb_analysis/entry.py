"""Benchmark offline inference throughput using SGLang."""

import time
import dataclasses
import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import torch
from sglang.srt.server_args import ServerArgs
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.entrypoints.engine import Engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment(server_args) -> None:
    """Set up environment variables for the benchmark."""
    os.environ["MODEL_PATH"] = server_args.model_path.lower()
    
    # directory of log files
    model_name = os.path.basename(server_args.model_path.lower())
    os.environ["LOG_DIR"] = f"/home/bingxche/log/{model_name}_ep4_mixtral_dataset_{CONFIG['num_samples']}_prompts_vanilla"
    
    # whether to log expert allocation, token distribution info...
    os.environ["RECORD_EXPERT_ALLOCATION"] = "False"
    os.environ["RECORD_TOKEN_DISTRIBUTION"] = "False"
    
    # whether to use custom expert allocation
    os.environ["CUSTOM_EXPERT_ALLOCATION"] = "False"
    # this should match the number of physical experts if we use custom expert allocation
    os.environ["NUM_EXPERTS"] = "8"
    # file path of custom_expert_allocation.csv
    os.environ["EXPERT_ALLOCATION_FILE_PATH"] = "/home/bingxche/log/mixtral8x7b_ep4_mixtral_dataset_5_prompts_vanilla/moe_token_dist_eplb_8replicas_1groups_1nodes_4gpus/phy2log_8replicas_1groups_1nodes_4gpus.json"


def get_engine_instance() -> Engine:
    """Create and return an Engine instance with the specified configuration."""
    server_args = ServerArgs(
        # model_path="/home/bingxche/deepseek-v3",
        model_path="/home/bingxche/Mixtral-8x7B-Instruct-v0.1",
        tp_size=4,
        # ep_size=8,
        enable_ep_moe=True,
        trust_remote_code=True,
        disable_cuda_graph=True,
    )
    
    setup_environment(server_args)

    return Engine(**dataclasses.asdict(server_args))

# Configuration constants
CONFIG = {
    "data_path": "/home/bingxche/data/09292024_mixtral_15k_mintoken2_v1.pkl",
    "num_samples": 5000,  # number of rows of dataset to use for inference
    "enable_profiling": False,
}

DEFAULT_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 0.001,
    "max_new_tokens": 1024,
    "ignore_eos": False,
}

def load_sample_requests() -> List[Tuple[str, int, int]]:
    """Load and sort sample requests by input length in descending order."""
    processed_data = pd.read_pickle(CONFIG["data_path"]).head(CONFIG["num_samples"])

    prompts = []
    for _, request in processed_data.iterrows():
        prompts.append(
            (request["input"], request["tok_input_len"], request["tok_ref_output_len"])
        )

    # Sort prompts by descending length for better batching efficiency
    return sorted(prompts, key=lambda x: x[1], reverse=True)


def monitor_trace_file(directory: str, interval: int = 1) -> None:
    """Monitor newly created trace files until they stop growing."""
    logger.info(f"Monitoring {directory} for new trace files...")
    known_files = set(os.listdir(directory))

    while True:
        flag = False
        time.sleep(interval)
        current_files = set(os.listdir(directory))

        new_files = current_files - known_files
        for new_file in new_files:
            new_file_path = os.path.join(directory, new_file)
            print(f"New file detected: {new_file}")

            previous_size = 0
            while True:
                try:
                    current_size = os.path.getsize(new_file_path)
                except FileNotFoundError:
                    print(f"File {new_file} is no longer accessible.")
                    break

                if current_size > previous_size:
                    previous_size = current_size
                else:
                    flag = True
                    break

                time.sleep(interval)
        if flag:
            break


def save_outputs_to_json(outputs: Any) -> None:
    """Save outputs to a JSON file."""
    path = os.path.join(os.getenv("LOG_DIR"), "outputs.json")
    with open(path, "a") as fout:
        fout.write(json.dumps(outputs) + "\n")

    output_count = len(outputs) if isinstance(outputs, list) else 1
    logger.info(f"Saved {output_count} entries to {path}")


def profile_run_sglang(
    requests: List[Tuple[str, int, int]], sampling_params: Dict[str, Any]
) -> None:
    """Run SGLang with profiling enabled."""
    # Setup profiling directory
    profile_dir = os.path.join(os.getenv("LOG_DIR"), "trace")
    os.environ["SGLANG_TORCH_PROFILER_DIR"] = profile_dir
    os.makedirs(os.getenv("SGLANG_TORCH_PROFILER_DIR"), exist_ok=True)

    engine = get_engine_instance()

    prompts = [req[0] for req in requests]

    # Warmup
    logger.info("Starting warmup...")
    engine.generate(
        prompt=[""],
        sampling_params=[{"temperature": 0, "max_new_tokens": 1, "ignore_eos": False}],
    )
    time.sleep(0.5)

    # Profiling run
    logger.info("Starting profiling...")
    engine.start_profile()
    start = time.perf_counter()
    outputs = engine.generate(prompt=prompts, sampling_params=sampling_params)
    end = time.perf_counter()
    engine.stop_profile()

    # Monitor trace files
    monitor_trace_file(os.getenv("SGLANG_TORCH_PROFILER_DIR"))

    latency = end - start
    logger.info(f"Profiling run completed in {latency:.2f} seconds")

    # Calculate metrics
    total_input_tokens = sum(req[1] for req in requests)
    total_output_tokens = sum(o["meta_info"]["completion_tokens"] for o in outputs)

    results = {
        "successful_requests": len(requests),
        "total_latency": latency,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": len(requests) / latency,
        "input_throughput": total_input_tokens / latency,
        "output_throughput": total_output_tokens / latency,
        "total_throughput": (total_input_tokens + total_output_tokens) / latency,
        "prompts": prompts,
        "sampling_params": sampling_params,
        "model_output": outputs,
    }

    save_outputs_to_json(results)
    engine.shutdown()


def run_sglang(
    requests: List[Tuple[str, int, int]], sampling_params: Dict[str, Any]
) -> Tuple[List[str], List[str], float]:
    """Run SGLang inference without profiling."""
    
    engine = get_engine_instance()
    
    prompts = [req[0] for req in requests]

    # Generate responses
    start = time.perf_counter()
    outputs = engine.generate(prompt=prompts, sampling_params=sampling_params)
    end = time.perf_counter()
    latency = end - start

    # Extract output texts
    output_texts = []
    if isinstance(outputs, list):
        for output in outputs:
            output_texts.append(output.get("text", output.get("meta_info", "")))
    else:
        output_texts.append(outputs.get("text", outputs.get("meta_info", "")))

    # Calculate metrics
    total_input_tokens = sum(req[1] for req in requests)
    total_output_tokens = sum(o["meta_info"]["completion_tokens"] for o in outputs)

    results = {
        "total_latency": latency,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": len(requests) / latency,
        "input_throughput": total_input_tokens / latency,
        "output_throughput": total_output_tokens / latency,
        "total_throughput": (total_input_tokens + total_output_tokens) / latency,
        "prompts": prompts,
        "sampling_params": sampling_params,
        "model_output": outputs,
    }

    save_outputs_to_json(outputs)
    engine.shutdown()

    return prompts, output_texts, latency


def calculate_throughput(
    requests: List[Tuple[str, int, int]],
    prompts: List[str],
    outputs: List[str],
    elapsed_time: float,
) -> None:
    """Calculate and log throughput metrics."""
    if len(requests) != len(outputs):
        raise ValueError("Number of requests doesn't match number of outputs")

    # Calculate token counts
    total_input_tokens = sum(req[1] for req in requests)
    total_output_tokens = sum(len(output) for output in outputs)
    total_tokens = total_input_tokens + total_output_tokens

    # Calculate throughput metrics
    req_throughput = len(requests) / elapsed_time
    token_throughput = total_tokens / elapsed_time
    output_token_throughput = total_output_tokens / elapsed_time

    # Log results
    logger.info(
        f"Throughput: {req_throughput:.2f} requests/s, "
        f"{token_throughput:.2f} total tokens/s, "
        f"{output_token_throughput:.2f} output tokens/s"
    )


def main(enable_profiling: bool = False) -> None:
    """Main function to run the benchmark."""
    requests = load_sample_requests()

    if enable_profiling:
        logger.info("Running with profiling enabled")
        profile_run_sglang(requests, DEFAULT_SAMPLING_PARAMS)
    else:
        logger.info("Running standard inference")
        prompts, outputs, latency = run_sglang(requests, DEFAULT_SAMPLING_PARAMS)
        calculate_throughput(requests, prompts, outputs, latency)


if __name__ == "__main__":
    main(enable_profiling=CONFIG["enable_profiling"]) 