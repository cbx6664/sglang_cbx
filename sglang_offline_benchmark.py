"""Benchmark offline inference throughput with SGLang using real dataset."""
import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from typing import Tuple

# Add the correct sglang python package path
sys.path.insert(0, '/sgl-workspace/sglang_private/python')

import torch
import pandas as pd
import sglang as sgl
from sglang.srt.server_args import ServerArgs

torch.manual_seed(0)
import logging
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BenchArgs:
    dataset_path: str = "/home/bingxche/data/09292024_mixtral_15k_mintoken2_v1.pkl"
    dataset_type: str = "all"  # "all", "GSM8K", "OpenOrca", "MBXP", or comma-separated like "GSM8K,OpenOrca"
    num_samples: int = 100
    result_filename: str = "result.jsonl"
    result_dir: str = "."
    max_new_tokens: int = 1024
    ignore_eos: bool = False
    record_expert_distribution: bool = True
    expert_distribution_dir: str = "/home/bingxche/log/expert_distribution"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--dataset-path", type=str, default=BenchArgs.dataset_path,
                           help="Path to the dataset pickle file")
        parser.add_argument("--dataset-type", type=str, default=BenchArgs.dataset_type,
                           help="Dataset type to use: 'all', 'GSM8K', 'OpenOrca', 'MBXP', or comma-separated like 'GSM8K,OpenOrca'")
        parser.add_argument("--num-samples", type=int, default=BenchArgs.num_samples,
                           help="Number of samples to use from dataset")
        parser.add_argument("--result-filename", type=str, default=BenchArgs.result_filename,
                           help="Output filename for results")
        parser.add_argument("--result-dir", type=str, default=BenchArgs.result_dir,
                           help="Directory to save result files")
        parser.add_argument("--max-new-tokens", type=int, default=BenchArgs.max_new_tokens,
                           help="Maximum number of new tokens to generate")
        parser.add_argument("--ignore-eos", action="store_true",
                           help="Ignore end of sequence tokens")
        parser.add_argument("--record-expert-distribution", action="store_true",
                           help="Enable expert distribution recording")
        parser.add_argument("--expert-distribution-dir", type=str, default=BenchArgs.expert_distribution_dir,
                           help="Directory to save expert distribution files")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def get_sglang_engine(server_args: ServerArgs, bench_args: BenchArgs):
    """Create SGLang engine instance using ServerArgs"""
    # Enable expert distribution recorder if requested
    if bench_args.record_expert_distribution:
        if server_args.expert_distribution_recorder_mode is None:
            server_args.expert_distribution_recorder_mode = "stat"
        # Set environment variable for output directory
        os.makedirs(bench_args.expert_distribution_dir, exist_ok=True)
        os.environ["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"] = bench_args.expert_distribution_dir
    
    return sgl.Engine(server_args=server_args)


def load_real_dataset(dataset_path, num_samples=100, dataset_type="all"):
    """Load the real dataset using raw text input with optional dataset type filtering"""
    print(f"Loading dataset from {dataset_path}")
    full_data = pd.read_pickle(dataset_path)
    
    # Filter by dataset type if specified
    if dataset_type != "all":
        # Split comma-separated dataset types
        requested_types = [dtype.strip() for dtype in dataset_type.split(",")]
        
        # Validate dataset types
        available_types = full_data['dataset'].unique()
        invalid_types = [dtype for dtype in requested_types if dtype not in available_types]
        if invalid_types:
            raise ValueError(f"Invalid dataset types: {invalid_types}. Available types: {list(available_types)}")
        
        # Filter the data
        filtered_data = full_data[full_data['dataset'].isin(requested_types)]
        print(f"Filtered to dataset types: {requested_types}")
        print(f"Available samples per dataset:")
        for dtype in requested_types:
            count = len(filtered_data[filtered_data['dataset'] == dtype])
            print(f"  {dtype}: {count} samples")
    else:
        filtered_data = full_data
        print(f"Using all dataset types:")
        print(f"Available samples per dataset:")
        for dtype in full_data['dataset'].unique():
            count = len(full_data[full_data['dataset'] == dtype])
            print(f"  {dtype}: {count} samples")
    
    # Sample the requested number of samples
    if len(filtered_data) > num_samples:
        processed_data = filtered_data.sample(n=num_samples, random_state=42).reset_index(drop=True)
        print(f"Randomly sampled {num_samples} from {len(filtered_data)} available samples")
    else:
        processed_data = filtered_data.reset_index(drop=True)
        print(f"Using all {len(processed_data)} available samples (requested {num_samples})")
    
    # Show final distribution
    print(f"Final sample distribution:")
    final_counts = processed_data['dataset'].value_counts()
    for dtype, count in final_counts.items():
        print(f"  {dtype}: {count} samples")
    
    prompts = []
    for idx, request in processed_data.iterrows():
        prompts.append(request['input'])  # Just use the raw text
   
    print(f"Loaded {len(prompts)} requests")
    return prompts


def run_sglang(prompts, sampling_params, server_args: ServerArgs, bench_args: BenchArgs):
    """Run SGLang inference"""
    llm = get_sglang_engine(server_args, bench_args)
    
    # Expert distribution recording
    if bench_args.record_expert_distribution:
        llm.start_expert_distribution_record()
        logger.info("start_expert_distribution_record()")
    
    start = time.perf_counter()
    responses = llm.generate(prompt=prompts, sampling_params=sampling_params)
    end = time.perf_counter()
    
    if bench_args.record_expert_distribution:
        llm.stop_expert_distribution_record()
        logger.info("stop_expert_distribution_record()")
        llm.dump_expert_distribution_record()
        logger.info(f"dump_expert_distribution_record()"
                    f"\nExpert distribution saved to: {bench_args.expert_distribution_dir}")
    
    # Extract token counts and analyze end reasons
    input_tokens = 0
    output_tokens = 0
    end_reason_stats = {}
    sample_responses = []
    
    for i, response in enumerate(responses):
        if 'meta_info' in response:
            input_tokens += response['meta_info'].get('prompt_tokens', 0)
            output_tokens += response['meta_info'].get('completion_tokens', 0)
            
            # Track end reasons - handle both string and dict cases
            finish_reason_raw = response['meta_info'].get('finish_reason', 'unknown')
            if isinstance(finish_reason_raw, dict):
                # If it's a dict, try to extract the actual reason
                finish_reason = str(finish_reason_raw.get('reason', finish_reason_raw))
            elif isinstance(finish_reason_raw, str):
                finish_reason = finish_reason_raw
            else:
                finish_reason = str(finish_reason_raw)
            
            end_reason_stats[finish_reason] = end_reason_stats.get(finish_reason, 0) + 1
        
        # Collect sample responses (first 3)
        if i < 3:
            # Extract finish reason for sample
            finish_reason_raw = response.get('meta_info', {}).get('finish_reason', 'unknown')
            if isinstance(finish_reason_raw, dict):
                finish_reason_display = str(finish_reason_raw.get('reason', finish_reason_raw))
            elif isinstance(finish_reason_raw, str):
                finish_reason_display = finish_reason_raw
            else:
                finish_reason_display = str(finish_reason_raw)
                
            sample_responses.append({
                'prompt_preview': prompts[i][:100] + "..." if len(prompts[i]) > 100 else prompts[i],
                'response': response.get('text', ''),
                'finish_reason': finish_reason_display,
                'prompt_tokens': response.get('meta_info', {}).get('prompt_tokens', 0),
                'completion_tokens': response.get('meta_info', {}).get('completion_tokens', 0)
            })
    
    total_tokens = input_tokens + output_tokens
    num_requests = len(prompts)
    elapsed_time = end - start
    
    # Save results to JSON file
    os.makedirs(bench_args.result_dir, exist_ok=True)
    result_path = os.path.join(bench_args.result_dir, bench_args.result_filename)
    result = {
        "model_path": server_args.model_path,
        "tp_size": server_args.tp_size,
        "dataset_type": bench_args.dataset_type,
        "num_requests": num_requests,
        "num_samples": bench_args.num_samples,
        "total_time": elapsed_time,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "requests_per_second": num_requests / elapsed_time,
        "tokens_per_second": total_tokens / elapsed_time,
        "input_tokens_per_second": input_tokens / elapsed_time,
        "output_tokens_per_second": output_tokens / elapsed_time,
        "avg_input_tokens": input_tokens / num_requests,
        "avg_output_tokens": output_tokens / num_requests,
        "end_reason_stats": end_reason_stats,
        "sample_responses": sample_responses,
        "server_args": dataclasses.asdict(server_args),
        "bench_args": dataclasses.asdict(bench_args),
        "sampling_params": {
            "max_new_tokens": bench_args.max_new_tokens,
            "ignore_eos": bench_args.ignore_eos,
        }
    }
    
    # Save to JSONL file
    with open(result_path, 'a') as f:
        f.write(json.dumps(result) + '\n')
    
    print(f"time: {elapsed_time:.2f}s")
    print(f"average input tokens: {input_tokens / num_requests:.1f}")
    print(f"average output tokens: {output_tokens / num_requests:.1f}")
    
    # Print end reason statistics
    print(f"\n=== END REASON STATISTICS ===")
    for reason, count in end_reason_stats.items():
        percentage = (count / num_requests) * 100
        print(f"{reason}: {count} ({percentage:.1f}%)")
    
    # Print sample responses
    print(f"\n=== SAMPLE RESPONSES ===")
    for i, sample in enumerate(sample_responses):
        print(f"\nSample {i+1}:")
        print(f"  Prompt: {sample['prompt_preview']}")
        print(f"  Finish reason: {sample['finish_reason']}")
        print(f"  Tokens: {sample['prompt_tokens']} -> {sample['completion_tokens']}")
        print(f"  Response: {sample['response'][:200]}..." if len(sample['response']) > 200 else f"  Response: {sample['response']}")
    
    print("sample response:", responses[0]['text'][:100] if responses else "No response")
    
    llm.shutdown()
    return input_tokens, output_tokens, elapsed_time, result_path


def main(server_args: ServerArgs, bench_args: BenchArgs):
    """Main benchmark function"""
    # Load dataset
    prompts = load_real_dataset(bench_args.dataset_path, bench_args.num_samples, bench_args.dataset_type)
    
    # Create sampling params
    sampling_params = {
        "max_new_tokens": bench_args.max_new_tokens,
        "ignore_eos": bench_args.ignore_eos,
    }
    
    print(f"\n=== BENCHMARK SETUP ===")
    print(f"Model: {server_args.model_path}")
    print(f"Dataset type: {bench_args.dataset_type}")
    print(f"Total requests: {len(prompts)}")
    
    input_tokens, output_tokens, elapsed_time, result_path = run_sglang(prompts, sampling_params, server_args, bench_args)
    
    total_tokens = input_tokens + output_tokens
    num_requests = len(prompts)
    
    print(f"\n=== THROUGHPUT RESULTS ===")
    print(f"Throughput: {num_requests / elapsed_time:.2f} requests/s, "
            f"{total_tokens / elapsed_time:.2f} total tokens/s, "
            f"{output_tokens / elapsed_time:.2f} output tokens/s")
    
    print(f"\n=== DETAILED METRICS ===")
    print(f"Total requests: {num_requests}")
    print(f"Total elapsed time: {elapsed_time:.2f}s")
    print(f"Requests/s: {num_requests / elapsed_time:.2f}")
    print(f"Total tokens/s: {total_tokens / elapsed_time:.2f}")
    print(f"Input tokens/s: {input_tokens / elapsed_time:.2f}")
    print(f"Output tokens/s: {output_tokens / elapsed_time:.2f}")
    print(f"Total tokens processed: {total_tokens:,}")
    
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGLang Offline Batch Inference Benchmark")
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)
    
    main(server_args, bench_args) 