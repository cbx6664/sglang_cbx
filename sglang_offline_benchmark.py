"""Benchmark offline inference throughput with SGLang using real dataset."""
import argparse
import dataclasses
import json
import logging
import time
from typing import Tuple

import torch
import pandas as pd
import sglang as sgl
from sglang.srt.server_args import ServerArgs

torch.manual_seed(0)


@dataclasses.dataclass
class BenchArgs:
    dataset_path: str = "/work1/amd/bingxche/data/09292024_mixtral_15k_mintoken2_v1.pkl"
    num_samples: int = 100
    result_filename: str = "result.jsonl"
    profile: bool = False
    profile_filename_prefix: str = "profile"
    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 0.001
    max_new_tokens: int = 1024
    ignore_eos: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--dataset-path", type=str, default=BenchArgs.dataset_path,
                           help="Path to the dataset pickle file")
        parser.add_argument("--num-samples", type=int, default=BenchArgs.num_samples,
                           help="Number of samples to use from dataset")
        parser.add_argument("--result-filename", type=str, default=BenchArgs.result_filename,
                           help="Output filename for results")
        parser.add_argument("--profile", action="store_true",
                           help="Enable profiling")
        parser.add_argument("--profile-filename-prefix", type=str, default=BenchArgs.profile_filename_prefix,
                           help="Prefix for profiling output files")
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature,
                           help="Sampling temperature")
        parser.add_argument("--top-k", type=int, default=BenchArgs.top_k,
                           help="Top-k sampling parameter")
        parser.add_argument("--top-p", type=float, default=BenchArgs.top_p,
                           help="Top-p sampling parameter")
        parser.add_argument("--max-new-tokens", type=int, default=BenchArgs.max_new_tokens,
                           help="Maximum number of new tokens to generate")
        parser.add_argument("--ignore-eos", action="store_true",
                           help="Ignore EOS token during generation")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def get_sglang_engine(server_args: ServerArgs):
    """Create SGLang engine instance using ServerArgs"""
    # The correct way: pass all ServerArgs as dict to Engine
    return sgl.Engine(**dataclasses.asdict(server_args))


def load_real_dataset(dataset_path, num_samples=100):
    """Load the real dataset using raw text input"""
    print(f"Loading dataset from {dataset_path}")
    processed_data = pd.read_pickle(dataset_path).iloc[0:num_samples]
    
    prompts = []
    for idx, request in processed_data.iterrows():
        prompts.append(request['input'])  # Just use the raw text
   
    print(f"Loaded {len(prompts)} requests")
    return prompts


def profile_run_sglang(prompts, sampling_params, server_args: ServerArgs, bench_args: BenchArgs):
    """Run SGLang with profiling enabled"""
    llm = get_sglang_engine(server_args)
    
    # Create profile output filename
    profile_filename = f"{bench_args.profile_filename_prefix}_samples{len(prompts)}.trace.json.gz"
    profile_dir = f"/tmp/sglang_trace_dir"
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir))
    ) as p:
        start = time.perf_counter()
        responses = llm.generate(prompt=prompts, sampling_params=sampling_params)
        end = time.perf_counter()
    
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    print(f"Profile saved to {profile_dir}")
    print(f"time: {end - start:.2f}s")
    llm.shutdown()


def run_sglang(prompts, sampling_params, server_args: ServerArgs, bench_args: BenchArgs):
    """Run SGLang inference"""
    llm = get_sglang_engine(server_args)
    
    start = time.perf_counter()
    responses = llm.generate(prompt=prompts, sampling_params=sampling_params)
    end = time.perf_counter()
    
    # Extract token counts from responses
    input_tokens = 0
    output_tokens = 0
    
    for response in responses:
        if 'meta_info' in response:
            input_tokens += response['meta_info'].get('prompt_tokens', 0)
            output_tokens += response['meta_info'].get('completion_tokens', 0)
    
    total_tokens = input_tokens + output_tokens
    num_requests = len(prompts)
    elapsed_time = end - start
    
    # Save results to JSON file
    result = {
        "model_path": server_args.model_path,
        "tp_size": server_args.tp_size,
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
        "sampling_params": {
            "temperature": bench_args.temperature,
            "top_k": bench_args.top_k,
            "top_p": bench_args.top_p,
            "max_new_tokens": bench_args.max_new_tokens,
            "ignore_eos": bench_args.ignore_eos,
        }
    }
    
    # Save to JSONL file
    with open(bench_args.result_filename, 'a') as f:
        f.write(json.dumps(result) + '\n')
    
    print(f"time: {elapsed_time:.2f}s")
    print(f"average input tokens: {input_tokens / num_requests:.1f}")
    print(f"average output tokens: {output_tokens / num_requests:.1f}")
    print("sample response:", responses[0]['text'][:100] if responses else "No response")
    
    llm.shutdown()
    return input_tokens, output_tokens, elapsed_time


def main(server_args: ServerArgs, bench_args: BenchArgs):
    """Main benchmark function"""
    # Load dataset
    prompts = load_real_dataset(bench_args.dataset_path, bench_args.num_samples)
    
    # Create sampling params
    sampling_params = {
        "temperature": bench_args.temperature,
        "top_k": bench_args.top_k,
        "top_p": bench_args.top_p,
        "max_new_tokens": bench_args.max_new_tokens,
        "ignore_eos": bench_args.ignore_eos,
    }
    
    print(f"\n=== BENCHMARK SETUP ===")
    print(f"Model: {server_args.model_path}")
    print(f"Total requests: {len(prompts)}")
    
    if bench_args.profile:
        profile_run_sglang(prompts, sampling_params, server_args, bench_args)
    else:
        input_tokens, output_tokens, elapsed_time = run_sglang(prompts, sampling_params, server_args, bench_args)
        
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
        
        print(f"\nResults saved to {bench_args.result_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGLang Offline Batch Inference Benchmark")
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)
    
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )
    
    main(server_args, bench_args) 