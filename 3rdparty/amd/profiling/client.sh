#!/bin/bash

# Start profiling via API
curl http://localhost:30000/start_profile -H "Content-Type: application/json"

# Benchmark serving using sglang with random dataset and tokenizer
# Define the log file with a timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="sglang_client_log_$TIMESTAMP.json"

# Run the benchmark with specified parameters and save logs
python3 -m sglang.bench_serving \
    --backend sglang \
    --model /home/bingxche/deepseek-v3-5layer \
    --dataset-name sharegpt \
    --sharegpt-output-len 10 \
    --num-prompts 1 \
    --output-file online.jsonl 2>&1 | tee "$LOGFILE"

# Stop profiling via API
curl http://localhost:30000/stop_profile -H "Content-Type: application/json"

# Convert tracing file to csv & json
sqlite3 trace.rpd ".mode csv" ".header on" ".output trace.csv" "select * from top;" ".output stdout"
python3 /home/bingxche/bind/sglang_cbx/3rdparty/amd/profiling/rocmProfileData/tools/rpd2tracing.py trace.rpd trace.json
