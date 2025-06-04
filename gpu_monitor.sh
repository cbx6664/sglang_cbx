#!/bin/bash

# GPU监控脚本 - 优化版
# 使用方法: ./gpu_monitor.sh [输出文件] [监控间隔秒数]

OUTPUT_FILE=${1:-"gpu_monitor_$(date +%Y%m%d_%H%M%S).csv"}
INTERVAL=${2:-1}

echo "开始GPU监控..."
echo "输出文件: $OUTPUT_FILE"
echo "监控间隔: ${INTERVAL}秒"
echo "按 Ctrl+C 停止监控"

# 创建CSV头部
echo "timestamp,$(rocm-smi --showmemuse --showuse --showpower --csv 2>/dev/null | head -1)" > "$OUTPUT_FILE"

# 监控循环
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 获取GPU数据，跳过头部，为每行添加时间戳
    rocm-smi --showmemuse --showuse --showpower --csv 2>/dev/null | tail -n +2 | while IFS= read -r line; do
        echo "$TIMESTAMP,$line" >> "$OUTPUT_FILE"
    done
    
    sleep "$INTERVAL"
done 