#!/usr/bin/env python3
"""
简化的实验启动脚本
"""

import os
import sys
from pathlib import Path

# 解决OpenMP冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 确保可以导入模块
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    # 修改为你的实际路径
    DATA_PATH = r"C:\Users\bingxche\data\log\deepseek-v3_tp8_mixtral_dataset_5000_prompts_vanilla\moe_token_dist"
    OUTPUT_ROOT = r"C:\Users\bingxche\data\log\analysis_results"
    
    print("🚀 Load Balance Analysis")
    print("=" * 50)
    print(f"Data Path: {DATA_PATH}")
    print(f"Output Path: {OUTPUT_ROOT}")
    
    try:
        from load_balance_analyzer import LoadBalanceAnalyzer
        
        # 1. Vanilla 8 GPU
        print("\n📊 Running Vanilla 8 GPU...")
        analyzer = LoadBalanceAnalyzer(DATA_PATH, 8, f"{OUTPUT_ROOT}/vanilla_8gpu")
        analyzer.load_data()
        vanilla_result = analyzer.analyze_vanilla_distribution()
        print(f"✅ Vanilla std_dev: {vanilla_result['metrics']['std_dev']:.6f}")
        
        # 2. EPLB 8 GPU
        print("\n🔄 Running EPLB 8 GPU...")
        config = {"num_replicas": 256, "num_gpus": 8, "num_groups": 1, "num_nodes": 1}
        eplb_result = analyzer.run_algorithm("eplb", config, measure_time=False)
        print(f"✅ EPLB std_dev: {eplb_result['metrics']['std_dev']:.6f}")
        
        # 3. EPLB 16 GPU
        print("\n🔄 Running EPLB 16 GPU...")
        analyzer16 = LoadBalanceAnalyzer(DATA_PATH, 16, f"{OUTPUT_ROOT}/eplb_16gpu")
        analyzer16.load_data()
        config16 = {"num_replicas": 256, "num_gpus": 16, "num_groups": 1, "num_nodes": 1}
        eplb16_result = analyzer16.run_algorithm("eplb", config16, measure_time=False)
        print(f"✅ EPLB 16GPU std_dev: {eplb16_result['metrics']['std_dev']:.6f}")
        
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {OUTPUT_ROOT}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 