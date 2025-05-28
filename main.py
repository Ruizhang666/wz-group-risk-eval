import subprocess
import sys
import os
import time
from datetime import datetime

# 根据 README 和现有脚本，更新/确认输出目录
required_directories = [
    "data", "model", "code", "outputs",
    "outputs/reports", "outputs/loop_results", "outputs/log",
    "outputs/loop_analysis", "outputs/loop_filter",
    "outputs/visualizations", "outputs/扩展画像"
]
for directory in required_directories:
    os.makedirs(directory, exist_ok=True)

steps = [
    ("数据预处理: 转换Excel为CSV", [sys.executable, "code/convert_excel_to_csv.py"]),
    ("图结构构建: 构建股权关系图", [sys.executable, "code/graph_builder.py"]),
    ("异构图整合: 整合交易数据", [sys.executable, "code/add_transaction.py"]),
    ("环路检测: 执行股权闭环检测 (NetworkX)", [sys.executable, "code/loop_detection_nx.py"]),
    ("环路分析: 执行闭环画像分析", [sys.executable, "code/loop_profiling.py"]),
    ("股权指标提取: 扩展环路画像 (多核并行)", [sys.executable, "code/equity_metrics_extractor_parallel.py"]),
    ("简化闭环筛选: 基于时间窗口的闭环筛选 (多核并行)", [sys.executable, "code/simplified_closure_filter_parallel.py", "--months", "3", "--max-nodes", "6", "--source-type", "natural_person"])
]

def run_step(desc, cmd):
    print(f"\n===== {desc} =====")
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"开始时间: {start_datetime}")
    
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(cmd, check=True, cwd=project_root)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{desc} 完成！")
        print(f"耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)\n")
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{desc} 失败，错误信息: {e}")
        print(f"耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
        sys.exit(e.returncode)
    except FileNotFoundError as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{desc} 失败，文件未找到: {e}. 请确保脚本路径正确并且位于 'code' 文件夹下。")
        print(f"耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
        sys.exit(1)

if __name__ == "__main__":
    total_start_time = time.time()
    print("一键执行全流程开始！\n")
    
    for desc, cmd in steps:
        run_step(desc, cmd)
    
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    
    print("\n全部流程执行完毕！")
    print(f"总耗时: {total_elapsed_time:.2f} 秒 ({total_elapsed_time/60:.2f} 分钟)")
    print("\n=== 输出文件说明 ===")
    print("📁 数据文件:")
    print("  - data/ 目录: 原始数据文件")
    print("  - model/ 目录: 图模型文件")
    print("\n📊 分析结果:")
    print("  - outputs/loop_results/ 目录: 环路检测结果")
    print("  - outputs/loop_analysis/ 目录: 环路画像分析结果")
    print("  - outputs/扩展画像/ 目录: 综合画像数据 (交易+股权指标)")
    print("  - outputs/loop_filter/ 目录: 简化闭环筛选结果")
    print("  - outputs/visualizations/ 目录: 可视化图表")
    print("\n📋 报告文件:")
    print("  - outputs/reports/ 目录: 各类分析报告")
    print("  - outputs/log/ 目录: 运行日志")
    print("\n🎯 核心输出:")
    print("  - outputs/扩展画像/loop_comprehensive_metrics.csv: 综合画像数据")
    print("  - outputs/loop_filter/simplified_filtered_loops.csv: 筛选后的闭环")
    print("  - 各目录下的统计报告和可视化图表\n") 