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
    "outputs/visualizations"
]
for directory in required_directories:
    os.makedirs(directory, exist_ok=True)

steps = [
    ("数据预处理: 转换Excel为CSV", [sys.executable, "code/convert_excel_to_csv.py"]),
    ("图结构构建: 构建股权关系图", [sys.executable, "code/graph_builder.py"]),
    ("异构图整合: 整合交易数据", [sys.executable, "code/add_transaction.py"]),
    ("环路检测: 执行股权闭环检测 (NetworkX)", [sys.executable, "code/loop_detection_nx.py"]),
    ("环路分析: 执行闭环画像分析", [sys.executable, "code/loop_profiling.py"]),
    ("环路筛选: 执行高性能环路筛选", [sys.executable, "code/loop_filter_script.py"]),
    ("探索性数据分析: 分析异构图并生成报告", [sys.executable, "code/exploratory_analysis.py"]),
    ("可视化分析: 生成环路指标可视化图表", [sys.executable, "code/loop_metrics_visualization.py"])
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
    print("- 数据文件位于 data 目录")
    print("- 模型文件位于 model 目录")
    print("- 代码文件位于 code 目录")
    print("- 分析报告位于 outputs/reports 目录 (由探索性数据分析等脚本生成)")
    print("- 闭环检测结果位于 outputs/loop_results 目录")
    print("- 筛选后的闭环位于 outputs/loop_filter 目录")
    print("- 闭环深度分析结果位于 outputs/loop_analysis 目录 (由环路画像分析等脚本生成)")
    print("- 可视化图表位于 outputs/visualizations 目录 (由可视化分析脚本生成)")
    print("- 日志文件位于 outputs/log 目录 (部分脚本可能生成)\n") 