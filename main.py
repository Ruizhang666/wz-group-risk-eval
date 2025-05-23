import subprocess
import sys
import os
import time
from datetime import datetime

# 确保必要的目录存在
for directory in ["data", "model", "code", "outputs", "outputs/reports", "outputs/loop_results", "outputs/log", "outputs/loop_analysis"]:
    os.makedirs(directory, exist_ok=True)

steps = [
    ("转换Excel为CSV", [sys.executable, "code/convert_excel_to_csv.py"]),
    ("构建股权关系图", [sys.executable, "code/graph_builder.py"]),
    ("整合交易数据并生成异构图", [sys.executable, "code/add_transaction.py"]),
    ("分析异构图并生成报告", [sys.executable, "code/exploratory_analysis.py"]),
    ("执行股权闭环检测", [sys.executable, "code/loop_detection_nx.py"]),
    ("执行闭环分析与统计", [sys.executable, "code/loop_profiling.py"]),
]

def run_step(desc, cmd):
    print(f"\n===== {desc} =====")
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"开始时间: {start_datetime}")
    
    try:
        result = subprocess.run(cmd, check=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{desc} 完成！")
        print(f"耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)\n")
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{desc} 失败，退出码：{e.returncode}")
        print(f"耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
        sys.exit(e.returncode)

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
    print("- 分析报告位于 outputs/reports 目录")
    print("- 闭环检测结果位于 outputs/loop_results 目录")
    print("- 闭环分析结果位于 outputs/loop_analysis 目录")
    print("- 日志文件位于 outputs/log 目录\n") 