import subprocess
import sys
import os

# 确保必要的目录存在
for directory in ["data", "model", "code", "outputs", "outputs/reports", "outputs/loop_results"]:
    os.makedirs(directory, exist_ok=True)

steps = [
    ("转换Excel为CSV", [sys.executable, "code/convert_excel_to_csv.py"]),
    ("构建股权关系图", [sys.executable, "code/graph_builder.py"]),
    ("整合交易数据并生成异构图", [sys.executable, "code/add_transaction.py"]),
    ("分析异构图并生成报告", [sys.executable, "code/exploratory_analysis.py"]),
]

def run_step(desc, cmd):
    print(f"\n===== {desc} =====")
    try:
        result = subprocess.run(cmd, check=True)
        print(f"{desc} 完成！\n")
    except subprocess.CalledProcessError as e:
        print(f"{desc} 失败，退出码：{e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    print("一键执行全流程开始！\n")
    for desc, cmd in steps:
        run_step(desc, cmd)
    print("\n全部流程执行完毕！")
    print("- 数据文件位于 data 目录")
    print("- 模型文件位于 model 目录")
    print("- 代码文件位于 code 目录")
    print("- 分析报告位于 outputs/reports 目录\n") 