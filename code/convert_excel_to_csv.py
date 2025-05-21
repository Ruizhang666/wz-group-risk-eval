import pandas as pd
import os

# --- 配置 ---
EXCEL_FILE_PATH = "data/化工公司样本测试数据0521.xlsx"
SHEET_NAME = "化工公司交易数据"
OUTPUT_CSV_PATH = "data/交易数据.csv" # 输出的CSV文件名

# 确保输出目录存在
output_dir = os.path.dirname(OUTPUT_CSV_PATH)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"正在将Excel文件 '{EXCEL_FILE_PATH}' 中的 '{SHEET_NAME}' 表转换为CSV...")

try:
    # 读取Excel文件
    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME)
    
    # 删除所有全空的行
    df = df.dropna(how='all')
    
    # 将数据保存为CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    
    print(f"转换成功! CSV文件已保存为 '{OUTPUT_CSV_PATH}'")
    print(f"CSV文件包含 {len(df)} 行数据。")
    
except Exception as e:
    print(f"转换过程中发生错误: {e}") 