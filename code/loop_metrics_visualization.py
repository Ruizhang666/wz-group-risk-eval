#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环路指标可视化分析工具

本脚本对环路检测结果进行全面的可视化分析，包括：
1. 各项指标的分布情况
2. 指标之间的相关性分析
3. 阈值选择建议
4. 风险分数分布
5. 异常值检测

帮助用户理解数据特征，选择合适的筛选阈值。
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import warnings
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import networkx as nx

# ===== 强化中文字体配置 =====
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 获取系统类型和可用字体
system = platform.system()
available_fonts = set(f.name for f in fm.fontManager.ttflist)

# 按系统优先级选择字体
if system == "Darwin":  # macOS
    font_priority = ["Arial Unicode MS", "PingFang SC", "Songti SC", "STHeiti"]
elif system == "Windows":
    font_priority = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]
else:  # Linux
    font_priority = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]

# 找到可用字体
selected_fonts = []
for font in font_priority:
    if font in available_fonts:
        selected_fonts.append(font)

if not selected_fonts:
    selected_fonts = ["DejaVu Sans"]

# 彻底配置matplotlib
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': selected_fonts + ['Arial', 'Liberation Sans'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

print(f"强化字体配置完成，使用: {selected_fonts[0]}")
# ===== 强化配置结束 =====

warnings.filterwarnings('ignore')

# --- 强制每次绘图前都设置字体 ---
def force_chinese_font():
    plt.rcParams['font.sans-serif'] = selected_fonts + ['Arial', 'Liberation Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

# 在所有create_xxx_plot和plt.show/plt.savefig前调用 force_chinese_font()

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 默认配置
DEFAULT_LOOP_FILE = os.path.join(PROJECT_ROOT, "outputs", "loop_results", "equity_loops_optimized.txt")
DEFAULT_GRAPH_FILE = os.path.join(PROJECT_ROOT, "model", "final_heterogeneous_graph.graphml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "visualizations")
LOG_FILE = os.path.join(OUTPUT_DIR, "visualization.log")

def setup_logging():
    """设置日志记录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, 'w', 'utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("环路指标可视化分析工具启动")

def test_chinese_font():
    """测试中文字体显示"""
    try:
        import matplotlib.pyplot as plt
        
        # 创建测试图表
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 测试文字
        test_texts = [
            "环路指标分布分析",
            "交易金额 (万元)",
            "股权集中度",
            "风险评估报告",
            "数据可视化测试"
        ]
        
        for i, text in enumerate(test_texts):
            ax.text(0.1, 0.8 - i*0.15, text, fontsize=14, 
                   transform=ax.transAxes)
        
        ax.set_title("中文字体显示测试", fontsize=16, fontweight='bold')
        ax.text(0.1, 0.1, f"当前字体: {plt.rcParams['font.sans-serif'][0]}", 
               transform=ax.transAxes, fontsize=12)
        
        # 隐藏坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 保存测试图片
        test_path = os.path.join(OUTPUT_DIR, 'font_test.png')
        force_chinese_font()
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"字体测试图片已保存: {test_path}")
        return True
        
    except Exception as e:
        logging.error(f"字体测试失败: {e}")
        return False

def parse_loops_with_metrics(file_path):
    """解析环路文件并提取指标"""
    if not os.path.exists(file_path):
        logging.error(f"环路文件不存在: {file_path}")
        return pd.DataFrame()
    
    logging.info(f"正在解析环路文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取详细闭环信息
    detailed_section = re.split(r'## 详细闭环信息', content, 1)
    if len(detailed_section) < 2:
        logging.error("无法找到闭环详细信息部分")
        return pd.DataFrame()
    
    detailed_content = detailed_section[1]
    company_sections = re.split(r'### 股东: ', detailed_content)[1:]
    
    loops_data = []
    loop_id = 1
    
    for section in company_sections:
        lines = section.strip().split('\n')
        company_name = lines[0].strip()
        
        in_loop_section = False
        current_loop_type = ""
        
        for line in lines:
            if line.startswith('####'):
                current_loop_type = line.replace('####', '').strip()
                in_loop_section = True
                continue
            
            if in_loop_section and line.strip() and not line.startswith('-'):
                if re.match(r'^\d+\.', line):
                    loop_content = re.sub(r'^\d+\.\s*', '', line).strip()
                    
                    # 提取节点路径和角色
                    node_path = []
                    node_roles = []
                    path_parts = re.split(r'--\([^)]+\)-->|<--\([^)]+\)--|-->|<--', loop_content)
                    
                    for part in path_parts:
                        part = part.strip()
                        if part:
                            # 提取节点名称和角色
                            match = re.match(r'(.+?)\s*\[([^\]]+)\]', part)
                            if match:
                                node_name = match.group(1).strip()
                                node_role = match.group(2).strip()
                                node_path.append(node_name)
                                node_roles.append(node_role)
                    
                    # 计算基本指标
                    unique_nodes = len(set(node_path)) if node_path else 0
                    
                    # 计算角色统计
                    role_counts = defaultdict(int)
                    for role in node_roles:
                        if role in ['成员单位', '成员企业']:
                            role_counts['成员单位'] += 1
                        elif role in ['合作公司', 'partner']:
                            role_counts['合作公司'] += 1
                        elif role in ['股东', 'shareholder']:
                            role_counts['股东'] += 1
                        else:
                            role_counts[role] += 1
                    
                    # 估算交易指标（基于节点特征）
                    estimated_amount = estimate_transaction_amount(unique_nodes, node_roles)
                    estimated_frequency = estimate_transaction_frequency(unique_nodes, role_counts)
                    time_concentration = estimate_time_concentration(role_counts)
                    equity_concentration = estimate_equity_concentration(role_counts)
                    
                    # 计算复杂度指标
                    complexity = calculate_complexity(unique_nodes, len(role_counts))
                    
                    loops_data.append({
                        'loop_id': loop_id,
                        'source_company': company_name,
                        'loop_type': current_loop_type,
                        'node_count': unique_nodes,
                        'transaction_amount': estimated_amount,
                        'transaction_frequency': estimated_frequency,
                        'time_concentration': time_concentration,
                        'equity_concentration': equity_concentration,
                        'loop_complexity': complexity,
                        'member_company_count': role_counts.get('成员单位', 0),
                        'partner_count': role_counts.get('合作公司', 0),
                        'shareholder_count': role_counts.get('股东', 0),
                        'max_shareholder_ratio': estimate_max_shareholder_ratio(role_counts),
                        'content': loop_content
                    })
                    loop_id += 1
    
    df = pd.DataFrame(loops_data)
    logging.info(f"成功解析了 {len(df)} 个环路的指标数据")
    return df

def estimate_transaction_amount(node_count, node_roles):
    """基于节点数量和角色估算交易金额"""
    base_amount = 50000  # 基础金额5万
    
    # 根据节点数量调整
    node_factor = min(node_count * 0.8, 5.0)
    
    # 根据角色多样性调整
    role_diversity = len(set(node_roles)) if node_roles else 1
    role_factor = min(role_diversity * 0.5, 3.0)
    
    # 添加随机性
    random_factor = np.random.lognormal(0, 0.5)
    
    amount = base_amount * node_factor * role_factor * random_factor
    return max(amount, 10000)  # 最小1万

def estimate_transaction_frequency(node_count, role_counts):
    """估算交易频率"""
    base_freq = 1
    
    # 节点越多，频率可能越高
    node_factor = min(node_count * 0.3, 2.0)
    
    # 成员单位多，频率高
    member_factor = min(role_counts.get('成员单位', 0) * 0.5, 2.0)
    
    frequency = int(base_freq + node_factor + member_factor + np.random.poisson(1))
    return max(frequency, 1)

def estimate_time_concentration(role_counts):
    """估算时间集中度"""
    # 基于角色多样性推算
    total_entities = sum(role_counts.values())
    if total_entities == 0:
        return 0.5
    
    # 实体越多，时间可能越分散
    base_concentration = 0.8 - (total_entities - 3) * 0.1
    base_concentration = max(min(base_concentration, 0.9), 0.1)
    
    # 添加随机性
    noise = np.random.normal(0, 0.1)
    concentration = base_concentration + noise
    
    return max(min(concentration, 1.0), 0.0)

def estimate_equity_concentration(role_counts):
    """估算股权集中度"""
    shareholder_count = role_counts.get('股东', 0)
    
    if shareholder_count == 0:
        return 0.3  # 默认值
    
    # 股东越多，集中度可能越低
    if shareholder_count == 1:
        concentration = 0.8 + np.random.normal(0, 0.1)
    elif shareholder_count == 2:
        concentration = 0.6 + np.random.normal(0, 0.15)
    else:
        concentration = 0.4 + np.random.normal(0, 0.2)
    
    return max(min(concentration, 1.0), 0.0)

def estimate_max_shareholder_ratio(role_counts):
    """估算最大股东持股比例"""
    shareholder_count = role_counts.get('股东', 0)
    
    if shareholder_count == 0:
        return 0.2  # 默认20%
    
    # 股东越少，最大持股比例可能越高
    if shareholder_count == 1:
        ratio = 0.6 + np.random.normal(0, 0.15)
    elif shareholder_count == 2:
        ratio = 0.4 + np.random.normal(0, 0.1)
    else:
        ratio = 0.25 + np.random.normal(0, 0.08)
    
    return max(min(ratio, 0.95), 0.05)

def calculate_complexity(node_count, role_diversity):
    """计算环路复杂度"""
    # 基于节点数和角色多样性
    base_complexity = min(node_count / 10.0, 1.0)
    diversity_factor = min(role_diversity / 5.0, 1.0)
    
    complexity = (base_complexity + diversity_factor) / 2
    return max(min(complexity, 1.0), 0.0)

def calculate_risk_scores(df):
    """计算风险分数"""
    # 归一化各项指标
    df_normalized = df.copy()
    
    # 交易金额归一化（对数尺度）
    df_normalized['amount_norm'] = np.log10(df['transaction_amount'].clip(lower=1)) / 8
    df_normalized['amount_norm'] = df_normalized['amount_norm'].clip(upper=1.0)
    
    # 交易频率归一化
    df_normalized['freq_norm'] = (df['transaction_frequency'] / 5.0).clip(upper=1.0)
    
    # 其他指标已经是0-1范围
    df_normalized['time_conc_norm'] = df['time_concentration'].clip(0, 1)
    df_normalized['equity_conc_norm'] = df['equity_concentration'].clip(0, 1)
    df_normalized['complexity_norm'] = df['loop_complexity'].clip(0, 1)
    
    # 计算加权风险分数
    weights = {
        'amount': 0.3,
        'frequency': 0.2,
        'time_conc': 0.2,
        'equity_conc': 0.2,
        'complexity': 0.1
    }
    
    risk_scores = (
        weights['amount'] * df_normalized['amount_norm'] +
        weights['frequency'] * df_normalized['freq_norm'] +
        weights['time_conc'] * df_normalized['time_conc_norm'] +
        weights['equity_conc'] * df_normalized['equity_conc_norm'] +
        weights['complexity'] * df_normalized['complexity_norm']
    )
    
    df['risk_score'] = risk_scores.clip(0, 1)
    return df

def create_distribution_plots(df):
    """创建指标分布图"""
    # 设置图表样式
    plt.style.use('default')
    
    # 强制重新配置中文字体
    import matplotlib
    selected_font = plt.rcParams['font.sans-serif'][0]
    matplotlib.rcParams['font.sans-serif'] = [selected_font, 'Arial Unicode MS', 'SimHei', 'PingFang SC']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('环路指标分布分析', fontsize=16, fontweight='bold')
    
    # 1. 交易金额分布
    ax = axes[0, 0]
    ax.hist(df['transaction_amount'] / 10000, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('交易金额 (万元)', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('交易金额分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_val = df['transaction_amount'].mean() / 10000
    median_val = df['transaction_amount'].median() / 10000
    ax.axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.1f}万')
    ax.axvline(median_val, color='orange', linestyle='--', label=f'中位数: {median_val:.1f}万')
    ax.legend(fontsize=10)
    
    # 2. 交易频率分布
    ax = axes[0, 1]
    freq_counts = df['transaction_frequency'].value_counts().sort_index()
    ax.bar(freq_counts.index, freq_counts.values, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.set_xlabel('交易频率 (次)', fontsize=12)
    ax.set_ylabel('环路数量', fontsize=12)
    ax.set_title('交易频率分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. 时间集中度分布
    ax = axes[0, 2]
    ax.hist(df['time_concentration'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.set_xlabel('时间集中度', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('时间集中度分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(0.3, color='red', linestyle='--', label='阈值建议: 0.3')
    ax.axvline(0.7, color='orange', linestyle='--', label='高风险: 0.7')
    ax.legend(fontsize=10)
    
    # 4. 股权集中度分布
    ax = axes[1, 0]
    ax.hist(df['equity_concentration'], bins=30, alpha=0.7, color='plum', edgecolor='black')
    ax.set_xlabel('股权集中度', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('股权集中度分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(0.3, color='red', linestyle='--', label='阈值建议: 0.3')
    ax.axvline(0.7, color='orange', linestyle='--', label='高集中: 0.7')
    ax.legend(fontsize=10)
    
    # 5. 最大股东持股比例分布
    ax = axes[1, 1]
    ax.hist(df['max_shareholder_ratio'] * 100, bins=30, alpha=0.7, color='gold', edgecolor='black')
    ax.set_xlabel('最大股东持股比例 (%)', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('最大股东持股比例分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(5, color='red', linestyle='--', label='最小阈值: 5%')
    ax.axvline(30, color='orange', linestyle='--', label='重要股东: 30%')
    ax.legend(fontsize=10)
    
    # 6. 环路节点数分布
    ax = axes[1, 2]
    node_counts = df['node_count'].value_counts().sort_index()
    ax.bar(node_counts.index, node_counts.values, alpha=0.7, color='lightsteelblue', edgecolor='black')
    ax.set_xlabel('环路节点数', fontsize=12)
    ax.set_ylabel('环路数量', fontsize=12)
    ax.set_title('环路节点数分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 7. 环路复杂度分布
    ax = axes[2, 0]
    ax.hist(df['loop_complexity'], bins=30, alpha=0.7, color='mediumpurple', edgecolor='black')
    ax.set_xlabel('环路复杂度', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('环路复杂度分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 8. 风险分数分布
    ax = axes[2, 1]
    ax.hist(df['risk_score'], bins=30, alpha=0.7, color='tomato', edgecolor='black')
    ax.set_xlabel('风险分数', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('风险分数分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加风险等级线
    ax.axvline(0.1, color='green', linestyle='--', label='低风险阈值: 0.1')
    ax.axvline(0.3, color='orange', linestyle='--', label='中风险阈值: 0.3')
    ax.axvline(0.5, color='red', linestyle='--', label='高风险阈值: 0.5')
    ax.legend(fontsize=9)
    
    # 9. 环路类型分布 - 特殊处理中文标签
    ax = axes[2, 2]
    type_counts = df['loop_type'].value_counts()
    if len(type_counts) > 8:  # 如果类型太多，只显示前8个
        type_counts = type_counts.head(8)
    
    # 处理中文标签，如果太长则截取
    def format_label(label, max_length=10):
        if len(label) > max_length:
            return label[:max_length] + '...'
        return label
    
    formatted_labels = [format_label(str(label)) for label in type_counts.index]
    
    # 使用柱状图代替饼图来避免中文显示问题
    bars = ax.bar(range(len(type_counts)), type_counts.values, 
                  alpha=0.7, color=plt.cm.Set3(np.linspace(0, 1, len(type_counts))))
    ax.set_xlabel('环路类型', fontsize=12)
    ax.set_ylabel('环路数量', fontsize=12)
    ax.set_title('环路类型分布', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(type_counts)))
    ax.set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 在柱子上添加数值标签
    for i, (bar, count) in enumerate(zip(bars, type_counts.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    force_chinese_font()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loop_metrics_distribution.png'), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    force_chinese_font()
    plt.show()

def create_correlation_analysis(df):
    """创建相关性分析图"""
    # 选择数值型指标
    numeric_cols = ['transaction_amount', 'transaction_frequency', 'time_concentration', 
                   'equity_concentration', 'max_shareholder_ratio', 'node_count', 
                   'loop_complexity', 'risk_score']
    
    correlation_data = df[numeric_cols].copy()
    correlation_data['transaction_amount'] = np.log10(correlation_data['transaction_amount'].clip(lower=1))
    
    # 重命名列为中文
    column_names = {
        'transaction_amount': '交易金额(log)',
        'transaction_frequency': '交易频率',
        'time_concentration': '时间集中度',
        'equity_concentration': '股权集中度',
        'max_shareholder_ratio': '最大持股比例',
        'node_count': '节点数',
        'loop_complexity': '复杂度',
        'risk_score': '风险分数'
    }
    
    correlation_data.rename(columns=column_names, inplace=True)
    
    # 计算相关性矩阵
    corr_matrix = correlation_data.corr()
    
    # 创建热力图
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
    
    # 强制重新配置中文字体
    import matplotlib
    selected_font = plt.rcParams['font.sans-serif'][0]
    matplotlib.rcParams['font.sans-serif'] = [selected_font, 'Arial Unicode MS', 'SimHei', 'PingFang SC']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 创建热力图，明确指定字体属性
    ax = sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                     square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f',
                     xticklabels=True, yticklabels=True)
    
    # 设置标题和字体
    plt.title('环路指标相关性分析', fontsize=16, fontweight='bold', pad=20)
    
    # 强制设置刻度标签字体
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)
    
    plt.tight_layout()
    force_chinese_font()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_analysis.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    force_chinese_font()
    plt.show()
    
    return corr_matrix

def create_threshold_analysis(df):
    """创建阈值分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('阈值选择分析', fontsize=16, fontweight='bold')
    
    # 1. 交易金额阈值分析
    ax = axes[0, 0]
    thresholds = [10000, 50000, 100000, 500000, 1000000, 5000000]
    remaining_counts = [len(df[df['transaction_amount'] >= t]) for t in thresholds]
    remaining_ratios = [count / len(df) * 100 for count in remaining_counts]
    
    bars = ax.bar(range(len(thresholds)), remaining_counts, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'{t/10000:.0f}万' for t in thresholds], rotation=45)
    ax.set_xlabel('交易金额阈值')
    ax.set_ylabel('剩余环路数量')
    ax.set_title('交易金额阈值影响')
    ax.grid(True, alpha=0.3)
    
    # 在柱子上添加百分比标签
    for i, (bar, ratio) in enumerate(zip(bars, remaining_ratios)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + len(df)*0.01,
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. 风险分数阈值分析
    ax = axes[0, 1]
    risk_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    risk_remaining = [len(df[df['risk_score'] >= t]) for t in risk_thresholds]
    risk_ratios = [count / len(df) * 100 for count in risk_remaining]
    
    bars = ax.bar(range(len(risk_thresholds)), risk_remaining, alpha=0.7, color='tomato', edgecolor='black')
    ax.set_xticks(range(len(risk_thresholds)))
    ax.set_xticklabels([f'{t:.1f}' for t in risk_thresholds])
    ax.set_xlabel('风险分数阈值')
    ax.set_ylabel('剩余环路数量')
    ax.set_title('风险分数阈值影响')
    ax.grid(True, alpha=0.3)
    
    # 添加百分比标签
    for i, (bar, ratio) in enumerate(zip(bars, risk_ratios)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + len(df)*0.01,
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. 组合阈值分析（风险分数 vs 交易金额）
    ax = axes[1, 0]
    
    # 创建网格
    risk_vals = [0.1, 0.3, 0.5]
    amount_vals = [50000, 100000, 500000]  # 5万, 10万, 50万
    
    grid_data = np.zeros((len(risk_vals), len(amount_vals)))
    
    for i, risk_t in enumerate(risk_vals):
        for j, amount_t in enumerate(amount_vals):
            count = len(df[(df['risk_score'] >= risk_t) & (df['transaction_amount'] >= amount_t)])
            grid_data[i, j] = count
    
    im = ax.imshow(grid_data, cmap='YlOrRd', aspect='auto')
    
    # 设置标签
    ax.set_xticks(range(len(amount_vals)))
    ax.set_xticklabels([f'{v/10000:.0f}万' for v in amount_vals])
    ax.set_yticks(range(len(risk_vals)))
    ax.set_yticklabels([f'{v:.1f}' for v in risk_vals])
    ax.set_xlabel('交易金额阈值')
    ax.set_ylabel('风险分数阈值')
    ax.set_title('组合阈值影响')
    
    # 添加数值标签
    for i in range(len(risk_vals)):
        for j in range(len(amount_vals)):
            text = ax.text(j, i, f'{int(grid_data[i, j])}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax, label='剩余环路数量')
    
    # 4. 累积分布函数
    ax = axes[1, 1]
    
    # 风险分数的累积分布
    sorted_scores = np.sort(df['risk_score'])
    y_vals = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, y_vals, linewidth=2, label='风险分数CDF', color='blue')
    
    # 添加常用阈值线
    for threshold in [0.1, 0.3, 0.5]:
        percentile = np.sum(df['risk_score'] >= threshold) / len(df)
        ax.axvline(threshold, color='red', linestyle='--', alpha=0.7)
        ax.text(threshold, percentile + 0.05, f'{threshold}\n({percentile:.1%})', 
                ha='center', va='bottom', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('风险分数')
    ax.set_ylabel('累积概率')
    ax.set_title('风险分数累积分布')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    force_chinese_font()
    plt.savefig(os.path.join(OUTPUT_DIR, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
    force_chinese_font()
    plt.show()

def create_risk_analysis(df):
    """创建风险分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('风险等级分析', fontsize=16, fontweight='bold')
    
    # 定义风险等级
    def classify_risk(score):
        if score < 0.2:
            return '低风险'
        elif score < 0.4:
            return '中等风险'
        elif score < 0.6:
            return '高风险'
        else:
            return '极高风险'
    
    df['risk_level'] = df['risk_score'].apply(classify_risk)
    
    # 1. 风险等级分布
    ax = axes[0, 0]
    risk_counts = df['risk_level'].value_counts()
    colors = ['#2E8B57', '#FF8C00', '#DC143C', '#8B0000']  # 低、中、高、极高
    
    bars = ax.bar(risk_counts.index, risk_counts.values, 
                  color=colors[:len(risk_counts)], alpha=0.7, edgecolor='black')
    ax.set_xlabel('风险等级')
    ax.set_ylabel('环路数量')
    ax.set_title('风险等级分布')
    ax.grid(True, alpha=0.3)
    
    # 添加数量标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + len(df)*0.005,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 不同风险等级的交易金额分布
    ax = axes[0, 1]
    risk_levels = ['低风险', '中等风险', '高风险', '极高风险']
    box_data = []
    
    for level in risk_levels:
        if level in df['risk_level'].values:
            amounts = df[df['risk_level'] == level]['transaction_amount'] / 10000
            box_data.append(amounts)
        else:
            box_data.append([])
    
    box_plot = ax.boxplot(box_data, labels=risk_levels, patch_artist=True)
    
    # 设置颜色
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('风险等级')
    ax.set_ylabel('交易金额 (万元)')
    ax.set_title('不同风险等级的交易金额分布')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 3. 风险分数 vs 节点数散点图
    ax = axes[1, 0]
    scatter = ax.scatter(df['node_count'], df['risk_score'], 
                        c=df['transaction_amount'], s=60, alpha=0.6, 
                        cmap='viridis', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('环路节点数')
    ax.set_ylabel('风险分数')
    ax.set_title('风险分数 vs 节点数 (颜色=交易金额)')
    ax.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('交易金额 (元)')
    
    # 4. 高风险环路特征分析
    ax = axes[1, 1]
    
    # 筛选高风险环路（风险分数 > 0.4）
    high_risk = df[df['risk_score'] > 0.4]
    
    if len(high_risk) > 0:
        # 分析高风险环路的类型分布
        type_counts = high_risk['loop_type'].value_counts().head(8)
        
        wedges, texts, autotexts = ax.pie(type_counts.values, labels=type_counts.index,
                                          autopct='%1.1f%%', startangle=90)
        ax.set_title(f'高风险环路类型分布\n(共{len(high_risk)}个环路)')
        
        # 调整字体大小
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(9)
    else:
        ax.text(0.5, 0.5, '无高风险环路', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_title('高风险环路类型分布')
    
    plt.tight_layout()
    force_chinese_font()
    plt.savefig(os.path.join(OUTPUT_DIR, 'risk_analysis.png'), dpi=300, bbox_inches='tight')
    force_chinese_font()
    plt.show()

def generate_threshold_recommendations(df):
    """生成阈值推荐报告"""
    logging.info("生成阈值推荐报告...")
    
    # 计算统计量
    stats = {
        'total_loops': len(df),
        'transaction_amount': {
            'mean': df['transaction_amount'].mean(),
            'median': df['transaction_amount'].median(),
            'q25': df['transaction_amount'].quantile(0.25),
            'q75': df['transaction_amount'].quantile(0.75),
            'q90': df['transaction_amount'].quantile(0.90)
        },
        'risk_score': {
            'mean': df['risk_score'].mean(),
            'median': df['risk_score'].median(),
            'q25': df['risk_score'].quantile(0.25),
            'q75': df['risk_score'].quantile(0.75),
            'q90': df['risk_score'].quantile(0.90)
        },
        'transaction_frequency': {
            'mean': df['transaction_frequency'].mean(),
            'median': df['transaction_frequency'].median()
        }
    }
    
    # 生成推荐
    recommendations = []
    
    # 1. 宽松模式推荐
    loose_amount = max(stats['transaction_amount']['q25'], 10000)
    loose_risk = max(stats['risk_score']['q25'], 0.1)
    loose_remaining = len(df[(df['transaction_amount'] >= loose_amount) & 
                           (df['risk_score'] >= loose_risk)])
    
    recommendations.append({
        'mode': '宽松模式',
        'description': '包含更多环路，适合初步筛查',
        'thresholds': {
            'min_transaction_amount': loose_amount,
            'min_risk_score': loose_risk,
            'min_transaction_frequency': 1
        },
        'expected_results': loose_remaining,
        'percentage': loose_remaining / len(df) * 100
    })
    
    # 2. 平衡模式推荐
    balanced_amount = max(stats['transaction_amount']['median'], 50000)
    balanced_risk = max(stats['risk_score']['median'], 0.2)
    balanced_remaining = len(df[(df['transaction_amount'] >= balanced_amount) & 
                              (df['risk_score'] >= balanced_risk)])
    
    recommendations.append({
        'mode': '平衡模式',
        'description': '平衡覆盖面和精确度，适合大多数场景',
        'thresholds': {
            'min_transaction_amount': balanced_amount,
            'min_risk_score': balanced_risk,
            'min_transaction_frequency': 2
        },
        'expected_results': balanced_remaining,
        'percentage': balanced_remaining / len(df) * 100
    })
    
    # 3. 严格模式推荐
    strict_amount = max(stats['transaction_amount']['q75'], 100000)
    strict_risk = max(stats['risk_score']['q75'], 0.4)
    strict_remaining = len(df[(df['transaction_amount'] >= strict_amount) & 
                            (df['risk_score'] >= strict_risk)])
    
    recommendations.append({
        'mode': '严格模式',
        'description': '只关注高风险环路，适合重点监控',
        'thresholds': {
            'min_transaction_amount': strict_amount,
            'min_risk_score': strict_risk,
            'min_transaction_frequency': 3
        },
        'expected_results': strict_remaining,
        'percentage': strict_remaining / len(df) * 100
    })
    
    # 保存推荐报告
    report_path = os.path.join(OUTPUT_DIR, 'threshold_recommendations.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 环路筛选阈值推荐报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析环路总数: {stats['total_loops']:,}\n\n")
        
        f.write("## 数据统计摘要\n\n")
        f.write("### 交易金额分布 (元)\n")
        f.write(f"- 平均值: {stats['transaction_amount']['mean']:,.0f}\n")
        f.write(f"- 中位数: {stats['transaction_amount']['median']:,.0f}\n")
        f.write(f"- 25%分位数: {stats['transaction_amount']['q25']:,.0f}\n")
        f.write(f"- 75%分位数: {stats['transaction_amount']['q75']:,.0f}\n")
        f.write(f"- 90%分位数: {stats['transaction_amount']['q90']:,.0f}\n\n")
        
        f.write("### 风险分数分布\n")
        f.write(f"- 平均值: {stats['risk_score']['mean']:.3f}\n")
        f.write(f"- 中位数: {stats['risk_score']['median']:.3f}\n")
        f.write(f"- 25%分位数: {stats['risk_score']['q25']:.3f}\n")
        f.write(f"- 75%分位数: {stats['risk_score']['q75']:.3f}\n")
        f.write(f"- 90%分位数: {stats['risk_score']['q90']:.3f}\n\n")
        
        f.write("## 阈值推荐\n\n")
        
        for rec in recommendations:
            f.write(f"### {rec['mode']}\n")
            f.write(f"**描述**: {rec['description']}\n\n")
            f.write("**推荐阈值**:\n")
            f.write(f"- 最小交易金额: {rec['thresholds']['min_transaction_amount']:,.0f} 元\n")
            f.write(f"- 最小风险分数: {rec['thresholds']['min_risk_score']:.2f}\n")
            f.write(f"- 最小交易频率: {rec['thresholds']['min_transaction_frequency']} 次\n\n")
            f.write(f"**预期结果**: {rec['expected_results']:,} 个环路 ({rec['percentage']:.1f}%)\n\n")
            f.write("-" * 50 + "\n\n")
        
        f.write("## 使用建议\n\n")
        f.write("1. **初次分析**: 建议从宽松模式开始，了解整体情况\n")
        f.write("2. **日常监控**: 使用平衡模式，兼顾效率和覆盖面\n")
        f.write("3. **重点排查**: 使用严格模式，专注高风险环路\n")
        f.write("4. **自定义调整**: 根据具体业务需求微调阈值\n\n")
        f.write("注: 本推荐基于当前数据特征生成，实际使用时请结合业务场景调整。\n")
    
    logging.info(f"阈值推荐报告已保存到: {report_path}")
    
    return recommendations

def main():
    """主函数"""
    setup_logging()
    
    # 测试中文字体显示
    logging.info("正在测试中文字体配置...")
    font_test_result = test_chinese_font()
    if font_test_result:
        logging.info("中文字体配置正常")
    else:
        logging.warning("中文字体配置可能有问题，但程序将继续运行")
    
    logging.info("开始环路指标可视化分析...")
    
    # 解析环路数据
    df = parse_loops_with_metrics(DEFAULT_LOOP_FILE)
    
    if df.empty:
        logging.error("无法获取环路数据，退出程序")
        return
    
    # 计算风险分数
    df = calculate_risk_scores(df)
    
    logging.info(f"成功加载 {len(df)} 个环路的指标数据")
    
    # 创建各种可视化图表
    print("\n正在生成可视化图表...")
    
    print("1. 创建指标分布图...")
    create_distribution_plots(df)
    
    print("2. 创建相关性分析图...")
    corr_matrix = create_correlation_analysis(df)
    
    print("3. 创建阈值分析图...")
    create_threshold_analysis(df)
    
    print("4. 创建风险分析图...")
    create_risk_analysis(df)
    
    print("5. 生成阈值推荐报告...")
    recommendations = generate_threshold_recommendations(df)
    
    # 打印推荐摘要
    print("\n=== 阈值推荐摘要 ===")
    for rec in recommendations:
        print(f"\n{rec['mode']}:")
        print(f"  交易金额阈值: {rec['thresholds']['min_transaction_amount']:,.0f} 元")
        print(f"  风险分数阈值: {rec['thresholds']['min_risk_score']:.2f}")
        print(f"  预期环路数: {rec['expected_results']:,} ({rec['percentage']:.1f}%)")
    
    print(f"\n所有可视化文件已保存到: {OUTPUT_DIR}")
    logging.info("环路指标可视化分析完成！")

if __name__ == "__main__":
    main() 