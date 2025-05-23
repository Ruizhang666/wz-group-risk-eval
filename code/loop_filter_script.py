#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环路优筛工具 - 基于多维度指标筛选高风险闭环 (优化版)

该脚本从闭环检测结果中筛选出最值得关注的可疑交易环路，
通过调整参数可以控制筛选的严格程度。
"""

import os
import re
import logging
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from collections import defaultdict
import json
import argparse
from typing import Dict, List, Tuple, Set
import numpy as np

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 配置常量 - 使用绝对路径
DEFAULT_LOOP_FILE = os.path.join(PROJECT_ROOT, "outputs", "loop_results", "equity_loops_optimized.txt")
DEFAULT_GRAPH_FILE = os.path.join(PROJECT_ROOT, "model", "final_heterogeneous_graph.graphml")
DEFAULT_METRICS_FILE = os.path.join(PROJECT_ROOT, "outputs", "loop_analysis", "loop_metrics.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "loop_filter")
LOG_FILE = os.path.join(OUTPUT_DIR, "loop_filter.log")
FILTERED_LOOPS_FILE = os.path.join(OUTPUT_DIR, "filtered_loops.txt")
FILTER_REPORT_FILE = os.path.join(OUTPUT_DIR, "filter_report.csv")

# ===== 筛选参数（降低阈值，更宽松的筛选条件）=====
class FilterConfig:
    """筛选配置类"""
    def __init__(self):
        # 交易相关参数 - 大幅降低阈值
        self.min_transaction_amount = 10000  # 最小交易总额（元） - 从100万降到1万
        self.min_transaction_frequency = 1  # 最小交易频率 - 从3降到1
        self.transaction_time_window_months = 12  # 交易时间窗口（月） - 从6增加到12
        self.time_concentration_threshold = 0.3  # 时间集中度阈值 - 从0.8降到0.3
        
        # 股权相关参数 - 更宽松
        self.min_shareholder_ratio = 0.01  # 最小股东持股比例 - 从10%降到1%
        self.max_shareholder_ratio = 0.99  # 最大股东持股比例 - 从90%增加到99%
        self.equity_concentration_threshold = 0.3  # 股权集中度阈值 - 从0.7降到0.3
        
        # 环路结构参数
        self.allowed_loop_types = None  # 允许的环路类型列表，None表示全部
        self.min_loop_nodes = 3  # 最小环路节点数 - 从4降到3
        self.max_loop_nodes = 10  # 最大环路节点数 - 从8增加到10
        self.min_member_companies = 0  # 最少成员单位数量 - 从1降到0
        
        # 风险评分权重
        self.weight_transaction_amount = 0.3
        self.weight_transaction_frequency = 0.2
        self.weight_time_concentration = 0.2
        self.weight_equity_concentration = 0.2
        self.weight_loop_complexity = 0.1
        
        # 输出控制
        self.top_k_results = 100  # 输出前K个高风险环路
        self.min_risk_score = 0.1  # 最小风险分数阈值 - 从0.5降到0.1

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
    logging.info("环路优筛工具启动 (优化版)")

def load_graph(file_path):
    """加载NetworkX图并预处理"""
    try:
        logging.info(f"正在加载图: {file_path}")
        graph = nx.read_graphml(file_path)
        
        # 预处理：创建节点名称到ID的映射，提高查找效率
        name_to_id = {}
        for node_id, attrs in graph.nodes(data=True):
            name = attrs.get('name', str(node_id))
            name_to_id[name] = node_id
        
        # 将映射保存到图对象中
        graph.graph['name_to_id'] = name_to_id
        
        logging.info(f"图加载成功: {len(graph.nodes())} 节点, {len(graph.edges())} 边")
        logging.info(f"创建了 {len(name_to_id)} 个节点名称映射")
        return graph
    except Exception as e:
        logging.error(f"加载图时出错: {e}")
        return None

def parse_loops(file_path):
    """从闭环检测结果文本解析环路"""
    if not os.path.exists(file_path):
        logging.error(f"闭环文件不存在: {file_path}")
        return {}
    
    logging.info(f"正在解析闭环文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取详细闭环信息
    detailed_section = re.split(r'## 详细闭环信息', content, 1)
    if len(detailed_section) < 2:
        logging.error("无法找到闭环详细信息部分")
        return {}
    
    detailed_content = detailed_section[1]
    company_sections = re.split(r'### 股东: ', detailed_content)[1:]
    
    loops = {}
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
                    
                    # 去重计算唯一节点数
                    unique_nodes = len(set(node_path)) if node_path else 0
                    
                    loops[loop_id] = {
                        'source': company_name,
                        'content': loop_content,
                        'type': current_loop_type,
                        'node_path': node_path,
                        'node_roles': node_roles,
                        'node_count': unique_nodes
                    }
                    loop_id += 1
    
    logging.info(f"成功解析了 {len(loops)} 个闭环")
    return loops

def load_loop_metrics(file_path):
    """加载闭环指标数据"""
    if os.path.exists(file_path):
        try:
            metrics_df = pd.read_csv(file_path)
            metrics_dict = metrics_df.set_index('loop_id').to_dict('index')
            logging.info(f"加载了 {len(metrics_dict)} 个闭环的指标数据")
            return metrics_dict
        except Exception as e:
            logging.warning(f"加载闭环指标失败: {e}")
    else:
        logging.warning(f"指标文件不存在: {file_path}")
    return {}

def calculate_transaction_metrics(loop_data, metrics_data, config):
    """计算交易相关指标 - 简化版本"""
    loop_id = loop_data.get('id')
    metrics = metrics_data.get(loop_id, {}) if metrics_data else {}
    
    # 如果没有指标数据，使用默认值或从节点路径推算
    if not metrics:
        # 基于节点数量估算交易金额 (简化逻辑)
        node_count = loop_data.get('node_count', 0)
        estimated_amount = node_count * 100000  # 每个节点估算10万
        
        return {
            'transaction_amount': estimated_amount,
            'transaction_frequency': 2,  # 默认频率
            'time_concentration': 0.5,  # 默认集中度
            'transaction_count': node_count
        }
    
    # 从指标数据中提取
    total_amount = metrics.get('total_transaction_amount', 0)
    
    # 简化的交易频率计算
    frequency = metrics.get('transaction_frequency', 1)
    if frequency == 0:
        frequency = 1  # 避免为0
    
    # 简化的时间集中度
    time_concentration = metrics.get('time_concentration', 0.5)
    
    return {
        'transaction_amount': float(total_amount) if total_amount else 50000,  # 默认5万
        'transaction_frequency': int(frequency),
        'time_concentration': float(time_concentration),
        'transaction_count': metrics.get('transaction_count', 1)
    }

def calculate_equity_metrics(loop_data, graph, config):
    """计算股权相关指标 - 优化版本"""
    node_path = loop_data.get('node_path', [])
    node_roles = loop_data.get('node_roles', [])
    
    # 使用预建的映射查找节点，大幅提升效率
    name_to_id = graph.graph.get('name_to_id', {})
    shareholder_ratios = []
    
    # 查找股东节点的持股信息
    for node_name, node_role in zip(node_path, node_roles):
        if node_role in ['股东', 'shareholder']:
            node_id = name_to_id.get(node_name)
            if node_id:
                # 获取持股比例
                for _, target, data in graph.out_edges(node_id, data=True):
                    edge_label = data.get('label', '')
                    if '控股' in edge_label or '投资' in edge_label:
                        percent = data.get('percent', 0)
                        if isinstance(percent, str):
                            try:
                                # 尝试从字符串提取百分比
                                percent = float(re.findall(r'[\d.]+', percent)[0]) if re.findall(r'[\d.]+', percent) else 0
                            except:
                                percent = 0
                        if isinstance(percent, (int, float)) and percent > 0:
                            shareholder_ratios.append(percent)
    
    # 如果没有找到股权信息，使用默认值
    if not shareholder_ratios:
        # 基于节点角色估算
        shareholder_count = sum(1 for role in node_roles if role in ['股东', 'shareholder'])
        if shareholder_count > 0:
            default_ratio = 20.0  # 默认20%持股
            shareholder_ratios = [default_ratio] * shareholder_count
    
    # 计算股权集中度（HHI指数）
    equity_concentration = 0
    if shareholder_ratios:
        total_ratio = sum(shareholder_ratios)
        if total_ratio > 0:
            normalized_ratios = [r / total_ratio for r in shareholder_ratios]
            equity_concentration = sum(r ** 2 for r in normalized_ratios)
    
    return {
        'max_shareholder_ratio': max(shareholder_ratios) / 100.0 if shareholder_ratios else 0.2,  # 转换为小数
        'min_shareholder_ratio': min(shareholder_ratios) / 100.0 if shareholder_ratios else 0.1,
        'equity_concentration': equity_concentration,
        'shareholder_count': len(shareholder_ratios)
    }

def calculate_structure_metrics(loop_data):
    """计算环路结构指标"""
    node_count = loop_data.get('node_count', 0)
    node_roles = loop_data.get('node_roles', [])
    
    # 计算各类节点数量
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
    
    # 环路复杂度（基于节点数和角色多样性）
    complexity = min(node_count * len(role_counts) / 20.0, 1.0)  # 归一化
    
    return {
        'node_count': node_count,
        'member_company_count': role_counts.get('成员单位', 0),
        'partner_count': role_counts.get('合作公司', 0),
        'shareholder_count': role_counts.get('股东', 0),
        'loop_complexity': complexity
    }

def calculate_risk_score(metrics, config):
    """计算综合风险分数"""
    # 归一化各项指标
    normalized_metrics = {}
    
    # 交易金额归一化（对数尺度）
    amount = metrics.get('transaction_amount', 0)
    if amount > 0:
        normalized_metrics['amount'] = min(np.log10(max(amount, 1)) / 8, 1.0)  # 调整缩放
    else:
        normalized_metrics['amount'] = 0
    
    # 交易频率归一化
    frequency = metrics.get('transaction_frequency', 0)
    normalized_metrics['frequency'] = min(frequency / 5.0, 1.0)  # 5次为满分
    
    # 时间集中度
    normalized_metrics['time_conc'] = min(metrics.get('time_concentration', 0), 1.0)
    
    # 股权集中度
    normalized_metrics['equity_conc'] = min(metrics.get('equity_concentration', 0), 1.0)
    
    # 环路复杂度
    normalized_metrics['complexity'] = min(metrics.get('loop_complexity', 0), 1.0)
    
    # 计算加权风险分数
    risk_score = (
        config.weight_transaction_amount * normalized_metrics['amount'] +
        config.weight_transaction_frequency * normalized_metrics['frequency'] +
        config.weight_time_concentration * normalized_metrics['time_conc'] +
        config.weight_equity_concentration * normalized_metrics['equity_conc'] +
        config.weight_loop_complexity * normalized_metrics['complexity']
    )
    
    return min(risk_score, 1.0)

def filter_loops(loops, graph, metrics_data, config):
    """根据配置筛选环路 - 优化版本"""
    filtered_results = []
    filter_stats = {
        'total': len(loops),
        'passed_type': 0,
        'passed_nodes': 0,
        'passed_amount': 0,
        'passed_frequency': 0,
        'passed_ratio': 0,
        'passed_members': 0,
        'passed_risk': 0
    }
    
    logging.info(f"开始筛选 {len(loops)} 个环路...")
    
    for loop_id, loop_data in loops.items():
        loop_data['id'] = loop_id
        
        # 基本筛选条件
        # 1. 环路类型
        if config.allowed_loop_types and loop_data['type'] not in config.allowed_loop_types:
            continue
        filter_stats['passed_type'] += 1
        
        # 2. 节点数量
        node_count = loop_data['node_count']
        if node_count < config.min_loop_nodes or node_count > config.max_loop_nodes:
            continue
        filter_stats['passed_nodes'] += 1
        
        # 计算各项指标
        try:
            transaction_metrics = calculate_transaction_metrics(loop_data, metrics_data, config)
            equity_metrics = calculate_equity_metrics(loop_data, graph, config)
            structure_metrics = calculate_structure_metrics(loop_data)
            
            # 合并所有指标
            all_metrics = {**transaction_metrics, **equity_metrics, **structure_metrics}
            
            # 应用筛选条件
            # 3. 交易金额
            if all_metrics['transaction_amount'] < config.min_transaction_amount:
                continue
            filter_stats['passed_amount'] += 1
            
            # 4. 交易频率
            if all_metrics['transaction_frequency'] < config.min_transaction_frequency:
                continue
            filter_stats['passed_frequency'] += 1
            
            # 5. 股东持股比例 - 只在有数据时检查
            max_ratio = all_metrics['max_shareholder_ratio']
            min_ratio = all_metrics['min_shareholder_ratio']
            if (max_ratio > 0 and 
                (min_ratio < config.min_shareholder_ratio or max_ratio > config.max_shareholder_ratio)):
                continue
            filter_stats['passed_ratio'] += 1
            
            # 6. 成员单位数量
            if all_metrics['member_company_count'] < config.min_member_companies:
                continue
            filter_stats['passed_members'] += 1
            
            # 计算风险分数
            risk_score = calculate_risk_score(all_metrics, config)
            
            # 7. 风险分数阈值
            if risk_score < config.min_risk_score:
                continue
            filter_stats['passed_risk'] += 1
            
            # 保存结果
            result = {
                'loop_id': loop_id,
                'risk_score': risk_score,
                'loop_data': loop_data,
                'metrics': all_metrics
            }
            filtered_results.append(result)
            
        except Exception as e:
            logging.warning(f"处理环路 {loop_id} 时出错: {e}")
            continue
        
        # 进度报告
        if loop_id % 5000 == 0:
            logging.info(f"已处理 {loop_id} / {len(loops)} 个环路...")
    
    # 打印筛选统计
    logging.info("=== 筛选统计 ===")
    logging.info(f"总环路数: {filter_stats['total']}")
    logging.info(f"通过类型筛选: {filter_stats['passed_type']}")
    logging.info(f"通过节点数筛选: {filter_stats['passed_nodes']}")
    logging.info(f"通过交易金额筛选: {filter_stats['passed_amount']}")
    logging.info(f"通过交易频率筛选: {filter_stats['passed_frequency']}")
    logging.info(f"通过持股比例筛选: {filter_stats['passed_ratio']}")
    logging.info(f"通过成员单位筛选: {filter_stats['passed_members']}")
    logging.info(f"通过风险分数筛选: {filter_stats['passed_risk']}")
    
    # 按风险分数排序
    filtered_results.sort(key=lambda x: x['risk_score'], reverse=True)
    
    # 返回前K个结果
    return filtered_results[:config.top_k_results]

def save_filtered_results(filtered_results, config):
    """保存筛选结果"""
    # 保存详细报告
    with open(FILTERED_LOOPS_FILE, 'w', encoding='utf-8') as f:
        f.write(f"# 高风险闭环筛选结果 (优化版)\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"筛选条件:\n")
        f.write(f"- 最小交易金额: {config.min_transaction_amount:,.0f} 元\n")
        f.write(f"- 最小交易频率: {config.min_transaction_frequency} 次/{config.transaction_time_window_months}个月\n")
        f.write(f"- 最小风险分数: {config.min_risk_score}\n")
        f.write(f"- 筛选出 {len(filtered_results)} 个高风险闭环\n\n")
        
        f.write("## 高风险闭环列表\n\n")
        
        for i, result in enumerate(filtered_results):
            loop_data = result['loop_data']
            metrics = result['metrics']
            
            f.write(f"### {i+1}. 风险分数: {result['risk_score']:.3f}\n\n")
            f.write(f"**基本信息:**\n")
            f.write(f"- 环路ID: {result['loop_id']}\n")
            f.write(f"- 类型: {loop_data['type']}\n")
            f.write(f"- 起始股东: {loop_data['source']}\n")
            f.write(f"- 节点数: {loop_data['node_count']}\n\n")
            
            f.write(f"**风险指标:**\n")
            f.write(f"- 交易总额: {metrics['transaction_amount']:,.0f} 元\n")
            f.write(f"- 交易频率: {metrics['transaction_frequency']} 次\n")
            f.write(f"- 时间集中度: {metrics['time_concentration']:.2%}\n")
            f.write(f"- 股权集中度: {metrics['equity_concentration']:.2%}\n")
            f.write(f"- 最大持股比例: {metrics['max_shareholder_ratio']:.2%}\n\n")
            
            f.write(f"**环路路径:**\n")
            f.write(f"{loop_data['content']}\n\n")
            f.write("-" * 80 + "\n\n")
    
    # 保存CSV摘要
    if filtered_results:
        summary_data = []
        for result in filtered_results:
            summary_data.append({
                'loop_id': result['loop_id'],
                'risk_score': result['risk_score'],
                'type': result['loop_data']['type'],
                'source': result['loop_data']['source'],
                'node_count': result['loop_data']['node_count'],
                'transaction_amount': result['metrics']['transaction_amount'],
                'transaction_frequency': result['metrics']['transaction_frequency'],
                'time_concentration': result['metrics']['time_concentration'],
                'equity_concentration': result['metrics']['equity_concentration'],
                'max_shareholder_ratio': result['metrics']['max_shareholder_ratio']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(FILTER_REPORT_FILE, index=False, encoding='utf-8')
        
        logging.info(f"筛选结果已保存到: {FILTERED_LOOPS_FILE}")
        logging.info(f"摘要报告已保存到: {FILTER_REPORT_FILE}")
    else:
        logging.warning("没有筛选结果，未生成文件")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='环路优筛工具 (优化版)')
    parser.add_argument('--loop-file', default=DEFAULT_LOOP_FILE, help='闭环检测结果文件')
    parser.add_argument('--graph-file', default=DEFAULT_GRAPH_FILE, help='异构图文件')
    parser.add_argument('--metrics-file', default=DEFAULT_METRICS_FILE, help='闭环指标文件')
    parser.add_argument('--min-amount', type=float, default=10000, help='最小交易金额')
    parser.add_argument('--min-frequency', type=int, default=1, help='最小交易频率')
    parser.add_argument('--time-window', type=int, default=12, help='交易时间窗口（月）')
    parser.add_argument('--min-risk-score', type=float, default=0.1, help='最小风险分数')
    parser.add_argument('--top-k', type=int, default=100, help='输出前K个结果')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 创建配置
    config = FilterConfig()
    config.min_transaction_amount = args.min_amount
    config.min_transaction_frequency = args.min_frequency
    config.transaction_time_window_months = args.time_window
    config.min_risk_score = args.min_risk_score
    config.top_k_results = args.top_k
    
    # 加载数据
    logging.info("加载数据...")
    loops = parse_loops(args.loop_file)
    graph = load_graph(args.graph_file)
    metrics_data = load_loop_metrics(args.metrics_file)
    
    if not loops or graph is None:
        logging.error("数据加载失败")
        return
    
    # 执行筛选
    logging.info("开始筛选环路...")
    start_time = datetime.now()
    filtered_results = filter_loops(loops, graph, metrics_data, config)
    end_time = datetime.now()
    
    # 保存结果
    save_filtered_results(filtered_results, config)
    
    # 打印摘要
    duration = (end_time - start_time).total_seconds()
    print(f"\n筛选完成！(耗时: {duration:.1f}秒)")
    print(f"- 总环路数: {len(loops)}")
    print(f"- 筛选后: {len(filtered_results)}")
    if filtered_results:
        avg_score = sum(r['risk_score'] for r in filtered_results) / len(filtered_results)
        print(f"- 平均风险分数: {avg_score:.3f}")
        print(f"- 最高风险分数: {max(r['risk_score'] for r in filtered_results):.3f}")
    print(f"\n结果已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
