#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
闭环分析工具 - 为检测到的股权闭环创建"画像"

该脚本读取闭环检测结果和异构图，计算各种统计指标，并生成结构化的闭环分析报告和数据表格。
"""

import os
import sys
import re
import logging
import pandas as pd
import networkx as nx
import json
from datetime import datetime
from collections import defaultdict
import igraph as ig
import random

# 配置常量
INPUT_LOOP_FILE = "outputs/loop_results/equity_loops_optimized.txt"  # 闭环检测结果文件
INPUT_GRAPH_FILE = "model/final_heterogeneous_graph.graphml"  # 异构图文件
OUTPUT_DIR = "outputs/loop_analysis"  # 输出目录
LOG_DIR = "outputs/log"
LOG_FILE = os.path.join(LOG_DIR, "loop_profiling.log")
LOOP_INFO_CSV = os.path.join(OUTPUT_DIR, "loop_basic_info.csv")  # 表一：闭环基本信息
LOOP_METRICS_CSV = os.path.join(OUTPUT_DIR, "loop_metrics.csv")  # 表二：闭环指标

def setup_logging():
    """设置日志记录"""
    for directory in [OUTPUT_DIR, LOG_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, 'w', 'utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"闭环分析工具启动，日志记录于 {LOG_FILE}")

def load_graph(file_path):
    """加载NetworkX图"""
    try:
        logging.info(f"正在加载图: {file_path}")
        graph = nx.read_graphml(file_path)
        logging.info(f"图加载成功: {len(graph.nodes())} 节点, {len(graph.edges())} 边")
        return graph
    except Exception as e:
        logging.error(f"加载图时出错: {e}")
        return None

def parse_loop_from_txt(file_path):
    """
    从闭环检测结果文本文件中解析闭环
    返回字典 {闭环ID: {源头: ..., 内容: ..., 节点路径: [...]}}
    """
    if not os.path.exists(file_path):
        logging.error(f"闭环文件不存在: {file_path}")
        return {}
    
    logging.info(f"正在解析闭环文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取闭环详细信息段落
    detailed_section = re.split(r'## 详细闭环信息', content, 1)
    if len(detailed_section) < 2:
        logging.error("无法找到闭环详细信息部分")
        return {}
    
    detailed_content = detailed_section[1]
    
    # 按股东分段
    company_sections = re.split(r'### 股东: ', detailed_content)[1:]
    
    loops = {}
    loop_id = 1
    
    for section in company_sections:
        # 提取股东名称
        lines = section.strip().split('\n')
        company_name = lines[0].strip()
        logging.info(f"处理股东 '{company_name}' 的闭环")
        
        # 提取闭环
        in_loop_section = False
        current_loop_type = ""
        for i, line in enumerate(lines):
            if line.startswith('####'):
                current_loop_type = line.replace('####', '').strip()
                in_loop_section = True
                continue
            
            if in_loop_section and line.strip() and not line.startswith('-'):
                if re.match(r'^\d+\.', line):  # 匹配形如 "1. 闭环内容" 的行
                    # 提取闭环内容
                    loop_content = re.sub(r'^\d+\.\s*', '', line).strip()
                    
                    # 提取节点路径
                    node_path = []
                    path_parts = re.split(r'--\([^)]+\)-->|<--\([^)]+\)--|-->', loop_content)
                    for part in path_parts:
                        part = part.strip()
                        if part:
                            # 提取节点名称，去除角色标签
                            node_name = re.sub(r'\s*\[[^\]]+\]', '', part).strip()
                            node_path.append(node_name)
                    
                    # 创建闭环记录
                    loop_data = {
                        'source': company_name,
                        'content': loop_content,
                        'type': current_loop_type,
                        'node_path': node_path
                    }
                    
                    loops[loop_id] = loop_data
                    loop_id += 1
    
    logging.info(f"成功解析了 {len(loops)} 个闭环")
    return loops

def analyze_loop_edges(graph, loops):
    """
    分析闭环中的边
    计算各种指标并返回包含指标的字典
    """
    logging.info("开始分析闭环边...")
    
    loop_metrics = {}
    
    for loop_id, loop_data in loops.items():
        node_path = loop_data['node_path']
        if not node_path or len(node_path) < 3:  # 闭环至少需要3个节点
            continue
        
        # 确保路径形成闭环
        if node_path[0] != node_path[-1]:
            node_path.append(node_path[0])
        
        # 初始化指标
        metrics = {
            'loop_id': loop_id,
            'node_count': len(set(node_path)),  # 去重后的节点数
            'edge_count': 0,
            'total_transaction_amount': 0.0,
            'avg_transaction_amount': 0.0,
            'max_transaction_amount': 0.0,
            'min_transaction_amount': float('inf'),
            'transaction_count': 0,
            'earliest_transaction_date': None,
            'latest_transaction_date': None,
            'time_span_days': 0,
            'has_bidirectional_edges': False,
            'partner_count': 0,
            'shareholder_count': 0,
            'member_company_count': 0,
            'cross_holding_count': 0,  # 交叉持股数量
            'max_edge_count_between_nodes': 0,  # 两节点间最大边数
        }
        
        # 节点类型计数
        node_types = defaultdict(int)
        for node in set(node_path):
            try:
                node_data = graph.nodes[node]
                node_type = node_data.get('label', '')
                node_types[node_type] += 1
            except:
                # 节点可能不在图中
                continue
        
        metrics['partner_count'] = node_types.get('partner', 0)
        metrics['shareholder_count'] = node_types.get('股东', 0)
        metrics['member_company_count'] = node_types.get('成员单位', 0)
        
        # 边分析
        edge_counts = defaultdict(int)  # 统计节点对之间的边数
        transaction_amounts = []
        transaction_dates = []
        bidirectional_edges = set()  # 双向边的节点对
        
        for i in range(len(node_path) - 1):
            source = node_path[i]
            target = node_path[i + 1]
            edge_key = (source, target)
            reverse_edge_key = (target, source)
            
            # 检查是否有双向边
            if reverse_edge_key in edge_counts:
                bidirectional_edges.add((min(source, target), max(source, target)))
            
            edge_counts[edge_key] += 1
            
            # 尝试获取边的信息
            try:
                if graph.has_edge(source, target):
                    edges = [(source, target)]
                elif graph.has_edge(target, source):
                    edges = [(target, source)]
                else:
                    edges = []
                
                for s, t in edges:
                    edge_data = graph.edges[s, t]
                    metrics['edge_count'] += 1
                    
                    # 交易金额
                    if 'transaction_amount' in edge_data:
                        try:
                            amount = float(edge_data['transaction_amount'])
                            transaction_amounts.append(amount)
                            metrics['total_transaction_amount'] += amount
                            metrics['max_transaction_amount'] = max(metrics['max_transaction_amount'], amount)
                            if amount > 0:  # 忽略零值
                                metrics['min_transaction_amount'] = min(metrics['min_transaction_amount'], amount)
                            metrics['transaction_count'] += 1
                        except (ValueError, TypeError):
                            pass
                    
                    # 交易日期
                    if 'transaction_date' in edge_data:
                        try:
                            date_str = edge_data['transaction_date']
                            # 尝试不同的日期格式
                            for date_format in ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y']:
                                try:
                                    date = datetime.strptime(date_str, date_format)
                                    transaction_dates.append(date)
                                    break
                                except ValueError:
                                    continue
                        except Exception as e:
                            pass
            except Exception as e:
                logging.warning(f"分析闭环 {loop_id} 的边 ({source} -> {target}) 时出错: {e}")
        
        # 计算交易金额平均值
        if transaction_amounts:
            metrics['avg_transaction_amount'] = sum(transaction_amounts) / len(transaction_amounts)
            if metrics['min_transaction_amount'] == float('inf'):
                metrics['min_transaction_amount'] = 0.0
        else:
            metrics['min_transaction_amount'] = 0.0
        
        # 计算交易时间跨度
        if transaction_dates:
            earliest = min(transaction_dates)
            latest = max(transaction_dates)
            metrics['earliest_transaction_date'] = earliest.strftime('%Y-%m-%d')
            metrics['latest_transaction_date'] = latest.strftime('%Y-%m-%d')
            metrics['time_span_days'] = (latest - earliest).days
        
        # 交叉持股和最大边数
        metrics['has_bidirectional_edges'] = len(bidirectional_edges) > 0
        metrics['cross_holding_count'] = len(bidirectional_edges)
        metrics['max_edge_count_between_nodes'] = max(edge_counts.values()) if edge_counts else 0
        
        loop_metrics[loop_id] = metrics
    
    logging.info(f"完成了 {len(loop_metrics)} 个闭环的指标计算")
    return loop_metrics

def create_output_tables(loops, metrics):
    """
    创建输出表格
    表一：闭环基本信息
    表二：闭环指标
    """
    logging.info("创建输出表格...")
    
    # 表一：闭环基本信息
    basic_info = []
    for loop_id, loop_data in loops.items():
        basic_info.append({
            'loop_id': loop_id,
            'source': loop_data['source'],
            'content': loop_data['content'],
            'type': loop_data.get('type', '')
        })
    
    basic_info_df = pd.DataFrame(basic_info)
    
    # 表二：闭环指标
    metrics_list = list(metrics.values())
    metrics_df = pd.DataFrame(metrics_list)
    
    # 保存到CSV
    basic_info_df.to_csv(LOOP_INFO_CSV, index=False, encoding='utf-8')
    metrics_df.to_csv(LOOP_METRICS_CSV, index=False, encoding='utf-8')
    
    logging.info(f"表格已保存到: {LOOP_INFO_CSV} 和 {LOOP_METRICS_CSV}")
    
    return basic_info_df, metrics_df

def generate_loop_samples(loops, metrics, sample_size=5):
    """生成闭环样本用于展示"""
    # 按不同类型抽样
    loop_types = {}
    for loop_id, loop_data in loops.items():
        loop_type = loop_data.get('type', 'unknown')
        if loop_type not in loop_types:
            loop_types[loop_type] = []
        loop_types[loop_type].append(loop_id)
    
    samples = []
    for loop_type, loop_ids in loop_types.items():
        sample_ids = random.sample(loop_ids, min(sample_size, len(loop_ids)))
        for loop_id in sample_ids:
            if loop_id in metrics:
                sample = {
                    'loop_id': loop_id,
                    'type': loop_type,
                    'content': loops[loop_id]['content'],
                    'node_count': metrics[loop_id]['node_count'],
                    'edge_count': metrics[loop_id]['edge_count'],
                    'total_transaction_amount': metrics[loop_id]['total_transaction_amount'],
                    'transaction_count': metrics[loop_id]['transaction_count'],
                    'has_bidirectional_edges': metrics[loop_id]['has_bidirectional_edges']
                }
                samples.append(sample)
    
    return samples

def print_summary(loops, metrics, samples):
    """打印分析摘要"""
    print("\n" + "=" * 80)
    print(f"闭环分析摘要")
    print("=" * 80)
    
    # 闭环类型统计
    loop_types = {}
    for loop_id, loop_data in loops.items():
        loop_type = loop_data.get('type', 'unknown')
        if loop_type not in loop_types:
            loop_types[loop_type] = 0
        loop_types[loop_type] += 1
    
    print("\n闭环类型分布:")
    for loop_type, count in sorted(loop_types.items()):
        print(f"  - {loop_type}: {count} 个")
    
    # 指标摘要
    print("\n指标摘要:")
    # 节点数
    node_counts = [m['node_count'] for m in metrics.values()]
    print(f"  - 平均节点数: {sum(node_counts) / len(node_counts):.2f}")
    # 交易金额
    transaction_amounts = [m['total_transaction_amount'] for m in metrics.values()]
    print(f"  - 平均交易总额: {sum(transaction_amounts) / len(transaction_amounts):.2f}")
    # 交叉持股
    cross_holding_counts = [m['cross_holding_count'] for m in metrics.values()]
    total_cross_holdings = sum(cross_holding_counts)
    loops_with_cross_holdings = sum(1 for c in cross_holding_counts if c > 0)
    print(f"  - 包含交叉持股的闭环: {loops_with_cross_holdings} 个 ({loops_with_cross_holdings / len(metrics) * 100:.2f}%)")
    print(f"  - 交叉持股总数: {total_cross_holdings}")
    
    # 样本展示
    print("\n闭环样本:")
    for i, sample in enumerate(samples[:5]):  # 只显示前5个样本
        print(f"\n样本 {i+1}:")
        print(f"  ID: {sample['loop_id']}")
        print(f"  类型: {sample['type']}")
        print(f"  内容: {sample['content'][:150]}...")
        print(f"  节点数: {sample['node_count']}")
        print(f"  边数: {sample['edge_count']}")
        print(f"  交易总额: {sample['total_transaction_amount']:.2f}")
        print(f"  交易次数: {sample['transaction_count']}")
        print(f"  是否存在交叉持股: {'是' if sample['has_bidirectional_edges'] else '否'}")
    
    print("\n" + "=" * 80)
    print(f"分析结果已保存到:")
    print(f"  - 闭环基本信息: {LOOP_INFO_CSV}")
    print(f"  - 闭环指标: {LOOP_METRICS_CSV}")
    print("=" * 80 + "\n")

def main():
    """主函数"""
    setup_logging()
    
    # 加载异构图
    graph = load_graph(INPUT_GRAPH_FILE)
    if graph is None:
        logging.error("无法加载异构图，退出程序")
        return
    
    # 解析闭环文件
    loops = parse_loop_from_txt(INPUT_LOOP_FILE)
    if not loops:
        logging.error("未找到闭环数据，退出程序")
        return
    
    # 分析闭环边
    metrics = analyze_loop_edges(graph, loops)
    
    # 创建输出表格
    basic_info_df, metrics_df = create_output_tables(loops, metrics)
    
    # 生成样本
    samples = generate_loop_samples(loops, metrics)
    
    # 打印摘要
    print_summary(loops, metrics, samples)
    
    logging.info("闭环分析完成")

if __name__ == "__main__":
    main() 