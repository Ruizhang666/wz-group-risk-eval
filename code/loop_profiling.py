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
                    # 改进解析模式以更好地处理各种边标签
                    path_parts = re.split(r'--\([^)]+\)-->|<--\([^)]+\)--|-->|<--', loop_content)
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

def find_node_by_name(graph, name):
    """
    根据节点名称在图中查找节点ID
    返回匹配的节点ID列表
    """
    matched_nodes = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('name', '') == name:
            matched_nodes.append(node_id)
    return matched_nodes

def analyze_loop_edges(graph, loops):
    """
    分析闭环中的边
    计算各种指标并返回包含指标的字典
    """
    logging.info("开始分析闭环边...")
    
    loop_metrics = {}
    
    # 检查图类型并记录日志
    graph_type = type(graph).__name__
    logging.info(f"图类型: {graph_type}")
    
    # 检查图中是否存在交易边
    transaction_edge_count = 0
    for u, v, key, data in graph.edges(data=True, keys=True):
        if data.get('label') == '交易':
            transaction_edge_count += 1
            if transaction_edge_count <= 10:
                # 输出前10个交易边的详细信息作为样本
                u_name = graph.nodes[u].get('name', u)
                v_name = graph.nodes[v].get('name', v)
                logging.info(f"交易边样本 {transaction_edge_count}: {u_name} -> {v_name}, 属性: {data}")
    
    logging.info(f"图中交易边总数: {transaction_edge_count}")
    
    # 创建名称到节点ID的映射以加速查找
    name_to_nodes = {}
    for node_id, attrs in graph.nodes(data=True):
        name = attrs.get('name', '')
        if name:
            if name not in name_to_nodes:
                name_to_nodes[name] = []
            name_to_nodes[name].append(node_id)
    
    # 额外添加节点类型统计，确认图中的节点类型分布
    node_label_count = {}
    for node, attrs in graph.nodes(data=True):
        label = attrs.get('label', 'unknown')
        node_label_count[label] = node_label_count.get(label, 0) + 1
    
    logging.info(f"图中节点标签分布: {node_label_count}")
    
    # 记录闭环中找到节点和边的统计信息
    node_found_count = 0
    node_not_found_count = 0
    edge_found_count = 0
    edge_not_found_count = 0
    
    for loop_id, loop_data in loops.items():
        node_path = loop_data['node_path']
        if not node_path or len(node_path) < 3:  # 闭环至少需要3个节点
            continue
        
        # 确保路径形成闭环
        if node_path[0] != node_path[-1]:
            node_path.append(node_path[0])
        
        # 将节点名称转换为节点ID
        node_id_path = []
        node_not_found_in_loop = False
        node_type_map = {}  # 记录节点ID到节点类型的映射
        
        for node_name in node_path:
            if node_name in name_to_nodes:
                # 为简单起见，如果有多个同名节点，使用第一个
                node_id = name_to_nodes[node_name][0]
                node_id_path.append(node_id)
                # 记录节点类型
                node_type = graph.nodes[node_id].get('label', 'unknown')
                node_type_map[node_id] = node_type
                node_found_count += 1
            else:
                # 找不到节点，记录并跳过此闭环
                if int(loop_id) <= 5:
                    logging.warning(f"在闭环 {loop_id} 中找不到节点 '{node_name}'")
                node_not_found_count += 1
                node_not_found_in_loop = True
                break
        
        if node_not_found_in_loop:
            if int(loop_id) <= 5:
                logging.warning(f"闭环 {loop_id} 中有节点未找到，跳过此闭环")
            continue
        
        # 初始化指标 - 只保留用户指定的指标
        metrics = {
            'loop_id': loop_id,
            'total_transaction_amount': 0.0,  # 交易总额
            'upstream_to_member_transaction_amount': 0.0,  # 上游到成员公司的交易总额
            'upstream_to_member_transaction_count': 0,     # 上游到成员公司的交易次数(用于计算平均值)
            'member_to_downstream_transaction_amount': 0.0, # 成员公司到下游的交易总额
            'member_to_downstream_transaction_count': 0,    # 成员公司到下游的交易次数(用于计算平均值)
            'upstream_to_member_avg_amount': 0.0,          # 上游到成员公司的平均交易金额
            'member_to_downstream_avg_amount': 0.0,        # 成员公司到下游的平均交易金额
            'upstream_to_member_transaction_times': [],     # 上游到成员公司的交易时间列表
            'member_to_downstream_transaction_times': [],   # 成员公司到下游的交易时间列表
        }
        
        # 找到成员单位节点的索引位置
        member_node_indices = []
        for i, node_id in enumerate(node_id_path):
            if node_type_map.get(node_id) == '成员单位':
                member_node_indices.append(i)
        
        # 用于调试特定闭环
        if int(loop_id) <= 5:
            logging.info(f"分析闭环 {loop_id} 的边...")
            node_names = [graph.nodes[nid].get('name', nid) for nid in node_id_path]
            logging.info(f"闭环 {loop_id} 的节点路径 (ID转名称): {node_names}")
            if member_node_indices:
                logging.info(f"成员单位节点在路径中的索引位置: {member_node_indices}")
        
        for i in range(len(node_id_path) - 1):
            source_id = node_id_path[i]
            target_id = node_id_path[i + 1]
            source_type = node_type_map.get(source_id, 'unknown')
            target_type = node_type_map.get(target_id, 'unknown')
            
            # 判断当前边是否涉及成员单位
            is_upstream_to_member = source_type != '成员单位' and target_type == '成员单位'
            is_member_to_downstream = source_type == '成员单位' and target_type != '成员单位'
            
            source_name = graph.nodes[source_id].get('name', source_id)
            target_name = graph.nodes[target_id].get('name', target_id)
            
            # 调试输出
            if int(loop_id) <= 5:
                logging.info(f"检查边: {source_name} -> {target_name} (ID: {source_id} -> {target_id})")
            
            # 检查边是否存在
            if graph.has_edge(source_id, target_id):
                edge_found_count += 1
                # 对于MultiDiGraph，一对节点之间可能有多条边
                if isinstance(graph, nx.MultiDiGraph):
                    for key in graph.get_edge_data(source_id, target_id):
                        edge_data = graph.get_edge_data(source_id, target_id, key)
                        
                        # 特别处理交易边
                        if edge_data.get('label') == '交易':
                            if int(loop_id) <= 5:
                                logging.info(f"闭环 {loop_id} 发现交易边: {source_name} -> {target_name}, 属性: {edge_data}")
                            
                            # 处理交易金额
                            amount_field = 'amount'
                            if amount_field in edge_data and edge_data[amount_field] not in ['', None]:
                                try:
                                    # 确保是数值
                                    amount = edge_data[amount_field]
                                    if isinstance(amount, str):
                                        amount = float(amount.strip())
                                    elif isinstance(amount, (int, float)):
                                        amount = float(amount)
                                    else:
                                        continue
                                    
                                    if amount > 0:  # 忽略非正值
                                        # 总交易额
                                        metrics['total_transaction_amount'] += amount
                                        
                                        # 区分上游到成员公司和成员公司到下游的交易
                                        if is_upstream_to_member:
                                            metrics['upstream_to_member_transaction_amount'] += amount
                                            metrics['upstream_to_member_transaction_count'] += 1
                                        elif is_member_to_downstream:
                                            metrics['member_to_downstream_transaction_amount'] += amount
                                            metrics['member_to_downstream_transaction_count'] += 1
                                        
                                        if int(loop_id) <= 5:
                                            logging.info(f"处理交易金额: {amount}" + 
                                                        (", 上游到成员公司" if is_upstream_to_member else 
                                                         ", 成员公司到下游" if is_member_to_downstream else ""))
                                except (ValueError, TypeError) as e:
                                    if int(loop_id) <= 5:
                                        logging.info(f"处理交易金额时出错: {e}, 金额值: {edge_data.get(amount_field)}")
                            
                            # 处理交易日期
                            year = edge_data.get('year', '')
                            month = edge_data.get('month', '')
                            
                            if year and month:
                                try:
                                    if isinstance(month, str) and month.strip():
                                        month_str = month.zfill(2)
                                    elif isinstance(month, (int, float)):
                                        month_str = str(int(month)).zfill(2)
                                    else:
                                        month_str = "01"  # 默认值
                                    
                                    date_str = f"{year}-{month_str}-01"  # 假设每月1号
                                    
                                    # 区分上游到成员公司和成员公司到下游的交易时间
                                    if is_upstream_to_member:
                                        metrics['upstream_to_member_transaction_times'].append(date_str)
                                    elif is_member_to_downstream:
                                        metrics['member_to_downstream_transaction_times'].append(date_str)
                                        
                                except Exception as e:
                                    if int(loop_id) <= 5:
                                        logging.info(f"处理交易日期时出错: {e}, 年份: {year}, 月份: {month}")
                else:
                    # 处理普通DiGraph
                    edge_data = graph.get_edge_data(source_id, target_id)
                    
                    # 特别处理交易边
                    if edge_data.get('label') == '交易':
                        if int(loop_id) <= 5:
                            logging.info(f"闭环 {loop_id} 发现交易边: {source_name} -> {target_name}, 属性: {edge_data}")
                        
                        # 处理交易金额
                        amount_field = 'amount'
                        if amount_field in edge_data and edge_data[amount_field] not in ['', None]:
                            try:
                                # 确保是数值
                                amount = edge_data[amount_field]
                                if isinstance(amount, str):
                                    amount = float(amount.strip())
                                elif isinstance(amount, (int, float)):
                                    amount = float(amount)
                                else:
                                    continue
                                
                                if amount > 0:  # 忽略非正值
                                    # 总交易额
                                    metrics['total_transaction_amount'] += amount
                                    
                                    # 区分上游到成员公司和成员公司到下游的交易
                                    if is_upstream_to_member:
                                        metrics['upstream_to_member_transaction_amount'] += amount
                                        metrics['upstream_to_member_transaction_count'] += 1
                                    elif is_member_to_downstream:
                                        metrics['member_to_downstream_transaction_amount'] += amount
                                        metrics['member_to_downstream_transaction_count'] += 1
                                    
                                    if int(loop_id) <= 5:
                                        logging.info(f"处理交易金额: {amount}" + 
                                                    (", 上游到成员公司" if is_upstream_to_member else 
                                                     ", 成员公司到下游" if is_member_to_downstream else ""))
                            except (ValueError, TypeError) as e:
                                if int(loop_id) <= 5:
                                    logging.info(f"处理交易金额时出错: {e}, 金额值: {edge_data.get(amount_field)}")
                        
                        # 处理交易日期
                        year = edge_data.get('year', '')
                        month = edge_data.get('month', '')
                        
                        if year and month:
                            try:
                                if isinstance(month, str) and month.strip():
                                    month_str = month.zfill(2)
                                elif isinstance(month, (int, float)):
                                    month_str = str(int(month)).zfill(2)
                                else:
                                    month_str = "01"  # 默认值
                                
                                date_str = f"{year}-{month_str}-01"  # 假设每月1号
                                
                                # 区分上游到成员公司和成员公司到下游的交易时间
                                if is_upstream_to_member:
                                    metrics['upstream_to_member_transaction_times'].append(date_str)
                                elif is_member_to_downstream:
                                    metrics['member_to_downstream_transaction_times'].append(date_str)
                                    
                            except Exception as e:
                                if int(loop_id) <= 5:
                                    logging.info(f"处理交易日期时出错: {e}, 年份: {year}, 月份: {month}")
            else:
                edge_not_found_count += 1
                # 如果直接边不存在，尝试查找反向边
                if graph.has_edge(target_id, source_id):
                    edge_found_count += 1
                    if int(loop_id) <= 5:
                        logging.info(f"找到反向边: {target_name} -> {source_name}")
                    
                    # 对于反向边，交换一下上游和下游的概念
                    is_upstream_to_member_reversed = target_type != '成员单位' and source_type == '成员单位'
                    is_member_to_downstream_reversed = target_type == '成员单位' and source_type != '成员单位'
                    
                    # 对于MultiDiGraph，一对节点之间可能有多条边
                    if isinstance(graph, nx.MultiDiGraph):
                        for key in graph.get_edge_data(target_id, source_id):
                            edge_data = graph.get_edge_data(target_id, source_id, key)
                            
                            # 特别处理交易边
                            if edge_data.get('label') == '交易':
                                if int(loop_id) <= 5:
                                    logging.info(f"闭环 {loop_id} 发现反向交易边: {target_name} -> {source_name}, 属性: {edge_data}")
                                
                                # 处理交易金额 - 与正向边处理相同
                                amount_field = 'amount'
                                if amount_field in edge_data and edge_data[amount_field] not in ['', None]:
                                    try:
                                        amount = float(edge_data[amount_field])
                                        if amount > 0:
                                            # 总交易额
                                            metrics['total_transaction_amount'] += amount
                                            
                                            # 区分上游到成员公司和成员公司到下游的交易（注意这是反向边）
                                            if is_upstream_to_member_reversed:
                                                metrics['upstream_to_member_transaction_amount'] += amount
                                                metrics['upstream_to_member_transaction_count'] += 1
                                            elif is_member_to_downstream_reversed:
                                                metrics['member_to_downstream_transaction_amount'] += amount
                                                metrics['member_to_downstream_transaction_count'] += 1
                                    except (ValueError, TypeError):
                                        pass
                                
                                # 处理交易日期 - 与正向边处理相同
                                year = edge_data.get('year', '')
                                month = edge_data.get('month', '')
                                
                                if year and month:
                                    try:
                                        month_str = str(month).zfill(2) if isinstance(month, (int, float)) else month.zfill(2)
                                        date_str = f"{year}-{month_str}-01"
                                        
                                        # 区分上游到成员公司和成员公司到下游的交易时间（注意这是反向边）
                                        if is_upstream_to_member_reversed:
                                            metrics['upstream_to_member_transaction_times'].append(date_str)
                                        elif is_member_to_downstream_reversed:
                                            metrics['member_to_downstream_transaction_times'].append(date_str)
                                    except Exception:
                                        pass
                    else:
                        # 处理普通DiGraph的反向边
                        edge_data = graph.get_edge_data(target_id, source_id)
                        
                        # 特别处理交易边
                        if edge_data.get('label') == '交易':
                            # 处理与正向边相同
                            pass
                else:
                    if int(loop_id) <= 5:
                        logging.info(f"未找到边: {source_name} -> {target_name} 或反向边")
        
        # 计算上游到成员公司和成员公司到下游的平均交易金额
        if metrics['upstream_to_member_transaction_count'] > 0:
            metrics['upstream_to_member_avg_amount'] = metrics['upstream_to_member_transaction_amount'] / metrics['upstream_to_member_transaction_count']
        
        if metrics['member_to_downstream_transaction_count'] > 0:
            metrics['member_to_downstream_avg_amount'] = metrics['member_to_downstream_transaction_amount'] / metrics['member_to_downstream_transaction_count']
        
        loop_metrics[loop_id] = metrics
    
    # 统计结果
    logging.info(f"节点统计: 找到 {node_found_count}, 未找到 {node_not_found_count}")
    logging.info(f"边统计: 找到 {edge_found_count}, 未找到 {edge_not_found_count}")
    
    # 计算并输出交易统计数据
    transaction_loops = sum(1 for m in loop_metrics.values() if m['total_transaction_amount'] > 0)
    total_transaction_amount = sum(m['total_transaction_amount'] for m in loop_metrics.values())
    
    if transaction_loops:
        logging.info(f"包含交易边的闭环数量: {transaction_loops} (占比 {transaction_loops/len(loop_metrics)*100:.2f}%)")
        logging.info(f"闭环中的交易总金额: {total_transaction_amount:.2f}")
        
        # 添加上游-成员公司和成员公司-下游交易统计
        upstream_member_loops = sum(1 for m in loop_metrics.values() if m['upstream_to_member_transaction_count'] > 0)
        member_downstream_loops = sum(1 for m in loop_metrics.values() if m['member_to_downstream_transaction_count'] > 0)
        
        if upstream_member_loops:
            logging.info(f"包含上游到成员公司交易的闭环数量: {upstream_member_loops} (占比 {upstream_member_loops/len(loop_metrics)*100:.2f}%)")
        
        if member_downstream_loops:
            logging.info(f"包含成员公司到下游交易的闭环数量: {member_downstream_loops} (占比 {member_downstream_loops/len(loop_metrics)*100:.2f}%)")
    else:
        logging.info("警告：未检测到任何闭环中包含交易边！")
    
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
                    'total_transaction_amount': metrics[loop_id]['total_transaction_amount'],
                    'upstream_to_member_transaction_amount': metrics[loop_id]['upstream_to_member_transaction_amount'],
                    'member_to_downstream_transaction_amount': metrics[loop_id]['member_to_downstream_transaction_amount'],
                    'upstream_to_member_avg_amount': metrics[loop_id]['upstream_to_member_avg_amount'],
                    'member_to_downstream_avg_amount': metrics[loop_id]['member_to_downstream_avg_amount'],
                    'upstream_to_member_transaction_times': metrics[loop_id]['upstream_to_member_transaction_times'],
                    'member_to_downstream_transaction_times': metrics[loop_id]['member_to_downstream_transaction_times']
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
    # 交易金额
    transaction_amounts = [m['total_transaction_amount'] for m in metrics.values()]
    print(f"  - 平均交易总额: {sum(transaction_amounts) / len(transaction_amounts):.2f}")
    
    # 上游-成员公司交易统计
    upstream_member_amounts = [m['upstream_to_member_transaction_amount'] for m in metrics.values() if m['upstream_to_member_transaction_count'] > 0]
    if upstream_member_amounts:
        print(f"  - 上游-成员公司平均交易总额: {sum(upstream_member_amounts) / len(upstream_member_amounts):.2f}")
    
    # 成员公司-下游交易统计
    member_downstream_amounts = [m['member_to_downstream_transaction_amount'] for m in metrics.values() if m['member_to_downstream_transaction_count'] > 0]
    if member_downstream_amounts:
        print(f"  - 成员公司-下游平均交易总额: {sum(member_downstream_amounts) / len(member_downstream_amounts):.2f}")
    
    # 样本展示
    print("\n闭环样本:")
    for i, sample in enumerate(samples[:5]):  # 只显示前5个样本
        print(f"\n样本 {i+1}:")
        print(f"  ID: {sample['loop_id']}")
        print(f"  类型: {sample['type']}")
        print(f"  内容: {sample['content'][:150]}...")
        print(f"  交易总额: {sample['total_transaction_amount']:.2f}")
        
        # 添加新指标展示
        if sample['upstream_to_member_transaction_amount'] > 0:
            print(f"  上游-成员公司交易: 总额 {sample['upstream_to_member_transaction_amount']:.2f}, 平均金额: {sample['upstream_to_member_avg_amount']:.2f}")
            transaction_times = sample['upstream_to_member_transaction_times']
            if transaction_times:
                print(f"  上游-成员公司交易时间: {', '.join(transaction_times[:5])}{'...' if len(transaction_times) > 5 else ''}")
        
        if sample['member_to_downstream_transaction_amount'] > 0:
            print(f"  成员公司-下游交易: 总额 {sample['member_to_downstream_transaction_amount']:.2f}, 平均金额: {sample['member_to_downstream_avg_amount']:.2f}")
            transaction_times = sample['member_to_downstream_transaction_times']
            if transaction_times:
                print(f"  成员公司-下游交易时间: {', '.join(transaction_times[:5])}{'...' if len(transaction_times) > 5 else ''}")
    
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