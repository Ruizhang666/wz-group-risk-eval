#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股权指标提取器 - 多核并行优化版本
从图模型和基础信息中提取股权相关指标，与交易指标合并形成综合画像
"""

import pandas as pd
import networkx as nx
import numpy as np
import re
import ast
import json
import os
import logging
import multiprocessing as mp
from multiprocessing import Pool, Manager, Lock
from datetime import datetime
import time
from functools import partial
import gc
import pickle
from tqdm import tqdm

# 全局变量，用于存储图模型
global_graph = None
global_lock = None

def setup_logging():
    """设置日志"""
    log_dir = "outputs/log"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/equity_metrics_extractor_parallel.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def init_worker(graph_data):
    """初始化工作进程，加载共享的图模型"""
    global global_graph
    try:
        # 反序列化图模型
        global_graph = pickle.loads(graph_data)
        print(f"工作进程 {os.getpid()} 已加载图模型：{global_graph.number_of_nodes()} 节点，{global_graph.number_of_edges()} 边")
    except Exception as e:
        print(f"工作进程初始化失败: {e}")
        global_graph = None

def load_graph_model(graph_path):
    """加载图模型并序列化为可共享的数据"""
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"加载图模型: {graph_path}")
        G = nx.read_graphml(graph_path)
        logger.info(f"图模型加载成功，节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
        
        # 序列化图模型以便在进程间共享
        logger.info("序列化图模型用于多进程共享...")
        graph_data = pickle.dumps(G)
        logger.info(f"图模型序列化完成，大小: {len(graph_data) / 1024 / 1024:.2f} MB")
        
        return G, graph_data
    except Exception as e:
        logger.error(f"加载图模型失败: {e}")
        return None, None

def extract_loop_nodes_from_content(content):
    """从环路内容中提取节点列表"""
    try:
        if pd.isna(content) or not isinstance(content, str):
            return []
        
        # 使用正则表达式提取所有实体名称（在[]标签前的内容）
        pattern = r'([^[\]]+?)\s*\[[^\]]+\]'
        matches = re.findall(pattern, content)
        
        # 清理和去重
        nodes = []
        for match in matches:
            node_name = match.strip()
            if node_name and node_name not in nodes:
                nodes.append(node_name)
        
        return nodes
    except Exception as e:
        return []

def find_node_ids_by_names(graph, node_names):
    """根据节点名称在图中查找对应的节点ID"""
    try:
        node_ids = []
        name_to_id_map = {}
        
        # 构建名称到ID的映射
        for node_id, node_data in graph.nodes(data=True):
            node_name = node_data.get('name', '')
            if node_name:
                name_to_id_map[node_name] = node_id
        
        # 查找每个名称对应的ID
        for name in node_names:
            if name in name_to_id_map:
                node_ids.append(name_to_id_map[name])
            else:
                # 如果找不到精确匹配，尝试模糊匹配
                for graph_name, graph_id in name_to_id_map.items():
                    if name in graph_name or graph_name in name:
                        node_ids.append(graph_id)
                        break
        
        return node_ids
    except Exception as e:
        return []

def calculate_ownership_concentration_metrics(graph, loop_node_ids):
    """计算股权集中度指标"""
    try:
        ownership_percentages = []
        
        # 从图中提取环路节点间的控股比例
        for node_id in loop_node_ids:
            if node_id in graph.nodes:
                # 获取该节点的所有入边（被控股关系）
                for predecessor in graph.predecessors(node_id):
                    if predecessor in loop_node_ids:  # 只考虑环路内的控股关系
                        edge_data = graph.get_edge_data(predecessor, node_id)
                        if edge_data:
                            for edge_key, edge_attrs in edge_data.items():
                                # 只考虑控股关系
                                if edge_attrs.get('label') == '控股':
                                    percent = edge_attrs.get('percent')
                                    if percent is not None and isinstance(percent, (int, float)):
                                        ownership_percentages.append(percent)
        
        if not ownership_percentages:
            return {
                'max_ownership_percent': None,
                'min_ownership_percent': None,
                'avg_ownership_percent': None,
                'ownership_concentration_index': None,
                'total_ownership_percent': None,
                'ownership_count': 0
            }
        
        # 计算各项指标
        max_percent = max(ownership_percentages)
        min_percent = min(ownership_percentages)
        avg_percent = np.mean(ownership_percentages)
        total_percent = sum(ownership_percentages)
        
        # 计算集中度指数（类似HHI指数）
        concentration_index = sum([p**2 for p in ownership_percentages])
        
        return {
            'max_ownership_percent': round(max_percent, 4),
            'min_ownership_percent': round(min_percent, 4),
            'avg_ownership_percent': round(avg_percent, 4),
            'ownership_concentration_index': round(concentration_index, 4),
            'total_ownership_percent': round(total_percent, 4),
            'ownership_count': len(ownership_percentages)
        }
        
    except Exception as e:
        return {
            'max_ownership_percent': None,
            'min_ownership_percent': None,
            'avg_ownership_percent': None,
            'ownership_concentration_index': None,
            'total_ownership_percent': None,
            'ownership_count': 0
        }

def calculate_shareholder_type_metrics(graph, loop_node_ids):
    """计算股东类型分布指标"""
    try:
        type_counts = {'P': 0, 'E': 0, 'UE': 0, 'Unknown': 0}
        total_nodes = 0
        
        # 统计环路中各类型节点数量
        for node_id in loop_node_ids:
            if node_id in graph.nodes:
                total_nodes += 1
                node_type = graph.nodes[node_id].get('type', 'Unknown')
                if node_type in type_counts:
                    type_counts[node_type] += 1
                else:
                    type_counts['Unknown'] += 1
        
        if total_nodes == 0:
            return {
                'natural_person_count': 0,
                'enterprise_count': 0,
                'unknown_enterprise_count': 0,
                'unknown_type_count': 0,
                'natural_person_ratio': 0.0,
                'enterprise_ratio': 0.0,
                'dominant_shareholder_type': 'Unknown',
                'total_shareholders': 0
            }
        
        # 计算比例
        natural_person_ratio = type_counts['P'] / total_nodes
        enterprise_ratio = (type_counts['E'] + type_counts['UE']) / total_nodes
        
        # 确定主导股东类型
        if type_counts['P'] > type_counts['E'] + type_counts['UE']:
            dominant_type = 'Natural_Person'
        elif type_counts['E'] + type_counts['UE'] > type_counts['P']:
            dominant_type = 'Enterprise'
        else:
            dominant_type = 'Mixed'
        
        return {
            'natural_person_count': type_counts['P'],
            'enterprise_count': type_counts['E'],
            'unknown_enterprise_count': type_counts['UE'],
            'unknown_type_count': type_counts['Unknown'],
            'natural_person_ratio': round(natural_person_ratio, 4),
            'enterprise_ratio': round(enterprise_ratio, 4),
            'dominant_shareholder_type': dominant_type,
            'total_shareholders': total_nodes
        }
        
    except Exception as e:
        return {
            'natural_person_count': 0,
            'enterprise_count': 0,
            'unknown_enterprise_count': 0,
            'unknown_type_count': 0,
            'natural_person_ratio': 0.0,
            'enterprise_ratio': 0.0,
            'dominant_shareholder_type': 'Unknown',
            'total_shareholders': 0
        }

def calculate_network_centrality_metrics(graph, loop_node_ids):
    """计算网络中心性指标"""
    try:
        if len(loop_node_ids) < 2:
            return {
                'max_degree_centrality': 0.0,
                'max_betweenness_centrality': 0.0,
                'max_closeness_centrality': 0.0,
                'network_density': 0.0,
                'key_node_id': None,
                'avg_degree': 0.0
            }
        
        # 创建环路子图
        valid_nodes = [node_id for node_id in loop_node_ids if node_id in graph.nodes]
        if len(valid_nodes) < 2:
            return {
                'max_degree_centrality': 0.0,
                'max_betweenness_centrality': 0.0,
                'max_closeness_centrality': 0.0,
                'network_density': 0.0,
                'key_node_id': None,
                'avg_degree': 0.0
            }
        
        subgraph = graph.subgraph(valid_nodes)
        
        # 计算度中心性
        degree_centrality = nx.degree_centrality(subgraph)
        max_degree_centrality = max(degree_centrality.values()) if degree_centrality else 0.0
        
        # 计算介数中心性
        try:
            betweenness_centrality = nx.betweenness_centrality(subgraph)
            max_betweenness_centrality = max(betweenness_centrality.values()) if betweenness_centrality else 0.0
        except:
            max_betweenness_centrality = 0.0
        
        # 计算接近中心性
        try:
            closeness_centrality = nx.closeness_centrality(subgraph)
            max_closeness_centrality = max(closeness_centrality.values()) if closeness_centrality else 0.0
        except:
            max_closeness_centrality = 0.0
        
        # 计算网络密度
        network_density = nx.density(subgraph)
        
        # 找到关键节点（度中心性最高的节点）
        key_node_id = max(degree_centrality, key=degree_centrality.get) if degree_centrality else None
        
        # 计算平均度
        degrees = [subgraph.degree(node) for node in subgraph.nodes()]
        avg_degree = np.mean(degrees) if degrees else 0.0
        
        return {
            'max_degree_centrality': round(max_degree_centrality, 4),
            'max_betweenness_centrality': round(max_betweenness_centrality, 4),
            'max_closeness_centrality': round(max_closeness_centrality, 4),
            'network_density': round(network_density, 4),
            'key_node_id': key_node_id,
            'avg_degree': round(avg_degree, 2)
        }
        
    except Exception as e:
        return {
            'max_degree_centrality': 0.0,
            'max_betweenness_centrality': 0.0,
            'max_closeness_centrality': 0.0,
            'network_density': 0.0,
            'key_node_id': None,
            'avg_degree': 0.0
        }

def extract_loop_structure_metrics(loop_info):
    """提取环路结构指标"""
    try:
        type_str = loop_info.get('type', '')
        content = loop_info.get('content', '')
        
        # 提取节点数量
        node_count_match = re.search(r'(\d+)节点环路', type_str)
        node_count = int(node_count_match.group(1)) if node_count_match else 0
        
        # 提取类型分类
        type_category_match = re.search(r'类型(\d+)', type_str)
        type_category = int(type_category_match.group(1)) if type_category_match else 0
        
        # 计算路径长度（基于content中的箭头数量）
        arrow_count = content.count('-->') + content.count('<--')
        path_length = arrow_count
        
        # 计算复杂度评分
        complexity_score = node_count * path_length if node_count > 0 and path_length > 0 else 0
        
        return {
            'loop_node_count': node_count,
            'loop_type_category': type_category,
            'loop_path_length': path_length,
            'loop_complexity_score': complexity_score
        }
        
    except Exception as e:
        return {
            'loop_node_count': 0,
            'loop_type_category': 0,
            'loop_path_length': 0,
            'loop_complexity_score': 0
        }

def process_single_loop(loop_id, loop_info):
    """处理单个环路，提取所有股权指标（工作进程版本）"""
    global global_graph
    
    try:
        if global_graph is None:
            return {'loop_id': loop_id, 'error': 'Graph not loaded in worker'}
        
        # 提取环路节点名称
        content = loop_info.get('content', '')
        loop_node_names = extract_loop_nodes_from_content(content)
        
        # 根据名称查找图中的节点ID
        loop_node_ids = find_node_ids_by_names(global_graph, loop_node_names)
        
        # 1. 环路结构指标
        structure_metrics = extract_loop_structure_metrics(loop_info)
        
        # 2. 股权集中度指标
        ownership_metrics = calculate_ownership_concentration_metrics(global_graph, loop_node_ids)
        
        # 3. 股东类型分布指标
        shareholder_metrics = calculate_shareholder_type_metrics(global_graph, loop_node_ids)
        
        # 4. 网络中心性指标
        centrality_metrics = calculate_network_centrality_metrics(global_graph, loop_node_ids)
        
        # 合并所有指标
        all_metrics = {
            'loop_id': loop_id,
            **structure_metrics,
            **ownership_metrics,
            **shareholder_metrics,
            **centrality_metrics
        }
        
        return all_metrics
        
    except Exception as e:
        return {'loop_id': loop_id, 'error': str(e)}

def process_loop_batch(loop_batch):
    """批量处理环路数据"""
    results = []
    for loop_id, loop_info in loop_batch:
        result = process_single_loop(loop_id, loop_info)
        results.append(result)
    return results

def create_batches(data, n_cores):
    """根据核心数和数据量动态分配批次"""
    total_loops = len(data)
    
    # 计算最优批次大小：总数据量 / 核心数，确保每个核心有足够的工作
    # 每个核心处理3-4批，以确保负载均衡
    base_batch_size = max(50, total_loops // (n_cores * 3))
    
    # 考虑内存限制，设置最大批次大小
    max_batch_size = min(8000, total_loops // n_cores if n_cores > 0 else total_loops)
    
    # 最终批次大小
    batch_size = min(base_batch_size, max_batch_size)
    
    print(f"数据分配策略: 总数据 {total_loops:,} 个环路，{n_cores} 个进程，批次大小 {batch_size}")
    
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i + batch_size])
    
    return batches

def extract_equity_metrics_parallel(basic_info_file, graph_file, output_dir, n_cores=None):
    """并行提取股权指标主函数"""
    logger = setup_logging()
    
    try:
        logger.info("=== 开始并行提取股权指标 ===")
        
        # 确定使用的核心数
        if n_cores is None:
            n_cores = max(1, mp.cpu_count() - 1)  # 留一个核心给系统
        logger.info(f"使用 {n_cores} 个进程进行并行处理")
        
        # 加载数据
        logger.info(f"读取基础环路信息: {basic_info_file}")
        basic_info_df = pd.read_csv(basic_info_file)
        logger.info(f"读取到 {len(basic_info_df)} 个环路的基础信息")
        
        # 加载图模型
        graph, graph_data = load_graph_model(graph_file)
        if graph is None or graph_data is None:
            raise Exception("图模型加载失败")
        
        # 准备数据
        logger.info("准备环路数据...")
        loop_data = []
        for idx, row in basic_info_df.iterrows():
            loop_id = row['loop_id']
            loop_info = row.to_dict()
            loop_data.append((loop_id, loop_info))
        
        # 动态创建批次
        logger.info("创建处理批次...")
        batches = create_batches(loop_data, n_cores)
        logger.info(f"共创建 {len(batches)} 个批次")
        
        # 启动多进程处理
        logger.info("启动多进程处理...")
        start_time = time.time()
        
        equity_metrics_list = []
        
        with Pool(n_cores, initializer=init_worker, initargs=(graph_data,)) as pool:
            # 使用tqdm显示进度条
            with tqdm(total=len(loop_data), desc="处理环路", unit="环路") as pbar:
                # 提交所有批次任务
                batch_results = []
                for batch in batches:
                    result = pool.apply_async(process_loop_batch, (batch,))
                    batch_results.append((result, len(batch)))  # 存储结果和批次大小
                
                # 收集结果
                for result, batch_size in batch_results:
                    batch_metrics = result.get()
                    equity_metrics_list.extend(batch_metrics)
                    pbar.update(batch_size)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"并行处理完成，耗时: {processing_time:.2f} 秒")
        logger.info(f"平均每个环路处理时间: {processing_time/len(loop_data)*1000:.2f} 毫秒")
        
        # 创建股权指标DataFrame
        equity_metrics_df = pd.DataFrame(equity_metrics_list)
        
        # 检查错误
        error_count = equity_metrics_df['error'].notna().sum() if 'error' in equity_metrics_df.columns else 0
        if error_count > 0:
            logger.warning(f"有 {error_count} 个环路处理失败")
            # 移除错误列用于后续处理
            if 'error' in equity_metrics_df.columns:
                equity_metrics_df = equity_metrics_df.drop('error', axis=1)
        
        logger.info("股权指标提取完成")
        
        # 强制垃圾回收
        del graph_data
        gc.collect()
        
        return equity_metrics_df
        
    except Exception as e:
        logger.error(f"并行提取股权指标失败: {e}")
        raise

def merge_with_transaction_metrics(equity_metrics_df, transaction_metrics_file, output_dir):
    """将股权指标与交易指标合并，只输出最终结果"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== 合并交易指标和股权指标 ===")
        
        # 读取交易指标
        logger.info(f"读取交易指标: {transaction_metrics_file}")
        transaction_df = pd.read_csv(transaction_metrics_file)
        logger.info(f"读取到 {len(transaction_df)} 个环路的交易指标")
        
        # 合并数据
        logger.info("合并指标数据...")
        merged_df = pd.merge(transaction_df, equity_metrics_df, on='loop_id', how='left')
        logger.info(f"合并后数据量: {len(merged_df)} 行")
        
        # 保存综合画像（唯一的CSV输出）
        comprehensive_output_file = os.path.join(output_dir, 'loop_comprehensive_metrics.csv')
        merged_df.to_csv(comprehensive_output_file, index=False, encoding='utf-8-sig')
        logger.info(f"综合画像已保存到: {comprehensive_output_file}")
        
        # 生成综合报告（唯一的报告输出）
        generate_comprehensive_report(merged_df, output_dir)
        
        return merged_df
        
    except Exception as e:
        logger.error(f"合并指标失败: {e}")
        raise

def generate_comprehensive_report(merged_df, output_dir):
    """生成综合画像报告"""
    logger = logging.getLogger(__name__)
    
    try:
        report_file = os.path.join(output_dir, 'comprehensive_metrics_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 环路综合画像报告（并行优化版本）===\n\n")
            f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总环路数量: {len(merged_df)}\n")
            f.write(f"总指标维度: {len(merged_df.columns)}\n\n")
            
            # 指标分类统计
            transaction_cols = [col for col in merged_df.columns if any(keyword in col.lower() for keyword in ['transaction', 'amount', 'count', 'avg', 'times'])]
            equity_cols = [col for col in merged_df.columns if any(keyword in col.lower() for keyword in ['ownership', 'shareholder', 'centrality', 'density', 'node', 'type'])]
            
            f.write("=== 指标维度分布 ===\n")
            f.write(f"交易相关指标: {len(transaction_cols)}个\n")
            f.write(f"股权相关指标: {len(equity_cols)}个\n")
            f.write(f"其他指标: {len(merged_df.columns) - len(transaction_cols) - len(equity_cols)}个\n\n")
            
            # 数据完整性分析
            f.write("=== 数据完整性分析 ===\n")
            missing_stats = merged_df.isnull().sum()
            complete_rate = (1 - missing_stats / len(merged_df)) * 100
            
            # 只显示缺失率较高的指标
            high_missing = missing_stats[missing_stats > len(merged_df) * 0.1]  # 缺失率>10%
            if len(high_missing) > 0:
                f.write("缺失率较高的指标 (>10%):\n")
                for col in high_missing.index:
                    f.write(f"  {col}: {complete_rate[col]:.1f}% 完整\n")
            else:
                f.write("所有指标完整率均>90%\n")
            
            # 关键统计信息
            f.write(f"\n=== 关键统计信息 ===\n")
            if 'total_transaction_amount' in merged_df.columns:
                f.write(f"平均交易金额: {merged_df['total_transaction_amount'].mean():,.2f}\n")
            if 'loop_node_count' in merged_df.columns:
                f.write(f"平均节点数: {merged_df['loop_node_count'].mean():.2f}\n")
            if 'max_ownership_percent' in merged_df.columns:
                valid_ownership = merged_df['max_ownership_percent'].dropna()
                if len(valid_ownership) > 0:
                    f.write(f"平均控股比例: {valid_ownership.mean():.4f}\n")
            if 'network_density' in merged_df.columns:
                valid_density = merged_df['network_density'].dropna()
                if len(valid_density) > 0:
                    f.write(f"平均网络密度: {valid_density.mean():.4f}\n")
            
            # 性能信息
            f.write(f"\n=== 性能信息 ===\n")
            f.write(f"处理方式: 多核并行处理\n")
            f.write(f"使用进程数: {mp.cpu_count() - 1} (系统核心数 - 1)\n")
            f.write(f"批处理大小: 1000 环路/批\n")
            
            # 股东类型分布
            if 'dominant_shareholder_type' in merged_df.columns:
                f.write(f"\n=== 股东类型分布 ===\n")
                type_dist = merged_df['dominant_shareholder_type'].value_counts()
                for stype, count in type_dist.items():
                    type_name = {
                        'Natural_Person': '自然人主导',
                        'Enterprise': '企业主导',
                        'Mixed': '混合类型',
                        'Unknown': '未知类型'
                    }.get(stype, stype)
                    f.write(f"{type_name}: {count:,} ({count/len(merged_df)*100:.1f}%)\n")
            
            # 高质量数据统计
            f.write(f"\n=== 高质量数据统计 ===\n")
            high_quality_mask = (
                merged_df['total_transaction_amount'].notna() &
                merged_df['loop_node_count'].notna() &
                merged_df['max_ownership_percent'].notna()
            )
            high_quality_count = high_quality_mask.sum()
            f.write(f"高质量环路数量: {high_quality_count:,}/{len(merged_df):,} ({high_quality_count/len(merged_df)*100:.1f}%)\n")
            f.write("(定义: 同时具有交易金额、节点数量、股权比例信息的环路)\n\n")
            
            # 推荐筛选策略
            f.write("=== 推荐筛选策略 ===\n")
            f.write("基于综合画像的多维度筛选建议:\n\n")
            
            f.write("1. 高风险环路筛选:\n")
            f.write("   - 节点数 ≤ 4\n")
            f.write("   - 交易金额 > 80%分位数\n")
            f.write("   - 控股比例 > 0.8\n")
            f.write("   - 网络密度 > 0.5\n\n")
            
            f.write("2. 复杂结构环路筛选:\n")
            f.write("   - 复杂度评分 > 80%分位数\n")
            f.write("   - 网络密度 > 70%分位数\n")
            f.write("   - 度中心性 > 80%分位数\n\n")
            
            f.write("3. 集中控制环路筛选:\n")
            f.write("   - 最大控股比例 > 0.8\n")
            f.write("   - 股权集中度指数 > 0.5\n")
            f.write("   - 主导股东类型筛选\n\n")
            
            f.write("=== 输出文件说明 ===\n")
            f.write("• loop_comprehensive_metrics.csv: 综合画像数据文件\n")
            f.write("• comprehensive_metrics_report.txt: 本分析报告\n")
            f.write("• 可配合简化闭环筛选脚本进行多维度筛选\n")
            f.write("\n=== 并行优化说明 ===\n")
            f.write("• 使用多进程并行处理，显著提升处理速度\n")
            f.write("• 图模型在进程间共享，减少内存占用\n")
            f.write("• 批量处理减少进程间通信开销\n")
            f.write("• 实时进度显示，便于监控处理状态\n")
        
        logger.info(f"综合画像报告已保存到: {report_file}")
        
    except Exception as e:
        logger.warning(f"生成综合画像报告失败: {e}")

def main():
    """主函数"""
    logger = setup_logging()
    
    try:
        # 文件路径配置
        basic_info_file = "outputs/loop_analysis/loop_basic_info.csv"
        transaction_metrics_file = "outputs/loop_analysis/loop_metrics.csv"
        graph_file = "model/final_heterogeneous_graph.graphml"
        output_dir = "outputs/扩展画像"
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("=== 环路画像扩展开始（并行优化版本）===")
        
        # 获取系统信息
        cpu_count = mp.cpu_count()
        logger.info(f"系统CPU核心数: {cpu_count}")
        logger.info(f"将使用 {cpu_count - 1} 个进程进行并行处理")
        
        # 第一步：并行提取股权指标
        logger.info("第一步：并行提取股权指标")
        start_time = time.time()
        equity_metrics_df = extract_equity_metrics_parallel(
            basic_info_file, 
            graph_file, 
            output_dir,
            n_cores=cpu_count - 1  # 使用系统核心数-1，批次大小自动计算
        )
        equity_extraction_time = time.time() - start_time
        
        # 第二步：合并交易指标和股权指标，直接输出最终结果
        logger.info("第二步：合并指标形成综合画像")
        merge_start_time = time.time()
        comprehensive_df = merge_with_transaction_metrics(equity_metrics_df, transaction_metrics_file, output_dir)
        merge_time = time.time() - merge_start_time
        
        total_time = time.time() - start_time
        
        logger.info("=== 环路画像扩展完成（并行优化版本）===")
        logger.info(f"原始交易指标: {pd.read_csv(transaction_metrics_file).shape[1]}个")
        logger.info(f"新增股权指标: {equity_metrics_df.shape[1]-1}个")  # 减去loop_id
        logger.info(f"综合画像指标: {comprehensive_df.shape[1]}个")
        logger.info(f"环路数量: {len(comprehensive_df):,}")
        logger.info(f"")
        logger.info(f"=== 性能统计 ===")
        logger.info(f"股权指标提取时间: {equity_extraction_time:.2f} 秒")
        logger.info(f"指标合并时间: {merge_time:.2f} 秒")
        logger.info(f"总处理时间: {total_time:.2f} 秒")
        logger.info(f"平均每环路处理时间: {total_time/len(comprehensive_df)*1000:.2f} 毫秒")
        logger.info(f"处理速度: {len(comprehensive_df)/total_time:.2f} 环路/秒")
        logger.info(f"")
        logger.info(f"输出文件:")
        logger.info(f"  - {output_dir}/loop_comprehensive_metrics.csv")
        logger.info(f"  - {output_dir}/comprehensive_metrics_report.txt")
        
        return comprehensive_df
        
    except Exception as e:
        logger.error(f"环路画像扩展失败: {e}")
        raise

if __name__ == "__main__":
    main() 