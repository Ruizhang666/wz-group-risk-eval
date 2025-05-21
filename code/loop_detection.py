#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股权闭环检测工具 - 检测一级、二级、三级股权闭环
"""

import os
import networkx as nx
import logging
from datetime import datetime
import time
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Sequence

# 配置
INPUT_GRAPH_PATH = "model/final_heterogeneous_graph.graphml"
SIMPLIFIED_GRAPH_PATH = "model/simplified_loop_detection_graph.graphml"  # 简化图路径
OUTPUT_DIR = "outputs"
LOG_DIR = os.path.join(OUTPUT_DIR, "log")
LOG_FILE = os.path.join(LOG_DIR, "loop_detection.log")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "loop_results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "equity_loops.txt")

# 设置日志
def setup_logging():
    """设置日志记录"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, 'w', 'utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    logging.info(f"开始分析，日志记录于 {LOG_FILE}")

# 确保输出目录存在
def ensure_output_dirs():
    """确保输出目录存在"""
    for directory in [OUTPUT_DIR, RESULTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"创建目录: {directory}")

def load_graph(file_path):
    """从GraphML文件加载图。"""
    if not os.path.exists(file_path):
        logging.error(f"图文件未找到于 '{file_path}'")
        return None
    try:
        logging.info(f"正在从 '{file_path}' 加载图...")
        graph = nx.read_graphml(file_path)
        logging.info(f"图加载成功，包含 {graph.number_of_nodes()} 个节点和 {graph.number_of_edges()} 条边。")
        return graph
    except Exception as e:
        logging.error(f"加载图 '{file_path}' 时发生错误: {e}")
        return None

def get_node_info(graph, node_id):
    """获取节点的详细信息"""
    node_attrs = graph.nodes[node_id]
    name = node_attrs.get('name', node_id)
    label = node_attrs.get('label', '未知')
    return f"{name} [{label}]"

def find_role_based_paths_recursive(graph, current_path, role_sequence, target_len, direction_sequence=None):
    """
    递归查找符合角色序列的路径
    
    参数:
    - graph: NetworkX图
    - current_path: 当前路径
    - role_sequence: 角色序列，例如 ['股东', 'partner', '成员单位', ...]
    - target_len: 目标路径长度(包含重复的起始节点)
    - direction_sequence: 可选，方向序列(True表示向前，False表示向后)，例如[True, True, False, ...]
    
    返回:
    - 找到的所有符合条件的路径列表
    """
    found_paths = []
    current_node = current_path[-1]
    current_pos = len(current_path) - 1
    
    # 检查是否达到目标长度
    if len(current_path) == target_len:
        # 确保最后一个节点是起始节点
        if current_node == current_path[0]:
            found_paths.append(current_path.copy())
        return found_paths
    
    # 检查下一步应该是哪个角色
    next_role = role_sequence[current_pos + 1]
    
    # 确定搜索方向
    forward = True  # 默认向前搜索
    if direction_sequence and len(direction_sequence) > current_pos:
        forward = direction_sequence[current_pos]  # 使用指定方向
    
    # 基于方向获取邻居节点
    neighbors = []
    if forward:
        # 向前搜索：检查所有出边
        neighbors = list(graph.successors(current_node))
    else:
        # 向后搜索：检查所有入边
        neighbors = list(graph.predecessors(current_node))
    
    # 检查每个邻居
    for neighbor in neighbors:
        # 如果已在路径中，跳过(除非是最后一步回到起点)
        if neighbor in current_path:
            # 特殊情况：如果是最后一步且邻居是起点，允许添加
            if len(current_path) == target_len - 1 and neighbor == current_path[0]:
                new_path = current_path + [neighbor]
                found_paths.append(new_path)
            continue
        
        # 检查邻居角色是否匹配
        neighbor_role = graph.nodes[neighbor].get('label', '未知')
        if neighbor_role == next_role:
            # 递归搜索
            new_paths = find_role_based_paths_recursive(
                graph, 
                current_path + [neighbor], 
                role_sequence, 
                target_len,
                direction_sequence
            )
            found_paths.extend(new_paths)
    
    return found_paths

def search_loops_with_role_sequence(graph, role_sequence, direction_sequence=None):
    """
    查找符合特定角色序列的环路
    
    参数:
    - graph: NetworkX图
    - role_sequence: 角色序列
    - direction_sequence: 方向序列
    
    返回:
    - 找到的环路列表
    """
    all_found_loops = []
    target_loop_len = len(role_sequence)  # 环路中的节点数
    start_role = role_sequence[0]
    
    if target_loop_len < 2 or role_sequence[0] != role_sequence[-1]:
        logging.warning("警告：角色序列长度小于2或首尾角色不一致，无法形成有效环路搜索。")
        return all_found_loops
    
    # 找出所有弱连通分量（忽略边的方向）
    wccs = list(nx.weakly_connected_components(graph))
    logging.info(f"图中共有 {len(wccs)} 个弱连通分量")
    
    # 过滤出大小大于等于环路长度的弱连通分量
    valid_wccs = [wcc for wcc in wccs if len(wcc) >= target_loop_len]
    logging.info(f"其中大小大于等于 {target_loop_len} 的弱连通分量有 {len(valid_wccs)} 个")
    
    processed_loops_sig = set()  # 用于去重
    
    for i, wcc in enumerate(valid_wccs):
        if (i+1) % 10 == 0 or i+1 == len(valid_wccs):
            logging.info(f"正在处理弱连通分量 {i+1}/{len(valid_wccs)} (大小: {len(wcc)})")
        
        # 创建WCC的子图进行搜索
        subG = graph.subgraph(wcc).copy()
        
        # 从符合起始角色的节点开始搜索
        start_nodes = [node for node in subG.nodes() if graph.nodes[node].get('label') == start_role]
        
        for start_node in start_nodes:
            # 调用递归函数查找路径
            found_paths = find_role_based_paths_recursive(
                subG,
                [start_node],
                role_sequence,
                target_loop_len,
                direction_sequence
            )
            
            for path in found_paths:
                # 验证路径是否满足方向要求
                if validate_path_directions(subG, path, direction_sequence):
                    # 对找到的环路进行签名去重（使用节点名称和标签组合作为标识）
                    path_info = tuple(f"{graph.nodes[node].get('name', node)}_{graph.nodes[node].get('label', '未知')}" 
                                     for node in path[:-1])  # 去掉最后一个重复的节点
                    loop_signature = tuple(sorted(path_info))
                    
                    if loop_signature not in processed_loops_sig:
                        all_found_loops.append(path)
                        processed_loops_sig.add(loop_signature)
    
    logging.info(f"搜索完成，找到 {len(all_found_loops)} 个符合角色序列的环路")
    return all_found_loops

def validate_path_directions(graph, path, direction_sequence):
    """验证路径是否满足方向要求"""
    if not direction_sequence:
        return True  # 如果没有方向要求，则默认有效
    
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        expected_direction = direction_sequence[i] if i < len(direction_sequence) else True
        
        if expected_direction:
            # 期望方向是向前(->)，检查是否存在从current_node到next_node的边
            if not graph.has_edge(current_node, next_node):
                return False
        else:
            # 期望方向是向后(<-)，检查是否存在从next_node到current_node的边
            if not graph.has_edge(next_node, current_node):
                return False
    
    return True

def format_loop_path(graph, loop):
    """将闭环路径格式化为可读的字符串"""
    path_str = ""
    for i in range(len(loop)):
        node = loop[i]
        node_info = get_node_info(graph, node)
        
        # 将 partner 改为 合作公司
        if "[partner]" in node_info:
            node_info = node_info.replace("[partner]", "[合作公司]")
        
        path_str += node_info
        
        # 如果不是最后一个节点，添加箭头
        if i < len(loop) - 1:
            # 获取边的信息
            edge_data = graph.get_edge_data(node, loop[i+1])
            
            # 可能有多个边
            if edge_data:
                # 如果是MultiDiGraph，可能有多个键
                try:
                    if isinstance(edge_data, dict) and not isinstance(edge_data, list):
                        if 0 in edge_data:  # 默认键
                            edge_attrs = edge_data[0]
                            if isinstance(edge_attrs, dict):
                                edge_label = edge_attrs.get('label', '关联')
                            else:
                                edge_label = '关联'
                        else:
                            # 取第一个边的标签
                            first_key = list(edge_data.keys())[0]
                            edge_attrs = edge_data[first_key]
                            if isinstance(edge_attrs, dict):
                                edge_label = edge_attrs.get('label', '关联')
                            else:
                                edge_label = '关联'
                    else:
                        if isinstance(edge_data, dict):
                            edge_label = edge_data.get('label', '关联')
                        else:
                            edge_label = '关联'
                    
                    path_str += f" --({edge_label})--> "
                except Exception as e:
                    logging.debug(f"处理边属性时出错: {e}，使用默认标签")
                    path_str += " --> "
            else:
                # 检查是否有反向边（对应<-情况）
                reverse_edge = graph.get_edge_data(loop[i+1], node)
                if reverse_edge:
                    # 获取反向边的标签
                    try:
                        if isinstance(reverse_edge, dict) and not isinstance(reverse_edge, list):
                            if 0 in reverse_edge:  # 默认键
                                edge_attrs = reverse_edge[0]
                                if isinstance(edge_attrs, dict):
                                    edge_label = edge_attrs.get('label', '关联')
                                else:
                                    edge_label = '关联'
                            else:
                                # 取第一个边的标签
                                first_key = list(reverse_edge.keys())[0]
                                edge_attrs = reverse_edge[first_key]
                                if isinstance(edge_attrs, dict):
                                    edge_label = edge_attrs.get('label', '关联')
                                else:
                                    edge_label = '关联'
                        else:
                            if isinstance(reverse_edge, dict):
                                edge_label = reverse_edge.get('label', '关联')
                            else:
                                edge_label = '关联'
                        
                        path_str += f" <--({edge_label})-- "
                    except Exception as e:
                        logging.debug(f"处理反向边属性时出错: {e}，使用默认标签")
                        path_str += " <-- "
                else:
                    path_str += " --> "  # 如果没有边信息，使用默认箭头
    
    return path_str

def analyze_and_save_loops(graph):
    """分析图中的闭环并保存结果"""
    start_time = time.time()
    
    # 定义各级别闭环的角色序列和方向
    # 格式为 [角色序列, 方向序列]
    # 方向序列中，True表示向前(->)，False表示向后(<-)
    
    # 一级闭环：股东->合作公司A->成员公司->合作公司B<-股东
    level1_config = [
        ['股东', 'partner', '成员单位', 'partner', '股东'],  # 角色序列
        [True, True, True, False]  # 方向序列
    ]
    
    # 二级闭环：股东A->股东C->合作公司A->成员公司<-合作公司B<-股东S<-股东A
    level2_config = [
        ['股东', '股东', 'partner', '成员单位', 'partner', '股东', '股东'],  # 角色序列
        [True, True, True, False, False, False]  # 方向序列
    ]
    
    # 三级闭环：股东h->股东e->股东C->合作公司A->成员公司->合作公司B<-股东S<-股东g<-股东h
    level3_config = [
        ['股东', '股东', '股东', 'partner', '成员单位', 'partner', '股东', '股东', '股东'],  # 角色序列
        [True, True, True, True, True, False, False, False]  # 方向序列
    ]
    
    # 也处理英文标签
    # 一级闭环：股东->合作公司A->成员公司->合作公司B<-股东
    level1_config_en = [
        ['shareholder', 'partner', '成员单位', 'partner', 'shareholder'],  # 角色序列
        [True, True, True, False]  # 方向序列
    ]
    
    # 二级闭环：股东A->股东C->合作公司A->成员公司<-合作公司B<-股东S<-股东A
    level2_config_en = [
        ['shareholder', 'shareholder', 'partner', '成员单位', 'partner', 'shareholder', 'shareholder'],  # 角色序列
        [True, True, True, False, False, False]  # 方向序列
    ]
    
    # 三级闭环：股东h->股东e->股东C->合作公司A->成员公司->合作公司B<-股东S<-股东g<-股东h
    level3_config_en = [
        ['shareholder', 'shareholder', 'shareholder', 'partner', '成员单位', 'partner', 'shareholder', 'shareholder', 'shareholder'],  # 角色序列
        [True, True, True, True, True, False, False, False]  # 方向序列
    ]
    
    # 查找各级别闭环
    logging.info("开始查找一级闭环...")
    level1_loops_cn = search_loops_with_role_sequence(graph, level1_config[0], level1_config[1])
    level1_loops_en = search_loops_with_role_sequence(graph, level1_config_en[0], level1_config_en[1])
    level1_loops = level1_loops_cn + level1_loops_en
    
    logging.info("开始查找二级闭环...")
    level2_loops_cn = search_loops_with_role_sequence(graph, level2_config[0], level2_config[1])
    level2_loops_en = search_loops_with_role_sequence(graph, level2_config_en[0], level2_config_en[1])
    level2_loops = level2_loops_cn + level2_loops_en
    
    logging.info("开始查找三级闭环...")
    level3_loops_cn = search_loops_with_role_sequence(graph, level3_config[0], level3_config[1])
    level3_loops_en = search_loops_with_role_sequence(graph, level3_config_en[0], level3_config_en[1])
    level3_loops = level3_loops_cn + level3_loops_en
    
    # 按公司分组
    company_loops = defaultdict(lambda: {'level1': [], 'level2': [], 'level3': []})
    
    # 处理一级闭环
    for loop in level1_loops:
        start_node = loop[0]
        start_name = graph.nodes[start_node].get('name', start_node)
        company_loops[start_name]['level1'].append(loop)
    
    # 处理二级闭环
    for loop in level2_loops:
        start_node = loop[0]
        start_name = graph.nodes[start_node].get('name', start_node)
        company_loops[start_name]['level2'].append(loop)
    
    # 处理三级闭环
    for loop in level3_loops:
        start_node = loop[0]
        start_name = graph.nodes[start_node].get('name', start_node)
        company_loops[start_name]['level3'].append(loop)
    
    # 保存结果到文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# 股权闭环检测报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## 总结\n\n")
        f.write(f"- 发现一级闭环: {len(level1_loops)} 个\n")
        f.write(f"- 发现二级闭环: {len(level2_loops)} 个\n")
        f.write(f"- 发现三级闭环: {len(level3_loops)} 个\n")
        f.write(f"- 共涉及 {len(company_loops)} 个股东\n\n")
        
        f.write("## 详细闭环信息\n\n")
        
        # 按公司名称排序
        for company_name in sorted(company_loops.keys()):
            loops_data = company_loops[company_name]
            total_loops = len(loops_data['level1']) + len(loops_data['level2']) + len(loops_data['level3'])
            
            f.write(f"### 股东: {company_name}\n\n")
            f.write(f"该股东共涉及 {total_loops} 个闭环:\n")
            f.write(f"- 一级闭环: {len(loops_data['level1'])} 个\n")
            f.write(f"- 二级闭环: {len(loops_data['level2'])} 个\n")
            f.write(f"- 三级闭环: {len(loops_data['level3'])} 个\n\n")
            
            # 一级闭环详情
            if loops_data['level1']:
                f.write("#### 一级闭环\n\n")
                for i, loop in enumerate(loops_data['level1']):
                    f.write(f"{i+1}. {format_loop_path(graph, loop)}\n\n")
            
            # 二级闭环详情
            if loops_data['level2']:
                f.write("#### 二级闭环\n\n")
                for i, loop in enumerate(loops_data['level2']):
                    f.write(f"{i+1}. {format_loop_path(graph, loop)}\n\n")
            
            # 三级闭环详情
            if loops_data['level3']:
                f.write("#### 三级闭环\n\n")
                for i, loop in enumerate(loops_data['level3']):
                    f.write(f"{i+1}. {format_loop_path(graph, loop)}\n\n")
            
            f.write("-" * 80 + "\n\n")
    
    logging.info(f"闭环分析完成，用时: {time.time() - start_time:.2f} 秒")
    logging.info(f"发现一级闭环: {len(level1_loops)} 个")
    logging.info(f"发现二级闭环: {len(level2_loops)} 个")
    logging.info(f"发现三级闭环: {len(level3_loops)} 个")
    logging.info(f"报告已保存到: {OUTPUT_FILE}")
    
    return True

def main():
    """主函数"""
    # 设置日志
    setup_logging()
    
    # 确保输出目录存在
    ensure_output_dirs()
    
    # 先尝试加载简化图，如果不存在则使用原图
    if os.path.exists(SIMPLIFIED_GRAPH_PATH):
        logging.info(f"发现简化图 {SIMPLIFIED_GRAPH_PATH}，使用简化图进行环路检测...")
        graph = load_graph(SIMPLIFIED_GRAPH_PATH)
        if graph:
            logging.info("成功加载简化图！")
            # 保存原始图的路径，在输出报告中可能需要查询详细信息
            original_graph_path = INPUT_GRAPH_PATH
        else:
            logging.warning(f"简化图加载失败，尝试加载原始图 {INPUT_GRAPH_PATH}...")
            graph = load_graph(INPUT_GRAPH_PATH)
            original_graph_path = None
    else:
        logging.info(f"未发现简化图，使用原始图 {INPUT_GRAPH_PATH} 进行环路检测...")
        graph = load_graph(INPUT_GRAPH_PATH)
        original_graph_path = None
    
    if graph is None:
        logging.error("图加载失败，退出程序。")
        return
    
    # 记录图的总体信息
    logging.info(f"图包含 {graph.number_of_nodes()} 个节点和 {graph.number_of_edges()} 条边")
    
    # 分析闭环
    logging.info("开始分析股权闭环...")
    analyze_and_save_loops(graph)
    
    logging.info("股权闭环分析完成。")
    if original_graph_path:
        logging.info(f"注意：本次分析使用了简化图，详细的交易信息请参考原始图 {original_graph_path}")

if __name__ == "__main__":
    main() 