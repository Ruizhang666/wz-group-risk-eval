#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股权闭环检测工具 - igraph 兼容性优化版 (性能优化版) - 并行配置处理与算法优化
"""

import os
import logging
import time
from datetime import datetime
from collections import defaultdict, deque
import igraph as ig
import multiprocessing
import numpy as np  # 添加用于向量化操作

# --- (Configuration constants like INPUT_GRAPH_PATH, OUTPUT_DIR, etc. remain the same) ---
INPUT_GRAPH_PATH = "model/final_heterogeneous_graph.graphml"
SIMPLIFIED_GRAPH_PATH = "model/simplified_loop_detection_graph.graphml"
OUTPUT_DIR = "outputs"
LOG_DIR = os.path.join(OUTPUT_DIR, "log")
LOG_FILE = os.path.join(LOG_DIR, "loop_detection_optimized.log")  # 修改日志文件名
RESULTS_DIR = os.path.join(OUTPUT_DIR, "loop_results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "equity_loops_optimized.txt")  # 修改输出文件名
# 添加一个缓存大小限制，用于控制内存使用
MAX_CACHE_SIZE = 10000

# ===== 用户可配置参数 =====
# 设置要分析的节点数，如果为None则分析所有节点数的环路
# 可选值: 4, 5, 6, 7, 8 或 None (分析所有环路)
NODE_COUNT = None  # 在这里修改要分析的节点数，例如: NODE_COUNT = 6
# ========================

def setup_logging():
    """设置日志记录"""
    for directory in [LOG_DIR, RESULTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, 'w', 'utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"开始分析，日志记录于 {LOG_FILE}")

# --- (load_graph, get_role, get_name functions remain the same) ---
def load_graph(file_path):
    """从GraphML文件加载igraph图"""
    if not os.path.exists(file_path):
        logging.error(f"图文件未找到于 '{file_path}'")
        return None
    try:
        logging.info(f"正在从 '{file_path}' 加载图...")
        graph = ig.Graph.Read_GraphML(file_path)
        logging.info(f"图加载成功，包含 {graph.vcount()} 个节点和 {graph.ecount()} 条边。")
        # Ensure 'name' and 'label' attributes exist or provide defaults gracefully
        if "name" not in graph.vertex_attributes():
            logging.warning("Vertex attribute 'name' not found. Using IDs as names.")
            graph.vs["name"] = [str(v.index) for v in graph.vs]
        if "label" not in graph.vertex_attributes():
            logging.warning("Vertex attribute 'label' (for role) not found. Using '未知' as role.")
            graph.vs["label"] = ["未知"] * graph.vcount()
        if "label" not in graph.edge_attributes():
            logging.warning("Edge attribute 'label' not found. Using '关联' as edge label.")
            graph.es["label"] = ["关联"] * graph.ecount()
            
        # 预处理步骤：创建角色索引和邻居缓存
        graph.vs["role_index"] = [-1] * graph.vcount()
        role_to_index = {}
        curr_index = 0
        for v in graph.vs:
            role = v["label"]
            if role not in role_to_index:
                role_to_index[role] = curr_index
                curr_index += 1
            v["role_index"] = role_to_index[role]
        
        # 创建角色索引映射
        graph["role_to_vertices"] = defaultdict(list)
        for v in graph.vs:
            graph["role_to_vertices"][v["label"]].append(v.index)
            
        # 预计算出边和入边索引，加速邻居查找
        graph["out_neighbors"] = [graph.neighbors(v.index, mode="out") for v in graph.vs]
        graph["in_neighbors"] = [graph.neighbors(v.index, mode="in") for v in graph.vs]

        return graph
    except Exception as e:
        logging.error(f"加载图 '{file_path}' 时发生错误: {e}")
        return None

def get_role(graph, v_id):
    """获取节点角色"""
    return graph.vs[v_id]["label"]

def get_name(graph, v_id):
    """获取节点名称"""
    return graph.vs[v_id]["name"]

# 添加缓存装饰器用于记忆化搜索
def lru_cache_with_limit(maxsize=MAX_CACHE_SIZE):
    """简单的LRU缓存实现，限制缓存大小"""
    def decorator(func):
        cache = {}
        queue = deque()
        
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key in cache:
                return cache[key]
            
            result = func(*args, **kwargs)
            
            # 如果缓存已满，移除最早的项
            if len(cache) >= maxsize:
                oldest = queue.popleft()
                if oldest in cache:
                    del cache[oldest]
            
            cache[key] = result
            queue.append(key)
            return result
        
        return wrapper
    return decorator

# 优化后的路径查找函数
def find_paths_with_role_sequence(graph, role_sequence, direction_sequence=None):
    """
    找出所有符合角色序列和方向序列的路径
    采用迭代BFS和预计算优化，提高性能
    """
    if not graph:
        logging.error("Graph object is None in find_paths_with_role_sequence.")
        return []
    if len(role_sequence) < 2 or role_sequence[0] != role_sequence[-1]:
        logging.warning("角色序列必须以相同角色开始和结束才能形成闭环")
        return []
    
    logging.info(f"查找环路: 角色序列=[{','.join(role_sequence)}], 目标长度={len(role_sequence)-1}")
    
    start_role = role_sequence[0]
    # 使用预计算的角色到节点映射
    start_vertices = graph["role_to_vertices"].get(start_role, [])
    logging.info(f"找到 {len(start_vertices)} 个起始角色为 '{start_role}' 的节点")
    
    if not start_vertices:
        return []
        
    if not direction_sequence:
        direction_sequence = [True] * (len(role_sequence) - 1)
    
    if len(direction_sequence) != len(role_sequence) - 1:
        logging.error(f"方向序列长度 ({len(direction_sequence)}) 与环路路径长度 ({len(role_sequence)-1}) 不匹配")
        return []
    
    all_paths = []
    path_count = 0
    start_time = time.time()
    
    # 预先过滤可能的节点，减少搜索空间
    filtered_nodes_by_role = {}
    for i, role in enumerate(role_sequence):
        filtered_nodes_by_role[i] = set(graph["role_to_vertices"].get(role, []))
    
    # 批量处理起始节点，使用更高效的数据结构
    batch_size = min(100, len(start_vertices))  # 批量处理大小
    
    for batch_start in range(0, len(start_vertices), batch_size):
        batch_end = min(batch_start + batch_size, len(start_vertices))
        current_batch = start_vertices[batch_start:batch_end]
        
        # 每个批次单独处理
        for start_vertex in current_batch:
            # 使用deque代替list，提高append/pop效率
            stack = deque([(start_vertex, [start_vertex], {start_vertex})])
            
            while stack:
                current, path, visited = stack.pop()
                
                if len(path) == len(role_sequence) - 1:
                    start_node = path[0]
                    is_connected = False
                    
                    # 使用预计算的邻居列表
                    if direction_sequence[-1]:
                        if start_node in graph["out_neighbors"][current]:
                            is_connected = True
                    else:
                        if start_node in graph["in_neighbors"][current]:
                            is_connected = True
                    
                    if is_connected and start_node in filtered_nodes_by_role[len(role_sequence)-1]:
                        all_paths.append(path.copy())
                        path_count += 1
                        if path_count % 1000 == 0:
                            logging.info(f"角色序列 [{','.join(role_sequence)}]: 已找到 {path_count} 个候选路径，耗时: {time.time() - start_time:.2f}秒")
                    continue
                
                next_role_idx = len(path)
                next_role = role_sequence[next_role_idx]
                is_forward = direction_sequence[len(path) - 1]
                
                # 使用预计算的邻居列表
                neighbors_indices = graph["out_neighbors"][current] if is_forward else graph["in_neighbors"][current]
                
                # 使用集合操作优化邻居过滤
                candidate_neighbors = set(neighbors_indices) & filtered_nodes_by_role[next_role_idx]
                candidate_neighbors = [n for n in candidate_neighbors if n not in visited or n == path[0]]
                
                # 根据角色索引快速过滤
                for neighbor in candidate_neighbors:
                    new_path = path + [neighbor]
                    new_visited = visited | {neighbor}
                    stack.append((neighbor, new_path, new_visited))
    
    # 使用更高效的方式去重路径
    seen_paths = set()
    unique_paths = []
    
    for path in all_paths:
        # 创建标准化的路径表示，用于去重
        path_key = tuple(sorted((get_name(graph, v_id), get_role(graph, v_id)) for v_id in path))
        if path_key not in seen_paths:
            unique_paths.append(path)
            seen_paths.add(path_key)
    
    logging.info(f"角色序列 [{','.join(role_sequence)}]: 找到 {len(unique_paths)} 个符合条件的独立环路 (耗时: {time.time() - start_time:.2f}秒)")
    return unique_paths

# --- (format_cycle_path function with minor optimizations) ---
def format_cycle_path(graph, cycle_indices):
    """格式化环路为可读字符串 (optimized version)"""
    if not cycle_indices:
        return "空环路"
    
    full_cycle_indices = cycle_indices + [cycle_indices[0]]
    path_parts = []
    
    # 预先获取所有节点信息，减少重复查询
    nodes_info = {}
    for v_id in set(full_cycle_indices):
        nodes_info[v_id] = (get_name(graph, v_id), get_role(graph, v_id))
    
    # 预先查询所有边的信息
    edge_info = {}
    for i in range(len(full_cycle_indices) - 1):
        src, dst = full_cycle_indices[i], full_cycle_indices[i+1]
        if graph.are_adjacent(src, dst):
            try:
                edge_id = graph.get_eid(src, dst, directed=True, error=True)
                edge_info[(src, dst)] = graph.es[edge_id].get("label", "关联")
            except:
                edge_info[(src, dst)] = "关联"
        elif graph.are_adjacent(dst, src):
            try:
                edge_id = graph.get_eid(dst, src, directed=True, error=True)
                edge_info[(dst, src)] = graph.es[edge_id].get("label", "关联")
            except:
                edge_info[(dst, src)] = "关联"
    
    for i in range(len(full_cycle_indices)):
        v_id = full_cycle_indices[i]
        name, label = nodes_info[v_id]
        node_info = f"{name} [{label}]"
        
        if "[partner]" in node_info:
            node_info = node_info.replace("[partner]", "[合作公司]")
        path_parts.append(node_info)
        
        if i < len(full_cycle_indices) - 1:
            next_v_id = full_cycle_indices[i+1]
            direction_str = " --> "
            
            if (v_id, next_v_id) in edge_info:
                edge_label_str = edge_info[(v_id, next_v_id)]
                direction_str = f" --({edge_label_str})--> "
            elif (next_v_id, v_id) in edge_info:
                edge_label_str = edge_info[(next_v_id, v_id)]
                direction_str = f" <--({edge_label_str})-- "
            
            path_parts.append(direction_str)
            
    return "".join(path_parts)

# Worker function for parallel processing - optimized
def _worker_find_loops(args):
    """优化后的工作进程函数"""
    graph_path, role_seq, dir_seq, config_name = args
    try:
        # 加载图并添加缓存
        graph = load_graph(graph_path)
        if graph is None:
            logging.error(f"[{config_name}] Worker failed to load graph from {graph_path}")
            return config_name, []

        logging.info(f"[{config_name}] Worker started processing.")
        loops = find_paths_with_role_sequence(graph, role_seq, dir_seq)
        logging.info(f"[{config_name}] Worker finished, found {len(loops)} loops.")
        return config_name, loops
    except Exception as e:
        logging.error(f"[{config_name}] Worker encountered error: {str(e)}")
        return config_name, []

# 创建所有支持的环路配置
def get_all_loop_configs():
    """返回所有支持的环路配置"""
    return {
        "4节点环路": (
            ['股东', 'partner', '成员单位', 'partner', '股东'],
            [True, True, True, False]
        ),
        "6节点环路": (
            ['股东', '股东', 'partner', '成员单位', 'partner', '股东', '股东'],
            [True, True, True, False, False, False]
        ),
        "8节点环路": (
            ['股东', '股东', '股东', 'partner', '成员单位', 'partner', '股东', '股东', '股东'],
            [True, True, True, True, True, False, False, False]
        ),
        # 新增环路配置
        "7节点环路(类型1)": (
            ['股东', '股东', '股东', 'partner', '成员单位', 'partner', '股东', '股东'],
            [True, True, True, True, True, False, False]
        ),
        "6节点环路(类型2)": (
            ['股东', '股东', '股东', 'partner', '成员单位', 'partner', '股东'],
            [True, True, True, True, True, False]
        ),
        "7节点环路(类型2)": (
            ['股东', '股东', 'partner', '成员单位', 'partner', '股东', '股东', '股东'],
            [True, True, True, True, False, False, False]
        ),
        "6节点环路(类型3)": (
            ['股东', 'partner', '成员单位', 'partner', '股东', '股东', '股东'],
            [True, True, True, False, False, False]
        ),
        "5节点环路(类型1)": (
            ['股东', '股东', 'partner', '成员单位', 'partner', '股东'],
            [True, True, True, True, False]
        ),
        "5节点环路(类型2)": (
            ['股东', 'partner', '成员单位', 'partner', '股东', '股东'],
            [True, True, True, False, False]
        )
    }

# 筛选特定节点数的环路配置
def filter_configs_by_node_count(configs, n=None):
    """
    根据节点数筛选环路配置
    
    参数:
        configs: 所有环路配置字典
        n: 节点数，如果为None则返回所有配置
        
    返回:
        筛选后的配置字典
    """
    if n is None:
        return configs
    
    filtered_configs = {}
    for name, (role_seq, dir_seq) in configs.items():
        # 节点数为角色序列长度减1（因为首尾是同一个节点）
        node_count = len(role_seq) - 1
        if node_count == n or (str(n) in name and "节点环路" in name):
            filtered_configs[name] = (role_seq, dir_seq)
    
    if not filtered_configs:
        logging.warning(f"未找到节点数为 {n} 的环路配置，将使用所有配置")
        return configs
        
    return filtered_configs

# 优化分析与保存函数
def analyze_and_save_loops(graph, graph_path_for_workers, n=None, output_file=None):
    """
    优化后的闭环分析与保存函数
    
    参数:
        graph: 图对象
        graph_path_for_workers: 图文件路径，用于并行处理
        n: 节点数，只分析并输出特定节点数的环路，如果为None则分析所有环路
        output_file: 输出文件路径，如果为None则使用默认路径
    
    返回:
        分析是否成功的布尔值
    """
    total_start_time = time.time()
    
    # 如果未指定输出文件，使用默认文件
    if output_file is None:
        output_file = OUTPUT_FILE
        if n is not None:
            # 如果指定了节点数，则修改输出文件名
            filename, ext = os.path.splitext(OUTPUT_FILE)
            output_file = f"{filename}_{n}nodes{ext}"
    
    # 获取所有环路配置
    all_configs = get_all_loop_configs()
    
    # 根据节点数筛选配置
    configs = filter_configs_by_node_count(all_configs, n)
    
    logging.info(f"使用 {len(configs)} 个环路配置进行分析" + (f" (节点数: {n})" if n else ""))
    
    # 并行处理任务
    tasks = [(graph_path_for_workers, role_seq, dir_seq, name) for name, (role_seq, dir_seq) in configs.items()]
    
    all_loops_results = {}
    
    # 自动确定最佳进程数
    cpu_count = os.cpu_count() or 1
    num_processes = min(len(configs), max(1, cpu_count - 1))  # 保留一个CPU核心给系统
    logging.info(f"启动并行闭环分析，使用 {num_processes} 个进程 (系统有 {cpu_count} 个CPU核心).")

    if num_processes > 0 and tasks:
        # 使用进程池并发执行任务
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(_worker_find_loops, tasks)
        
        for config_name, loops in results:
            all_loops_results[config_name] = loops
    else:
        logging.info("由于进程数为0或无任务，改为顺序执行。")
        for task_args in tasks:
            config_name_res, loops_res = _worker_find_loops(task_args)
            all_loops_results[config_name_res] = loops_res

    # 创建环路类型到键的映射
    loops_by_type = {}
    for config_name in configs.keys():
        key = config_name.replace("节点环路", "").replace("(", "_").replace(")", "").replace("类型", "type")
        key = "node" + key
        loops_by_type[key] = all_loops_results.get(config_name, [])
    
    company_loops = defaultdict(lambda: {key: [] for key in loops_by_type.keys()})
    unique_sigs_global = set() 
    
    # 优化公司闭环处理函数
    def process_loops_for_company(loops_list, level_key, graph_obj):
        # 批量处理以提高效率
        for batch_start in range(0, len(loops_list), 100):
            batch_end = min(batch_start + 100, len(loops_list))
            batch = loops_list[batch_start:batch_end]
            
            for cycle_indices in batch:
                if not cycle_indices: continue
                start_v_id = cycle_indices[0]
                start_name = get_name(graph_obj, start_v_id)
                
                # 更高效的签名生成
                sig_parts = tuple(sorted((get_name(graph_obj, v_id), get_role(graph_obj, v_id)) for v_id in cycle_indices))
                
                if sig_parts not in unique_sigs_global:
                    company_loops[start_name][level_key].append(cycle_indices)
                    unique_sigs_global.add(sig_parts)

    # 处理闭环数据
    process_start = time.time()
    logging.info("开始处理闭环数据...")
    for loop_type, loops in loops_by_type.items():
        process_loops_for_company(loops, loop_type, graph)
    logging.info(f"闭环数据处理完成，耗时: {time.time() - process_start:.2f} 秒")
    
    # 统计各类型环路数量
    loop_counts = {loop_type: sum(len(data[loop_type]) for data in company_loops.values()) 
                 for loop_type in loops_by_type.keys()}
    
    # 创建显示名称映射
    display_names = {}
    for config_name in configs.keys():
        key = config_name.replace("节点环路", "").replace("(", "_").replace(")", "").replace("类型", "type")
        key = "node" + key
        display_names[key] = config_name
    
    # 保存结果到文件
    save_start = time.time()
    logging.info(f"开始保存结果到文件: {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        title_suffix = f"({n}节点)" if n else "(扩展版)"
        f.write(f"# 股权闭环检测报告 {title_suffix}\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 总结\n\n")
        
        for loop_type in loops_by_type.keys():
            f.write(f"- 发现{display_names[loop_type]}: {loop_counts[loop_type]} 个\n")
        
        f.write(f"- 共涉及 {len(company_loops)} 个股东 (基于去重后的环路起点)\n\n")
        
        f.write("## 详细闭环信息\n\n")
        for company_name in sorted(company_loops.keys()):
            loops_data = company_loops[company_name]
            total_company_loops = sum(len(loops_data[loop_type]) for loop_type in loops_by_type.keys())
            
            if total_company_loops == 0: continue

            f.write(f"### 股东: {company_name}\n\n")
            f.write(f"该股东共涉及 {total_company_loops} 个已识别的闭环:\n")
            for loop_type in loops_by_type.keys():
                if len(loops_data[loop_type]) > 0:
                    f.write(f"- {display_names[loop_type]}: {len(loops_data[loop_type])} 个\n")
            f.write("\n")
            
            # 写入各类型环路的详细信息
            for loop_type in loops_by_type.keys():
                if loops_data[loop_type]:
                    f.write(f"#### {display_names[loop_type]}\n\n")
                    for i, cycle_idx in enumerate(loops_data[loop_type]):
                        f.write(f"{i+1}. {format_cycle_path(graph, cycle_idx)}\n\n")
            f.write("-" * 80 + "\n\n")
    
    logging.info(f"结果保存完成，耗时: {time.time() - save_start:.2f} 秒")
    
    total_end_time = time.time()
    logging.info(f"闭环分析完成，总用时: {total_end_time - total_start_time:.2f} 秒")
    logging.info(f"最终结果 (去重后):")
    for loop_type in loops_by_type.keys():
        logging.info(f"- {display_names[loop_type]}: {loop_counts[loop_type]} 个")
    logging.info(f"报告已保存到: {output_file}")
    
    return True

def main():
    """主函数"""
    setup_logging()
    # Windows多进程支持
    multiprocessing.freeze_support() 
    
    graph_path_to_use = SIMPLIFIED_GRAPH_PATH if os.path.exists(SIMPLIFIED_GRAPH_PATH) else INPUT_GRAPH_PATH
    logging.info(f"使用图文件: {graph_path_to_use}")
    
    # 加载主图
    start_load = time.time()
    main_graph_for_reporting = load_graph(graph_path_to_use)
    if main_graph_for_reporting is None:
        logging.error("主图加载失败，退出程序。")
        return
    
    logging.info(f"主图加载完成，耗时: {time.time() - start_load:.2f} 秒")
    logging.info(f"主图包含: {main_graph_for_reporting.vcount()} 节点, {main_graph_for_reporting.ecount()} 边")
    
    # 使用全局配置的节点数
    if NODE_COUNT is not None:
        logging.info(f"根据配置，分析 {NODE_COUNT} 节点环路...")
        analyze_and_save_loops(main_graph_for_reporting, graph_path_to_use, n=NODE_COUNT)
    else:
        logging.info("根据配置，分析所有环路类型...")
        analyze_and_save_loops(main_graph_for_reporting, graph_path_to_use)
    
    logging.info("股权闭环分析完成。")

if __name__ == "__main__":
    main()