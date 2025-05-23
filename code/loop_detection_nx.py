#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股权闭环检测工具 - NetworkX 兼容版 (适用于 Windows 环境)
基于 code/loop_detection.py 改写，核心逻辑保持一致，只将底层图库替换为 networkx。
"""

import os
import logging
import time
from datetime import datetime
from collections import defaultdict, deque
import multiprocessing
import networkx as nx  # 使用 networkx 替代 igraph

# --- 配置常量（保持与原脚本一致，方便复用） ---
INPUT_GRAPH_PATH = "model/final_heterogeneous_graph.graphml"
SIMPLIFIED_GRAPH_PATH = "model/simplified_loop_detection_graph.graphml"
OUTPUT_DIR = "outputs"
LOG_DIR = os.path.join(OUTPUT_DIR, "log")
LOG_FILE = os.path.join(LOG_DIR, "loop_detection_nx.log")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "loop_results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "equity_loops_nx.txt")
MAX_CACHE_SIZE = 10000  # 目前未使用，但保留以保持接口一致

# ===== 用户可配置参数 =====
NODE_COUNT = None  # 指定要分析的环节点数；为 None 时分析全部
# ============================

def setup_logging():
    """初始化日志系统"""
    for directory in [LOG_DIR, RESULTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, 'w', 'utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"开始分析，日志记录于 {LOG_FILE}")

# -----------------------------------------------------------------------------
# 图加载与预处理
# -----------------------------------------------------------------------------

def load_graph(file_path):
    """从 GraphML 文件加载 NetworkX 图，并做预处理"""
    if not os.path.exists(file_path):
        logging.error(f"图文件未找到于 '{file_path}'")
        return None
    try:
        logging.info(f"正在从 '{file_path}' 加载图...")
        G = nx.read_graphml(file_path)

        # 确保得到的是有向图
        if not isinstance(G, nx.DiGraph):
            G = nx.DiGraph(G)  # 强制转换为有向图

        # 补充节点属性
        for n in G.nodes:
            if 'name' not in G.nodes[n]:
                G.nodes[n]['name'] = str(n)
            if 'label' not in G.nodes[n]:
                G.nodes[n]['label'] = '未知'

        # 补充边属性
        for u, v, data in G.edges(data=True):
            if 'label' not in data:
                data['label'] = '关联'

        # 构建角色到节点的映射
        role_to_vertices = defaultdict(list)
        for n in G.nodes:
            role_to_vertices[G.nodes[n]['label']].append(n)
        G.graph['role_to_vertices'] = role_to_vertices

        # 预计算出入邻接表，提升查找效率
        G.graph['out_neighbors'] = {n: list(G.successors(n)) for n in G.nodes}
        G.graph['in_neighbors'] = {n: list(G.predecessors(n)) for n in G.nodes}

        logging.info(f"图加载成功，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")
        return G
    except Exception as e:
        logging.error(f"加载图 '{file_path}' 时发生错误: {e}")
        return None

# -----------------------------------------------------------------------------
# 基础工具函数
# -----------------------------------------------------------------------------

def get_role(G, node):
    return G.nodes[node]['label']

def get_name(G, node):
    return G.nodes[node]['name']

# -----------------------------------------------------------------------------
# 环路搜索核心逻辑
# -----------------------------------------------------------------------------

def find_paths_with_role_sequence(G, role_sequence, direction_sequence=None):
    """按给定角色序列/方向序列搜索闭环路径"""
    if not G:
        logging.error("Graph object is None in find_paths_with_role_sequence.")
        return []

    if len(role_sequence) < 2 or role_sequence[0] != role_sequence[-1]:
        logging.warning("角色序列必须以相同角色开始和结束才能形成闭环")
        return []

    logging.info(f"查找环路: 角色序列=[{','.join(role_sequence)}], 目标长度={len(role_sequence)-1}")

    start_role = role_sequence[0]
    start_vertices = G.graph['role_to_vertices'].get(start_role, [])
    logging.info(f"找到 {len(start_vertices)} 个起始角色为 '{start_role}' 的节点")
    if not start_vertices:
        return []

    if not direction_sequence:
        direction_sequence = [True] * (len(role_sequence) - 1)
    if len(direction_sequence) != len(role_sequence) - 1:
        logging.error("方向序列长度与角色序列不匹配")
        return []

    filtered_nodes_by_role = {
        idx: set(G.graph['role_to_vertices'].get(role, []))
        for idx, role in enumerate(role_sequence)
    }

    all_paths = []
    path_count = 0
    start_time = time.time()

    batch_size = min(100, len(start_vertices))

    for batch_start in range(0, len(start_vertices), batch_size):
        batch_end = min(batch_start + batch_size, len(start_vertices))
        current_batch = start_vertices[batch_start:batch_end]

        for start_node in current_batch:
            stack = deque([(start_node, [start_node], {start_node})])
            while stack:
                current, path, visited = stack.pop()

                if len(path) == len(role_sequence) - 1:
                    # 判断最后一步能否回到起点
                    is_forward = direction_sequence[-1]
                    if (is_forward and G.has_edge(current, start_node)) or (not is_forward and G.has_edge(start_node, current)):
                        if start_node in filtered_nodes_by_role[len(role_sequence)-1]:
                            all_paths.append(path.copy())
                            path_count += 1
                            if path_count % 1000 == 0:
                                logging.info(
                                    f"角色序列 [{','.join(role_sequence)}]: 已找到 {path_count} 个候选路径，耗时: {time.time() - start_time:.2f}秒")
                    continue

                next_role_idx = len(path)
                next_role = role_sequence[next_role_idx]
                is_forward = direction_sequence[len(path) - 1]

                neighbors = G.graph['out_neighbors'][current] if is_forward else G.graph['in_neighbors'][current]
                candidate_neighbors = [n for n in neighbors if n in filtered_nodes_by_role[next_role_idx] and (n not in visited or n == path[0])]

                for neighbor in candidate_neighbors:
                    new_path = path + [neighbor]
                    new_visited = visited | {neighbor}
                    stack.append((neighbor, new_path, new_visited))

    # 去重
    seen_sigs = set()
    unique_paths = []
    for path in all_paths:
        sig = tuple(sorted((get_name(G, n), get_role(G, n)) for n in path))
        if sig not in seen_sigs:
            unique_paths.append(path)
            seen_sigs.add(sig)

    logging.info(
        f"角色序列 [{','.join(role_sequence)}]: 找到 {len(unique_paths)} 个符合条件的独立环路 (耗时: {time.time() - start_time:.2f}秒)")
    return unique_paths

# -----------------------------------------------------------------------------
# 环路格式化（输出友好字符串）
# -----------------------------------------------------------------------------

def _first_edge_label(edge_attributes):
    """辅助函数：从边的属性字典中获取 'label'。"""
    if edge_attributes is None:
        return '关联'
    # G 被强制转换为 DiGraph, 因此 edge_attributes 是边的属性字典
    return edge_attributes.get('label', '关联')

def format_cycle_path(G, cycle_nodes):
    if not cycle_nodes:
        return "空环路"
    full_cycle = cycle_nodes + [cycle_nodes[0]]

    nodes_info = {n: (get_name(G, n), get_role(G, n)) for n in set(full_cycle)}

    edge_info = {}
    for i in range(len(full_cycle) - 1):
        u, v = full_cycle[i], full_cycle[i+1]
        if G.has_edge(u, v):
            edge_info[(u, v)] = _first_edge_label(G.get_edge_data(u, v))
        elif G.has_edge(v, u):
            edge_info[(v, u)] = _first_edge_label(G.get_edge_data(v, u))

    parts = []
    for i, n in enumerate(full_cycle):
        name, label = nodes_info[n]
        node_repr = f"{name} [{label}]"
        if "[partner]" in node_repr:
            node_repr = node_repr.replace("[partner]", "[合作公司]")
        parts.append(node_repr)

        if i < len(full_cycle) - 1:
            nxt = full_cycle[i+1]
            if (n, nxt) in edge_info:
                parts.append(f" --({edge_info[(n, nxt)]})--> ")
            elif (nxt, n) in edge_info:
                parts.append(f" <--({edge_info[(nxt, n)]})-- ")
            else:
                parts.append(" --> ")
    return "".join(parts)

# -----------------------------------------------------------------------------
# 并行工作进程
# -----------------------------------------------------------------------------

def _worker_find_loops(args):
    graph_path, role_seq, dir_seq, config_name = args
    try:
        G = load_graph(graph_path)
        if G is None:
            logging.error(f"[{config_name}] Worker failed to load graph from {graph_path}")
            return config_name, []
        logging.info(f"[{config_name}] Worker started processing.")
        loops = find_paths_with_role_sequence(G, role_seq, dir_seq)
        logging.info(f"[{config_name}] Worker finished, found {len(loops)} loops.")
        return config_name, loops
    except Exception as e:
        logging.error(f"[{config_name}] Worker encountered error: {str(e)}")
        return config_name, []

# -----------------------------------------------------------------------------
# 环路配置（与原脚本保持一致）
# -----------------------------------------------------------------------------

def get_all_loop_configs():
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

# 根据节点数过滤配置
def filter_configs_by_node_count(configs, n=None):
    if n is None:
        return configs
    filtered = {}
    for name, (role_seq, dir_seq) in configs.items():
        node_count = len(role_seq) - 1
        if node_count == n or (str(n) in name and "节点环路" in name):
            filtered[name] = (role_seq, dir_seq)
    if not filtered:
        logging.warning(f"未找到节点数为 {n} 的环路配置，将使用所有配置")
        return configs
    return filtered

# -----------------------------------------------------------------------------
# 主分析函数
# -----------------------------------------------------------------------------

def analyze_and_save_loops(G, graph_path_for_workers, n=None, output_file=None):
    total_start = time.time()

    if output_file is None:
        output_file = OUTPUT_FILE
        if n is not None:
            base, ext = os.path.splitext(OUTPUT_FILE)
            output_file = f"{base}_{n}nodes{ext}"

    all_configs = get_all_loop_configs()
    configs = filter_configs_by_node_count(all_configs, n)

    logging.info(f"使用 {len(configs)} 个环路配置进行分析" + (f" (节点数: {n})" if n else ""))

    tasks = [(graph_path_for_workers, role_seq, dir_seq, name) for name, (role_seq, dir_seq) in configs.items()]
    all_loops_results = {}

    cpu_cnt = os.cpu_count() or 1
    num_procs = min(len(configs), max(1, cpu_cnt - 1))
    logging.info(f"启动并行闭环分析，使用 {num_procs} 个进程 (系统有 {cpu_cnt} 个CPU核心).")

    if num_procs > 0 and tasks:
        with multiprocessing.Pool(processes=num_procs) as pool:
            results = pool.map(_worker_find_loops, tasks)
        for config_name, loops in results:
            all_loops_results[config_name] = loops
    else:
        logging.info("由于进程数为0或无任务，改为顺序执行。")
        for t in tasks:
            cfg_name, loops_res = _worker_find_loops(t)
            all_loops_results[cfg_name] = loops_res

    # 环路结果整理
    loops_by_type = {}
    for cfg_name in configs.keys():
        key = cfg_name.replace("节点环路", "").replace("(", "_").replace(")", "").replace("类型", "type")
        key = "node" + key
        loops_by_type[key] = all_loops_results.get(cfg_name, [])

    company_loops = defaultdict(lambda: {k: [] for k in loops_by_type.keys()})
    unique_sigs_global = set()

    def process_loops_for_company(loops_list, level_key):
        for batch_start in range(0, len(loops_list), 100):
            batch_end = min(batch_start + 100, len(loops_list))
            for cycle_nodes in loops_list[batch_start:batch_end]:
                if not cycle_nodes:
                    continue
                start_node = cycle_nodes[0]
                start_name = get_name(G, start_node)
                sig_parts = tuple(sorted((get_name(G, n), get_role(G, n)) for n in cycle_nodes))
                if sig_parts not in unique_sigs_global:
                    company_loops[start_name][level_key].append(cycle_nodes)
                    unique_sigs_global.add(sig_parts)

    process_start = time.time()
    logging.info("开始处理闭环数据...")
    for loop_type, loops in loops_by_type.items():
        process_loops_for_company(loops, loop_type)
    logging.info(f"闭环数据处理完成，耗时: {time.time() - process_start:.2f} 秒")

    loop_counts = {lt: sum(len(d[lt]) for d in company_loops.values()) for lt in loops_by_type.keys()}

    display_names = {}
    for cfg_name in configs.keys():
        key = cfg_name.replace("节点环路", "").replace("(", "_").replace(")", "").replace("类型", "type")
        key = "node" + key
        display_names[key] = cfg_name

    save_start = time.time()
    logging.info(f"开始保存结果到文件: {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        title_suffix = f"({n}节点)" if n else "(扩展版)"
        f.write(f"# 股权闭环检测报告 {title_suffix}\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 总结\n\n")
        for lt in loops_by_type.keys():
            f.write(f"- 发现{display_names[lt]}: {loop_counts[lt]} 个\n")
        f.write(f"- 共涉及 {len(company_loops)} 个股东 (基于去重后的环路起点)\n\n")

        f.write("## 详细闭环信息\n\n")
        for company in sorted(company_loops.keys()):
            loops_data = company_loops[company]
            total = sum(len(loops_data[lt]) for lt in loops_by_type.keys())
            if total == 0:
                continue
            f.write(f"### 股东: {company}\n\n")
            f.write(f"该股东共涉及 {total} 个已识别的闭环:\n")
            for lt in loops_by_type.keys():
                if loops_data[lt]:
                    f.write(f"- {display_names[lt]}: {len(loops_data[lt])} 个\n")
            f.write("\n")
            for lt in loops_by_type.keys():
                if loops_data[lt]:
                    f.write(f"#### {display_names[lt]}\n\n")
                    for i, cyc in enumerate(loops_data[lt]):
                        f.write(f"{i+1}. {format_cycle_path(G, cyc)}\n\n")
            f.write("-" * 80 + "\n\n")
    logging.info(f"结果保存完成，耗时: {time.time() - save_start:.2f} 秒")

    total_end = time.time()
    logging.info(f"闭环分析完成，总用时: {total_end - total_start:.2f} 秒")
    logging.info("最终结果 (去重后):")
    for lt in loops_by_type.keys():
        logging.info(f"- {display_names[lt]}: {loop_counts[lt]} 个")
    logging.info(f"报告已保存到: {output_file}")

    return True

# -----------------------------------------------------------------------------
# 主入口
# -----------------------------------------------------------------------------

def main():
    setup_logging()
    multiprocessing.freeze_support()  # Windows 兼容

    graph_path = SIMPLIFIED_GRAPH_PATH if os.path.exists(SIMPLIFIED_GRAPH_PATH) else INPUT_GRAPH_PATH
    logging.info(f"使用图文件: {graph_path}")

    G = load_graph(graph_path)
    if G is None:
        logging.error("主图加载失败，退出程序。")
        return

    logging.info(f"主图加载完成，包含 {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")

    if NODE_COUNT is not None:
        logging.info(f"根据配置，分析 {NODE_COUNT} 节点环路...")
        analyze_and_save_loops(G, graph_path, n=NODE_COUNT)
    else:
        logging.info("根据配置，分析所有环路类型...")
        analyze_and_save_loops(G, graph_path)

    logging.info("股权闭环分析完成。")

if __name__ == "__main__":
    main() 