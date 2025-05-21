#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异构图分析工具 - 不包含可视化，仅生成文本报告
"""

import os
import networkx as nx
import pandas as pd
from collections import defaultdict, Counter
import community  # type: ignore
import time
import logging
from datetime import datetime

# 配置
INPUT_GRAPH_PATH = "model/final_heterogeneous_graph.graphml"
OUTPUT_DIR = "outputs"
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
LOG_FILE = os.path.join(OUTPUT_DIR, "exploratory_analysis.log")
OUTPUT_REPORT_PATH = os.path.join(REPORTS_DIR, "exploratory_analysis_report.md")
OUTPUT_SUMMARY_CSV = os.path.join(REPORTS_DIR, "graph_summary.csv")
OUTPUT_TOP_ENTITIES_CSV = os.path.join(REPORTS_DIR, "top_entities.csv")

# 设置日志
def setup_logging():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
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
    for directory in [OUTPUT_DIR, REPORTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"创建目录: {directory}")

# 保存报告
def save_report(df, filename):
    filepath = os.path.join(REPORTS_DIR, filename)
    df.to_csv(filepath, index=True, encoding='utf-8-sig')
    logging.info(f"报告已保存: {filepath}")

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

def analyze_heterogeneous_graph(graph, report_logger):
    """
    分析异构图并生成报告
    """
    start_time = time.time()
    logging.info("正在创建报告 {}...".format(OUTPUT_REPORT_PATH))
    
    if graph is None or graph.number_of_nodes() == 0:
        logging.error("图为空，无法进行分析。请检查输入文件。")
        return
    
    # 基本统计信息
    report_logger.info("## 1. 基本统计信息\n")
    report_logger.info("-" * 50)
    report_logger.info(f"节点总数: {graph.number_of_nodes()}")
    report_logger.info(f"边总数: {graph.number_of_edges()}")
    
    # 保存基本统计信息到CSV
    stats_df = pd.DataFrame([{
        '统计项': '节点总数',
        '数值': graph.number_of_nodes(),
        '说明': '图中包含的所有实体的数量，包括公司、股东等各类型节点'
    }, {
        '统计项': '边总数',
        '数值': graph.number_of_edges(),
        '说明': '图中所有关系的数量，包括股权关系、交易关系等'
    }])
    
    # 节点类型统计
    node_labels = defaultdict(int)
    for node, attrs in graph.nodes(data=True):
        label = attrs.get('label', '未知')
        node_labels[label] += 1
    
    report_logger.info("\n节点类型分布:")
    for label, count in sorted(node_labels.items(), key=lambda x: x[1], reverse=True):
        percent = count/graph.number_of_nodes()*100
        report_logger.info(f"  - {label}: {count} 个节点 ({percent:.2f}%)")
        
        # 提供不同节点类型的解释
        explanation = ""
        if label == "company":
            explanation = "参与股权或交易关系的企业实体"
        elif label == "shareholder":
            explanation = "持有企业股份的个人或企业实体"
        elif label == "partner":
            explanation = "与核心企业有交易关系的合作伙伴，可能是上游供应商或下游客户"
        elif label == "数据缺失":
            explanation = "数据不完整的实体，缺少类型标识"
        
        # 使用concat代替append，创建一个新的DataFrame并进行合并
        stats_df = pd.concat([
            stats_df, 
            pd.DataFrame([{
                '统计项': f'节点类型 - {label}',
                '数值': count,
                '说明': f'{explanation}，占总节点的{percent:.2f}%'
            }])
        ], ignore_index=True)
    
    # 边类型统计
    edge_labels = defaultdict(int)
    for _, _, attrs in graph.edges(data=True):
        label = attrs.get('label', '未知')
        edge_labels[label] += 1
    
    report_logger.info("\n边类型分布:")
    for label, count in sorted(edge_labels.items(), key=lambda x: x[1], reverse=True):
        percent = count/graph.number_of_edges()*100
        report_logger.info(f"  - {label}: {count} 条边 ({percent:.2f}%)")
        
        # 提供不同边类型的解释
        explanation = ""
        if label == "holds":
            explanation = "表示股权持有关系，从股东指向被投资企业"
        elif label == "transaction":
            explanation = "表示交易关系，边上附带交易金额、频次等属性"
        elif label.lower() == "supplier":
            explanation = "表示供应商关系，从供应商指向客户"
        elif label.lower() == "customer":
            explanation = "表示客户关系，从客户指向供应商"
        
        # 使用concat代替append
        stats_df = pd.concat([
            stats_df, 
            pd.DataFrame([{
                '统计项': f'边类型 - {label}',
                '数值': count,
                '说明': f'{explanation}，占总边数的{percent:.2f}%'
            }])
        ], ignore_index=True)
    
    # 保存统计数据
    stats_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding='utf-8-sig')
    report_logger.info(f"\n基本统计信息已保存至 {OUTPUT_SUMMARY_CSV}")
    
    # 图的连通性分析
    report_logger.info("\n\n## 2. 连通性分析\n")
    report_logger.info("-" * 50)
    
    # 创建无向图进行连通分量分析
    undirected_graph = graph.to_undirected()
    connected_components = list(nx.connected_components(undirected_graph))
    
    report_logger.info(f"连通分量数量: {len(connected_components)}")
    report_logger.info("连通分量解释: 连通分量是图中相互连接的节点集合，不同连通分量之间没有连接。多个连通分量表示网络中存在相互隔离的企业关系群体。")
    
    # 分析主要连通分量
    largest_cc = max(connected_components, key=len)
    percent_largest = len(largest_cc)/graph.number_of_nodes()*100
    report_logger.info(f"最大连通分量包含 {len(largest_cc)} 个节点 ({percent_largest:.2f}% 的总节点)")
    report_logger.info("最大连通分量解释: 这是图中最大的相互连接的企业网络，代表了主要的企业关系群体。")
    
    # 分析连通分量内的节点类型
    report_logger.info("\n最大连通分量的节点类型分布:")
    cc_node_labels = defaultdict(int)
    for node in largest_cc:
        label = graph.nodes[node].get('label', '未知')
        cc_node_labels[label] += 1
    
    for label, count in sorted(cc_node_labels.items(), key=lambda x: x[1], reverse=True):
        percent = count/len(largest_cc)*100
        report_logger.info(f"  - {label}: {count} 个节点 ({percent:.2f}%)")
    
    # 中心性分析
    report_logger.info("\n\n## 3. 中心性分析\n")
    report_logger.info("-" * 50)
    report_logger.info("计算中，请稍候...")
    report_logger.info("中心性分析解释: 中心性指标衡量节点在网络中的重要性，不同指标反映不同类型的重要性。")
    
    # 使用最大连通分量的子图进行中心性分析
    largest_cc_subgraph = graph.subgraph(largest_cc).copy()
    
    # 度中心性
    report_logger.info("\n### 3.1 度中心性分析")
    report_logger.info("度中心性解释: 度中心性表示节点的连接数量。在有向图中，入度表示指向该节点的边数（如被投资次数或下游企业数量），出度表示从该节点出发的边数（如投资次数或上游企业数量）。")
    
    in_degree = {node: val for node, val in largest_cc_subgraph.in_degree()}
    out_degree = {node: val for node, val in largest_cc_subgraph.out_degree()}
    
    # 入度最高的节点（被最多实体指向）
    top_in_degree = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    report_logger.info("\n入度最高的节点（被最多实体指向）:")
    report_logger.info("解释: 这些企业被最多其他实体指向，可能是被广泛投资的企业或拥有众多供应商的核心企业。")
    for node, degree_val in top_in_degree:
        node_name = largest_cc_subgraph.nodes[node].get('name', node)
        node_type = largest_cc_subgraph.nodes[node].get('label', '未知')
        report_logger.info(f"  - {node_name} [{node_type}]: 入度 = {degree_val}")
    
    # 出度最高的节点（指向最多实体）
    top_out_degree = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    report_logger.info("\n出度最高的节点（指向最多实体）:")
    report_logger.info("解释: 这些企业指向最多其他实体，可能是大型投资方或拥有众多客户的供应商。")
    for node, degree_val in top_out_degree:
        node_name = largest_cc_subgraph.nodes[node].get('name', node)
        node_type = largest_cc_subgraph.nodes[node].get('label', '未知')
        report_logger.info(f"  - {node_name} [{node_type}]: 出度 = {degree_val}")
    
    # PageRank中心性
    report_logger.info("\n### 3.2 PageRank分析")
    report_logger.info("PageRank解释: PageRank算法评估节点的全局重要性，不仅考虑连接数量，还考虑连接质量。高PageRank值的节点通常是网络中的核心枢纽，连接到其他重要节点。")
    
    try:
        pagerank_vals = nx.pagerank(largest_cc_subgraph, alpha=0.85, max_iter=100)
        top_pagerank = sorted(pagerank_vals.items(), key=lambda x: x[1], reverse=True)[:10]
        
        report_logger.info("\nPageRank值最高的节点（网络重要性）:")
        for node, pr_val in top_pagerank:
            node_name = largest_cc_subgraph.nodes[node].get('name', node)
            node_type = largest_cc_subgraph.nodes[node].get('label', '未知')
            report_logger.info(f"  - {node_name} [{node_type}]: PageRank = {pr_val:.6f}")
    except:
        report_logger.info("PageRank计算失败，可能是图结构问题。")
    
    # 保存重要实体信息
    top_entities = []
    
    # 合并度中心性数据
    for node, in_deg in top_in_degree:
        out_deg = out_degree.get(node, 0)
        pagerank = pagerank_vals.get(node, 0) if 'pagerank_vals' in locals() else 0
        node_name = largest_cc_subgraph.nodes[node].get('name', node)
        node_type = largest_cc_subgraph.nodes[node].get('label', '未知')
        
        top_entities.append({
            '实体ID': node,
            '实体名称': node_name,
            '实体类型': node_type,
            '入度': in_deg,
            '出度': out_deg,
            'PageRank': pagerank,
            '说明': f"该实体在网络中的重要性排名靠前，有{in_deg}个入连接和{out_deg}个出连接"
        })
    
    # 添加可能缺失的出度高的节点
    for node, out_deg in top_out_degree:
        if not any(e['实体ID'] == node for e in top_entities):
            in_deg = in_degree.get(node, 0)
            pagerank = pagerank_vals.get(node, 0) if 'pagerank_vals' in locals() else 0
            node_name = largest_cc_subgraph.nodes[node].get('name', node)
            node_type = largest_cc_subgraph.nodes[node].get('label', '未知')
            
            top_entities.append({
                '实体ID': node,
                '实体名称': node_name,
                '实体类型': node_type,
                '入度': in_deg,
                '出度': out_deg,
                'PageRank': pagerank,
                '说明': f"该实体在网络中主动连接其他实体能力强，有{in_deg}个入连接和{out_deg}个出连接"
            })
    
    # 保存为CSV
    if top_entities:
        pd.DataFrame(top_entities).to_csv(OUTPUT_TOP_ENTITIES_CSV, index=False, encoding='utf-8-sig')
        report_logger.info(f"\n重要实体信息已保存至 {OUTPUT_TOP_ENTITIES_CSV}")
    
    # 社区检测
    report_logger.info("\n\n## 4. 社区检测\n")
    report_logger.info("-" * 50)
    report_logger.info("社区检测解释: 社区是图中节点的集合，内部连接紧密但与其他社区连接较少。识别社区有助于了解企业关系网络中的自然分组，如产业集群或投资集团。")
    
    # 使用无向图进行社区检测
    undirected_largest_cc = largest_cc_subgraph.to_undirected()
    
    try:
        report_logger.info("正在使用Louvain算法进行社区检测...")
        report_logger.info("Louvain算法解释: 这是一种广泛使用的社区检测算法，通过优化模块度来识别网络中的社区结构。")
        communities = community.best_partition(undirected_largest_cc)
        
        # 统计社区
        community_sizes = Counter(communities.values())
        
        report_logger.info(f"检测到 {len(community_sizes)} 个社区")
        
        # 分析最大的社区
        top_communities = community_sizes.most_common(5)
        
        for i, (comm_id, size) in enumerate(top_communities):
            percent = size/len(largest_cc)*100
            report_logger.info(f"\n社区 {i+1}: 包含 {size} 个节点 ({percent:.2f}% 的最大连通分量)")
            report_logger.info(f"社区解释: 这是第{i+1}大的企业关系集群，内部企业间关系紧密。")
            
            # 统计社区内的节点类型
            comm_node_types = defaultdict(int)
            comm_members = [node for node, comm in communities.items() if comm == comm_id]
            
            for node in comm_members:
                node_type = largest_cc_subgraph.nodes[node].get('label', '未知')
                comm_node_types[node_type] += 1
            
            report_logger.info("节点类型分布:")
            for node_type, count in sorted(comm_node_types.items(), key=lambda x: x[1], reverse=True):
                type_percent = count/size*100
                report_logger.info(f"  - {node_type}: {count} 个节点 ({type_percent:.2f}%)")
            
            # 展示社区内的重要节点
            community_nodes = [node for node, comm in communities.items() if comm == comm_id]
            subgraph = largest_cc_subgraph.subgraph(community_nodes)
            
            # 计算子图内的度
            degree_in_community = {node: subgraph.degree(node) for node in subgraph.nodes()}
            top_nodes = sorted(degree_in_community.items(), key=lambda x: x[1], reverse=True)[:5]
            
            report_logger.info("社区内最重要的节点:")
            report_logger.info("解释: 这些节点在该社区内连接最多，是社区的核心成员。")
            for node, degree_val in top_nodes:
                node_name = subgraph.nodes[node].get('name', node)
                node_type = subgraph.nodes[node].get('label', '未知')
                report_logger.info(f"  - {node_name} [{node_type}]: 度 = {degree_val}")
    except Exception as e:
        report_logger.error(f"社区检测过程中出错: {e}")
    
    # 风险分析
    report_logger.info("\n\n## 5. 风险分析\n")
    report_logger.info("-" * 50)
    
    # 分析跨社区交易
    report_logger.info("\n### 5.1 跨社区关系分析")
    report_logger.info("跨社区关系解释: 这些关系连接不同社区的实体，代表了不同企业群体之间的桥梁。这些关系对整个网络的连通性至关重要。")
    
    if 'communities' in locals():
        # 找出跨社区的边
        cross_community_edges = []
        for u, v, attrs in largest_cc_subgraph.edges(data=True):
            if u in communities and v in communities and communities[u] != communities[v]:
                cross_community_edges.append((u, v, attrs))
        
        report_logger.info(f"发现 {len(cross_community_edges)} 条跨社区连接")
        
        # 分析前10条跨社区边
        if cross_community_edges:
            report_logger.info("\n重要的跨社区连接:")
            report_logger.info("解释: 这些连接是不同企业群体之间的重要桥梁，可能代表关键的跨集团投资或交易关系。")
            for i, (u, v, attrs) in enumerate(cross_community_edges[:10]):
                u_name = largest_cc_subgraph.nodes[u].get('name', u)
                u_type = largest_cc_subgraph.nodes[u].get('label', '未知')
                u_community = communities[u]
                
                v_name = largest_cc_subgraph.nodes[v].get('name', v)
                v_type = largest_cc_subgraph.nodes[v].get('label', '未知')
                v_community = communities[v]
                
                edge_label = attrs.get('label', '关联')
                nature = attrs.get('nature', '未定义')
                
                relation_description = f"{u_name} [{u_type}, 社区 {u_community}] --({edge_label})"
                if nature and nature != '未定义':
                    relation_description += f" [性质: {nature}]"
                relation_description += f"--> {v_name} [{v_type}, 社区 {v_community}]"
                
                report_logger.info(f"  - {relation_description}")
    else:
        report_logger.info("由于社区检测失败，无法进行跨社区分析")
    
    # 总结
    analysis_time = time.time() - start_time
    report_logger.info(f"\n\n## 总结\n")
    report_logger.info(f"分析完成，用时: {analysis_time:.2f} 秒")
    report_logger.info("此报告提供了企业关系网络的全面分析，包括基本统计信息、连通性分析、中心性分析、社区检测和风险分析。")
    report_logger.info("这些指标有助于识别网络中的重要实体、关键关系和潜在风险，为企业决策提供数据支持。")
    
    return True

def main():
    # 设置日志
    setup_logging()
    
    # 确保输出目录存在
    ensure_output_dirs()
    
    logging.info(f"开始加载异构图 {INPUT_GRAPH_PATH}...")
    G = load_graph(INPUT_GRAPH_PATH)
    
    if G is None:
        logging.error("图加载失败，退出程序。")
        return
    
    logging.info(f"正在创建报告 {OUTPUT_REPORT_PATH}...")
    
    # 创建一个自定义日志处理器，将日志内容同时写入报告文件
    report_logger = logging.getLogger("report_logger")
    report_logger.setLevel(logging.INFO)
    
    # 同时将输出写入控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    report_logger.addHandler(console_handler)
    
    # 打开报告文件
    with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as report_file:
        # 添加报告文件处理器
        file_handler = logging.StreamHandler(report_file)
        file_handler.setLevel(logging.INFO)
        report_logger.addHandler(file_handler)
        
        # 写入报告标题
        report_logger.info("# 企业股权与交易关系异构图分析报告\n")
        report_logger.info(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 执行分析
        analyze_heterogeneous_graph(G, report_logger)
        
        # 移除文件处理器
        report_logger.removeHandler(file_handler)
    
    logging.info(f"分析完成，报告已保存到 {OUTPUT_REPORT_PATH}")
    logging.info(f"同时生成了CSV文件: {OUTPUT_SUMMARY_CSV} 和 {OUTPUT_TOP_ENTITIES_CSV}")

if __name__ == "__main__":
    main() 