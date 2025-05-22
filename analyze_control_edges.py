#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析异构图中的控股边关系，特别是股东-股东和股东-partner之间的控股关系。
"""

import networkx as nx
import pandas as pd
import collections
import os

def main():
    """主函数"""
    print("开始分析异构图中的控股边关系...")
    
    # 加载异构图
    graph_path = "model/final_heterogeneous_graph.graphml"
    if not os.path.exists(graph_path):
        print(f"错误：找不到异构图文件 {graph_path}")
        return
        
    try:
        graph = nx.read_graphml(graph_path)
        print(f"成功加载异构图，包含 {graph.number_of_nodes()} 个节点和 {graph.number_of_edges()} 条边。")
    except Exception as e:
        print(f"加载异构图时出错: {e}")
        return
    
    # 1. 统计控股边的分布
    control_edges_dist = {}
    for u, v, key, data in graph.edges(data=True, keys=True):
        if data.get('label') == '控股':
            u_label = graph.nodes[u].get('label', 'unknown')
            v_label = graph.nodes[v].get('label', 'unknown')
            edge_type = f'{u_label}->{v_label}'
            control_edges_dist[edge_type] = control_edges_dist.get(edge_type, 0) + 1
    
    print("\n控股边分布:")
    for edge_type, count in sorted(control_edges_dist.items(), key=lambda x: x[1], reverse=True):
        print(f'{edge_type}: {count}个')
    
    # 2. 提取各类型的控股边
    shareholder_to_shareholder = [(u, v, data) for u, v, key, data in graph.edges(data=True, keys=True) 
                                 if data.get('label') == '控股' 
                                 and graph.nodes[u].get('label') == '股东' 
                                 and graph.nodes[v].get('label') == '股东']
    
    shareholder_to_partner = [(u, v, data) for u, v, key, data in graph.edges(data=True, keys=True) 
                             if data.get('label') == '控股' 
                             and graph.nodes[u].get('label') == '股东' 
                             and graph.nodes[v].get('label') == 'partner']
    
    partner_to_shareholder = [(u, v, data) for u, v, key, data in graph.edges(data=True, keys=True) 
                             if data.get('label') == '控股' 
                             and graph.nodes[u].get('label') == 'partner' 
                             and graph.nodes[v].get('label') == '股东']
    
    # 3. 分析持股比例分布
    def percent_distribution(edges):
        """分析持股比例的分布"""
        ranges = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, float('inf'))]
        dist = collections.Counter()
        for _, _, data in edges:
            percent = data.get('percent')
            if percent is not None:
                for low, high in ranges:
                    if low <= percent < high:
                        dist[(low, high)] += 1
                        break
                    elif percent >= high and high == 1.0:
                        dist[(1.0, float('inf'))] += 1
                        break
        return dist
    
    print("\n股东->股东持股比例分布:")
    s2s_dist = percent_distribution(shareholder_to_shareholder)
    for (low, high), count in sorted(s2s_dist.items()):
        if high == float('inf'):
            print(f'{low*100:.1f}%+: {count}个 ({count/len(shareholder_to_shareholder)*100:.2f}%)')
        else:
            print(f'{low*100:.1f}%-{high*100:.1f}%: {count}个 ({count/len(shareholder_to_shareholder)*100:.2f}%)')
    
    print("\n股东->partner持股比例分布:")
    s2p_dist = percent_distribution(shareholder_to_partner)
    for (low, high), count in sorted(s2p_dist.items()):
        if high == float('inf'):
            print(f'{low*100:.1f}%+: {count}个 ({count/len(shareholder_to_partner)*100:.2f}%)')
        else:
            print(f'{low*100:.1f}%-{high*100:.1f}%: {count}个 ({count/len(shareholder_to_partner)*100:.2f}%)')
    
    print("\npartner->股东持股比例分布:")
    p2s_dist = percent_distribution(partner_to_shareholder)
    for (low, high), count in sorted(p2s_dist.items()):
        if high == float('inf'):
            print(f'{low*100:.1f}%+: {count}个 ({count/len(partner_to_shareholder)*100:.2f}%)')
        else:
            print(f'{low*100:.1f}%-{high*100:.1f}%: {count}个 ({count/len(partner_to_shareholder)*100:.2f}%)')
    
    # 4. 分析持股金额分布
    def amount_examples(edges, n=3):
        """提取持股金额示例"""
        examples = []
        for u, v, data in edges:
            amount = data.get('amount', '')
            if amount:
                u_name = graph.nodes[u].get('name', u)
                v_name = graph.nodes[v].get('name', v)
                examples.append((u_name, v_name, amount, data.get('percent')))
                if len(examples) >= n:
                    break
        return examples
    
    print("\n股东->股东持股金额示例:")
    for u_name, v_name, amount, percent in amount_examples(shareholder_to_shareholder):
        print(f'从 {u_name} 到 {v_name} 的控股边, 金额: {amount}, 比例: {percent if percent is not None else "未知"}')
    
    print("\n股东->partner持股金额示例:")
    for u_name, v_name, amount, percent in amount_examples(shareholder_to_partner):
        print(f'从 {u_name} 到 {v_name} 的控股边, 金额: {amount}, 比例: {percent if percent is not None else "未知"}')
    
    print("\npartner->股东持股金额示例:")
    for u_name, v_name, amount, percent in amount_examples(partner_to_shareholder):
        print(f'从 {u_name} 到 {v_name} 的控股边, 金额: {amount}, 比例: {percent if percent is not None else "未知"}')
    
    # 5. 闭环中的控股边分析
    # 这里可以添加对闭环中控股边的特殊分析，例如在闭环中股东-股东控股边的特点
    
    print("\n分析完成！")

if __name__ == "__main__":
    main() 