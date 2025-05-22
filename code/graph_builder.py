# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx
import json
import ast
import os # Added for path operations

# 新增辅助函数：规范化百分比数据
def _normalize_percent(value):
    """
    将各种格式的百分比值规范化为0.0到1.0之间的小数。
    如果无法解析，则返回 None。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if 0 <= value <= 1: # 假设已经是0-1之间的小数
            return float(value)
        elif 1 < value <= 100: # 假设是1-100之间的百分比数字
             return float(value) / 100.0
        else: # 其他范围的数字，可能代表错误或特殊含义，暂定为None
            # print(f"GraphBuilder Warn: Numeric percent value {value} is out of expected range (0-1 or 1-100), treating as None.")
            return None 
    
    if isinstance(value, str):
        original_value_str = value # 保存原始字符串用于可能的警告
        value = value.strip()
        if not value: # 空字符串
            return None
        
        has_percent_symbol = value.endswith('%')
        if has_percent_symbol:
            value = value[:-1].strip() # 去掉百分号

        try:
            num_val = float(value)
            if has_percent_symbol: # 如果原始有百分号，直接除以100
                return num_val / 100.0
            else: # 如果原始没有百分号
                if 0 <= num_val <= 1: # 已经是0-1的小数
                    return num_val
                elif 1 < num_val <= 100: # 是1-100的数字，当作百分比
                    return num_val / 100.0
                else: # 其他范围，视为无效
                    # print(f"GraphBuilder Warn: String percent value '{original_value_str}' (parsed as {num_val}) is out of expected range, treating as None.")
                    return None
        except ValueError:
            # print(f"GraphBuilder Warn: Could not convert percent string '{original_value_str}' to float, treating as None.")
            return None # 无法转换为浮点数
    
    # print(f"GraphBuilder Warn: Unhandled percent value type: {value} (type: {type(value)}), treating as None.")
    return None # 其他未处理的类型

# 新增函数：检查是否应该添加边
def _should_add_edge(graph, source, target, edge_attrs):
    """
    检查是否应该在给定的source和target之间添加边。
    规则简化为：
    1. 每个方向最多只能有一条边
    2. 两个节点之间最多只有两条边（一条A->B，一条B->A）
    """
    # 检查source->target方向是否已有边
    forward_edges = graph.get_edge_data(source, target)
    if forward_edges:  # 如果已经存在从source到target的边
        return False  # 同方向已有边，不再添加
    
    # source->target方向没有边，可以添加
    return True

# 辅助函数，用于递归解析children字段并添加节点和边
def _parse_children_recursive(main_row_entity_id, children_data, graph, level_0_ids): # Added level_0_ids
    if isinstance(children_data, str):
        try:
            # 首先尝试使用 ast.literal_eval，它可以更安全地处理包含单引号的 Python 字面量
            children_list = ast.literal_eval(children_data)
            if not isinstance(children_list, list): # 确保结果是列表
                # 如果解析结果不是列表（例如，如果字符串只是一个字典），将其包装在列表中
                # 或者根据您的数据结构决定如何处理这种情况。这里假设我们总是期望一个子节点列表。
                print(f"GraphBuilder Warn: ast.literal_eval on children_data for {main_row_entity_id} did not return a list. Data: {children_data[:100]}") # Log a snippet
                # 根据实际情况，你可能需要将其视为错误并返回，或尝试其他解析。
                # 为了与后续逻辑兼容，如果不是list，尝试将其包装成list或者进行json解析的回退
                raise ValueError("ast.literal_eval did not result in a list.")
        except (ValueError, SyntaxError) as e_ast: # ast.literal_eval 失败
            # print(f"GraphBuilder Info: ast.literal_eval failed for '{children_data[:100]}...' for main entity '{main_row_entity_id}'. Error: {e_ast}. Falling back to JSON parsing.")
            try:
                # 替换策略：
                # 1. 将 CSV 中的 \\' (代表实际的单引号) 替换为特殊占位符
                # 2. 将结构性的 ' 替换为 "
                # 3. 将占位符替换回 JSON 字符串中合法的 '
                processed_str = children_data.replace("\\\\\'", "__TEMP_SINGLE_QUOTE__") 
                processed_str = processed_str.replace("'", '"')
                processed_str = processed_str.replace("__TEMP_SINGLE_QUOTE__", "'") 
                children_list = json.loads(processed_str)
            except json.JSONDecodeError as e_json_primary:
                try:
                    # 备用：简单替换，如果上面的方法因为复杂引号失败
                    children_list = json.loads(children_data.replace("'", '"'))
                except json.JSONDecodeError as e_json_fallback:
                    print(f"GraphBuilder Warn: Could not parse children JSON string '{children_data}' for main entity '{main_row_entity_id}'. AST error: {e_ast}, Primary JSON error: {e_json_primary}. Fallback JSON error: {e_json_fallback}")
                    return
            except Exception as e_general:
                print(f"GraphBuilder Warn: Unknown error parsing children string '{children_data}' for main entity '{main_row_entity_id}' after AST eval. Error: {e_general}")
                return
    elif isinstance(children_data, list):
        children_list = children_data
    else:
        print(f"GraphBuilder Warn: Children data is not a string or list for main entity '{main_row_entity_id}'. Type: {type(children_data)}")
        return

    for child_info_from_json in children_list:
        shareholder_name = child_info_from_json.get('name')
        shareholder_eid = child_info_from_json.get('eid')
        
        if not pd.notna(shareholder_name) or shareholder_name == '':
            # print(f"GraphBuilder Info: Skipping child with no name from children list of {main_row_entity_id}.")
            continue

        shareholder_node_id = shareholder_eid if pd.notna(shareholder_eid) and shareholder_eid != '' else shareholder_name
        
        # 新增逻辑：为children中的非level0节点添加 '股东' 标签
        # level_0_ids 包含所有主CSV中 level=0 的公司的ID
        is_level_0_company_in_children = shareholder_node_id in level_0_ids
        
        if not is_level_0_company_in_children:
            node_attrs = {
                'name': shareholder_name,
                'type': child_info_from_json.get('type', ''),
                'short_name': child_info_from_json.get('short_name', ''),
                'level': child_info_from_json.get('level', ''),
                'label': '股东'
            }
        else:
            node_attrs = {
                'name': shareholder_name,
                'type': child_info_from_json.get('type', ''),
                'short_name': child_info_from_json.get('short_name', ''),
                'level': child_info_from_json.get('level', ''),
            }

        if not graph.has_node(shareholder_node_id):
            graph.add_node(shareholder_node_id, **node_attrs)
        else:
            # 如果节点已存在，用children中的信息补充或更新
            for attr, value in node_attrs.items():
                 # Do not overwrite existing label if it's already set (e.g. from a previous pass or if it was a level 0 company)
                 if attr == 'label' and graph.nodes[shareholder_node_id].get('label'):
                     continue 
                 if value or not graph.nodes[shareholder_node_id].get(attr):
                    graph.nodes[shareholder_node_id][attr] = value
            # Ensure 'label' is applied if it was determined above and node already existed
            if 'label' in node_attrs and not graph.nodes[shareholder_node_id].get('label'):
                 graph.nodes[shareholder_node_id]['label'] = node_attrs['label']

        # 添加从股东 (shareholder_node_id from JSON) 到被投资公司 (main_row_entity_id) 的边
        edge_attrs = {
            'amount': child_info_from_json.get('amount', ''),
            'percent': _normalize_percent(child_info_from_json.get('percent')), # 使用规范化函数
            'sh_type': child_info_from_json.get('sh_type', ''),
            'source_info': 'children_field',
            'label': '控股' # 新增：为边添加 '控股' 标签
        }
        
        # 修改：使用_should_add_edge函数检查是否应该添加边
        if _should_add_edge(graph, shareholder_node_id, main_row_entity_id, edge_attrs):
            graph.add_edge(shareholder_node_id, main_row_entity_id, **edge_attrs)

        # 递归处理孙子节点 (即当前股东的股东)
        # main_row_entity_id for the recursive call will be the current shareholder_node_id
        # grand_children_data will be the 'children' field of the current shareholder_node_id (if any)
        grand_children_data = child_info_from_json.get('children')
        if grand_children_data:
            _parse_children_recursive(shareholder_node_id, grand_children_data, graph, level_0_ids) # Pass level_0_ids

def build_graph(csv_path='data/三层股权穿透输出数据.csv'):
    """
    从指定的CSV文件读取股权数据并构建一个NetworkX MultiDiGraph。
    支持真正的多边特性，允许在相同节点对之间存在多条边。
    """
    encodings_to_try = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'gb2312', 'big5']
    df = None
    read_successful = False

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=encoding, dtype=str) # 读取所有列为字符串以保留原始格式
            # print(f"GraphBuilder: Successfully read CSV '{csv_path}' with encoding: {encoding}") # Commented out
            read_successful = True
            break
        except UnicodeDecodeError:
            # print(f"GraphBuilder: Failed to decode CSV '{csv_path}' with encoding: {encoding}") # Commented out
            pass # Continue to the next encoding
        except pd.errors.EmptyDataError:
            print(f"GraphBuilder Warn: CSV file '{csv_path}' is empty or could not be read with encoding {encoding}.")
            # 如果文件就是空的，不应该继续尝试其他编码或报错，而是返回空图或相应处理
            G = nx.MultiDiGraph()
            print(f"GraphBuilder: Returning empty graph due to empty or unreadable CSV: {csv_path}")
            return G
        except Exception as e:
            print(f"GraphBuilder: An unexpected error occurred while reading '{csv_path}' with {encoding}: {e}")
            # 对于其他pandas读取错误或一般错误，也记录并尝试下一种编码
    
    if not read_successful or df is None:
        print(f"GraphBuilder Error: Could not read CSV file '{csv_path}' with any of the attempted encodings.")
        # 可以选择抛出异常或者返回一个空图，这里选择后者以便调用方可以处理
        G = nx.MultiDiGraph()
        print(f"GraphBuilder: Returning empty graph as CSV could not be loaded: {csv_path}")
        return G
    
    # Check if DataFrame is empty after successful read (e.g. header only or all rows filtered out previously)
    if df.empty:
        print(f"GraphBuilder Warn: CSV file '{csv_path}' was read successfully but resulted in an empty DataFrame.")
        G = nx.MultiDiGraph()
        print(f"GraphBuilder: Returning empty graph due to empty DataFrame from: {csv_path}")
        return G

    G = nx.MultiDiGraph()

    # 搜集所有 level 0 公司的ID (eid 或 name)
    level_0_ids = set()
    for _, row in df.iterrows():
        if pd.notna(row.get('level')) and str(row.get('level')).strip() == '0':
            level_0_node_id = row['eid'] if pd.notna(row['eid']) and row['eid'] != '' else row['name']
            if pd.notna(level_0_node_id) and level_0_node_id != '':
                 level_0_ids.add(level_0_node_id)
    print(f"GraphBuilder: Found {len(level_0_ids)} unique level 0 entities.")

    # 第一遍：添加所有在主行中定义了name的节点，并建立基于parent_id的边
    # Rule: Child (current_node_id) -> Parent (parent_id_val)
    for _, row in df.iterrows():
        if pd.notna(row['name']):
            current_node_id = row['eid'] if pd.notna(row['eid']) and row['eid'] != '' else row['name']
            
            node_attrs = {
                'name': row['name'],
                'type': row['type'] if pd.notna(row['type']) else '',
                'short_name': row['short_name'] if pd.notna(row['short_name']) else '',
                'level': row['level'] if pd.notna(row['level']) else '' # level is from the main row
            }
            # Level 0 公司不在这里加 '股东' 标签
            if not G.has_node(current_node_id):
                G.add_node(current_node_id, **node_attrs)
            else: 
                for attr, value in node_attrs.items():
                    G.nodes[current_node_id][attr] = value 

            parent_id_val = row.get('parent_id')
            if pd.notna(parent_id_val) and parent_id_val != '':
                # 如果父节点不存在，暂时不创建，期望它有自己的主行数据
                edge_attrs = {
                    'amount': row['amount'] if pd.notna(row['amount']) else '',
                    'percent': _normalize_percent(row.get('percent')), # 使用规范化函数
                    'sh_type': row['sh_type'] if pd.notna(row['sh_type']) else '',
                    'source_info': 'parent_id_field',
                    'label': '控股' # 新增：为边添加 '控股' 标签
                }
                # If parent_id_val node doesn't exist, it will be created by add_edge
                if not G.has_node(parent_id_val):
                     G.add_node(parent_id_val, name=str(parent_id_val)) # Simplified name for now
                     # 如果parent_id_val不在level_0_ids中，且现在才被创建，它应该被视为股东
                     if parent_id_val not in level_0_ids:
                         G.nodes[parent_id_val]['label'] = '股东'

                # MODIFIED EDGE DIRECTION HERE: current_node_id (Child) -> parent_id_val (Parent)
                # 修改：使用_should_add_edge函数检查是否应该添加边
                if _should_add_edge(G, current_node_id, parent_id_val, edge_attrs):
                    G.add_edge(current_node_id, parent_id_val, **edge_attrs)
    
    # 第二遍：处理children字段，补充可能的节点和边
    # shareholder_in_children_json -> current_node_id
    for _, row in df.iterrows():
        if pd.notna(row['name']):
            current_node_id = row['eid'] if pd.notna(row['eid']) and row['eid'] != '' else row['name']
            children_json_str = row.get('children')
            if pd.notna(children_json_str) and children_json_str not in [[], '[]', '']:
                # 确保父节点（current_node_id）在图中，如果它只有children而没有parent_id，第一遍可能没覆盖到
                if not G.has_node(current_node_id):
                     G.add_node(current_node_id, name=row['name'], type=row.get('type', ''), short_name=row.get('short_name', ''), level=row.get('level', ''))
                _parse_children_recursive(current_node_id, children_json_str, G, level_0_ids) # Pass level_0_ids
    
    # 统计平行边数据
    parallel_edges_count = 0       # 平行边总数
    nodes_with_parallel_edges = 0  # 有平行边的节点对数量
    max_parallel_edges = 0
    parallel_edge_pairs = {}
    
    for u, v in G.edges():
        edge_count = G.number_of_edges(u, v)
        if edge_count > 1:
            if (u, v) not in parallel_edge_pairs:
                parallel_edge_pairs[(u, v)] = edge_count
                nodes_with_parallel_edges += 1
                parallel_edges_count += edge_count  # 累计所有平行边
            max_parallel_edges = max(max_parallel_edges, edge_count)
    
    print(f"GraphBuilder: Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"GraphBuilder: Found {nodes_with_parallel_edges} node pairs with parallel edges.")
    print(f"GraphBuilder: Total parallel edges: {parallel_edges_count} (including all edges between these node pairs).")
    print(f"GraphBuilder: Maximum parallel edges between any node pair: {max_parallel_edges}")
    
    # 显示一些平行边示例
    if parallel_edge_pairs:
        print("\nGraphBuilder: Examples of parallel edges:")
        shown = 0
        for (u, v), count in sorted(parallel_edge_pairs.items(), key=lambda x: x[1], reverse=True):
            if shown < 5:  # 只显示前5个示例
                u_name = G.nodes[u].get('name', u)
                v_name = G.nodes[v].get('name', v)
                u_type = G.nodes[u].get('label', '未知')
                v_type = G.nodes[v].get('label', '未知')
                print(f"  {u_name} [{u_type}] -> {v_name} [{v_type}]: {count} edges")
                shown += 1
    
    # 新增：打印有两条边的节点对的详细边属性
    if parallel_edge_pairs:
        print("\nGraphBuilder: 详细显示有两条边的节点对及其边属性:")
        shown_pairs = 0
        for (u, v), count in sorted(parallel_edge_pairs.items(), key=lambda x: x[1], reverse=True):
            if count == 2 and shown_pairs < 10:  # 只显示恰好有两条边的节点对，最多10个
                u_name = G.nodes[u].get('name', u)
                v_name = G.nodes[v].get('name', v)
                u_type = G.nodes[u].get('label', '未知')
                v_type = G.nodes[v].get('label', '未知')
                
                print(f"\n节点对 {shown_pairs+1}: {u_name} [{u_type}] -> {v_name} [{v_type}]")
                
                # 获取两个节点之间的所有边
                edges_data = G.get_edge_data(u, v)
                for key, data in edges_data.items():
                    percent_value = data.get('percent', '未知')
                    if isinstance(percent_value, float):
                        percent_display = f"{percent_value*100:.2f}%"
                    else:
                        percent_display = str(percent_value)
                    
                    print(f"  边 {key}: 百分比={percent_display}, 金额={data.get('amount', '未知')}, 来源={data.get('source_info', '未知')}")
                
                shown_pairs += 1
    
    # 统计带标签的节点和边
    shareholder_nodes_count = 0
    for node_id, attrs in G.nodes(data=True):
        if attrs.get('label') == '股东':
            shareholder_nodes_count += 1
    
    holding_edges_count = 0
    for u, v, attrs in G.edges(data=True):
        if attrs.get('label') == '控股':
            holding_edges_count += 1
            
    print(f"GraphBuilder: Number of nodes labeled '股东': {shareholder_nodes_count}")
    print(f"GraphBuilder: Number of edges labeled '控股': {holding_edges_count}")
    
    return G

def save_graph(graph, file_path):
    """
    将NetworkX图保存到指定的文件路径 (GraphML格式)。
    在保存前处理None值，并确保输出目录存在。
    """
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"GraphBuilder: Created directory: {output_dir}")

        graph_copy = graph.copy() # 操作副本

        for _, data in graph_copy.nodes(data=True):
            for key, value in data.items():
                if value is None:
                    data[key] = "" 
        
        for _, _, data in graph_copy.edges(data=True):
            for key, value in data.items():
                if value is None:
                    data[key] = ""

        nx.write_graphml(graph_copy, file_path)
        print(f"GraphBuilder: Graph successfully saved to {file_path}")
    except Exception as e:
        print(f"GraphBuilder Error: Could not save graph to {file_path}. Error: {e}")

if __name__ == '__main__':
    # 构建图并保存
    print("\n===== 构建多边图(MultiDiGraph) =====")
    graph = build_graph("data/三层股权穿透输出数据_1.csv")
    
    # 确保输出目录存在
    os.makedirs('model', exist_ok=True)
    
    # 保存图
    save_graph(graph, "model/shareholder_graph.graphml")

    print(f"多边图已保存到 model/shareholder_graph.graphml") 