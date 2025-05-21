import pandas as pd
import networkx as nx
import os
import sys
from datetime import datetime

# --- 配置 ---
SHAREHOLDER_GRAPH_PATH = "model/shareholder_graph.graphml"
TRANSACTION_CSV_PATH = "data/交易数据.csv" # 由 convert_excel_to_csv.py 生成
FINAL_GRAPH_PATH = "model/final_heterogeneous_graph.graphml"
LOG_FILE_PATH = "outputs/log/add_transaction.log"  # 日志文件路径

# 列名映射 (根据交易数据.csv的实际列名)
# 从之前的探索，我们知道列名是： 年份, 月份, 类型, 成员单位, 交易对象, 交易金额
COL_YEAR = "年份"
COL_MONTH = "月份"
COL_TYPE = "类型" # 包含 "Customer" 或 "Supplier"
COL_MEMBER_UNIT = "成员单位" # 核心成员供应商
COL_TRADING_PARTNER = "交易对象" # Customer 或 Supplier
COL_AMOUNT = "交易金额"

# 全局日志文件句柄
log_file = None

def log_setup():
    """设置日志文件"""
    global log_file
    # 确保日志目录存在
    log_dir = os.path.dirname(LOG_FILE_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    try:
        log_file = open(LOG_FILE_PATH, 'w', encoding='utf-8')
        log_message(f"日志开始记录: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", print_console=True)
        log_message(f"Python版本: {sys.version}", print_console=True)
        log_message("-" * 80, print_console=True)
    except Exception as e:
        print(f"无法创建日志文件: {e}")
        log_file = None

def log_message(message, print_console=False):
    """记录消息到日志文件和控制台(可选)"""
    if print_console:
        print(message)  # 打印到控制台
    
    # 写入日志文件
    global log_file
    if log_file:
        try:
            log_file.write(message + "\n")
            log_file.flush()  # 确保立即写入文件
        except Exception as e:
            print(f"写入日志失败: {e}")

def close_log():
    """关闭日志文件"""
    global log_file
    if log_file:
        try:
            log_message(f"日志记录结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", print_console=True)
            log_file.close()
        except Exception as e:
            print(f"关闭日志文件失败: {e}")

def load_graph(file_path):
    """从GraphML文件加载图。"""
    if not os.path.exists(file_path):
        log_message(f"错误：图文件未找到于 '{file_path}'", print_console=True)
        return None
    try:
        log_message(f"正在从 '{file_path}' 加载图...", print_console=True)
        graph = nx.read_graphml(file_path)
        log_message(f"图加载成功，包含 {graph.number_of_nodes()} 个节点和 {graph.number_of_edges()} 条边。", print_console=True)
        return graph
    except Exception as e:
        log_message(f"加载图 '{file_path}' 时发生错误: {e}", print_console=True)
        return None

def save_final_graph(graph, file_path):
    """将最终的图保存到GraphML文件，并处理None值。"""
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log_message(f"创建目录: {output_dir}", print_console=True)

        graph_copy = graph.copy()

        for _, data in graph_copy.nodes(data=True):
            for key, value in data.items():
                if value is None:
                    data[key] = ""
        
        for _, _, data in graph_copy.edges(data=True):
            for key, value in data.items():
                if value is None:
                    data[key] = ""

        nx.write_graphml(graph_copy, file_path)
        log_message(f"最终异构图已成功保存至 {file_path}", print_console=True)
    except Exception as e:
        log_message(f"保存最终图到 '{file_path}' 时发生错误: {e}", print_console=True)

def find_node_by_name(graph, name):
    """根据name属性查找节点ID"""
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('name') == name:
            return node_id
    return None

def add_transaction_data_to_graph(graph, transaction_csv_path):
    """
    读取交易数据CSV，将其信息添加到现有图中。
    - 添加/更新 "成员单位" 节点，标签为 "成员单位"。
    - 更新 "交易对象" 节点的标签为 "partner"。
    - 添加 "交易" 边及属性，包括nature属性表明关系本质（customer/supplier）。
    """
    if graph is None:
        log_message("错误：输入的图为空，无法添加交易数据。", print_console=True)
        return None

    try:
        log_message(f"正在从 '{transaction_csv_path}' 读取交易数据...", print_console=True)
        df_transactions = pd.read_csv(transaction_csv_path, dtype=str) # 先全部按字符串读取，后续转换
        log_message(f"成功读取 {len(df_transactions)} 条交易记录。", print_console=True)
    except FileNotFoundError:
        log_message(f"错误：交易数据CSV文件 '{transaction_csv_path}' 未找到。", print_console=True)
        return graph # 返回原图
    except Exception as e:
        log_message(f"读取交易数据CSV '{transaction_csv_path}' 时发生错误: {e}", print_console=True)
        return graph

    # 初始化计数器
    new_member_units_count = 0
    updated_partner_labels_count = 0
    new_transaction_edges_count = 0
    
    # 记录已标记为 "成员单位" 的节点，避免重复打印信息
    member_units_processed = set()
    
    # 记录节点标签变化情况
    label_changes = {}

    for _, row in df_transactions.iterrows():
        try:
            member_unit_name = str(row[COL_MEMBER_UNIT]).strip()
            trading_partner_name = str(row[COL_TRADING_PARTNER]).strip()
            transaction_type = str(row[COL_TYPE]).strip() # "Customer" 或 "Supplier"
            
            year = str(row[COL_YEAR]).strip()
            month = str(row[COL_MONTH]).strip()
            amount_str = str(row[COL_AMOUNT]).strip()
            amount = None
            try:
                if amount_str:
                    amount = float(amount_str)
            except ValueError:
                log_message(f"警告：交易金额 '{amount_str}' 无法转换为数字，行: {row.to_dict()}。此交易金额将设为None。")

            if not member_unit_name or not trading_partner_name:
                log_message(f"警告：成员单位或交易对象名称为空，跳过此行: {row.to_dict()}")
                continue

            # 1. 处理 "成员单位" 节点
            # 首先尝试根据name属性查找已存在的成员单位节点
            member_unit_node_id = find_node_by_name(graph, member_unit_name)
            
            if member_unit_node_id is None:
                # 如果未找到已存在的节点，则使用名称作为节点ID创建新节点
                member_unit_node_id = member_unit_name
                graph.add_node(member_unit_node_id, name=member_unit_name, label="成员单位", type="企业")
                if member_unit_node_id not in member_units_processed:
                    new_member_units_count += 1
                    member_units_processed.add(member_unit_node_id)
                    log_message(f"信息：成员单位 '{member_unit_name}' 在原图中不存在，作为新节点添加。")
            else:
                # 如果节点已存在，确保其标签是 "成员单位"
                old_label = graph.nodes[member_unit_node_id].get('label')
                if old_label != "成员单位":
                    log_message(f"信息：已存在的节点 '{member_unit_node_id}' (name={member_unit_name}) 更新标签为 '成员单位'。")
                graph.nodes[member_unit_node_id]['label'] = "成员单位"
                # 确保name属性存在并正确
                graph.nodes[member_unit_node_id]['name'] = member_unit_name
                if 'type' not in graph.nodes[member_unit_node_id] or not graph.nodes[member_unit_node_id]['type']:
                    graph.nodes[member_unit_node_id]['type'] = "企业" # 补充类型

            # 2. 处理 "交易对象" 节点，统一标签为 "partner"
            # 首先尝试根据name属性查找已存在的交易对象节点
            trading_partner_node_id = find_node_by_name(graph, trading_partner_name)
            
            relation_nature = ""
            if transaction_type.lower() == "customer":
                relation_nature = "customer"
            elif transaction_type.lower() == "supplier":
                relation_nature = "supplier"
            else:
                log_message(f"警告：未知的交易类型 '{transaction_type}' ({type(transaction_type)})，无法为交易对象 '{trading_partner_name}' 设置关系性质。")
            
            if relation_nature:
                if trading_partner_node_id is None:
                    # 如果未找到已存在的节点，则使用名称作为节点ID创建新节点
                    trading_partner_node_id = trading_partner_name
                    graph.add_node(trading_partner_node_id, name=trading_partner_name, label="partner", type="企业")
                    updated_partner_labels_count += 1
                    log_message(f"信息：交易对象 '{trading_partner_name}' 在原图中不存在，作为新节点添加并标记为 'partner'。")
                else:
                    # 如果节点已存在，更新其标签为 "partner"
                    old_label = graph.nodes[trading_partner_node_id].get('label')
                    if old_label != "partner" and old_label not in ["成员单位", "股东"]:
                        # 记录标签变化
                        if trading_partner_name not in label_changes:
                            label_changes[trading_partner_name] = set()
                        label_changes[trading_partner_name].add((old_label, "partner"))
                        
                        log_message(f"信息：已存在的节点 '{trading_partner_node_id}' (name={trading_partner_name}) 更新标签为 'partner'。")
                        updated_partner_labels_count += 1
                    
                    if old_label not in ["成员单位", "股东"]:
                        graph.nodes[trading_partner_node_id]['label'] = "partner"
                    # 确保name属性存在并正确
                    graph.nodes[trading_partner_node_id]['name'] = trading_partner_name
                    if 'type' not in graph.nodes[trading_partner_node_id] or not graph.nodes[trading_partner_node_id]['type']:
                        graph.nodes[trading_partner_node_id]['type'] = "企业" # 补充类型

            # 3. 添加 "交易" 边
            # 假设交易方向：
            # 如果交易对象是 Customer, 边: 成员单位 -> 交易对象 (成员单位卖给客户)
            # 如果交易对象是 Supplier, 边: 交易对象 -> 成员单位 (供应商卖给成员单位)
            
            edge_attrs = {
                'label': "交易",
                'nature': relation_nature,  # 新增nature字段，记录关系本质
                'year': year,
                'month': month,
                'amount': amount if amount is not None else "" # 确保是字符串或数字，而不是None
            }

            u_node, v_node = None, None
            if relation_nature == "customer": # 成员单位 -> 客户
                u_node, v_node = member_unit_node_id, trading_partner_node_id
            elif relation_nature == "supplier": # 供应商 -> 成员单位
                u_node, v_node = trading_partner_node_id, member_unit_node_id
            
            if u_node and v_node:
                # NetworkX MultiDiGraph允许相同节点间有多个边，只要它们的key不同或属性不同。
                # 为了简单起见，如果只关心是否存在交易关系，可以先检查边是否存在。
                # 但由于交易有年月金额，同一对实体间可能有多笔交易，所以允许多重边是合理的。
                # add_edge 会自动处理多重边，每次都是一条新的边。
                graph.add_edge(u_node, v_node, **edge_attrs)
                new_transaction_edges_count += 1
            else:
                if relation_nature: # 仅当关系性质有效时才警告，否则上面已有类型警告
                    log_message(f"警告：由于交易类型 '{transaction_type}' 或节点问题，未能在 '{member_unit_name}' 和 '{trading_partner_name}' 之间添加交易边。")

        except Exception as e_row:
            log_message(f"处理行数据 '{row.to_dict()}' 时发生错误: {e_row}。跳过此行。")
            continue
            
    log_message("\n--- 数据整合统计 ---", print_console=True)
    log_message(f"新增的独立'成员单位'节点数量: {new_member_units_count}", print_console=True)
    log_message(f"更新/新增为'partner'标签的节点数量: {updated_partner_labels_count}", print_console=True)
    log_message(f"新增的'交易'边数量: {new_transaction_edges_count}", print_console=True)
    
    # 分析标签变化情况
    if label_changes:
        log_message("\n--- 节点标签变化分析 ---", print_console=True)
        log_message("以下节点的标签在处理过程中发生了变化(表明公司既是上游又是下游):", print_console=True)
        for node_name, changes in label_changes.items():
            if len(changes) > 1:  # 多次变化
                changes_str = ", ".join([f"'{old}'-->'{new}'" for old, new in changes])
                log_message(f"  公司: {node_name}, 标签变化: {changes_str}", print_console=True)
            elif list(changes)[0][0] and list(changes)[0][0] != list(changes)[0][1]:  # 有意义的变化
                old, new = list(changes)[0]
                log_message(f"  公司: {node_name}, 标签变化: '{old}'--->'{new}'", print_console=True)
    
    # 统计各种节点类型
    node_types = {}
    for node, attrs in graph.nodes(data=True):
        label = attrs.get('label', '数据缺失')  # 将未知改为数据缺失
        if label == '未知':
            attrs['label'] = '数据缺失'  # 修改节点标签
            label = '数据缺失'
        node_types[label] = node_types.get(label, 0) + 1
    
    log_message("\n--- 节点类型统计 ---", print_console=True)
    for label, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        log_message(f"节点类型 '{label}': {count} 个", print_console=True)
        
        # 对于"数据缺失"类型的节点，列出部分示例以便分析
        if label == '数据缺失' and count > 0:
            log_message("\n数据缺失类型节点的示例:", print_console=True)
            unknown_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get('label') == '数据缺失']
            for i, node in enumerate(unknown_nodes[:min(5, len(unknown_nodes))]):
                attrs = graph.nodes[node]
                log_message(f"  示例 {i+1}: ID={node}, 属性={attrs}", print_console=True)
    
    return graph


if __name__ == "__main__":
    # 设置日志
    log_setup()
    
    log_message("开始将交易数据整合到股东图中...", print_console=True)
    
    # 1. 加载股东图
    shareholder_graph = load_graph(SHAREHOLDER_GRAPH_PATH)
    
    if shareholder_graph:
        # 2. 添加交易数据
        final_graph = add_transaction_data_to_graph(shareholder_graph, TRANSACTION_CSV_PATH)
        
        if final_graph:
            # 3. 保存最终的异构图
            save_final_graph(final_graph, FINAL_GRAPH_PATH)
            log_message(f"\n最终图包含 {final_graph.number_of_nodes()} 个节点和 {final_graph.number_of_edges()} 条边。", print_console=True)
    else:
        log_message("未能加载股东图，处理中止。", print_console=True)
        
    log_message("\nadd_transaction.py 处理完成。", print_console=True)
    
    # 关闭日志
    close_log() 