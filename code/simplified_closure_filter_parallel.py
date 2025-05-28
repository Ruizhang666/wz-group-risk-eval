#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化闭环筛选脚本 - 多核并行优化版本
只判断指定时间窗口内是否存在闭环，不匹配具体交易对
"""

import pandas as pd
import ast
import os
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import argparse
import logging
import multiprocessing as mp
from multiprocessing import Pool
import time
from tqdm import tqdm
import gc

def setup_logging():
    """设置日志"""
    log_dir = "outputs/log"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/simplified_closure_filter_parallel.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_transaction_times(time_str):
    """解析交易时间字符串为日期列表"""
    try:
        if pd.isna(time_str) or time_str == '[]':
            return []
        
        if isinstance(time_str, str):
            time_str = time_str.strip().strip('"\'')
            if time_str.startswith('[') and time_str.endswith(']'):
                time_list = ast.literal_eval(time_str)
            else:
                time_list = [t.strip().strip('"\'') for t in time_str.split(',')]
        else:
            time_list = time_str
        
        dates = []
        for date_str in time_list:
            if date_str and date_str.strip():
                try:
                    date_obj = datetime.strptime(date_str.strip().strip('"\''), '%Y-%m-%d')
                    dates.append(date_obj)
                except ValueError:
                    continue
        
        return sorted(dates)
    
    except Exception:
        return []

def check_closure_exists(upstream_times, downstream_times, n_months=None, n_days=None):
    """
    简化的闭环检查：只判断时间窗口内是否同时存在上游和下游交易
    不尝试匹配具体的交易对
    """
    if not upstream_times or not downstream_times:
        return {
            'has_closure': False,
            'reason': 'missing_transactions',
            'total_upstream': len(upstream_times),
            'total_downstream': len(downstream_times)
        }
    
    # 设置默认值
    if n_months is None and n_days is None:
        n_months = 6
    
    # 找到最早的上游交易和最晚的下游交易
    earliest_upstream = min(upstream_times)
    latest_downstream = max(downstream_times)
    
    # 如果下游交易都在上游交易之前，肯定没有闭环
    if latest_downstream < earliest_upstream:
        return {
            'has_closure': False,
            'reason': 'downstream_before_upstream',
            'total_upstream': len(upstream_times),
            'total_downstream': len(downstream_times)
        }
    
    # 检查时间窗口内是否存在闭环可能性
    for upstream_time in upstream_times:
        # 计算该上游交易的时间窗口
        if n_months is not None:
            window_end = upstream_time + relativedelta(months=n_months)
        else:  # n_days is not None
            window_end = upstream_time + timedelta(days=n_days)
        
        # 检查是否有下游交易在这个时间窗口内
        for downstream_time in downstream_times:
            if upstream_time <= downstream_time <= window_end:
                return {
                    'has_closure': True,
                    'reason': 'closure_found',
                    'total_upstream': len(upstream_times),
                    'total_downstream': len(downstream_times),
                    'earliest_upstream': earliest_upstream.strftime('%Y-%m-%d'),
                    'latest_downstream': latest_downstream.strftime('%Y-%m-%d')
                }
    
    # 没有发现闭环
    return {
        'has_closure': False,
        'reason': 'no_closure_in_timeframe',
        'total_upstream': len(upstream_times),
        'total_downstream': len(downstream_times),
        'earliest_upstream': earliest_upstream.strftime('%Y-%m-%d'),
        'latest_downstream': latest_downstream.strftime('%Y-%m-%d')
    }

def extract_node_count(type_str):
    """从type字符串中提取节点数量"""
    try:
        if pd.isna(type_str) or not isinstance(type_str, str):
            return None
        
        import re
        match = re.search(r'(\d+)节点环路', type_str)
        if match:
            return int(match.group(1))
        return None
    except Exception:
        return None

def analyze_loop_source_type(loop_content, source_name=None):
    """分析环路源头类型"""
    import re
    
    try:
        if pd.isna(loop_content) or not isinstance(loop_content, str):
            return {
                'has_natural_person_source': False,
                'source_type': 'enterprise',
                'source_entities': []
            }
        
        # 提取所有标记为[股东]的实体
        shareholder_pattern = r'([^[\]]+)\s*\[股东\]'
        shareholders = re.findall(shareholder_pattern, loop_content)
        
        # 判断是否包含自然人特征
        natural_person_indicators = [
            r'^[一-龥]{2,4}$',  # 2-4个中文字符的姓名
            r'^[A-Z]股$',  # A股、B股等
            r'.*[先生|女士|总|董事长|总经理|主席]$',  # 包含职务的个人
        ]
        
        has_natural_person = False
        source_entities = []
        
        for shareholder in shareholders:
            shareholder = shareholder.strip()
            source_entities.append(shareholder)
            
            for pattern in natural_person_indicators:
                if re.match(pattern, shareholder):
                    has_natural_person = True
                    break
        
        source_type = 'natural_person' if has_natural_person else 'enterprise'
        
        return {
            'has_natural_person_source': has_natural_person,
            'source_type': source_type,
            'source_entities': source_entities
        }
        
    except Exception:
        return {
            'has_natural_person_source': False,
            'source_type': 'enterprise',
            'source_entities': []
        }

def process_loop_batch(args):
    """处理一批环路数据"""
    loop_batch, basic_info_dict, n_months, n_days, max_nodes, source_type = args
    
    filtered_results = []
    
    for _, row in loop_batch.iterrows():
        loop_id = row.get('loop_id', row.name)
        
        # 首先检查节点数量条件和源头类型条件
        if max_nodes is not None or source_type is not None:
            if loop_id in basic_info_dict:
                basic_info = basic_info_dict[loop_id]
                
                # 检查节点数量
                if max_nodes is not None:
                    type_str = basic_info.get('type', '')
                    node_count = extract_node_count(type_str)
                    if node_count is not None and node_count > max_nodes:
                        continue
                
                # 检查源头类型
                if source_type is not None:
                    content_str = basic_info.get('content', '')
                    source_name = basic_info.get('source', '')
                    source_analysis = analyze_loop_source_type(content_str, source_name)
                    
                    if source_type == 'natural_person' and not source_analysis['has_natural_person_source']:
                        continue
                    elif source_type == 'enterprise' and source_analysis['has_natural_person_source']:
                        continue
        
        # 解析交易时间
        upstream_times = parse_transaction_times(row['upstream_to_member_transaction_times'])
        downstream_times = parse_transaction_times(row['member_to_downstream_transaction_times'])
        
        # 简化的闭环检查
        closure_result = check_closure_exists(upstream_times, downstream_times, n_months, n_days)
        
        # 如果存在闭环，添加到结果中
        if closure_result['has_closure']:
            # 添加原始数据
            result_row = row.copy()
            
            # 从基础信息中获取环路详情
            if loop_id in basic_info_dict:
                basic_info = basic_info_dict[loop_id]
                result_row['source'] = basic_info.get('source', '')
                result_row['content'] = basic_info.get('content', '')
                result_row['type'] = basic_info.get('type', '')
            
            # 添加闭环检查结果
            result_row['closure_reason'] = closure_result['reason']
            if 'earliest_upstream' in closure_result:
                result_row['earliest_upstream'] = closure_result['earliest_upstream']
                result_row['latest_downstream'] = closure_result['latest_downstream']
            
            filtered_results.append(result_row)
    
    return filtered_results

def create_batches(df, n_cores):
    """根据核心数动态分配批次"""
    total_loops = len(df)
    
    # 计算最优批次大小：总数据量 / 核心数，确保每个核心有足够的工作
    base_batch_size = max(100, total_loops // (n_cores * 4))  # 每个核心处理4批，最少100个
    
    # 考虑内存限制，设置最大批次大小
    max_batch_size = min(5000, total_loops // n_cores if n_cores > 0 else total_loops)
    
    # 最终批次大小
    batch_size = min(base_batch_size, max_batch_size)
    
    print(f"数据分配策略: 总数据 {total_loops:,} 个环路，{n_cores} 个进程，批次大小 {batch_size}")
    
    batches = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        batches.append(batch)
    
    return batches

def filter_loops_by_closure_parallel(input_file, output_file, n_months=None, n_days=None, max_nodes=None, source_type=None, n_cores=None):
    """根据闭环条件并行筛选环路"""
    logger = logging.getLogger(__name__)
    
    # 确定使用的核心数
    if n_cores is None:
        n_cores = max(1, mp.cpu_count() - 1)
    
    # 构建筛选条件描述
    time_condition = ""
    if n_months is not None:
        time_condition = f"{n_months}个月内"
    elif n_days is not None:
        time_condition = f"{n_days}天内"
    else:
        time_condition = "6个月内（默认）"
    
    node_condition = f"，节点数≤{max_nodes}" if max_nodes is not None else ""
    source_condition = ""
    if source_type == 'natural_person':
        source_condition = "，自然人源头"
    elif source_type == 'enterprise':
        source_condition = "，企业源头"
    
    condition_desc = time_condition + node_condition + source_condition
    
    logger.info(f"开始并行简化闭环筛选，筛选条件: {condition_desc}")
    logger.info(f"使用 {n_cores} 个进程进行并行处理")
    
    try:
        # 读取数据
        logger.info(f"读取数据文件: {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"总共读取 {len(df)} 个环路")
        
        # 读取基础环路信息
        basic_info_file = input_file.replace('loop_metrics.csv', 'loop_basic_info.csv')
        basic_info_dict = {}
        
        if os.path.exists(basic_info_file):
            logger.info(f"读取基础环路信息: {basic_info_file}")
            basic_info_df = pd.read_csv(basic_info_file)
            logger.info(f"读取到 {len(basic_info_df)} 个环路的基础信息")
            
            # 转换为字典便于快速查找
            for _, row in basic_info_df.iterrows():
                basic_info_dict[row['loop_id']] = row.to_dict()
        else:
            logger.warning(f"基础环路信息文件不存在: {basic_info_file}")
        
        # 动态分配批次
        logger.info("创建处理批次...")
        batches = create_batches(df, n_cores)
        logger.info(f"共创建 {len(batches)} 个批次")
        
        # 准备参数
        batch_args = [
            (batch, basic_info_dict, n_months, n_days, max_nodes, source_type)
            for batch in batches
        ]
        
        # 启动多进程处理
        logger.info("启动多进程筛选...")
        start_time = time.time()
        
        all_filtered_results = []
        
        with Pool(n_cores) as pool:
            # 使用tqdm显示进度条，改进为实时更新
            with tqdm(total=len(df), desc="筛选环路", unit="环路") as pbar:
                # 使用imap实现平滑的进度更新
                batch_count = 0
                for batch_filtered in pool.imap(process_loop_batch, batch_args):
                    batch_count += 1
                    all_filtered_results.extend(batch_filtered)
                    # 计算当前批次处理的数据量
                    current_batch_size = len(batch_args[batch_count-1][0])
                    pbar.update(current_batch_size)
                    # 更新描述显示当前批次信息
                    pbar.set_description(f"筛选环路 (批次 {batch_count}/{len(batches)})")
                
                # 确保进度条显示完成状态
                pbar.set_description("筛选环路 (已完成)")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"并行筛选完成，耗时: {processing_time:.2f} 秒")
        logger.info(f"平均每个环路处理时间: {processing_time/len(df)*1000:.2f} 毫秒")
        
        # 保存筛选后的环路
        if all_filtered_results:
            filtered_df = pd.DataFrame(all_filtered_results)
            filtered_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"筛选后的环路已保存到: {output_file}")
            logger.info(f"筛选结果: {len(all_filtered_results)}/{len(df)} 个环路满足闭环条件")
        else:
            logger.warning("没有环路满足闭环条件")
            pd.DataFrame().to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 生成统计报告
        generate_statistics_report(df, all_filtered_results, n_months, n_days, max_nodes, source_type, output_file, processing_time, n_cores)
        
        # 清理内存
        del basic_info_dict
        gc.collect()
        
        return len(all_filtered_results)
        
    except Exception as e:
        logger.error(f"并行闭环筛选过程中发生错误: {e}")
        raise

def generate_statistics_report(original_df, filtered_results, n_months, n_days, max_nodes, source_type, output_file, processing_time, n_cores):
    """生成统计报告"""
    logger = logging.getLogger(__name__)
    
    report_file = output_file.replace('.csv', '_statistics_report.txt')
    
    # 构建筛选条件描述
    time_condition = ""
    if n_months is not None:
        time_condition = f"{n_months}个月内的闭环"
    elif n_days is not None:
        time_condition = f"{n_days}天内的闭环"
    else:
        time_condition = "6个月内的闭环（默认）"
    
    node_condition = f"，节点数≤{max_nodes}" if max_nodes is not None else ""
    source_condition = ""
    if source_type == 'natural_person':
        source_condition = "，自然人源头"
    elif source_type == 'enterprise':
        source_condition = "，企业源头"
    
    condition_desc = time_condition + node_condition + source_condition
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== 简化闭环筛选统计报告（并行优化版本）===\n\n")
        f.write(f"筛选参数: {condition_desc}\n")
        f.write(f"原始环路数量: {len(original_df)}\n")
        f.write(f"满足闭环条件的环路数量: {len(filtered_results)}\n")
        f.write(f"闭环筛选率: {len(filtered_results)/len(original_df)*100:.2f}%\n\n")
        
        f.write("=== 性能信息 ===\n")
        f.write(f"处理方式: 多核并行处理\n")
        f.write(f"使用进程数: {n_cores}\n")
        f.write(f"处理时间: {processing_time:.2f} 秒\n")
        f.write(f"处理速度: {len(original_df)/processing_time:.2f} 环路/秒\n")
        f.write(f"平均每环路: {processing_time/len(original_df)*1000:.2f} 毫秒\n\n")
        
        f.write("=== 筛选逻辑说明 ===\n")
        f.write("• 使用多进程并行处理，显著提升筛选速度\n")
        f.write("• 不尝试匹配具体的交易对\n")
        f.write("• 只判断时间窗口内是否同时存在上游和下游交易\n")
        f.write("• 避免了交易对应关系的错误假设\n")
        f.write("• 结果更加可靠和有意义\n\n")
        
        if filtered_results:
            filtered_df = pd.DataFrame(filtered_results)
            
            f.write("=== 环路类型分布 ===\n")
            if 'type' in filtered_df.columns:
                type_distribution = filtered_df['type'].value_counts()
                for loop_type, count in type_distribution.items():
                    f.write(f"{loop_type}: {count}个闭环 ({count/len(filtered_results)*100:.1f}%)\n")
    
    logger.info(f"统计报告已保存到: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简化闭环筛选脚本 - 多核并行优化版本')
    parser.add_argument('--input', '-i', 
                       default='outputs/loop_analysis/loop_metrics.csv',
                       help='输入CSV文件路径')
    parser.add_argument('--output', '-o',
                       default='outputs/loop_filter/simplified_filtered_loops.csv', 
                       help='输出CSV文件路径')
    parser.add_argument('--months', '-m', type=int, default=None,
                       help='时间间隔阈值（月）')
    parser.add_argument('--days', '-d', type=int, default=None,
                       help='时间间隔阈值（天）')
    parser.add_argument('--max-nodes', '-n', type=int, default=None,
                       help='最大节点数量')
    parser.add_argument('--source-type', '-s', choices=['natural_person', 'enterprise'], default=None,
                       help='源头类型筛选')
    parser.add_argument('--cores', '-c', type=int, default=None,
                       help='使用的CPU核心数')
    
    args = parser.parse_args()
    
    # 检查参数冲突
    if args.months is not None and args.days is not None:
        print("错误: 不能同时指定 --months 和 --days 参数")
        sys.exit(1)
    
    # 设置默认值
    if args.months is None and args.days is None and args.max_nodes is None:
        args.months = 6
    
    # 设置日志
    logger = setup_logging()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        logger.info("=== 开始并行简化闭环筛选 ===")
        logger.info(f"输入文件: {args.input}")
        logger.info(f"输出文件: {args.output}")
        
        # 检查输入文件是否存在
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"输入文件不存在: {args.input}")
        
        # 获取系统信息
        cpu_count = mp.cpu_count()
        use_cores = args.cores if args.cores is not None else max(1, cpu_count - 1)
        logger.info(f"系统CPU核心数: {cpu_count}")
        logger.info(f"将使用 {use_cores} 个进程进行并行处理")
        
        # 执行并行筛选
        filtered_count = filter_loops_by_closure_parallel(
            args.input, args.output, args.months, args.days, 
            args.max_nodes, args.source_type, use_cores
        )
        
        logger.info("=== 并行简化闭环筛选完成 ===")
        logger.info(f"筛选出 {filtered_count} 个满足条件的环路")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main() 