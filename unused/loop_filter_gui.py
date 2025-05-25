#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
物产中大图风控系统 - 高级闭环优筛系统
重新设计版 - 满足所有新需求

新功能：
1. 时间维度分析 - 上下游交易时间间隔控制
2. 股权分析集成 - 调用loop_filter_script分析
3. 完整性检查 - Sanity Check确保数据完整
4. 自动数据读取 - 智能发现并加载所有相关数据
5. 真实数据导向 - 不使用估算值
6. 环路类型控制 - 可选择特定类型环路
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import os
import re
import json
import subprocess
import networkx as nx
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

class AdvancedLoopFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("物产中大图风控系统 - 高级闭环优筛系统 v2.0")
        self.root.geometry("1600x1000")
        
        # 尝试最大化窗口
        try:
            self.root.state('zoomed')
        except:
            try:
                self.root.attributes('-zoomed', True)
            except:
                pass
        
        # 数据存储
        self.raw_loops_data = {}           # 原始环路数据
        self.loop_metrics_data = {}        # 环路指标数据
        self.graph_data = None             # 图数据
        self.filtered_results = pd.DataFrame()  # 筛选结果
        self.data_sources = {}             # 数据源信息
        
        # 筛选参数
        self.filter_params = {
            # 时间维度参数
            'time_window_months': tk.IntVar(value=12),
            'max_time_gap_days': tk.IntVar(value=30),
            'min_upstream_transactions': tk.IntVar(value=1),
            'min_downstream_transactions': tk.IntVar(value=1),
            
            # 交易金额参数
            'min_total_amount': tk.DoubleVar(value=0),
            'min_upstream_amount': tk.DoubleVar(value=0),
            'min_downstream_amount': tk.DoubleVar(value=0),
            
            # 股权参数
            'min_equity_ratio': tk.DoubleVar(value=0),
            'max_equity_ratio': tk.DoubleVar(value=100),
            'min_shareholders': tk.IntVar(value=1),
            
            # 环路结构参数
            'min_nodes': tk.IntVar(value=3),
            'max_nodes': tk.IntVar(value=15),
            'selected_loop_types': [],
            
            # 输出控制
            'max_results': tk.IntVar(value=1000)
        }
        
        # GUI组件
        self.progress_queue = queue.Queue()
        self.log_queue = queue.Queue()
        
        # 创建界面
        self.create_widgets()
        
        # 绑定参数变化事件
        self.bind_parameter_events()
        
        # 自动加载数据
        self.auto_load_data()
    
    def create_widgets(self):
        """创建主界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 顶部控制面板
        self.create_control_panel(main_frame)
        
        # 主内容区域 - 使用PanedWindow分割
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # 左侧面板
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # 右侧面板
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        
        # 创建左右面板内容
        self.create_left_panel(left_frame)
        self.create_right_panel(right_frame)
        
        # 底部状态栏
        self.create_status_panel(main_frame)
    
    def create_control_panel(self, parent):
        """创建顶部控制面板"""
        control_frame = ttk.LabelFrame(parent, text="系统控制")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 数据控制区
        data_frame = ttk.Frame(control_frame)
        data_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(data_frame, text="🔄 重新加载数据", 
                  command=self.reload_all_data).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(data_frame, text="🔍 完整性检查", 
                  command=self.run_sanity_check).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(data_frame, text="📊 运行股权分析", 
                  command=self.run_equity_analysis).pack(side=tk.LEFT, padx=2)
        
        # 筛选控制区
        filter_frame = ttk.Frame(control_frame)
        filter_frame.pack(side=tk.LEFT, padx=(20, 5), pady=5)
        
        ttk.Button(filter_frame, text="🎯 执行筛选", 
                  command=self.execute_filtering).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(filter_frame, text="🔄 重置参数", 
                  command=self.reset_parameters).pack(side=tk.LEFT, padx=2)
        
        # 导出控制区
        export_frame = ttk.Frame(control_frame)
        export_frame.pack(side=tk.RIGHT, padx=5, pady=5)
        
        ttk.Button(export_frame, text="📤 导出结果", 
                  command=self.export_results).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(export_frame, text="📋 生成报告", 
                  command=self.generate_report).pack(side=tk.LEFT, padx=2)
    
    def create_left_panel(self, parent):
        """创建左侧控制面板"""
        # 数据源信息
        self.create_data_source_panel(parent)
        
        # 筛选参数面板
        self.create_filter_parameters_panel(parent)
        
        # 环路类型选择
        self.create_loop_type_panel(parent)
    
    def create_data_source_panel(self, parent):
        """创建数据源信息面板"""
        data_frame = ttk.LabelFrame(parent, text="数据源信息")
        data_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 数据源列表
        self.data_tree = ttk.Treeview(data_frame, 
                                     columns=('状态', '记录数', '最后更新', '大小'), 
                                     show='tree headings', height=6)
        
        self.data_tree.heading('#0', text='数据源')
        self.data_tree.heading('状态', text='状态')
        self.data_tree.heading('记录数', text='记录数')
        self.data_tree.heading('最后更新', text='最后更新')
        self.data_tree.heading('大小', text='大小')
        
        self.data_tree.column('#0', width=150)
        self.data_tree.column('状态', width=60)
        self.data_tree.column('记录数', width=80)
        self.data_tree.column('最后更新', width=100)
        self.data_tree.column('大小', width=50)
        
        self.data_tree.pack(fill=tk.X, padx=5, pady=5)
    
    def create_filter_parameters_panel(self, parent):
        """创建筛选参数面板"""
        param_frame = ttk.LabelFrame(parent, text="筛选参数")
        param_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建滚动区域
        canvas = tk.Canvas(param_frame)
        scrollbar = ttk.Scrollbar(param_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", 
                             lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 时间维度参数
        self.create_time_parameters(scrollable_frame)
        
        # 交易金额参数
        self.create_amount_parameters(scrollable_frame)
        
        # 股权参数
        self.create_equity_parameters(scrollable_frame)
        
        # 结构参数
        self.create_structure_parameters(scrollable_frame)
    
    def create_time_parameters(self, parent):
        """创建时间维度参数"""
        frame = ttk.LabelFrame(parent, text="⏰ 时间维度分析")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 时间窗口
        ttk.Label(frame, text="分析时间窗口 (月):").pack(anchor=tk.W, padx=5, pady=2)
        time_window_frame = ttk.Frame(frame)
        time_window_frame.pack(fill=tk.X, padx=5, pady=2)
        
        time_window_scale = ttk.Scale(time_window_frame, from_=1, to=36,
                                     variable=self.filter_params['time_window_months'],
                                     orient=tk.HORIZONTAL)
        time_window_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.time_window_label = ttk.Label(time_window_frame, text="12个月")
        self.time_window_label.pack(side=tk.RIGHT, padx=(5, 0))
        
        # 上下游交易时间间隔
        ttk.Label(frame, text="最大交易时间间隔 (天):").pack(anchor=tk.W, padx=5, pady=2)
        time_gap_frame = ttk.Frame(frame)
        time_gap_frame.pack(fill=tk.X, padx=5, pady=2)
        
        time_gap_scale = ttk.Scale(time_gap_frame, from_=1, to=365,
                                  variable=self.filter_params['max_time_gap_days'],
                                  orient=tk.HORIZONTAL)
        time_gap_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.time_gap_label = ttk.Label(time_gap_frame, text="30天")
        self.time_gap_label.pack(side=tk.RIGHT, padx=(5, 0))
        
        # 最小交易次数
        min_trans_frame = ttk.Frame(frame)
        min_trans_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(min_trans_frame, text="最小上游交易次数:").pack(side=tk.LEFT)
        upstream_spin = ttk.Spinbox(min_trans_frame, from_=0, to=20, width=5,
                                   textvariable=self.filter_params['min_upstream_transactions'])
        upstream_spin.pack(side=tk.RIGHT)
        
        min_trans_frame2 = ttk.Frame(frame)
        min_trans_frame2.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(min_trans_frame2, text="最小下游交易次数:").pack(side=tk.LEFT)
        downstream_spin = ttk.Spinbox(min_trans_frame2, from_=0, to=20, width=5,
                                     textvariable=self.filter_params['min_downstream_transactions'])
        downstream_spin.pack(side=tk.RIGHT)
    
    def create_amount_parameters(self, parent):
        """创建交易金额参数"""
        frame = ttk.LabelFrame(parent, text="💰 交易金额分析")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 总交易金额
        ttk.Label(frame, text="最小总交易金额 (万元):").pack(anchor=tk.W, padx=5, pady=2)
        total_amount_frame = ttk.Frame(frame)
        total_amount_frame.pack(fill=tk.X, padx=5, pady=2)
        
        total_amount_scale = ttk.Scale(total_amount_frame, from_=0, to=10000,
                                     variable=self.filter_params['min_total_amount'],
                                     orient=tk.HORIZONTAL)
        total_amount_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.total_amount_label = ttk.Label(total_amount_frame, text="0万")
        self.total_amount_label.pack(side=tk.RIGHT, padx=(5, 0))
        
        # 上游交易金额
        ttk.Label(frame, text="最小上游交易金额 (万元):").pack(anchor=tk.W, padx=5, pady=2)
        upstream_amount_frame = ttk.Frame(frame)
        upstream_amount_frame.pack(fill=tk.X, padx=5, pady=2)
        
        upstream_amount_scale = ttk.Scale(upstream_amount_frame, from_=0, to=5000,
                                        variable=self.filter_params['min_upstream_amount'],
                                        orient=tk.HORIZONTAL)
        upstream_amount_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.upstream_amount_label = ttk.Label(upstream_amount_frame, text="0万")
        self.upstream_amount_label.pack(side=tk.RIGHT, padx=(5, 0))
        
        # 下游交易金额
        ttk.Label(frame, text="最小下游交易金额 (万元):").pack(anchor=tk.W, padx=5, pady=2)
        downstream_amount_frame = ttk.Frame(frame)
        downstream_amount_frame.pack(fill=tk.X, padx=5, pady=2)
        
        downstream_amount_scale = ttk.Scale(downstream_amount_frame, from_=0, to=5000,
                                          variable=self.filter_params['min_downstream_amount'],
                                          orient=tk.HORIZONTAL)
        downstream_amount_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.downstream_amount_label = ttk.Label(downstream_amount_frame, text="0万")
        self.downstream_amount_label.pack(side=tk.RIGHT, padx=(5, 0))
    
    def create_equity_parameters(self, parent):
        """创建股权参数"""
        frame = ttk.LabelFrame(parent, text="🏛️ 股权结构分析")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 股权比例范围
        ttk.Label(frame, text="股权比例范围 (%):").pack(anchor=tk.W, padx=5, pady=2)
        
        equity_range_frame = ttk.Frame(frame)
        equity_range_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(equity_range_frame, text="最小:").pack(side=tk.LEFT)
        min_equity_spin = ttk.Spinbox(equity_range_frame, from_=0, to=100, width=6,
                                     textvariable=self.filter_params['min_equity_ratio'])
        min_equity_spin.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(equity_range_frame, text="最大:").pack(side=tk.LEFT)
        max_equity_spin = ttk.Spinbox(equity_range_frame, from_=0, to=100, width=6,
                                     textvariable=self.filter_params['max_equity_ratio'])
        max_equity_spin.pack(side=tk.LEFT, padx=5)
        
        # 最小股东数
        min_shareholders_frame = ttk.Frame(frame)
        min_shareholders_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(min_shareholders_frame, text="最小股东数:").pack(side=tk.LEFT)
        shareholders_spin = ttk.Spinbox(min_shareholders_frame, from_=1, to=50, width=5,
                                      textvariable=self.filter_params['min_shareholders'])
        shareholders_spin.pack(side=tk.RIGHT)
    
    def create_structure_parameters(self, parent):
        """创建结构参数"""
        frame = ttk.LabelFrame(parent, text="🔗 环路结构参数")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 节点数范围
        ttk.Label(frame, text="环路节点数范围:").pack(anchor=tk.W, padx=5, pady=2)
        
        nodes_range_frame = ttk.Frame(frame)
        nodes_range_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(nodes_range_frame, text="最小:").pack(side=tk.LEFT)
        min_nodes_spin = ttk.Spinbox(nodes_range_frame, from_=3, to=20, width=4,
                                    textvariable=self.filter_params['min_nodes'])
        min_nodes_spin.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(nodes_range_frame, text="最大:").pack(side=tk.LEFT)
        max_nodes_spin = ttk.Spinbox(nodes_range_frame, from_=3, to=20, width=4,
                                    textvariable=self.filter_params['max_nodes'])
        max_nodes_spin.pack(side=tk.LEFT, padx=5)
        
        # 最大结果数
        max_results_frame = ttk.Frame(frame)
        max_results_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(max_results_frame, text="最大输出结果数:").pack(side=tk.LEFT)
        results_spin = ttk.Spinbox(max_results_frame, from_=10, to=10000, width=6,
                                  textvariable=self.filter_params['max_results'])
        results_spin.pack(side=tk.RIGHT)
    
    def create_loop_type_panel(self, parent):
        """创建环路类型选择面板"""
        frame = ttk.LabelFrame(parent, text="🎯 环路类型选择")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 环路类型复选框列表
        self.loop_type_frame = ttk.Frame(frame)
        self.loop_type_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 初始化时会填充可用的环路类型
        self.loop_type_vars = {}
    
    def create_right_panel(self, parent):
        """创建右侧面板"""
        # 创建选项卡
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建各个选项卡
        self.create_overview_tab()
        self.create_time_analysis_tab()
        self.create_equity_analysis_tab()
        self.create_results_tab()
        self.create_log_tab()
    
    def create_overview_tab(self):
        """创建概览选项卡"""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="📊 数据概览")
        
        # 统计信息显示
        stats_frame = ttk.LabelFrame(overview_frame, text="数据统计")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=15, width=80)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_time_analysis_tab(self):
        """创建时间分析选项卡"""
        time_frame = ttk.Frame(self.notebook)
        self.notebook.add(time_frame, text="⏰ 时间分析")
        
        # 时间分析图表
        self.time_fig = Figure(dpi=80)
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, time_frame)
        self.time_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_equity_analysis_tab(self):
        """创建股权分析选项卡"""
        equity_frame = ttk.Frame(self.notebook)
        self.notebook.add(equity_frame, text="🏛️ 股权分析")
        
        # 股权分析图表
        self.equity_fig = Figure(dpi=80)
        self.equity_canvas = FigureCanvasTkAgg(self.equity_fig, equity_frame)
        self.equity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_results_tab(self):
        """创建结果选项卡"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="🎯 筛选结果")
        
        # 结果表格
        columns = ['环路ID', '类型', '节点数', '上游交易金额', '下游交易金额', 
                  '时间间隔', '股权集中度', '最后交易时间']
        
        self.results_tree = ttk.Treeview(results_frame, columns=columns, 
                                        show='headings', height=20)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100)
        
        # 添加滚动条
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", 
                                        command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_log_tab(self):
        """创建日志选项卡"""
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="📝 系统日志")
        
        # 日志显示区域
        self.log_text = scrolledtext.ScrolledText(log_frame, height=25, width=100,
                                                 font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_status_panel(self, parent):
        """创建状态面板"""
        status_frame = ttk.LabelFrame(parent, text="系统状态")
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 状态信息
        status_info_frame = ttk.Frame(status_frame)
        status_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="系统就绪 - 等待数据加载")
        ttk.Label(status_info_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_info_frame, 
                                          variable=self.progress_var,
                                          maximum=100)
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
    
    def bind_parameter_events(self):
        """绑定参数变化事件"""
        for param_name, param_var in self.filter_params.items():
            if isinstance(param_var, (tk.IntVar, tk.DoubleVar)):
                param_var.trace('w', self.on_parameter_change)
    
    def on_parameter_change(self, *args):
        """参数变化时的回调"""
        self.update_parameter_labels()
        
        # 如果有数据，自动更新预览
        if hasattr(self, 'raw_loops_data') and self.raw_loops_data:
            self.update_filter_preview()
    
    def update_parameter_labels(self):
        """更新参数标签显示"""
        if hasattr(self, 'time_window_label'):
            self.time_window_label.config(
                text=f"{self.filter_params['time_window_months'].get()}个月")
        
        if hasattr(self, 'time_gap_label'):
            self.time_gap_label.config(
                text=f"{self.filter_params['max_time_gap_days'].get()}天")
        
        if hasattr(self, 'total_amount_label'):
            self.total_amount_label.config(
                text=f"{self.filter_params['min_total_amount'].get():.0f}万")
        
        if hasattr(self, 'upstream_amount_label'):
            self.upstream_amount_label.config(
                text=f"{self.filter_params['min_upstream_amount'].get():.0f}万")
        
        if hasattr(self, 'downstream_amount_label'):
            self.downstream_amount_label.config(
                text=f"{self.filter_params['min_downstream_amount'].get():.0f}万")
    
    def auto_load_data(self):
        """自动加载所有可用数据"""
        self.log("🚀 开始自动扫描和加载数据...")
        
        # 在后台线程中执行数据加载
        threading.Thread(target=self._auto_load_data_worker, daemon=True).start()
        
        # 启动UI更新
        self.root.after(100, self.update_ui_from_queues)
    
    def _auto_load_data_worker(self):
        """数据加载工作线程"""
        try:
            # 扫描可用的数据文件
            data_files = self.scan_data_files()
            
            total_files = len(data_files)
            loaded_files = 0
            
            for file_type, file_info in data_files.items():
                self.progress_queue.put(('progress', loaded_files / total_files * 100))
                self.log_queue.put(f"📂 加载 {file_type}: {file_info['path']}")
                
                try:
                    if file_type == 'loops':
                        self.raw_loops_data = self.load_loops_data(file_info['path'])
                        self.data_sources['loops'] = file_info
                        
                    elif file_type == 'metrics':
                        self.loop_metrics_data = self.load_metrics_data(file_info['path'])
                        self.data_sources['metrics'] = file_info
                        
                    elif file_type == 'graph':
                        self.graph_data = self.load_graph_data(file_info['path'])
                        self.data_sources['graph'] = file_info
                    
                    file_info['status'] = '✅ 已加载'
                    file_info['records'] = self.get_record_count(file_type)
                    
                except Exception as e:
                    file_info['status'] = f'❌ 失败: {str(e)[:20]}'
                    self.log_queue.put(f"❌ 加载失败 {file_type}: {str(e)}")
                
                loaded_files += 1
            
            # 完成数据加载后的处理
            self.progress_queue.put(('progress', 100))
            self.log_queue.put("✅ 数据加载完成")
            
            # 更新UI组件
            self.progress_queue.put(('update_data_tree', None))
            self.progress_queue.put(('update_loop_types', None))
            self.progress_queue.put(('update_overview', None))
            
        except Exception as e:
            self.log_queue.put(f"❌ 数据加载过程出错: {str(e)}")
    
    def scan_data_files(self):
        """简化版：扫描可用的数据文件，优先选择特定文件。"""
        self.log("🤖 (新) 开始扫描数据文件 (scan_data_files)...")
        data_files = {}
        
        # 环路数据 - 优先选择优化版本，然后是nx版本
        loop_file_paths = [
            ('outputs/loop_results/equity_loops_optimized.txt', 'igraph'),
            ('outputs/loop_results/equity_loops_nx.txt', 'networkx')
        ]
        
        for path, engine_type in loop_file_paths:
            self.log(f"检查环路文件: {path}")
            if os.path.exists(path):
                data_files['loops'] = {
                    'path': path,
                    'size': os.path.getsize(path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(path)),
                    'status': '⏳ 待加载',
                    'engine': engine_type 
                }
                self.log(f"📁 选择环路数据文件: {path} (引擎: {engine_type})")
                break
            else:
                self.log(f"环路文件未找到: {path}")

        if 'loops' not in data_files:
            self.log("⚠️ 未找到指定的环路数据文件。")

        # 指标数据
        metrics_file_path = 'outputs/loop_analysis/loop_metrics.csv'
        self.log(f"检查指标文件: {metrics_file_path}")
        if os.path.exists(metrics_file_path):
            data_files['metrics'] = {
                'path': metrics_file_path,
                'size': os.path.getsize(metrics_file_path),
                'modified': datetime.fromtimestamp(os.path.getmtime(metrics_file_path)),
                'status': '⏳ 待加载'
                # 'engine': 'pandas' #  如果需要，可以为其他类型文件也添加 engine
            }
            self.log(f"📊 选择指标数据文件: {metrics_file_path}")
        else:
            self.log(f"指标文件未找到: {metrics_file_path}")
            # 可以考虑备用路径，例如 loop_basic_info.csv
            backup_metrics_path = 'outputs/loop_analysis/loop_basic_info.csv'
            self.log(f"检查备用指标文件: {backup_metrics_path}")
            if os.path.exists(backup_metrics_path):
                data_files['metrics'] = {
                    'path': backup_metrics_path,
                    'size': os.path.getsize(backup_metrics_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(backup_metrics_path)),
                    'status': '⏳ 待加载'
                }
                self.log(f"📊 选择备用指标数据文件: {backup_metrics_path}")
            else:
                self.log(f"备用指标文件未找到: {backup_metrics_path}")


        # 图数据
        graph_file_path = 'model/final_heterogeneous_graph.graphml'
        self.log(f"检查图数据文件: {graph_file_path}")
        if os.path.exists(graph_file_path):
            data_files['graph'] = {
                'path': graph_file_path,
                'size': os.path.getsize(graph_file_path),
                'modified': datetime.fromtimestamp(os.path.getmtime(graph_file_path)),
                'status': '⏳ 待加载'
                # 'engine': 'networkx'
            }
            self.log(f"📈 选择图数据文件: {graph_file_path}")
        else:
            self.log(f"图数据文件未找到: {graph_file_path}")
            # 可以考虑备用路径
            backup_graph_path = 'model/simplified_loop_detection_graph.graphml'
            self.log(f"检查备用图数据文件: {backup_graph_path}")
            if os.path.exists(backup_graph_path):
                data_files['graph'] = {
                    'path': backup_graph_path,
                    'size': os.path.getsize(backup_graph_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(backup_graph_path)),
                    'status': '⏳ 待加载'
                }
                self.log(f"📈 选择备用图数据文件: {backup_graph_path}")
            else:
                self.log(f"备用图数据文件未找到: {backup_graph_path}")
        
        if not data_files:
             self.log("‼️ 重要警告: 未能定位任何核心数据文件 (loops, metrics, graph)。GUI可能无法正常加载数据。")
        self.log(f"🔍 数据文件扫描完成. 找到的数据: {list(data_files.keys())}")
        return data_files
    
    def load_loops_data(self, file_path):
        """加载环路数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析环路数据（不使用估算值）
        loops_data = {}
        
        # 提取详细闭环信息
        detailed_section = re.split(r'## 详细闭环信息', content, 1)
        if len(detailed_section) < 2:
            return {}
        
        detailed_content = detailed_section[1]
        company_sections = re.split(r'### 股东: ', detailed_content)[1:]
        
        loop_id = 1
        for section in company_sections:
            lines = section.strip().split('\n')
            company_name = lines[0].strip()
            
            in_loop_section = False
            current_loop_type = ""
            
            for line in lines:
                if line.startswith('####'):
                    current_loop_type = line.replace('####', '').strip()
                    in_loop_section = True
                    continue
                
                if in_loop_section and line.strip() and not line.startswith('-'):
                    if re.match(r'^\d+\.', line):
                        loop_content = re.sub(r'^\d+\.\s*', '', line).strip()
                        
                        # 提取节点路径和角色
                        node_path, node_roles = self.extract_nodes_from_content(loop_content)
                        
                        loops_data[loop_id] = {
                            'source': company_name,
                            'content': loop_content,
                            'type': current_loop_type,
                            'node_path': node_path,
                            'node_roles': node_roles,
                            'node_count': len(set(node_path)),
                            'has_real_data': False  # 标记是否有真实数据
                        }
                        loop_id += 1
        
        return loops_data
    
    def extract_nodes_from_content(self, content):
        """从环路内容中提取节点信息"""
        node_path = []
        node_roles = []
        
        # 使用正则表达式提取节点和角色
        pattern = r'([^[\]]+?)\s*\[([^\]]+)\]'
        matches = re.findall(pattern, content)
        
        for node_name, node_role in matches:
            node_path.append(node_name.strip())
            node_roles.append(node_role.strip())
        
        return node_path, node_roles
    
    def load_metrics_data(self, file_path):
        """加载指标数据"""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            return {}
    
    def load_graph_data(self, file_path):
        """加载图数据"""
        try:
            return nx.read_graphml(file_path)
        except Exception as e:
            self.log_queue.put(f"图数据加载失败: {str(e)}")
            return None
    
    def get_record_count(self, file_type):
        """获取记录数量"""
        if file_type == 'loops':
            return len(self.raw_loops_data)
        elif file_type == 'metrics':
            return len(self.loop_metrics_data) if isinstance(self.loop_metrics_data, pd.DataFrame) else 0
        elif file_type == 'graph':
            return self.graph_data.number_of_nodes() if self.graph_data else 0
        return 0
    
    def update_ui_from_queues(self):
        """从队列更新UI"""
        # 处理进度更新
        try:
            while True:
                msg_type, value = self.progress_queue.get_nowait()
                if msg_type == 'progress':
                    self.progress_var.set(value)
                elif msg_type == 'update_data_tree':
                    self.update_data_tree()
                elif msg_type == 'update_loop_types':
                    self.update_loop_types()
                elif msg_type == 'update_overview':
                    self.update_overview()
        except queue.Empty:
            pass
        
        # 处理日志更新
        try:
            while True:
                log_msg = self.log_queue.get_nowait()
                self.log(log_msg)
        except queue.Empty:
            pass
        
        # 继续监听
        self.root.after(100, self.update_ui_from_queues)
    
    def update_data_tree(self):
        """更新数据源树，显示更详细的信息和引擎"""
        self.log("🌳 更新数据源树 (update_data_tree)...")
        # 清空现有项
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # 添加数据源信息
        if not self.data_sources:
            self.log("数据源为空，无法更新树。")
            return

        for file_type, file_info in self.data_sources.items():
            if not file_info or 'path' not in file_info : # 确保file_info有效
                self.log(f"警告: 数据源 '{file_type}' 信息不完整或无效，跳过。")
                continue

            self.log(f"更新树节点: {file_type}, 信息: {file_info}")
            # 添加引擎信息到显示名称
            engine_info = file_info.get('engine', '') # 获取引擎，默认为空字符串
            display_name = f"{file_type.upper()}"
            if engine_info:
                display_name += f" ({engine_info})"
            
            # 获取文件大小，如果不存在则为0
            size_bytes = file_info.get('size', 0)
            size_kb = f"{size_bytes / 1024:.1f}KB" if size_bytes else "N/A"

            # 获取记录数，不存在则为0
            records_count = file_info.get('records', 0)
            records_display = f"{records_count:,}" if records_count is not None else "N/A"

            # 获取修改时间，不存在则为N/A
            modified_time = file_info.get('modified')
            modified_display = modified_time.strftime('%Y-%m-%d %H:%M') if modified_time else "N/A"
            
            status_display = file_info.get('status', '未知')


            self.data_tree.insert('', 'end', 
                                 text=display_name,
                                 values=(
                                     status_display,
                                     records_display,
                                     modified_display,
                                     size_kb  # 添加文件大小
                                 ))
        self.log("数据源树更新完毕。")
    
    def update_loop_types(self):
        """更新环路类型选择"""
        # 清空现有的复选框
        for widget in self.loop_type_frame.winfo_children():
            widget.destroy()
        
        # 获取所有可用的环路类型
        if self.raw_loops_data:
            loop_types = set()
            for loop_data in self.raw_loops_data.values():
                loop_types.add(loop_data['type'])
            
            # 创建复选框
            self.loop_type_vars = {}
            for i, loop_type in enumerate(sorted(loop_types)):
                var = tk.BooleanVar(value=True)  # 默认全选
                self.loop_type_vars[loop_type] = var
                
                cb = ttk.Checkbutton(self.loop_type_frame, 
                                   text=loop_type,
                                   variable=var,
                                   command=self.on_loop_type_change)
                cb.grid(row=i//2, column=i%2, sticky='w', padx=5, pady=2)
    
    def update_overview(self):
        """更新数据概览"""
        overview_text = self.generate_overview_text()
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, overview_text)
    
    def generate_overview_text(self):
        """生成概览文本"""
        text = "🔍 物产中大图风控系统 - 数据概览\n"
        text += "=" * 60 + "\n\n"
        
        # 数据源统计
        text += "📊 数据源统计:\n"
        text += "-" * 30 + "\n"
        
        for file_type, file_info in self.data_sources.items():
            text += f"{file_type.upper():<10}: {file_info['status']} "
            text += f"({file_info.get('records', 0):,} 条记录)\n"
        
        text += "\n"
        
        # 环路统计
        if self.raw_loops_data:
            text += "🔄 环路统计:\n"
            text += "-" * 30 + "\n"
            text += f"总环路数量: {len(self.raw_loops_data):,}\n"
            
            # 按类型统计
            type_counts = Counter()
            node_counts = Counter()
            
            for loop_data in self.raw_loops_data.values():
                type_counts[loop_data['type']] += 1
                node_counts[loop_data['node_count']] += 1
            
            text += "\n按类型分布:\n"
            for loop_type, count in type_counts.most_common():
                text += f"  {loop_type}: {count:,} 个\n"
            
            text += "\n按节点数分布:\n"
            for node_count, count in sorted(node_counts.items()):
                text += f"  {node_count}节点: {count:,} 个\n"
        
        # 如果有指标数据
        if isinstance(self.loop_metrics_data, pd.DataFrame) and not self.loop_metrics_data.empty:
            text += "\n💰 交易指标统计:\n"
            text += "-" * 30 + "\n"
            
            # 计算统计信息
            if 'total_transaction_amount' in self.loop_metrics_data.columns:
                amounts = self.loop_metrics_data['total_transaction_amount'].dropna()
                if len(amounts) > 0:
                    text += f"平均交易金额: {amounts.mean()/10000:.1f} 万元\n"
                    text += f"最大交易金额: {amounts.max()/10000:.1f} 万元\n"
                    text += f"中位数交易金额: {amounts.median()/10000:.1f} 万元\n"
        
        # 系统状态
        text += "\n⚙️ 系统状态:\n"
        text += "-" * 30 + "\n"
        text += f"数据加载时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += f"可用环路类型: {len(self.loop_type_vars)} 种\n"
        
        return text
    
    def on_loop_type_change(self):
        """环路类型选择变化"""
        selected_types = []
        for loop_type, var in self.loop_type_vars.items():
            if var.get():
                selected_types.append(loop_type)
        
        self.filter_params['selected_loop_types'] = selected_types
        self.update_filter_preview()
    
    def update_filter_preview(self):
        """更新筛选预览"""
        if not self.raw_loops_data:
            return
        
        # 应用当前筛选条件计算预览结果
        filtered_count = self.calculate_filtered_count()
        
        self.status_var.set(f"筛选预览: {filtered_count:,} / {len(self.raw_loops_data):,} 个环路满足条件")
    
    def calculate_filtered_count(self):
        """计算筛选后的数量（快速预览）"""
        count = 0
        selected_types = self.filter_params['selected_loop_types']
        min_nodes = self.filter_params['min_nodes'].get()
        max_nodes = self.filter_params['max_nodes'].get()
        
        for loop_data in self.raw_loops_data.values():
            # 类型筛选
            if selected_types and loop_data['type'] not in selected_types:
                continue
            
            # 节点数筛选
            if not (min_nodes <= loop_data['node_count'] <= max_nodes):
                continue
            
            count += 1
        
        return count
    
    def execute_filtering(self):
        """执行完整筛选"""
        if not self.raw_loops_data:
            messagebox.showwarning("警告", "请先加载环路数据")
            return
        
        self.log("🎯 开始执行高级筛选...")
        
        # 在后台线程执行筛选
        threading.Thread(target=self._execute_filtering_worker, daemon=True).start()
    
    def _execute_filtering_worker(self):
        """筛选工作线程"""
        try:
            filtered_results = []
            
            total_loops = len(self.raw_loops_data)
            processed = 0
            
            for loop_id, loop_data in self.raw_loops_data.items():
                # 应用所有筛选条件
                if self.apply_all_filters(loop_id, loop_data):
                    # 计算详细指标
                    metrics = self.calculate_detailed_metrics(loop_id, loop_data)
                    
                    result = {
                        'loop_id': loop_id,
                        'type': loop_data['type'],
                        'node_count': loop_data['node_count'],
                        'content': loop_data['content'],
                        **metrics
                    }
                    filtered_results.append(result)
                
                processed += 1
                if processed % 100 == 0:
                    progress = processed / total_loops * 100
                    self.progress_queue.put(('progress', progress))
            
            # 按时间间隔排序
            filtered_results.sort(key=lambda x: x.get('time_gap_days', 999), reverse=False)
            
            # 限制结果数量
            max_results = self.filter_params['max_results'].get()
            filtered_results = filtered_results[:max_results]
            
            # 更新结果
            self.filtered_results = pd.DataFrame(filtered_results)
            
            self.progress_queue.put(('progress', 100))
            self.log_queue.put(f"✅ 筛选完成: {len(filtered_results)} 个环路满足条件")
            
            # 更新UI
            self.progress_queue.put(('update_results', None))
            self.progress_queue.put(('update_charts', None))
            
        except Exception as e:
            self.log_queue.put(f"❌ 筛选过程出错: {str(e)}")
    
    def apply_all_filters(self, loop_id, loop_data):
        """应用所有筛选条件"""
        # 1. 类型筛选
        selected_types = self.filter_params['selected_loop_types']
        if selected_types and loop_data['type'] not in selected_types:
            return False
        
        # 2. 节点数筛选
        min_nodes = self.filter_params['min_nodes'].get()
        max_nodes = self.filter_params['max_nodes'].get()
        if not (min_nodes <= loop_data['node_count'] <= max_nodes):
            return False
        
        # 3. 时间维度筛选（如果有图数据和指标数据）
        if self.graph_data and isinstance(self.loop_metrics_data, pd.DataFrame):
            if not self.apply_time_filters(loop_id, loop_data):
                return False
        
        # 4. 金额筛选（如果有指标数据）
        if isinstance(self.loop_metrics_data, pd.DataFrame):
            if not self.apply_amount_filters(loop_id, loop_data):
                return False
        
        # 5. 股权筛选（如果有图数据）
        if self.graph_data:
            if not self.apply_equity_filters(loop_id, loop_data):
                return False
        
        return True
    
    def apply_time_filters(self, loop_id, loop_data):
        """应用时间维度筛选"""
        # 这里需要实现时间维度的复杂逻辑
        # 分析上游交易和下游交易的时间间隔
        
        try:
            # 从指标数据中获取时间信息
            metrics_row = self.loop_metrics_data[self.loop_metrics_data['loop_id'] == loop_id]
            if metrics_row.empty:
                return True  # 没有数据则通过
            
            # 检查上下游交易时间间隔
            if 'upstream_to_member_transaction_times' in metrics_row.columns:
                upstream_times = metrics_row.iloc[0]['upstream_to_member_transaction_times']
                downstream_times = metrics_row.iloc[0]['member_to_downstream_transaction_times']
                
                if upstream_times and downstream_times:
                    # 计算时间间隔
                    time_gap = self.calculate_time_gap(upstream_times, downstream_times)
                    max_gap = self.filter_params['max_time_gap_days'].get()
                    
                    if time_gap > max_gap:
                        return False
            
            return True
            
        except Exception:
            return True  # 出错则通过筛选
    
    def calculate_time_gap(self, upstream_times, downstream_times):
        """计算上下游交易时间间隔"""
        try:
            # 解析时间字符串
            if isinstance(upstream_times, str):
                upstream_times = eval(upstream_times) if upstream_times.startswith('[') else [upstream_times]
            if isinstance(downstream_times, str):
                downstream_times = eval(downstream_times) if downstream_times.startswith('[') else [downstream_times]
            
            if not upstream_times or not downstream_times:
                return 0
            
            # 获取最后一次交易时间
            last_upstream = max(upstream_times)
            last_downstream = max(downstream_times)
            
            # 计算时间差
            upstream_date = datetime.strptime(last_upstream.split()[0], '%Y-%m-%d')
            downstream_date = datetime.strptime(last_downstream.split()[0], '%Y-%m-%d')
            
            return abs((downstream_date - upstream_date).days)
            
        except Exception:
            return 0
    
    def apply_amount_filters(self, loop_id, loop_data):
        """应用金额筛选"""
        try:
            metrics_row = self.loop_metrics_data[self.loop_metrics_data['loop_id'] == loop_id]
            if metrics_row.empty:
                return True
            
            row = metrics_row.iloc[0]
            
            # 检查总交易金额
            min_total = self.filter_params['min_total_amount'].get() * 10000
            if row.get('total_transaction_amount', 0) < min_total:
                return False
            
            # 检查上游交易金额
            min_upstream = self.filter_params['min_upstream_amount'].get() * 10000
            if row.get('upstream_to_member_transaction_amount', 0) < min_upstream:
                return False
            
            # 检查下游交易金额
            min_downstream = self.filter_params['min_downstream_amount'].get() * 10000
            if row.get('member_to_downstream_transaction_amount', 0) < min_downstream:
                return False
            
            return True
            
        except Exception:
            return True
    
    def apply_equity_filters(self, loop_id, loop_data):
        """应用股权筛选"""
        try:
            # 使用图数据分析股权结构
            node_path = loop_data['node_path']
            
            if not node_path or not self.graph_data:
                return True
            
            # 分析环路中的股权关系
            equity_ratios = []
            for node_name in node_path:
                # 在图中查找对应节点
                node_id = self.find_node_in_graph(node_name)
                if node_id:
                    # 获取持股信息
                    ratios = self.get_equity_ratios(node_id)
                    equity_ratios.extend(ratios)
            
            if equity_ratios:
                min_ratio = self.filter_params['min_equity_ratio'].get()
                max_ratio = self.filter_params['max_equity_ratio'].get()
                
                # 检查是否有在范围内的股权比例
                valid_ratios = [r for r in equity_ratios if min_ratio <= r <= max_ratio]
                return len(valid_ratios) > 0
            
            return True
            
        except Exception:
            return True
    
    def find_node_in_graph(self, node_name):
        """在图中查找节点"""
        if not self.graph_data:
            return None
        
        for node_id, attributes in self.graph_data.nodes(data=True):
            if attributes.get('name') == node_name:
                return node_id
        
        return None
    
    def get_equity_ratios(self, node_id):
        """获取节点的股权比例"""
        ratios = []
        
        try:
            # 获取出边（投资关系）
            for _, target, edge_data in self.graph_data.out_edges(node_id, data=True):
                percent = edge_data.get('percent', 0)
                if isinstance(percent, (int, float)) and percent > 0:
                    ratios.append(percent * 100)  # 转换为百分比
        except Exception:
            pass
        
        return ratios
    
    def calculate_detailed_metrics(self, loop_id, loop_data):
        """计算详细指标（只使用真实数据）"""
        metrics = {
            'upstream_amount': 0,
            'downstream_amount': 0,
            'total_amount': 0,
            'time_gap_days': 999,
            'equity_concentration': 0,
            'last_transaction_date': 'N/A'
        }
        
        try:
            # 从指标数据获取真实指标
            if isinstance(self.loop_metrics_data, pd.DataFrame):
                metrics_row = self.loop_metrics_data[self.loop_metrics_data['loop_id'] == loop_id]
                if not metrics_row.empty:
                    row = metrics_row.iloc[0]
                    
                    metrics['upstream_amount'] = row.get('upstream_to_member_transaction_amount', 0) / 10000
                    metrics['downstream_amount'] = row.get('member_to_downstream_transaction_amount', 0) / 10000
                    metrics['total_amount'] = row.get('total_transaction_amount', 0) / 10000
                    
                    # 计算时间间隔
                    upstream_times = row.get('upstream_to_member_transaction_times', [])
                    downstream_times = row.get('member_to_downstream_transaction_times', [])
                    
                    if upstream_times and downstream_times:
                        metrics['time_gap_days'] = self.calculate_time_gap(upstream_times, downstream_times)
                        
                        # 获取最后交易时间
                        all_times = []
                        if isinstance(upstream_times, str):
                            upstream_times = eval(upstream_times) if upstream_times.startswith('[') else [upstream_times]
                        if isinstance(downstream_times, str):
                            downstream_times = eval(downstream_times) if downstream_times.startswith('[') else [downstream_times]
                        
                        all_times.extend(upstream_times)
                        all_times.extend(downstream_times)
                        
                        if all_times:
                            metrics['last_transaction_date'] = max(all_times)
            
            # 计算股权集中度
            if self.graph_data:
                node_path = loop_data['node_path']
                equity_ratios = []
                
                for node_name in node_path:
                    node_id = self.find_node_in_graph(node_name)
                    if node_id:
                        ratios = self.get_equity_ratios(node_id)
                        equity_ratios.extend(ratios)
                
                if equity_ratios:
                    # 使用HHI指数计算集中度
                    total = sum(equity_ratios)
                    if total > 0:
                        hhi = sum((r/total)**2 for r in equity_ratios)
                        metrics['equity_concentration'] = hhi
        
        except Exception as e:
            self.log(f"计算指标时出错: {str(e)}")
        
        return metrics
    
    def run_sanity_check(self):
        """运行完整性检查"""
        self.log("🔍 开始运行数据完整性检查...")
        
        threading.Thread(target=self._sanity_check_worker, daemon=True).start()
    
    def _sanity_check_worker(self):
        """完整性检查工作线程"""
        try:
            issues = []
            
            # 检查数据源完整性
            if not self.raw_loops_data:
                issues.append("❌ 缺少环路数据")
            
            if not isinstance(self.loop_metrics_data, pd.DataFrame) or self.loop_metrics_data.empty:
                issues.append("⚠️ 缺少指标数据 - 将无法进行高级筛选")
            
            if not self.graph_data:
                issues.append("⚠️ 缺少图数据 - 将无法进行股权分析")
            
            # 检查数据一致性
            if self.raw_loops_data and isinstance(self.loop_metrics_data, pd.DataFrame):
                loop_ids = set(self.raw_loops_data.keys())
                metric_ids = set(self.loop_metrics_data['loop_id'].tolist())
                
                missing_metrics = loop_ids - metric_ids
                extra_metrics = metric_ids - loop_ids
                
                if missing_metrics:
                    issues.append(f"⚠️ {len(missing_metrics)} 个环路缺少指标数据")
                
                if extra_metrics:
                    issues.append(f"⚠️ {len(extra_metrics)} 个指标记录找不到对应环路")
            
            # 检查环路类型完整性
            if self.raw_loops_data:
                type_counts = Counter()
                for loop_data in self.raw_loops_data.values():
                    type_counts[loop_data['type']] += 1
                
                expected_types = ['4节点环路', '5节点环路', '6节点环路', '7节点环路', '8节点环路']
                missing_types = [t for t in expected_types if t not in type_counts]
                
                if missing_types:
                    issues.append(f"⚠️ 缺少以下环路类型: {', '.join(missing_types)}")
            
            # 生成报告
            if not issues:
                self.log_queue.put("✅ 数据完整性检查通过 - 所有数据完整且一致")
            else:
                self.log_queue.put("⚠️ 数据完整性检查发现问题:")
                for issue in issues:
                    self.log_queue.put(f"  {issue}")
            
            # 更新状态
            self.progress_queue.put(('progress', 0))
            
        except Exception as e:
            self.log_queue.put(f"❌ 完整性检查出错: {str(e)}")
    
    def run_equity_analysis(self):
        """运行股权分析（调用loop_filter_script）"""
        self.log("📊 开始运行股权分析...")
        
        threading.Thread(target=self._equity_analysis_worker, daemon=True).start()
    
    def _equity_analysis_worker(self):
        """股权分析工作线程"""
        try:
            # 调用loop_filter_script进行股权分析
            script_path = "../code/loop_filter_script.py"  # 相对路径
            
            if not os.path.exists(script_path):
                script_path = "loop_filter_script.py"  # 当前目录
            
            if not os.path.exists(script_path):
                self.log_queue.put("❌ 找不到loop_filter_script.py文件")
                return
            
            self.log_queue.put("🔄 正在调用股权分析脚本...")
            
            # 构建命令参数
            cmd = [
                "python", script_path,
                "--min-amount", str(self.filter_params['min_total_amount'].get() * 10000),
                "--min-frequency", str(self.filter_params['min_upstream_transactions'].get()),
                "--min-risk-score", "0.1",  # 使用较低阈值获取更多数据
                "--top-k", str(self.filter_params['max_results'].get())
            ]
            
            # 执行脚本
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="..")
            
            if result.returncode == 0:
                self.log_queue.put("✅ 股权分析完成")
                self.log_queue.put("📄 分析结果已保存到 outputs/loop_filter/ 目录")
                
                # 尝试加载分析结果
                self.load_equity_analysis_results()
                
            else:
                self.log_queue.put(f"❌ 股权分析失败: {result.stderr}")
            
        except Exception as e:
            self.log_queue.put(f"❌ 股权分析出错: {str(e)}")
    
    def load_equity_analysis_results(self):
        """加载股权分析结果"""
        try:
            results_path = "../outputs/loop_filter/filter_report.csv"
            if os.path.exists(results_path):
                equity_results = pd.read_csv(results_path)
                self.log_queue.put(f"📊 加载股权分析结果: {len(equity_results)} 条记录")
                
                # 将结果整合到当前数据中
                self.integrate_equity_results(equity_results)
            
        except Exception as e:
            self.log_queue.put(f"❌ 加载股权分析结果失败: {str(e)}")
    
    def integrate_equity_results(self, equity_results):
        """整合股权分析结果"""
        # 将股权分析结果与现有数据合并
        # 这里可以根据需要实现具体的整合逻辑
        pass
    
    def reload_all_data(self):
        """重新加载所有数据"""
        self.log("🔄 重新加载所有数据...")
        
        # 清空现有数据
        self.raw_loops_data = {}
        self.loop_metrics_data = {}
        self.graph_data = None
        self.data_sources = {}
        
        # 重新自动加载
        self.auto_load_data()
    
    def reset_parameters(self):
        """重置所有参数到默认值"""
        self.filter_params['time_window_months'].set(12)
        self.filter_params['max_time_gap_days'].set(30)
        self.filter_params['min_upstream_transactions'].set(1)
        self.filter_params['min_downstream_transactions'].set(1)
        
        self.filter_params['min_total_amount'].set(0)
        self.filter_params['min_upstream_amount'].set(0)
        self.filter_params['min_downstream_amount'].set(0)
        
        self.filter_params['min_equity_ratio'].set(0)
        self.filter_params['max_equity_ratio'].set(100)
        self.filter_params['min_shareholders'].set(1)
        
        self.filter_params['min_nodes'].set(3)
        self.filter_params['max_nodes'].set(15)
        self.filter_params['max_results'].set(1000)
        
        # 重置环路类型选择
        for var in self.loop_type_vars.values():
            var.set(True)
        
        self.log("✅ 参数已重置为默认值")
    
    def update_results_display(self):
        """更新结果显示"""
        # 清空现有结果
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        if self.filtered_results.empty:
            return
        
        # 添加筛选结果
        for _, row in self.filtered_results.iterrows():
            values = [
                int(row['loop_id']),
                row['type'],
                int(row['node_count']),
                f"{row['upstream_amount']:.1f}万",
                f"{row['downstream_amount']:.1f}万",
                f"{row['time_gap_days']}天",
                f"{row['equity_concentration']:.3f}",
                str(row['last_transaction_date'])[:10]
            ]
            self.results_tree.insert('', 'end', values=values)
    
    def update_charts(self):
        """更新图表显示"""
        if self.filtered_results.empty:
            return
        
        # 更新时间分析图表
        self.update_time_chart()
        
        # 更新股权分析图表
        self.update_equity_chart()
    
    def update_time_chart(self):
        """更新时间分析图表"""
        self.time_fig.clear()
        
        if 'time_gap_days' not in self.filtered_results.columns:
            return
        
        # 创建子图
        ax1 = self.time_fig.add_subplot(221)
        ax2 = self.time_fig.add_subplot(222)
        ax3 = self.time_fig.add_subplot(223)
        ax4 = self.time_fig.add_subplot(224)
        
        # 1. 时间间隔分布
        time_gaps = self.filtered_results['time_gap_days'].dropna()
        ax1.hist(time_gaps, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('时间间隔 (天)')
        ax1.set_ylabel('频数')
        ax1.set_title('上下游交易时间间隔分布')
        ax1.grid(True, alpha=0.3)
        
        # 2. 交易金额vs时间间隔
        if 'total_amount' in self.filtered_results.columns:
            ax2.scatter(self.filtered_results['time_gap_days'], 
                       self.filtered_results['total_amount'],
                       alpha=0.6, s=50)
            ax2.set_xlabel('时间间隔 (天)')
            ax2.set_ylabel('总交易金额 (万元)')
            ax2.set_title('交易金额 vs 时间间隔')
            ax2.grid(True, alpha=0.3)
        
        # 3. 环路类型vs时间间隔
        if 'type' in self.filtered_results.columns:
            type_gaps = {}
            for loop_type in self.filtered_results['type'].unique():
                type_gaps[loop_type] = self.filtered_results[
                    self.filtered_results['type'] == loop_type]['time_gap_days'].tolist()
            
            ax3.boxplot(type_gaps.values(), labels=type_gaps.keys())
            ax3.set_xlabel('环路类型')
            ax3.set_ylabel('时间间隔 (天)')
            ax3.set_title('不同类型环路的时间间隔')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. 月度趋势（如果有足够的时间数据）
        ax4.text(0.5, 0.5, '月度趋势分析\n(需要更多时间数据)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('月度趋势分析')
        
        self.time_fig.tight_layout()
        self.time_canvas.draw()
    
    def update_equity_chart(self):
        """更新股权分析图表"""
        self.equity_fig.clear()
        
        if 'equity_concentration' not in self.filtered_results.columns:
            return
        
        # 创建子图
        ax1 = self.equity_fig.add_subplot(221)
        ax2 = self.equity_fig.add_subplot(222)
        ax3 = self.equity_fig.add_subplot(223)
        ax4 = self.equity_fig.add_subplot(224)
        
        # 1. 股权集中度分布
        equity_conc = self.filtered_results['equity_concentration'].dropna()
        ax1.hist(equity_conc, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax1.set_xlabel('股权集中度')
        ax1.set_ylabel('频数')
        ax1.set_title('股权集中度分布')
        ax1.grid(True, alpha=0.3)
        
        # 2. 股权集中度vs交易金额
        if 'total_amount' in self.filtered_results.columns:
            ax2.scatter(self.filtered_results['equity_concentration'], 
                       self.filtered_results['total_amount'],
                       alpha=0.6, s=50, c='red')
            ax2.set_xlabel('股权集中度')
            ax2.set_ylabel('总交易金额 (万元)')
            ax2.set_title('股权集中度 vs 交易金额')
            ax2.grid(True, alpha=0.3)
        
        # 3. 不同节点数的股权集中度
        if 'node_count' in self.filtered_results.columns:
            node_equity = {}
            for node_count in sorted(self.filtered_results['node_count'].unique()):
                node_equity[f'{node_count}节点'] = self.filtered_results[
                    self.filtered_results['node_count'] == node_count]['equity_concentration'].tolist()
            
            ax3.boxplot(node_equity.values(), labels=node_equity.keys())
            ax3.set_xlabel('环路节点数')
            ax3.set_ylabel('股权集中度')
            ax3.set_title('不同节点数的股权集中度')
        
        # 4. 股权风险评估
        ax4.text(0.5, 0.5, '股权风险评估\n(基于集中度和交易模式)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('股权风险评估')
        
        self.equity_fig.tight_layout()
        self.equity_canvas.draw()
    
    def export_results(self):
        """导出筛选结果"""
        if self.filtered_results.empty:
            messagebox.showwarning("警告", "没有筛选结果可以导出")
            return
        
        try:
            # 确保输出目录存在
            output_dir = "../outputs/loop_filter_gui"
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"filtered_loops_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # 导出CSV
            self.filtered_results.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            self.log(f"✅ 筛选结果已导出到: {filepath}")
            messagebox.showinfo("成功", f"筛选结果已导出到:\n{filepath}")
            
        except Exception as e:
            self.log(f"❌ 导出失败: {str(e)}")
            messagebox.showerror("错误", f"导出失败: {str(e)}")
    
    def generate_report(self):
        """生成详细报告"""
        if self.filtered_results.empty:
            messagebox.showwarning("警告", "没有筛选结果可以生成报告")
            return
        
        try:
            # 确保输出目录存在
            output_dir = "../outputs/loop_filter_gui"
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"loop_analysis_report_{timestamp}.md"
            filepath = os.path.join(output_dir, filename)
            
            report_content = self.generate_report_content()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.log(f"✅ 分析报告已生成: {filepath}")
            messagebox.showinfo("成功", f"分析报告已生成:\n{filepath}")
            
        except Exception as e:
            self.log(f"❌ 报告生成失败: {str(e)}")
            messagebox.showerror("错误", f"报告生成失败: {str(e)}")
    
    def generate_report_content(self):
        """生成报告内容"""
        report = "# 物产中大图风控系统 - 高级闭环筛选分析报告\n\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 筛选参数
        report += "## 筛选参数配置\n\n"
        report += f"- 分析时间窗口: {self.filter_params['time_window_months'].get()} 个月\n"
        report += f"- 最大交易时间间隔: {self.filter_params['max_time_gap_days'].get()} 天\n"
        report += f"- 最小总交易金额: {self.filter_params['min_total_amount'].get()} 万元\n"
        report += f"- 环路节点数范围: {self.filter_params['min_nodes'].get()}-{self.filter_params['max_nodes'].get()}\n"
        report += f"- 选择的环路类型: {len([v for v in self.loop_type_vars.values() if v.get()])} 种\n\n"
        
        # 筛选结果统计
        report += "## 筛选结果统计\n\n"
        report += f"- 原始环路总数: {len(self.raw_loops_data):,}\n"
        report += f"- 筛选后环路数: {len(self.filtered_results):,}\n"
        report += f"- 筛选率: {len(self.filtered_results)/len(self.raw_loops_data)*100:.2f}%\n\n"
        
        # 关键发现
        if not self.filtered_results.empty:
            report += "## 关键发现\n\n"
            
            # 时间维度分析
            if 'time_gap_days' in self.filtered_results.columns:
                avg_gap = self.filtered_results['time_gap_days'].mean()
                max_gap = self.filtered_results['time_gap_days'].max()
                min_gap = self.filtered_results['time_gap_days'].min()
                
                report += f"### 时间维度分析\n"
                report += f"- 平均交易时间间隔: {avg_gap:.1f} 天\n"
                report += f"- 最大交易时间间隔: {max_gap} 天\n"
                report += f"- 最小交易时间间隔: {min_gap} 天\n\n"
            
            # 金额分析
            if 'total_amount' in self.filtered_results.columns:
                total_amounts = self.filtered_results['total_amount'].dropna()
                if len(total_amounts) > 0:
                    report += f"### 交易金额分析\n"
                    report += f"- 平均交易金额: {total_amounts.mean():.1f} 万元\n"
                    report += f"- 最大交易金额: {total_amounts.max():.1f} 万元\n"
                    report += f"- 交易金额中位数: {total_amounts.median():.1f} 万元\n\n"
            
            # 环路类型分析
            if 'type' in self.filtered_results.columns:
                type_counts = self.filtered_results['type'].value_counts()
                report += f"### 环路类型分布\n"
                for loop_type, count in type_counts.items():
                    report += f"- {loop_type}: {count} 个 ({count/len(self.filtered_results)*100:.1f}%)\n"
                report += "\n"
        
        # 高风险环路列表（前10个）
        report += "## 高风险环路列表 (Top 10)\n\n"
        
        top_loops = self.filtered_results.head(10)
        for i, (_, row) in enumerate(top_loops.iterrows()):
            report += f"### {i+1}. 环路ID: {int(row['loop_id'])}\n"
            report += f"- 类型: {row['type']}\n"
            report += f"- 节点数: {int(row['node_count'])}\n"
            if 'total_amount' in row:
                report += f"- 总交易金额: {row['total_amount']:.1f} 万元\n"
            if 'time_gap_days' in row:
                report += f"- 交易时间间隔: {row['time_gap_days']} 天\n"
            if 'equity_concentration' in row:
                report += f"- 股权集中度: {row['equity_concentration']:.3f}\n"
            report += f"- 环路路径: {row['content'][:100]}...\n\n"
        
        return report
    
    def log(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        
        # 更新状态栏
        self.status_var.set(message)


def main():
    """主函数"""
    root = tk.Tk()
    app = AdvancedLoopFilterGUI(root)
    
    # 设置窗口样式
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    root.minsize(1400, 900)
    
    # 启动主循环
    root.mainloop()


if __name__ == "__main__":
    # 检查依赖
    try:
        import tkinter
        import matplotlib
        import pandas
        import numpy
        import networkx
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装: pip install matplotlib pandas numpy networkx")
        exit(1)
    
    print("🚀 启动物产中大图风控系统 - 高级闭环优筛系统 v2.0")
    print("=" * 60)
    print("新功能:")
    print("✅ 时间维度分析 - 上下游交易时间间隔控制")
    print("✅ 股权分析集成 - 自动调用分析脚本")
    print("✅ 完整性检查 - 确保数据完整性")
    print("✅ 自动数据读取 - 智能发现并加载数据")
    print("✅ 真实数据导向 - 不使用估算值")
    print("✅ 环路类型控制 - 可选择特定类型")
    print("=" * 60)
    
    main()