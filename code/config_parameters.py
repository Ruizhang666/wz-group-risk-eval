#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图风控系统参数配置中心

本脚本集中管理所有可调参数，方便统一配置和维护。
参数分为几个主要类别：环路检测、环路筛选、性能控制、可视化配置等。
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

@dataclass
class LoopDetectionConfig:
    """环路检测参数配置"""
    
    # === 基本输入输出配置 ===
    input_graph_path: str = os.path.join(PROJECT_ROOT, "model", "final_heterogeneous_graph.graphml")
    simplified_graph_path: str = os.path.join(PROJECT_ROOT, "model", "simplified_loop_detection_graph.graphml")
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs")
    
    # === 环路检测范围配置 ===
    min_cycle_length: int = 3  # 最小环长度（3节点环）
    max_cycle_length: int = 8  # 最大环长度（8节点环）
    search_depth_limit: int = 12  # DFS搜索深度限制
    
    # === 性能控制参数 ===
    max_cycles_to_process: int = 100000  # 最多处理的环数量
    max_paths_per_search: int = 10000  # 每次搜索最多路径数
    memory_limit_mb: int = 4096  # 内存使用上限(MB)
    timeout_seconds: int = 3600  # 单次搜索超时时间(秒)
    
    # === 并行处理配置 ===
    enable_multiprocessing: bool = True  # 是否启用多进程
    max_workers: int = 4  # 最大工作进程数
    chunk_size: int = 1000  # 数据块大小
    
    # === 搜索策略配置 ===
    enable_bidirectional_search: bool = True  # 是否启用双向搜索
    enable_path_caching: bool = True  # 是否启用路径缓存
    cache_size_limit: int = 50000  # 缓存大小限制
    
    # === 过滤条件配置 ===
    exclude_self_loops: bool = True  # 排除自环
    exclude_duplicate_edges: bool = True  # 排除重复边
    min_unique_entities: int = 3  # 最少唯一实体数
    
    def __post_init__(self):
        """参数验证和后处理"""
        assert self.min_cycle_length >= 3, "最小环长度必须大于等于3"
        assert self.max_cycle_length <= 20, "最大环长度不能超过20"
        assert self.min_cycle_length <= self.max_cycle_length, "最小环长度不能大于最大环长度"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

@dataclass 
class LoopFilterConfig:
    """环路筛选参数配置"""
    
    # === 交易金额相关参数 ===
    min_transaction_amount: float = 10000  # 最小交易总额（元）
    """
    说明：筛选交易总额大于此阈值的环路
    - 10,000: 较宽松，包含小额交易
    - 100,000: 中等，过滤极小交易
    - 1,000,000: 严格，只关注大额交易
    """
    
    max_transaction_amount: float = float('inf')  # 最大交易总额（元）
    
    # === 交易频率相关参数 ===
    min_transaction_frequency: int = 1  # 最小交易频率（次）
    """
    说明：在指定时间窗口内的最小交易次数
    - 1: 只要有交易就算
    - 3: 需要有持续的交易行为
    - 5: 要求频繁交易
    """
    
    transaction_time_window_months: int = 12  # 交易时间窗口（月）
    time_concentration_threshold: float = 0.3  # 时间集中度阈值
    """
    说明：交易在时间窗口内的集中程度
    - 0.3: 相对分散的交易时间
    - 0.5: 中等集中度
    - 0.8: 高度集中，可能存在异常
    """
    
    # === 股权相关参数 ===
    min_shareholder_ratio: float = 0.01  # 最小股东持股比例
    """
    说明：参与环路的股东最小持股比例
    - 0.01 (1%): 很宽松，包含小股东
    - 0.05 (5%): 中等，有一定影响力
    - 0.10 (10%): 严格，只关注大股东
    """
    
    max_shareholder_ratio: float = 0.99  # 最大股东持股比例
    equity_concentration_threshold: float = 0.3  # 股权集中度阈值（HHI指数）
    """
    说明：股权集中度（赫芬达尔指数）
    - 0.3: 相对分散的股权结构
    - 0.5: 中等集中度
    - 0.7: 高度集中，可能存在控制关系
    """
    
    # === 环路结构参数 ===
    min_loop_nodes: int = 3  # 最小环路节点数
    max_loop_nodes: int = 10  # 最大环路节点数
    min_member_companies: int = 0  # 最少成员单位数量
    """
    说明：环路中成员单位的最少数量
    - 0: 不限制成员单位数量
    - 1: 至少包含一个成员单位
    - 2: 需要多个成员单位参与
    """
    
    allowed_loop_types: Optional[List[str]] = None  # 允许的环路类型
    """
    说明：只保留指定类型的环路，None表示不过滤
    例如：['控股环路', '担保环路', '关联交易环路']
    """
    
    # === 风险评分权重 ===
    weight_transaction_amount: float = 0.3  # 交易金额权重
    weight_transaction_frequency: float = 0.2  # 交易频率权重  
    weight_time_concentration: float = 0.2  # 时间集中度权重
    weight_equity_concentration: float = 0.2  # 股权集中度权重
    weight_loop_complexity: float = 0.1  # 环路复杂度权重
    """
    说明：各项风险指标的权重，总和应为1.0
    可根据业务重点调整各权重：
    - 重视交易规模：增加transaction_amount权重
    - 重视频繁交易：增加transaction_frequency权重
    - 重视股权控制：增加equity_concentration权重
    """
    
    # === 输出控制参数 ===
    min_risk_score: float = 0.1  # 最小风险分数阈值
    """
    说明：只输出风险分数大于此阈值的环路
    - 0.1: 非常宽松，包含低风险环路
    - 0.3: 中等，过滤明显的低风险
    - 0.5: 严格，只关注中高风险
    - 0.7: 非常严格，只输出高风险
    """
    
    top_k_results: int = 100  # 输出前K个高风险环路
    
    def __post_init__(self):
        """参数验证"""
        # 权重总和检查
        total_weight = (self.weight_transaction_amount + 
                       self.weight_transaction_frequency +
                       self.weight_time_concentration + 
                       self.weight_equity_concentration +
                       self.weight_loop_complexity)
        
        assert abs(total_weight - 1.0) < 0.01, f"权重总和应为1.0，当前为{total_weight:.3f}"
        assert self.min_shareholder_ratio <= self.max_shareholder_ratio
        assert self.min_loop_nodes <= self.max_loop_nodes

@dataclass
class VisualizationConfig:
    """可视化参数配置"""
    
    # === 基本配置 ===
    figure_size: Tuple[int, int] = (12, 8)  # 图表大小
    dpi: int = 300  # 图像分辨率
    style: str = 'seaborn-v0_8'  # 图表样式
    
    # === 颜色配置 ===
    color_palette: str = 'viridis'  # 调色板
    risk_colors: Dict[str, str] = field(default_factory=lambda: {
        'low': '#2E8B57',      # 低风险 - 海绿色
        'medium': '#FF8C00',   # 中风险 - 橙色  
        'high': '#DC143C',     # 高风险 - 红色
        'critical': '#8B0000'  # 极高风险 - 深红色
    })
    
    # === 图表配置 ===
    show_grid: bool = True  # 显示网格
    show_legend: bool = True  # 显示图例
    font_size: int = 12  # 字体大小
    title_font_size: int = 16  # 标题字体大小
    
    # === 直方图配置 ===
    hist_bins: int = 50  # 直方图分箱数
    hist_alpha: float = 0.7  # 透明度
    
    # === 散点图配置 ===
    scatter_size: int = 50  # 散点大小
    scatter_alpha: float = 0.6  # 透明度
    
    # === 箱型图配置 ===
    box_showfliers: bool = True  # 显示异常值
    box_patch_alpha: float = 0.7  # 箱体透明度
    
    # === 输出配置 ===
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs", "visualizations")
    save_format: str = 'png'  # 保存格式: png, pdf, svg
    save_dpi: int = 300  # 保存时的DPI
    
    def __post_init__(self):
        """创建输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)

@dataclass
class SystemConfig:
    """系统级参数配置"""
    
    # === 日志配置 ===
    log_level: str = 'INFO'  # 日志级别: DEBUG, INFO, WARNING, ERROR
    log_format: str = '%(asctime)s - %(levelname)s - %(message)s'
    enable_file_logging: bool = True  # 启用文件日志
    enable_console_logging: bool = True  # 启用控制台日志
    log_rotation_size: int = 10  # 日志轮转大小(MB)
    
    # === 性能监控 ===
    enable_memory_monitoring: bool = True  # 启用内存监控
    memory_check_interval: int = 1000  # 内存检查间隔(操作次数)
    enable_progress_bar: bool = True  # 启用进度条
    progress_update_interval: int = 1000  # 进度更新间隔
    
    # === 文件路径配置 ===
    temp_dir: str = os.path.join(PROJECT_ROOT, "temp")  # 临时文件目录
    backup_dir: str = os.path.join(PROJECT_ROOT, "backup")  # 备份目录
    
    # === 错误处理 ===
    enable_error_recovery: bool = True  # 启用错误恢复
    max_retry_attempts: int = 3  # 最大重试次数
    retry_delay_seconds: float = 1.0  # 重试延迟(秒)
    
    def __post_init__(self):
        """创建必要的目录"""
        for directory in [self.temp_dir, self.backup_dir]:
            os.makedirs(directory, exist_ok=True)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.loop_detection = LoopDetectionConfig()
        self.loop_filter = LoopFilterConfig()  
        self.visualization = VisualizationConfig()
        self.system = SystemConfig()
        
    def save_to_file(self, filepath: str):
        """保存配置到文件"""
        config_dict = {
            'loop_detection': self.loop_detection.__dict__,
            'loop_filter': self.loop_filter.__dict__,
            'visualization': self.visualization.__dict__,
            'system': self.system.__dict__,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
    
    def load_from_file(self, filepath: str):
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 更新各配置对象
        for key, value in config_dict.get('loop_detection', {}).items():
            if hasattr(self.loop_detection, key):
                setattr(self.loop_detection, key, value)
                
        for key, value in config_dict.get('loop_filter', {}).items():
            if hasattr(self.loop_filter, key):
                setattr(self.loop_filter, key, value)
                
        for key, value in config_dict.get('visualization', {}).items():
            if hasattr(self.visualization, key):
                setattr(self.visualization, key, value)
                
        for key, value in config_dict.get('system', {}).items():
            if hasattr(self.system, key):
                setattr(self.system, key, value)
    
    def get_quick_configs(self) -> Dict[str, Dict]:
        """获取几种预设的快速配置"""
        return {
            "strict_mode": {
                "description": "严格模式 - 高阈值，只输出高风险环路",
                "loop_filter": {
                    "min_transaction_amount": 1000000,  # 100万
                    "min_transaction_frequency": 3,
                    "min_shareholder_ratio": 0.05,  # 5%
                    "equity_concentration_threshold": 0.7,
                    "min_risk_score": 0.5,
                    "top_k_results": 50
                }
            },
            
            "loose_mode": {
                "description": "宽松模式 - 低阈值，包含更多环路",  
                "loop_filter": {
                    "min_transaction_amount": 10000,  # 1万
                    "min_transaction_frequency": 1,
                    "min_shareholder_ratio": 0.01,  # 1%
                    "equity_concentration_threshold": 0.3,
                    "min_risk_score": 0.1,
                    "top_k_results": 200
                }
            },
            
            "balanced_mode": {
                "description": "平衡模式 - 中等阈值，适合大多数场景",
                "loop_filter": {
                    "min_transaction_amount": 100000,  # 10万
                    "min_transaction_frequency": 2,
                    "min_shareholder_ratio": 0.03,  # 3%
                    "equity_concentration_threshold": 0.5,
                    "min_risk_score": 0.3,
                    "top_k_results": 100
                }
            },
            
            "performance_mode": {
                "description": "性能模式 - 优化速度，降低精度",
                "loop_detection": {
                    "max_cycle_length": 6,  # 减少环长度
                    "max_cycles_to_process": 50000,
                    "enable_multiprocessing": True,
                    "max_workers": 8
                },
                "loop_filter": {
                    "min_risk_score": 0.2,
                    "top_k_results": 50
                }
            }
        }
    
    def apply_quick_config(self, config_name: str):
        """应用快速配置"""
        quick_configs = self.get_quick_configs()
        
        if config_name not in quick_configs:
            raise ValueError(f"未知的配置名称: {config_name}")
        
        config = quick_configs[config_name]
        
        # 应用配置
        for section_name, section_config in config.items():
            if section_name == "description":
                continue
                
            if section_name == "loop_detection":
                for key, value in section_config.items():
                    if hasattr(self.loop_detection, key):
                        setattr(self.loop_detection, key, value)
                        
            elif section_name == "loop_filter":
                for key, value in section_config.items():
                    if hasattr(self.loop_filter, key):
                        setattr(self.loop_filter, key, value)
    
    def print_current_config(self):
        """打印当前配置"""
        print("=== 当前配置参数 ===\n")
        
        print("【环路检测配置】")
        print(f"- 环长度范围: {self.loop_detection.min_cycle_length} - {self.loop_detection.max_cycle_length}")
        print(f"- 最大处理环数: {self.loop_detection.max_cycles_to_process:,}")
        print(f"- 多进程: {self.loop_detection.enable_multiprocessing}")
        print(f"- 工作进程数: {self.loop_detection.max_workers}")
        print()
        
        print("【环路筛选配置】")
        print(f"- 最小交易金额: {self.loop_filter.min_transaction_amount:,.0f} 元")
        print(f"- 最小交易频率: {self.loop_filter.min_transaction_frequency} 次")
        print(f"- 最小持股比例: {self.loop_filter.min_shareholder_ratio:.1%}")
        print(f"- 股权集中度阈值: {self.loop_filter.equity_concentration_threshold:.1%}")
        print(f"- 最小风险分数: {self.loop_filter.min_risk_score}")
        print(f"- 输出环路数: {self.loop_filter.top_k_results}")
        print()
        
        print("【可视化配置】")
        print(f"- 图表大小: {self.visualization.figure_size}")
        print(f"- 分辨率: {self.visualization.dpi} DPI")
        print(f"- 直方图分箱: {self.visualization.hist_bins}")
        print()

# 创建全局配置实例
config = ConfigManager()

def main():
    """示例用法"""
    print("图风控系统参数配置中心")
    print("=" * 50)
    
    # 显示当前配置
    config.print_current_config()
    
    # 显示可用的快速配置
    print("【可用的快速配置】")
    quick_configs = config.get_quick_configs()
    for name, cfg in quick_configs.items():
        print(f"- {name}: {cfg['description']}")
    print()
    
    # 保存当前配置
    config_file = os.path.join(PROJECT_ROOT, "config", "current_config.json")
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    config.save_to_file(config_file)
    print(f"配置已保存到: {config_file}")
    
    # 示例：应用严格模式配置
    print("\n应用严格模式配置...")
    config.apply_quick_config("strict_mode")
    print("严格模式配置已应用！")
    
    config.print_current_config()

if __name__ == "__main__":
    main() 