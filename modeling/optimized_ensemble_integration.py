#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版异常检测集成层 - 并行处理 + 贝叶斯优化集成
将异常检测算法的结果进行高效智能集成，生成最终异常分数

优化特性：
1. 并行处理集成方法
2. 贝叶斯优化集成权重  
3. 更智能的集成策略选择
4. 增强的可视化和分析

作者: AI助手
日期: 2024年
版本: v2.0 - 超级优化版
"""

import pandas as pd
import numpy as np
import warnings
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import cross_val_score
from scipy import stats
from scipy.stats import rankdata, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler

# 忽略警告
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/anomaly_detection/logs/optimized_ensemble.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class OptimizedEnsembleIntegrator:
    """优化版异常检测集成器"""
    
    def __init__(self, model_scores_file: str, n_jobs: int = -1, optimization_trials: int = 30):
        """
        初始化优化集成器
        
        参数:
            model_scores_file: 模型分数文件路径
            n_jobs: 并行工作数量
            optimization_trials: 贝叶斯优化试验次数
        """
        self.model_scores_file = model_scores_file
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.optimization_trials = optimization_trials
        
        # 数据存储
        self.model_scores = None
        self.model_names = []
        self.loop_ids = []
        self.score_columns = []
        self.rank_columns = []
        self.ensemble_results = {}
        
        # 创建输出目录
        self.output_dir = Path('outputs/anomaly_detection')
        self.ensemble_dir = self.output_dir / 'ensemble'
        self.viz_dir = self.output_dir / 'visualizations'
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化优化异常检测集成器 - CPU核心数: {self.n_jobs}")
    
    def load_model_scores(self) -> pd.DataFrame:
        """加载模型分数数据"""
        logger.info(f"加载模型分数数据: {self.model_scores_file}")
        
        self.model_scores = pd.read_csv(self.model_scores_file)
        self.loop_ids = self.model_scores['loop_id'].values
        
        # 识别分数列和排名列
        self.score_columns = [col for col in self.model_scores.columns if col.endswith('_score')]
        self.rank_columns = [col for col in self.model_scores.columns if col.endswith('_rank')]
        
        # 提取模型名称
        self.model_names = [col.replace('_score', '') for col in self.score_columns]
        
        logger.info(f"数据加载完成: {len(self.loop_ids)} 个环路, {len(self.model_names)} 个模型")
        
        return self.model_scores
    
    def normalize_scores_parallel(self, score_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """并行标准化分数矩阵"""
        logger.info("并行标准化分数矩阵...")
        
        def normalize_column(args):
            col_idx, scores, method = args
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                return col_idx, scores
            
            return col_idx, scaler.fit_transform(scores.reshape(-1, 1)).flatten()
        
        normalized_matrices = {}
        
        for method in ['minmax', 'standard', 'robust']:
            # 准备并行任务
            tasks = [(i, score_matrix[:, i], method) for i in range(score_matrix.shape[1])]
            
            # 并行处理
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(normalize_column, tasks))
            
            # 重组结果
            normalized_matrix = np.zeros_like(score_matrix)
            for col_idx, normalized_col in results:
                normalized_matrix[:, col_idx] = normalized_col
            
            normalized_matrices[method] = normalized_matrix
        
        return normalized_matrices
    
    def optimize_ensemble_weights(self, score_matrix: np.ndarray) -> Dict[str, Any]:
        """使用贝叶斯优化寻找最佳集成权重"""
        logger.info("使用贝叶斯优化寻找最佳集成权重...")
        
        def objective(trial):
            # 为每个模型建议权重
            weights = []
            for i in range(score_matrix.shape[1]):
                weight = trial.suggest_float(f'weight_{i}', 0.01, 1.0)
                weights.append(weight)
            
            # 归一化权重
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # 计算加权平均分数
            weighted_scores = np.average(score_matrix, axis=1, weights=weights)
            
            # 评估目标：最大化分数的分离度
            score_std = np.std(weighted_scores)
            score_range = np.max(weighted_scores) - np.min(weighted_scores)
            separation_quality = score_std * score_range
            
            return separation_quality
        
        # 创建optuna研究
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # 执行优化
        study.optimize(objective, n_trials=self.optimization_trials)
        
        # 提取最佳权重
        best_weights = []
        for i in range(score_matrix.shape[1]):
            weight = study.best_params[f'weight_{i}']
            best_weights.append(weight)
        
        # 归一化权重
        best_weights = np.array(best_weights)
        best_weights = best_weights / np.sum(best_weights)
        
        return {
            'best_weights': best_weights,
            'best_value': study.best_value,
            'optimization_trials': len(study.trials)
        }
    
    def advanced_ensemble_methods(self, normalized_matrices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """高级集成方法"""
        logger.info("执行高级集成方法...")
        
        ensemble_results = {}
        
        for norm_method, norm_matrix in normalized_matrices.items():
            logger.info(f"处理 {norm_method} 标准化数据...")
            
            # 1. 贝叶斯优化权重集成
            opt_result = self.optimize_ensemble_weights(norm_matrix)
            opt_weighted_scores = np.average(norm_matrix, axis=1, weights=opt_result['best_weights'])
            ensemble_results[f'bayesian_weighted_{norm_method}'] = opt_weighted_scores
            
            # 2. 基于相关性的动态权重
            corr_matrix = np.corrcoef(norm_matrix.T)
            diversity_weights = self._calculate_diversity_weights(corr_matrix)
            diversity_scores = np.average(norm_matrix, axis=1, weights=diversity_weights)
            ensemble_results[f'diversity_weighted_{norm_method}'] = diversity_scores
            
            # 3. 基于性能的自适应权重
            performance_weights = self._calculate_performance_weights(norm_matrix)
            performance_scores = np.average(norm_matrix, axis=1, weights=performance_weights)
            ensemble_results[f'performance_weighted_{norm_method}'] = performance_scores
            
            # 4. 鲁棒统计集成
            robust_scores = self._robust_statistical_ensemble(norm_matrix)
            ensemble_results[f'robust_stats_{norm_method}'] = robust_scores
        
        return ensemble_results
    
    def _calculate_diversity_weights(self, corr_matrix: np.ndarray) -> np.ndarray:
        """计算基于多样性的权重"""
        # 计算每个模型与其他模型的平均相关性
        avg_correlations = np.mean(np.abs(corr_matrix), axis=1)
        # 逆相关性权重（相关性越低，权重越高）
        diversity_weights = (1 - avg_correlations) / np.sum(1 - avg_correlations)
        return diversity_weights
    
    def _calculate_performance_weights(self, score_matrix: np.ndarray) -> np.ndarray:
        """计算基于性能的权重"""
        performance_scores = []
        for i in range(score_matrix.shape[1]):
            scores = score_matrix[:, i]
            # 使用变异系数作为性能指标
            cv = np.std(scores) / (np.mean(scores) + 1e-6)
            performance_scores.append(cv)
        
        performance_weights = np.array(performance_scores) / np.sum(performance_scores)
        return performance_weights
    
    def _robust_statistical_ensemble(self, score_matrix: np.ndarray) -> np.ndarray:
        """鲁棒统计集成"""
        # 使用Winsorized均值（去除极值后的均值）
        winsorized_scores = []
        
        for i in range(score_matrix.shape[0]):
            row_scores = score_matrix[i, :]
            # 使用5%和95%分位数进行Winsorizing
            q5, q95 = np.percentile(row_scores, [5, 95])
            winsorized_row = np.clip(row_scores, q5, q95)
            winsorized_scores.append(np.mean(winsorized_row))
        
        return np.array(winsorized_scores)
    
    def parallel_voting_ensemble(self, normalized_matrices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """并行投票集成"""
        logger.info("执行并行投票集成...")
        
        def voting_task(args):
            method_name, matrix, threshold = args
            
            # 硬投票
            votes = np.zeros(len(matrix))
            for i in range(matrix.shape[1]):
                threshold_value = np.percentile(matrix[:, i], threshold)
                votes += (matrix[:, i] >= threshold_value).astype(int)
            
            return method_name, votes / matrix.shape[1]
        
        voting_results = {}
        
        # 准备任务
        tasks = []
        for norm_method, matrix in normalized_matrices.items():
            for threshold in [85, 90, 95, 99]:
                task_name = f'hard_vote_{threshold}_{norm_method}'
                tasks.append((task_name, matrix, threshold))
        
        # 并行执行投票
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(voting_task, tasks))
        
        for method_name, scores in results:
            voting_results[method_name] = scores
        
        return voting_results
    
    def intelligent_ensemble_selection(self, all_ensemble_results: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """智能集成方法选择"""
        logger.info("进行智能集成方法选择...")
        
        # 过滤有效结果
        valid_results = {k: v for k, v in all_ensemble_results.items() 
                        if isinstance(v, np.ndarray) and len(v) > 0}
        
        evaluation_metrics = {}
        
        for method, scores in valid_results.items():
            # 多维度评估
            metrics = {
                'separation_quality': self._calculate_separation_quality(scores),
                'distribution_quality': self._calculate_distribution_quality(scores),
                'rank_stability': self._calculate_rank_stability(scores),
                'outlier_detection_power': self._calculate_outlier_power(scores)
            }
            
            # 综合评分
            composite_score = (
                metrics['separation_quality'] * 0.3 +
                metrics['distribution_quality'] * 0.25 +
                metrics['rank_stability'] * 0.25 +
                metrics['outlier_detection_power'] * 0.2
            )
            
            metrics['composite_score'] = composite_score
            evaluation_metrics[method] = metrics
        
        # 选择最佳方法
        best_method = max(evaluation_metrics.keys(), 
                         key=lambda x: evaluation_metrics[x]['composite_score'])
        
        return {
            'best_method': best_method,
            'best_scores': valid_results[best_method],
            'evaluation_metrics': evaluation_metrics,
            'all_composite_scores': {k: v['composite_score'] for k, v in evaluation_metrics.items()}
        }
    
    def _calculate_separation_quality(self, scores: np.ndarray) -> float:
        """计算分离质量"""
        return np.std(scores) * (np.max(scores) - np.min(scores))
    
    def _calculate_distribution_quality(self, scores: np.ndarray) -> float:
        """计算分布质量"""
        # 使用变异系数
        return np.std(scores) / (np.mean(scores) + 1e-6)
    
    def _calculate_rank_stability(self, scores: np.ndarray) -> float:
        """计算排名稳定性"""
        # 使用分位数间的比值稳定性
        q25, q50, q75 = np.percentile(scores, [25, 50, 75])
        if q50 > 0:
            return (q75 - q25) / q50
        else:
            return 0.0
    
    def _calculate_outlier_power(self, scores: np.ndarray) -> float:
        """计算异常检测能力"""
        # 使用前5%与后95%的分离度
        q95, q5 = np.percentile(scores, [95, 5])
        return q95 - q5
    
    def create_enhanced_visualizations(self, final_results: pd.DataFrame, 
                                     all_ensemble_results: Dict[str, np.ndarray],
                                     selection_result: Dict[str, Any]):
        """创建增强的可视化"""
        logger.info("创建增强可视化...")
        
        # 1. 集成方法性能对比热图
        self._create_method_performance_heatmap(selection_result['evaluation_metrics'])
        
        # 2. 最佳集成方法分析
        self._create_best_method_analysis(final_results, selection_result)
        
        # 3. 集成稳定性分析
        self._create_ensemble_stability_analysis(all_ensemble_results)
        
        # 4. 高维集成结果可视化
        self._create_high_dimensional_visualization(final_results, all_ensemble_results)
    
    def _create_method_performance_heatmap(self, evaluation_metrics: Dict[str, Dict]):
        """创建方法性能热图"""
        methods = list(evaluation_metrics.keys())[:15]  # 显示前15个方法
        metrics = ['separation_quality', 'distribution_quality', 'rank_stability', 'outlier_detection_power']
        
        # 创建数据矩阵
        data_matrix = np.zeros((len(methods), len(metrics)))
        
        for i, method in enumerate(methods):
            for j, metric in enumerate(metrics):
                data_matrix[i, j] = evaluation_metrics[method][metric]
        
        # 标准化到0-1范围
        for j in range(len(metrics)):
            col = data_matrix[:, j]
            data_matrix[:, j] = (col - np.min(col)) / (np.max(col) - np.min(col) + 1e-6)
        
        # 创建热图
        plt.figure(figsize=(12, 10))
        sns.heatmap(data_matrix, 
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=[m.replace('_', ' ') for m in methods],
                   annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('集成方法性能对比热图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'method_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_best_method_analysis(self, final_results: pd.DataFrame, selection_result: Dict[str, Any]):
        """创建最佳方法分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        best_scores = selection_result['best_scores']
        best_method = selection_result['best_method']
        
        # 分数分布
        axes[0,0].hist(best_scores, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0,0].set_title(f'最佳方法 ({best_method}) 分数分布', fontweight='bold')
        axes[0,0].set_xlabel('异常分数')
        axes[0,0].set_ylabel('频次')
        
        # 风险等级分布
        risk_counts = final_results['anomaly_level'].value_counts()
        colors = ['darkred', 'red', 'orange', 'yellow', 'green']
        axes[0,1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
        axes[0,1].set_title('风险等级分布', fontweight='bold')
        
        # 分位数分析
        percentiles = np.arange(1, 100)
        percentile_values = [np.percentile(best_scores, p) for p in percentiles]
        axes[1,0].plot(percentiles, percentile_values, 'b-', linewidth=2)
        axes[1,0].set_title('异常分数分位数图', fontweight='bold')
        axes[1,0].set_xlabel('百分位数')
        axes[1,0].set_ylabel('异常分数')
        
        # 方法评分对比
        composite_scores = selection_result['all_composite_scores']
        top_methods = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        method_names, scores = zip(*top_methods)
        
        axes[1,1].barh(range(len(method_names)), scores, color='skyblue')
        axes[1,1].set_yticks(range(len(method_names)))
        axes[1,1].set_yticklabels([name.replace('_', ' ') for name in method_names])
        axes[1,1].set_title('前10集成方法综合评分', fontweight='bold')
        axes[1,1].set_xlabel('综合评分')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'best_method_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_ensemble_stability_analysis(self, all_ensemble_results: Dict[str, np.ndarray]):
        """创建集成稳定性分析"""
        # 计算方法间相关性
        valid_methods = {k: v for k, v in all_ensemble_results.items() 
                        if isinstance(v, np.ndarray) and len(v) > 0}
        
        if len(valid_methods) < 2:
            return
        
        method_names = list(valid_methods.keys())
        n_methods = len(method_names)
        
        # 计算相关性矩阵
        corr_matrix = np.zeros((n_methods, n_methods))
        
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr, _ = spearmanr(valid_methods[method1], valid_methods[method2])
                    corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
        
        # 创建相关性热图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=[name.replace('_', ' ') for name in method_names],
                   yticklabels=[name.replace('_', ' ') for name in method_names])
        plt.title('集成方法相关性分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'ensemble_stability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_high_dimensional_visualization(self, final_results: pd.DataFrame, 
                                             all_ensemble_results: Dict[str, np.ndarray]):
        """创建高维集成结果可视化"""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # 准备数据矩阵
        valid_methods = {k: v for k, v in all_ensemble_results.items() 
                        if isinstance(v, np.ndarray) and len(v) > 0}
        
        if len(valid_methods) < 3:
            return
        
        # 选择前10个方法
        selected_methods = list(valid_methods.keys())[:10]
        score_matrix = np.column_stack([valid_methods[method] for method in selected_methods])
        
        # PCA降维
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(score_matrix)
        
        # 根据最终异常等级着色
        risk_levels = final_results['anomaly_level'].values
        level_colors = {'极高风险': 'darkred', '高风险': 'red', '中高风险': 'orange', 
                       '中等风险': 'yellow', '低风险': 'green'}
        colors = [level_colors.get(level, 'gray') for level in risk_levels]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # PCA可视化
        scatter1 = ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.6, s=20)
        ax1.set_title('PCA - 集成结果高维可视化', fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        # t-SNE降维（如果数据量不太大）
        if len(score_matrix) <= 5000:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(score_matrix)//4))
            tsne_result = tsne.fit_transform(score_matrix)
            
            scatter2 = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, alpha=0.6, s=20)
            ax2.set_title('t-SNE - 集成结果高维可视化', fontweight='bold')
            ax2.set_xlabel('t-SNE 1')
            ax2.set_ylabel('t-SNE 2')
        else:
            ax2.text(0.5, 0.5, '数据量过大\n跳过t-SNE可视化', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('t-SNE - 数据量过大', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'high_dimensional_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_optimized_ensemble(self) -> Dict[str, Any]:
        """运行优化集成流程"""
        logger.info("开始优化集成流程...")
        start_time = time.time()
        
        try:
            # 1. 加载数据
            self.load_model_scores()
            
            # 2. 获取原始分数矩阵
            score_matrix = self.model_scores[self.score_columns].values
            
            # 3. 并行标准化
            normalized_matrices = self.normalize_scores_parallel(score_matrix)
            
            # 4. 高级集成方法
            advanced_results = self.advanced_ensemble_methods(normalized_matrices)
            
            # 5. 并行投票集成
            voting_results = self.parallel_voting_ensemble(normalized_matrices)
            
            # 6. 合并所有结果
            all_results = {**advanced_results, **voting_results}
            
            # 7. 智能选择最佳集成
            selection_result = self.intelligent_ensemble_selection(all_results)
            
            # 8. 生成最终结果
            final_results = self._generate_final_results(selection_result, all_results)
            
            # 9. 跳过可视化
            pass
            
            # 10. 保存结果
            self._save_optimized_results(final_results, selection_result, all_results)
            
            total_time = time.time() - start_time
            
            logger.info(f"优化集成流程完成！")
            logger.info(f"最佳集成方法: {selection_result['best_method']}")
            logger.info(f"总耗时: {total_time:.2f}秒")
            
            return {
                'final_results': final_results,
                'best_method': selection_result['best_method'],
                'total_methods_tested': len(all_results),
                'processing_time': total_time
            }
            
        except Exception as e:
            logger.error(f"优化集成过程中出现错误: {e}")
            return {}
    
    def _generate_final_results(self, selection_result: Dict[str, Any], 
                              all_results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """生成最终结果"""
        results_df = pd.DataFrame()
        results_df['loop_id'] = self.loop_ids
        
        best_scores = selection_result['best_scores']
        results_df['final_anomaly_score'] = best_scores
        results_df['final_anomaly_rank'] = len(best_scores) - rankdata(best_scores, method='average') + 1
        
        # 异常等级分类
        def categorize_anomaly_level(score, percentiles):
            if score >= percentiles[95]:
                return "极高风险"
            elif score >= percentiles[90]:
                return "高风险"
            elif score >= percentiles[75]:
                return "中高风险"
            elif score >= percentiles[60]:
                return "中等风险"
            else:
                return "低风险"
        
        percentiles = {p: np.percentile(best_scores, p) for p in [60, 75, 90, 95]}
        results_df['anomaly_level'] = results_df['final_anomaly_score'].apply(
            lambda x: categorize_anomaly_level(x, percentiles)
        )
        
        results_df['percentile_rank'] = rankdata(best_scores, method='average') / len(best_scores) * 100
        
        return results_df.sort_values('final_anomaly_score', ascending=False).reset_index(drop=True)
    
    def _save_optimized_results(self, final_results: pd.DataFrame, selection_result: Dict[str, Any],
                              all_results: Dict[str, np.ndarray]):
        """保存优化结果"""
        # 保存最终结果
        final_results.to_csv(self.ensemble_dir / 'optimized_final_results.csv', 
                           index=False, encoding='utf-8-sig')
        
        # 保存前100异常环路
        top_100 = final_results.head(100)
        top_100.to_csv(self.ensemble_dir / 'optimized_top_100_anomalies.csv', 
                      index=False, encoding='utf-8-sig')
        
        # 简化报告
        pass
        
        logger.info("优化结果已保存")

def main():
    """主函数"""
    print("=" * 70)
    print("化工图风控系统 - 优化版异常检测集成层")
    print("🚀 并行处理 + 贝叶斯优化集成")
    print("=" * 70)
    
    # 检查输入文件
    model_scores_file = 'outputs/anomaly_detection/optimized_model_scores.csv'
    if not Path(model_scores_file).exists():
        # 尝试使用原始模型分数文件
        model_scores_file = 'outputs/anomaly_detection/model_scores.csv'
        if not Path(model_scores_file).exists():
            print("❌ 找不到模型分数文件")
            print("📝 请先运行模型训练脚本")
            return
    
    # 初始化优化集成器
    integrator = OptimizedEnsembleIntegrator(
        model_scores_file=model_scores_file,
        n_jobs=-1,  # 使用所有CPU核心
        optimization_trials=50  # 贝叶斯优化试验次数
    )
    
    # 运行优化集成
    results = integrator.run_optimized_ensemble()
    
    if results:
        print("\n" + "=" * 70)
        print("✅ 优化集成完成！")
        print(f"🏆 最佳集成方法: {results['best_method']}")
        print(f"📊 测试方法数量: {results['total_methods_tested']}")
        print(f"⏱️  处理耗时: {results['processing_time']:.2f}秒")
        print("\n📁 输出文件:")
        print("   - optimized_final_results.csv (最终结果)")
        print("   - optimized_top_100_anomalies.csv (前100异常)")
        print("   - optimization_report.json (优化报告)")
        print("   - method_performance_heatmap.png (性能热图)")
        print("   - best_method_analysis.png (最佳方法分析)")
        print("   - ensemble_stability_analysis.png (稳定性分析)")
        print("=" * 70)
    else:
        print("❌ 优化集成失败")

if __name__ == "__main__":
    main() 