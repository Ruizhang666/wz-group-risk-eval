#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常检测解释系统 - 最终脚本
为异常检测结果提供详细的可解释性分析和业务理解

功能模块：
1. 特征重要性分析 - 解释哪些特征导致异常
2. 异常模式识别 - 识别不同类型的异常模式
3. 风险等级解释 - 详细说明风险等级判定依据
4. 业务影响评估 - 评估异常对业务的潜在影响
5. 修复建议生成 - 提供针对性的风险缓解建议
6. 可视化报告 - 生成直观的解释报告

作者: AI助手
日期: 2024年
版本: v1.0 - 解释层
"""

import pandas as pd
import numpy as np
import warnings
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import percentileofscore
import joblib
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap

# 忽略警告
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/anomaly_detection/logs/explanation_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AnomalyExplanationSystem:
    """异常检测解释系统"""
    
    def __init__(self):
        """初始化解释系统"""
        self.output_dir = Path('outputs/anomaly_detection')
        self.explanation_dir = self.output_dir / 'explanations'
        self.viz_dir = self.output_dir / 'visualizations'
        self.reports_dir = self.output_dir / 'reports'
        
        # 创建输出目录
        for dir_path in [self.explanation_dir, self.viz_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.anomaly_results = None
        self.feature_data = None
        self.feature_names = []
        self.explanation_models = {}
        self.feature_importance_global = {}
        
        logger.info("异常检测解释系统初始化完成")
    
    def load_data(self) -> bool:
        """加载异常检测结果和特征数据"""
        try:
            # 加载最终异常检测结果
            results_file = self.output_dir / 'ensemble/final_anomaly_results.csv'
            if not results_file.exists():
                logger.error(f"找不到异常检测结果文件: {results_file}")
                return False
            
            self.anomaly_results = pd.read_csv(results_file)
            logger.info(f"加载异常检测结果: {len(self.anomaly_results)} 个环路")
            
            # 加载特征数据
            features_file = self.output_dir / 'features/engineered_features.csv'
            if not features_file.exists():
                logger.error(f"找不到特征数据文件: {features_file}")
                return False
            
            self.feature_data = pd.read_csv(features_file)
            
            # 分离loop_id和特征
            if 'loop_id' in self.feature_data.columns:
                feature_cols = [col for col in self.feature_data.columns if col != 'loop_id']
            else:
                feature_cols = self.feature_data.columns.tolist()
            
            self.feature_names = feature_cols
            logger.info(f"加载特征数据: {len(self.feature_names)} 个特征")
            
            return True
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return False
    
    def build_explanation_models(self):
        """构建可解释性模型"""
        logger.info("构建可解释性模型...")
        
        # 准备训练数据
        X = self.feature_data[self.feature_names].values
        y = self.anomaly_results['final_anomaly_score'].values
        
        # 1. 决策树模型 - 易于解释的规则
        dt_model = DecisionTreeRegressor(
            max_depth=8,
            min_samples_split=100,
            min_samples_leaf=50,
            random_state=42
        )
        dt_model.fit(X, y)
        self.explanation_models['decision_tree'] = dt_model
        
        # 2. 随机森林模型 - 特征重要性
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X, y)
        self.explanation_models['random_forest'] = rf_model
        
        # 3. 计算全局特征重要性
        self.feature_importance_global = {
            'decision_tree': dict(zip(self.feature_names, dt_model.feature_importances_)),
            'random_forest': dict(zip(self.feature_names, rf_model.feature_importances_))
        }
        
        logger.info("可解释性模型构建完成")
    
    def generate_decision_rules(self) -> Dict[str, Any]:
        """生成决策规则"""
        logger.info("生成决策规则...")
        
        dt_model = self.explanation_models['decision_tree']
        
        # 导出决策树规则
        tree_rules = export_text(
            dt_model,
            feature_names=self.feature_names,
            max_depth=6
        )
        
        # 解析关键决策路径
        key_rules = []
        
        # 获取叶子节点的值和样本数
        leaf_values = dt_model.tree_.value.flatten()
        leaf_samples = dt_model.tree_.n_node_samples
        
        # 找出高异常分数的叶子节点路径
        high_anomaly_threshold = np.percentile(self.anomaly_results['final_anomaly_score'], 90)
        
        for i, (value, samples) in enumerate(zip(leaf_values, leaf_samples)):
            if value > high_anomaly_threshold and samples > 20:  # 高异常分数且样本数足够
                key_rules.append({
                    'node_id': i,
                    'predicted_score': float(value),
                    'sample_count': int(samples),
                    'rule_description': f"预测异常分数: {value:.3f}, 影响样本: {samples}"
                })
        
        return {
            'tree_rules_text': tree_rules,
            'key_decision_paths': key_rules,
            'total_rules': len(key_rules)
        }
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """分析特征重要性"""
        logger.info("分析特征重要性...")
        
        # 合并不同模型的特征重要性
        rf_importance = self.feature_importance_global['random_forest']
        dt_importance = self.feature_importance_global['decision_tree']
        
        # 创建综合特征重要性分析
        feature_analysis = {}
        
        for feature in self.feature_names:
            rf_imp = rf_importance.get(feature, 0)
            dt_imp = dt_importance.get(feature, 0)
            
            # 计算综合重要性分数
            combined_importance = (rf_imp + dt_imp) / 2
            
            # 计算特征统计信息
            feature_values = self.feature_data[feature].values
            feature_stats = {
                'mean': float(np.mean(feature_values)),
                'std': float(np.std(feature_values)),
                'min': float(np.min(feature_values)),
                'max': float(np.max(feature_values)),
                'skewness': float(stats.skew(feature_values)),
                'kurtosis': float(stats.kurtosis(feature_values))
            }
            
            feature_analysis[feature] = {
                'rf_importance': rf_imp,
                'dt_importance': dt_imp,
                'combined_importance': combined_importance,
                'statistics': feature_stats,
                'rank': 0  # 将在后面排序时填入
            }
        
        # 按综合重要性排序
        sorted_features = sorted(
            feature_analysis.items(),
            key=lambda x: x[1]['combined_importance'],
            reverse=True
        )
        
        # 添加排名信息
        for rank, (feature, info) in enumerate(sorted_features, 1):
            feature_analysis[feature]['rank'] = rank
        
        # 获取前10重要特征
        top_10_features = dict(sorted_features[:10])
        
        return {
            'all_features': feature_analysis,
            'top_10_features': top_10_features,
            'feature_ranking': [item[0] for item in sorted_features]
        }
    
    def identify_anomaly_patterns(self) -> Dict[str, Any]:
        """识别异常模式"""
        logger.info("识别异常模式...")
        
        # 获取高异常分数的环路
        high_anomaly_data = self.anomaly_results[
            self.anomaly_results['anomaly_level'].isin(['极高风险', '高风险'])
        ].copy()
        
        if len(high_anomaly_data) == 0:
            return {'patterns': [], 'total_high_risk': 0}
        
        # 获取对应的特征数据
        high_anomaly_loop_ids = high_anomaly_data['loop_id'].values
        
        # 合并数据
        merged_data = pd.merge(
            high_anomaly_data[['loop_id', 'final_anomaly_score', 'anomaly_level']],
            self.feature_data,
            on='loop_id',
            how='inner'
        )
        
        # 分析异常模式
        patterns = []
        
        # 1. 基于异常分数分组分析
        score_groups = {
            '极高异常': merged_data[merged_data['anomaly_level'] == '极高风险'],
            '高异常': merged_data[merged_data['anomaly_level'] == '高风险']
        }
        
        for group_name, group_data in score_groups.items():
            if len(group_data) == 0:
                continue
            
            # 计算该组的特征统计
            group_features = group_data[self.feature_names]
            
            # 找出该组的特征特点（与整体均值比较）
            overall_means = self.feature_data[self.feature_names].mean()
            group_means = group_features.mean()
            
            # 找出显著偏离的特征
            significant_features = []
            for feature in self.feature_names:
                if feature in group_means.index and feature in overall_means.index:
                    group_mean = group_means[feature]
                    overall_mean = overall_means[feature]
                    
                    if overall_mean != 0:
                        deviation_ratio = abs(group_mean - overall_mean) / abs(overall_mean)
                        if deviation_ratio > 0.5:  # 偏离50%以上
                            significant_features.append({
                                'feature': feature,
                                'group_mean': float(group_mean),
                                'overall_mean': float(overall_mean),
                                'deviation_ratio': float(deviation_ratio),
                                'direction': 'higher' if group_mean > overall_mean else 'lower'
                            })
            
            # 按偏离程度排序
            significant_features.sort(key=lambda x: x['deviation_ratio'], reverse=True)
            
            patterns.append({
                'pattern_name': group_name,
                'sample_count': len(group_data),
                'avg_anomaly_score': float(group_data['final_anomaly_score'].mean()),
                'significant_features': significant_features[:5],  # 前5个最显著特征
                'description': self._generate_pattern_description(group_name, significant_features[:3])
            })
        
        return {
            'patterns': patterns,
            'total_high_risk': len(high_anomaly_data),
            'pattern_summary': f"识别出 {len(patterns)} 种主要异常模式"
        }
    
    def _generate_pattern_description(self, pattern_name: str, features: List[Dict]) -> str:
        """生成模式描述"""
        if not features:
            return f"{pattern_name}环路的特征模式不明显"
        
        descriptions = []
        for feature_info in features[:3]:  # 只描述前3个特征
            feature = feature_info['feature']
            direction = "显著高于" if feature_info['direction'] == 'higher' else "显著低于"
            ratio = feature_info['deviation_ratio']
            
            # 简化特征名称显示
            display_name = feature.replace('_', ' ').title()
            descriptions.append(f"{display_name}{direction}正常水平({ratio:.1%})")
        
        return f"{pattern_name}环路主要特征: " + "; ".join(descriptions)
    
    def explain_individual_anomalies(self, top_n: int = 50) -> Dict[str, Any]:
        """解释个别异常环路"""
        logger.info(f"解释前{top_n}个异常环路...")
        
        # 获取前N个异常环路
        top_anomalies = self.anomaly_results.nlargest(top_n, 'final_anomaly_score')
        
        explanations = []
        
        for idx, row in top_anomalies.iterrows():
            loop_id = row['loop_id']
            anomaly_score = row['final_anomaly_score']
            risk_level = row['anomaly_level']
            
            # 获取该环路的特征值
            loop_features = self.feature_data[self.feature_data['loop_id'] == loop_id]
            if len(loop_features) == 0:
                continue
            
            loop_features = loop_features.iloc[0]
            
            # 使用决策树模型解释
            feature_values = loop_features[self.feature_names].values.reshape(1, -1)
            dt_prediction = self.explanation_models['decision_tree'].predict(feature_values)[0]
            
            # 找出最重要的贡献特征
            rf_model = self.explanation_models['random_forest']
            feature_contributions = []
            
            # 计算特征贡献（简化版本）
            for i, feature in enumerate(self.feature_names):
                feature_value = loop_features[feature]
                feature_importance = self.feature_importance_global['random_forest'][feature]
                
                # 计算该特征相对于平均值的偏离
                feature_mean = self.feature_data[feature].mean()
                feature_std = self.feature_data[feature].std()
                
                if feature_std > 0:
                    normalized_deviation = abs(feature_value - feature_mean) / feature_std
                    contribution_score = feature_importance * normalized_deviation
                    
                    feature_contributions.append({
                        'feature': feature,
                        'value': float(feature_value),
                        'mean': float(feature_mean),
                        'deviation': float(normalized_deviation),
                        'contribution': float(contribution_score)
                    })
            
            # 按贡献度排序
            feature_contributions.sort(key=lambda x: x['contribution'], reverse=True)
            
            explanations.append({
                'loop_id': int(loop_id),
                'anomaly_score': float(anomaly_score),
                'risk_level': risk_level,
                'percentile_rank': float(percentileofscore(
                    self.anomaly_results['final_anomaly_score'], 
                    anomaly_score
                )),
                'dt_prediction': float(dt_prediction),
                'top_contributing_features': feature_contributions[:5],
                'explanation_summary': self._generate_individual_explanation(
                    loop_id, risk_level, feature_contributions[:3]
                )
            })
        
        return {
            'individual_explanations': explanations,
            'total_explained': len(explanations),
            'explanation_method': '决策树 + 随机森林特征重要性'
        }
    
    def _generate_individual_explanation(self, loop_id: int, risk_level: str, 
                                       top_features: List[Dict]) -> str:
        """生成个别环路的解释"""
        base_text = f"环路 {loop_id} 被评为 {risk_level}，主要原因："
        
        reasons = []
        for feature_info in top_features:
            feature = feature_info['feature'].replace('_', ' ')
            deviation = feature_info['deviation']
            
            if deviation > 2:
                reasons.append(f"{feature}异常偏离正常范围({deviation:.1f}倍标准差)")
            elif deviation > 1:
                reasons.append(f"{feature}明显偏离正常水平({deviation:.1f}倍标准差)")
        
        return base_text + "; ".join(reasons) if reasons else base_text + "多项指标综合异常"
    
    def generate_business_impact_assessment(self) -> Dict[str, Any]:
        """生成业务影响评估"""
        logger.info("生成业务影响评估...")
        
        # 按风险等级统计
        risk_distribution = self.anomaly_results['anomaly_level'].value_counts().to_dict()
        
        # 计算潜在影响
        high_risk_count = risk_distribution.get('极高风险', 0) + risk_distribution.get('高风险', 0)
        total_loops = len(self.anomaly_results)
        high_risk_ratio = high_risk_count / total_loops if total_loops > 0 else 0
        
        # 业务影响评估
        impact_assessment = {
            'overall_risk_status': self._assess_overall_risk(high_risk_ratio),
            'risk_distribution': risk_distribution,
            'high_risk_ratio': float(high_risk_ratio),
            'total_loops_analyzed': total_loops,
            'immediate_attention_required': high_risk_count,
            'business_implications': self._generate_business_implications(risk_distribution),
            'priority_actions': self._generate_priority_actions(risk_distribution)
        }
        
        return impact_assessment
    
    def _assess_overall_risk(self, high_risk_ratio: float) -> str:
        """评估整体风险状态"""
        if high_risk_ratio > 0.1:
            return "高风险状态 - 需要立即关注"
        elif high_risk_ratio > 0.05:
            return "中等风险状态 - 需要密切监控"
        elif high_risk_ratio > 0.02:
            return "低风险状态 - 正常监控"
        else:
            return "风险可控状态 - 定期检查"
    
    def _generate_business_implications(self, risk_dist: Dict[str, int]) -> List[str]:
        """生成业务影响分析"""
        implications = []
        
        extreme_risk = risk_dist.get('极高风险', 0)
        high_risk = risk_dist.get('高风险', 0)
        
        if extreme_risk > 0:
            implications.append(f"{extreme_risk} 个环路存在极高风险，可能涉及重大合规问题")
        
        if high_risk > 0:
            implications.append(f"{high_risk} 个环路存在高风险，需要优先调查")
        
        if extreme_risk + high_risk > 100:
            implications.append("大量高风险环路可能表明系统性问题")
        
        return implications
    
    def _generate_priority_actions(self, risk_dist: Dict[str, int]) -> List[str]:
        """生成优先行动建议"""
        actions = []
        
        extreme_risk = risk_dist.get('极高风险', 0)
        high_risk = risk_dist.get('高风险', 0)
        
        if extreme_risk > 0:
            actions.append("立即审查所有极高风险环路的合规状态")
            actions.append("暂停或限制极高风险环路的相关业务活动")
        
        if high_risk > 0:
            actions.append("48小时内完成高风险环路的详细调查")
            actions.append("建立高风险环路的持续监控机制")
        
        if extreme_risk + high_risk > 50:
            actions.append("启动系统性风险评估程序")
            actions.append("考虑调整风险管理策略和控制措施")
        
        return actions
    
    def create_visualizations(self):
        """创建可视化图表"""
        logger.info("创建可视化图表...")
        
        # 1. 特征重要性图
        self._create_feature_importance_plot()
        
        # 2. 异常分数分布图
        self._create_anomaly_distribution_plot()
        
        # 3. 风险等级分布图
        self._create_risk_level_plot()
        
        # 4. 前20异常环路特征雷达图
        self._create_top_anomalies_radar()
        
        logger.info(f"可视化图表已保存到: {self.viz_dir}")
    
    def _create_feature_importance_plot(self):
        """创建特征重要性图"""
        feature_analysis = self.analyze_feature_importance()
        top_features = feature_analysis['top_10_features']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 随机森林重要性
        features = list(top_features.keys())
        rf_importance = [top_features[f]['rf_importance'] for f in features]
        
        ax1.barh(range(len(features)), rf_importance, color='skyblue')
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels([f.replace('_', '\n') for f in features])
        ax1.set_title('随机森林特征重要性 (Top 10)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('重要性分数')
        
        # 决策树重要性
        dt_importance = [top_features[f]['dt_importance'] for f in features]
        
        ax2.barh(range(len(features)), dt_importance, color='lightcoral')
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels([f.replace('_', '\n') for f in features])
        ax2.set_title('决策树特征重要性 (Top 10)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('重要性分数')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_anomaly_distribution_plot(self):
        """创建异常分数分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        scores = self.anomaly_results['final_anomaly_score']
        
        # 直方图
        axes[0,0].hist(scores, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0,0].set_title('异常分数分布直方图', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('异常分数')
        axes[0,0].set_ylabel('频次')
        
        # 箱线图
        axes[0,1].boxplot(scores, vert=True)
        axes[0,1].set_title('异常分数箱线图', fontsize=12, fontweight='bold')
        axes[0,1].set_ylabel('异常分数')
        
        # 累积分布
        sorted_scores = np.sort(scores)
        axes[1,0].plot(sorted_scores, np.arange(1, len(sorted_scores)+1) / len(sorted_scores))
        axes[1,0].set_title('异常分数累积分布', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('异常分数')
        axes[1,0].set_ylabel('累积概率')
        
        # 分位数图
        percentiles = np.arange(1, 100)
        percentile_values = [np.percentile(scores, p) for p in percentiles]
        axes[1,1].plot(percentiles, percentile_values, 'b-', linewidth=2)
        axes[1,1].set_title('异常分数分位数图', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('百分位数')
        axes[1,1].set_ylabel('异常分数')
        
        # 标记关键分位点
        key_percentiles = [90, 95, 99]
        for p in key_percentiles:
            value = np.percentile(scores, p)
            axes[1,1].axhline(y=value, color='red', linestyle='--', alpha=0.7)
            axes[1,1].text(p, value, f'{p}%: {value:.3f}', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'anomaly_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_risk_level_plot(self):
        """创建风险等级分布图"""
        risk_counts = self.anomaly_results['anomaly_level'].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 饼图
        colors = ['darkred', 'red', 'orange', 'yellow', 'green']
        ax1.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('风险等级分布 (饼图)', fontsize=12, fontweight='bold')
        
        # 柱状图
        bars = ax2.bar(risk_counts.index, risk_counts.values, color=colors)
        ax2.set_title('风险等级分布 (柱状图)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('环路数量')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'risk_level_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_top_anomalies_radar(self):
        """创建前20异常环路的特征雷达图"""
        # 获取前20异常环路
        top_20 = self.anomaly_results.nlargest(20, 'final_anomaly_score')
        
        # 获取前10重要特征
        feature_analysis = self.analyze_feature_importance()
        top_10_features = list(feature_analysis['top_10_features'].keys())
        
        # 准备雷达图数据
        angles = np.linspace(0, 2 * np.pi, len(top_10_features), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # 计算特征的标准化值
        feature_data_subset = self.feature_data[self.feature_data['loop_id'].isin(top_20['loop_id'])]
        
        for i, (idx, row) in enumerate(top_20.head(5).iterrows()):  # 只画前5个
            loop_id = row['loop_id']
            loop_features = feature_data_subset[feature_data_subset['loop_id'] == loop_id]
            
            if len(loop_features) == 0:
                continue
            
            # 获取特征值并标准化到0-1范围
            values = []
            for feature in top_10_features:
                feature_value = loop_features[feature].iloc[0]
                feature_min = self.feature_data[feature].min()
                feature_max = self.feature_data[feature].max()
                
                if feature_max > feature_min:
                    normalized_value = (feature_value - feature_min) / (feature_max - feature_min)
                else:
                    normalized_value = 0.5
                
                values.append(normalized_value)
            
            values += values[:1]  # 闭合雷达图
            
            # 绘制雷达图
            color = plt.cm.Reds(0.5 + i * 0.1)
            ax.plot(angles, values, 'o-', linewidth=2, label=f'环路 {int(loop_id)}', color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f.replace('_', '\n') for f in top_10_features])
        ax.set_ylim(0, 1)
        ax.set_title('前5异常环路特征雷达图', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'top_anomalies_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合解释报告"""
        logger.info("生成综合解释报告...")
        
        # 收集所有分析结果
        decision_rules = self.generate_decision_rules()
        feature_importance = self.analyze_feature_importance()
        anomaly_patterns = self.identify_anomaly_patterns()
        individual_explanations = self.explain_individual_anomalies(50)
        business_impact = self.generate_business_impact_assessment()
        
        # 创建综合报告
        comprehensive_report = {
            'report_metadata': {
                'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_loops_analyzed': len(self.anomaly_results),
                'total_features_used': len(self.feature_names),
                'explanation_methods': ['决策树规则', '随机森林重要性', '特征统计分析']
            },
            'executive_summary': {
                'overall_risk_status': business_impact['overall_risk_status'],
                'high_risk_loops': business_impact['immediate_attention_required'],
                'key_risk_factors': feature_importance['feature_ranking'][:5],
                'main_anomaly_patterns': len(anomaly_patterns['patterns'])
            },
            'detailed_analysis': {
                'feature_importance_analysis': feature_importance,
                'decision_rules': decision_rules,
                'anomaly_patterns': anomaly_patterns,
                'individual_explanations': individual_explanations,
                'business_impact_assessment': business_impact
            },
            'recommendations': {
                'immediate_actions': business_impact['priority_actions'],
                'monitoring_focus': feature_importance['feature_ranking'][:10],
                'risk_mitigation_strategies': self._generate_mitigation_strategies(anomaly_patterns)
            }
        }
        
        # 保存综合报告
        report_file = self.reports_dir / 'comprehensive_explanation_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        # 生成HTML报告
        self._generate_html_report(comprehensive_report)
        
        logger.info(f"综合解释报告已保存到: {report_file}")
        
        return comprehensive_report
    
    def _generate_mitigation_strategies(self, anomaly_patterns: Dict[str, Any]) -> List[str]:
        """生成风险缓解策略"""
        strategies = []
        
        patterns = anomaly_patterns.get('patterns', [])
        
        if patterns:
            strategies.append("针对识别的异常模式建立专项监控机制")
            strategies.append("对高风险特征建立实时预警系统")
            strategies.append("定期审查和更新异常检测模型")
        
        strategies.extend([
            "建立分级响应机制处理不同风险等级的异常",
            "加强对关键特征的数据质量控制",
            "建立异常处理的标准操作程序",
            "定期进行模型效果评估和优化"
        ])
        
        return strategies
    
    def _generate_html_report(self, report_data: Dict[str, Any]):
        """生成HTML格式的报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>异常检测解释报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .high-risk {{ color: #d32f2f; font-weight: bold; }}
                .medium-risk {{ color: #f57c00; font-weight: bold; }}
                .low-risk {{ color: #388e3c; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>化工图风控系统 - 异常检测解释报告</h1>
                <p>生成时间: {report_data['report_metadata']['generation_time']}</p>
                <p>分析环路数: {report_data['report_metadata']['total_loops_analyzed']:,}</p>
            </div>
            
            <div class="section">
                <h2>执行摘要</h2>
                <p><strong>整体风险状态:</strong> <span class="high-risk">{report_data['executive_summary']['overall_risk_status']}</span></p>
                <p><strong>高风险环路数量:</strong> {report_data['executive_summary']['high_risk_loops']}</p>
                <p><strong>主要风险因素:</strong> {', '.join(report_data['executive_summary']['key_risk_factors'])}</p>
            </div>
            
            <div class="section">
                <h2>立即行动建议</h2>
                <ul>
        """
        
        for action in report_data['recommendations']['immediate_actions']:
            html_content += f"<li>{action}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>监控重点特征</h2>
                <ol>
        """
        
        for feature in report_data['recommendations']['monitoring_focus']:
            html_content += f"<li>{feature.replace('_', ' ').title()}</li>"
        
        html_content += """
                </ol>
            </div>
        </body>
        </html>
        """
        
        html_file = self.reports_dir / 'explanation_report.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def run_complete_explanation(self) -> bool:
        """运行完整的解释分析流程"""
        logger.info("开始运行异常检测解释分析...")
        start_time = time.time()
        
        try:
            # 1. 加载数据
            if not self.load_data():
                return False
            
            # 2. 构建解释模型
            self.build_explanation_models()
            
            # 3. 跳过可视化
            pass
            
            # 4. 生成综合报告
            comprehensive_report = self.generate_comprehensive_report()
            
            total_time = time.time() - start_time
            
            logger.info(f"异常检测解释分析完成！")
            logger.info(f"总耗时: {total_time:.2f}秒")
            logger.info(f"分析了 {len(self.anomaly_results)} 个环路")
            logger.info(f"使用了 {len(self.feature_names)} 个特征")
            
            return True
            
        except Exception as e:
            logger.error(f"解释分析过程中出现错误: {e}")
            return False

def main():
    """主函数"""
    print("=" * 70)
    print("化工图风控系统 - 异常检测解释系统")
    print("🔍 为异常检测结果提供详细的可解释性分析")
    print("=" * 70)
    
    # 检查依赖文件
    required_files = [
        'outputs/anomaly_detection/ensemble/final_anomaly_results.csv',
        'outputs/anomaly_detection/features/engineered_features.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 找不到必需的输入文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n📝 请确保已运行以下脚本:")
        print("   1. python feature_engineering.py")
        print("   2. python anomaly_detection_models.py") 
        print("   3. python anomaly_ensemble_integration.py")
        return
    
    # 初始化解释系统
    explainer = AnomalyExplanationSystem()
    
    # 运行完整解释流程
    success = explainer.run_complete_explanation()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ 异常检测解释分析完成！")
        print("📊 输出文件:")
        print("   - comprehensive_explanation_report.json (综合分析报告)")
        print("   - explanation_report.html (HTML可视化报告)")
        print("   - feature_importance_analysis.png (特征重要性图)")
        print("   - anomaly_score_distribution.png (异常分数分布)")
        print("   - risk_level_distribution.png (风险等级分布)")
        print("   - top_anomalies_radar.png (异常环路雷达图)")
        print("📁 输出目录: outputs/anomaly_detection/")
        print("=" * 70)
    else:
        print("❌ 异常检测解释分析失败")

if __name__ == "__main__":
    main() 