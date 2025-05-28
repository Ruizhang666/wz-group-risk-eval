#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程脚本 - 异常检测系统第一步
从综合画像数据中提取和构造特征，为异常检测模型准备输入数据

该脚本实现了：
1. 原始特征的标准化和变换
2. 构造新的组合特征和统计特征
3. 特征选择和降维
4. 输出增强后的特征矩阵
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression
from scipy import stats
from scipy.stats import skew, kurtosis
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# 设置日志
def setup_logging():
    """配置日志系统"""
    log_dir = "outputs/anomaly_detection/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/feature_engineering.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class FeatureEngineer:
    """
    特征工程类 - 负责从原始画像数据中提取、变换和构造特征
    
    主要功能：
    1. 特征变换：标准化、归一化、对数变换等
    2. 特征构造：比率、统计量、交叉特征等
    3. 特征选择：去除低方差、高相关性特征
    """
    
    def __init__(self, input_file=None):
        """初始化特征工程器"""
        self.logger = setup_logging()
        self.input_file = input_file or "outputs/扩展画像/loop_comprehensive_metrics.csv"
        self.output_dir = "outputs/anomaly_detection/features"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 定义不同类型的特征列
        self.transaction_features = [
            'total_transaction_amount',
            'upstream_to_member_transaction_amount',
            'member_to_downstream_transaction_amount',
            'upstream_to_member_avg_amount',
            'member_to_downstream_avg_amount',
            'upstream_to_member_transaction_count',
            'member_to_downstream_transaction_count'
        ]
        
        self.equity_features = [
            'max_ownership_percent',
            'min_ownership_percent',
            'avg_ownership_percent',
            'ownership_concentration_index',
            'total_ownership_percent',
            'ownership_count'
        ]
        
        self.network_features = [
            'max_degree_centrality',
            'max_betweenness_centrality',
            'max_closeness_centrality',
            'network_density',
            'avg_degree'
        ]
        
        self.structure_features = [
            'loop_node_count',
            'loop_path_length',
            'loop_complexity_score',
            'natural_person_count',
            'enterprise_count',
            'total_shareholders'
        ]
        
    def load_data(self):
        """加载数据并进行基本预处理"""
        try:
            self.logger.info(f"加载数据文件: {self.input_file}")
            self.df = pd.read_csv(self.input_file)
            
            self.logger.info(f"成功加载 {len(self.df)} 条环路数据，包含 {len(self.df.columns)} 个原始特征")
            
            # 排除已知的字符串类型列（包含日期列表的列）
            string_columns = [
                'upstream_to_member_transaction_times',
                'member_to_downstream_transaction_times',
                'key_node_id',  # 节点ID通常是字符串
                'dominant_shareholder_type'  # 分类特征
            ]
            
            # 删除字符串列，避免后续处理错误
            columns_to_drop = [col for col in string_columns if col in self.df.columns]
            if columns_to_drop:
                self.logger.info(f"删除字符串类型列: {columns_to_drop}")
                self.df.drop(columns=columns_to_drop, inplace=True)
            
            # 识别数值特征
            self.numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'loop_id' in self.numeric_features:
                self.numeric_features.remove('loop_id')
            
            self.logger.info(f"识别到 {len(self.numeric_features)} 个数值型特征")
            
            # 数据质量检查
            self.logger.info(f"数据形状: {self.df.shape}")
            self.logger.info(f"缺失值总数: {self.df.isnull().sum().sum()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return False
    
    def handle_missing_values(self):
        """处理缺失值"""
        self.logger.info("开始处理缺失值...")
        
        # 统计缺失值情况
        missing_stats = self.df[self.numeric_features].isnull().sum()
        missing_ratio = missing_stats / len(self.df)
        
        # 对于缺失率高的特征，记录日志
        high_missing = missing_ratio[missing_ratio > 0.3]
        if len(high_missing) > 0:
            self.logger.warning(f"以下特征缺失率超过30%: {high_missing.to_dict()}")
        
        # 分类处理缺失值
        for col in self.numeric_features:
            if col in self.transaction_features:
                # 交易特征：缺失值填充为0（表示没有交易）
                self.df[col].fillna(0, inplace=True)
            elif col in self.equity_features:
                # 股权特征：使用中位数填充
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif col in self.network_features:
                # 网络特征：使用均值填充
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            else:
                # 其他特征：使用前向填充
                self.df[col].fillna(method='ffill', inplace=True)
                self.df[col].fillna(0, inplace=True)  # 如果还有缺失，填0
                
        self.logger.info("缺失值处理完成")
    
    def transform_features(self):
        """特征变换 - 包括标准化、归一化、对数变换等"""
        self.logger.info("开始特征变换...")
        
        # 1. 对数变换 - 处理偏态分布的特征（主要是金额类）
        log_features = []
        for col in self.transaction_features:
            if col in self.df.columns:
                # 对数变换（加1避免log(0)）
                new_col = f"{col}_log"
                self.df[new_col] = np.log1p(self.df[col])
                log_features.append(new_col)
                
                # 计算偏度，如果偏度降低则保留对数特征
                original_skew = abs(skew(self.df[col].dropna()))
                log_skew = abs(skew(self.df[new_col].dropna()))
                if log_skew < original_skew:
                    self.logger.info(f"{col} 对数变换后偏度从 {original_skew:.2f} 降至 {log_skew:.2f}")
        
        # 2. 平方根变换 - 对于计数类特征
        sqrt_features = []
        count_cols = [col for col in self.numeric_features if 'count' in col.lower()]
        for col in count_cols:
            if col in self.df.columns:
                new_col = f"{col}_sqrt"
                self.df[new_col] = np.sqrt(self.df[col])
                sqrt_features.append(new_col)
        
        # 3. Box-Cox变换 - 自动选择最佳变换
        boxcox_features = []
        for col in self.numeric_features[:5]:  # 只对前5个特征做Box-Cox示例
            if col in self.df.columns and self.df[col].min() > 0:  # Box-Cox要求数据为正
                try:
                    transformed_data, lambda_param = stats.boxcox(self.df[col])
                    new_col = f"{col}_boxcox"
                    self.df[new_col] = transformed_data
                    boxcox_features.append(new_col)
                    self.logger.info(f"{col} Box-Cox变换，lambda={lambda_param:.3f}")
                except:
                    pass
        
        # 4. 标准化 - 使用不同的标准化方法
        # Z-score标准化
        scaler_standard = StandardScaler()
        features_to_scale = self.numeric_features + log_features + sqrt_features
        scaled_features = scaler_standard.fit_transform(self.df[features_to_scale])
        scaled_df = pd.DataFrame(scaled_features, columns=[f"{col}_zscore" for col in features_to_scale])
        self.df = pd.concat([self.df, scaled_df], axis=1)
        
        # Min-Max标准化
        scaler_minmax = MinMaxScaler()
        minmax_features = scaler_minmax.fit_transform(self.df[self.numeric_features])
        minmax_df = pd.DataFrame(minmax_features, columns=[f"{col}_minmax" for col in self.numeric_features])
        self.df = pd.concat([self.df, minmax_df], axis=1)
        
        # Robust标准化（对异常值不敏感）
        scaler_robust = RobustScaler()
        robust_features = scaler_robust.fit_transform(self.df[self.numeric_features])
        robust_df = pd.DataFrame(robust_features, columns=[f"{col}_robust" for col in self.numeric_features])
        self.df = pd.concat([self.df, robust_df], axis=1)
        
        self.logger.info(f"特征变换完成，新增 {len(self.df.columns) - len(self.numeric_features) - 1} 个变换特征")
    
    def create_ratio_features(self):
        """创建比率特征"""
        self.logger.info("创建比率特征...")
        
        # 1. 上下游交易比率
        if all(col in self.df.columns for col in ['upstream_to_member_transaction_amount', 
                                                   'member_to_downstream_transaction_amount']):
            # 避免除零
            downstream = self.df['member_to_downstream_transaction_amount'].replace(0, 1e-6)
            self.df['upstream_downstream_amount_ratio'] = (
                self.df['upstream_to_member_transaction_amount'] / downstream
            )
            
            # 对数比率（更稳定）
            self.df['upstream_downstream_amount_log_ratio'] = np.log1p(
                self.df['upstream_to_member_transaction_amount']
            ) - np.log1p(self.df['member_to_downstream_transaction_amount'])
        
        # 2. 交易集中度
        if 'total_transaction_amount' in self.df.columns:
            self.df['upstream_concentration'] = (
                self.df['upstream_to_member_transaction_amount'] / 
                self.df['total_transaction_amount'].replace(0, 1e-6)
            )
            self.df['downstream_concentration'] = (
                self.df['member_to_downstream_transaction_amount'] / 
                self.df['total_transaction_amount'].replace(0, 1e-6)
            )
        
        # 3. 股权集中度比率
        if all(col in self.df.columns for col in ['max_ownership_percent', 'avg_ownership_percent']):
            avg = self.df['avg_ownership_percent'].replace(0, 1e-6)
            self.df['ownership_concentration_ratio'] = self.df['max_ownership_percent'] / avg
        
        # 4. 平均交易规模比率
        if all(col in self.df.columns for col in ['upstream_to_member_avg_amount', 
                                                   'member_to_downstream_avg_amount']):
            downstream_avg = self.df['member_to_downstream_avg_amount'].replace(0, 1e-6)
            self.df['avg_transaction_size_ratio'] = (
                self.df['upstream_to_member_avg_amount'] / downstream_avg
            )
        
        # 5. 网络密度与节点数比率
        if all(col in self.df.columns for col in ['network_density', 'loop_node_count']):
            nodes = self.df['loop_node_count'].replace(0, 1)
            self.df['density_per_node'] = self.df['network_density'] / nodes
            self.df['theoretical_max_edges_ratio'] = (
                self.df['network_density'] * nodes * (nodes - 1)
            )
        
        # 6. 自然人控制比率
        if all(col in self.df.columns for col in ['natural_person_count', 'total_shareholders']):
            total = self.df['total_shareholders'].replace(0, 1)
            self.df['natural_person_control_ratio'] = self.df['natural_person_count'] / total
        
        self.logger.info(f"创建了 {len([col for col in self.df.columns if 'ratio' in col])} 个比率特征")
    
    def create_statistical_features(self):
        """创建统计特征"""
        self.logger.info("创建统计特征...")
        
        # 1. 交易金额的统计特征
        amount_cols = [col for col in self.df.columns if 'amount' in col and 'log' not in col]
        if amount_cols:
            # 变异系数
            self.df['amount_cv'] = self.df[amount_cols].std(axis=1) / (self.df[amount_cols].mean(axis=1) + 1e-6)
            # 极差
            self.df['amount_range'] = self.df[amount_cols].max(axis=1) - self.df[amount_cols].min(axis=1)
            # 峰度
            self.df['amount_kurtosis'] = self.df[amount_cols].apply(lambda x: kurtosis(x), axis=1)
            # 偏度
            self.df['amount_skewness'] = self.df[amount_cols].apply(lambda x: skew(x), axis=1)
        
        # 2. 中心性指标的统计特征
        centrality_cols = [col for col in self.df.columns if 'centrality' in col]
        if centrality_cols:
            self.df['centrality_mean'] = self.df[centrality_cols].mean(axis=1)
            self.df['centrality_std'] = self.df[centrality_cols].std(axis=1)
            self.df['centrality_max_min_diff'] = (
                self.df[centrality_cols].max(axis=1) - self.df[centrality_cols].min(axis=1)
            )
        
        # 3. 所有权指标的统计特征
        ownership_cols = [col for col in self.df.columns if 'ownership' in col and 'percent' in col]
        if ownership_cols:
            self.df['ownership_entropy'] = self.df[ownership_cols].apply(
                lambda x: -np.sum(x * np.log(x + 1e-10)) if x.sum() > 0 else 0, axis=1
            )
        
        # 4. 分位数特征
        for feature_group in [amount_cols, centrality_cols]:
            if feature_group:
                group_name = 'amount' if 'amount' in feature_group[0] else 'centrality'
                self.df[f'{group_name}_q25'] = self.df[feature_group].quantile(0.25, axis=1)
                self.df[f'{group_name}_q75'] = self.df[feature_group].quantile(0.75, axis=1)
                self.df[f'{group_name}_iqr'] = self.df[f'{group_name}_q75'] - self.df[f'{group_name}_q25']
        
        self.logger.info("统计特征创建完成")
    
    def create_interaction_features(self):
        """创建交互特征 - 捕捉不同维度之间的相互作用"""
        self.logger.info("创建交互特征...")
        
        # 1. 交易与股权的交互
        if all(col in self.df.columns for col in ['total_transaction_amount', 'max_ownership_percent']):
            # 高交易额 × 高股权集中度（风险信号）
            self.df['high_amount_high_ownership'] = (
                self.df['total_transaction_amount'] * self.df['max_ownership_percent']
            )
            
            # 交易金额与股权集中度的一致性
            amount_rank = self.df['total_transaction_amount'].rank(pct=True)
            ownership_rank = self.df['max_ownership_percent'].rank(pct=True)
            self.df['amount_ownership_rank_diff'] = abs(amount_rank - ownership_rank)
        
        # 2. 网络结构与交易的交互
        if all(col in self.df.columns for col in ['network_density', 'total_transaction_amount']):
            # 密集网络中的高额交易
            self.df['dense_network_high_amount'] = (
                self.df['network_density'] * np.log1p(self.df['total_transaction_amount'])
            )
        
        # 3. 节点类型与交易模式的交互
        if all(col in self.df.columns for col in ['natural_person_ratio', 'upstream_downstream_amount_ratio']):
            # 自然人控制下的交易不平衡
            self.df['natural_person_transaction_imbalance'] = (
                self.df['natural_person_ratio'] * abs(np.log1p(self.df['upstream_downstream_amount_ratio']))
            )
        
        # 4. 复杂度与控制力的交互
        if all(col in self.df.columns for col in ['loop_complexity_score', 'ownership_concentration_index']):
            # 复杂结构下的高度集中控制
            self.df['complex_concentrated_control'] = (
                self.df['loop_complexity_score'] * self.df['ownership_concentration_index']
            )
        
        # 5. 时间跨度与交易频率的交互（如果有时间特征）
        if 'upstream_to_member_transaction_count' in self.df.columns:
            # 创建交易强度指标
            self.df['transaction_intensity'] = (
                self.df['upstream_to_member_transaction_count'] + 
                self.df.get('member_to_downstream_transaction_count', 0)
            ) / self.df.get('loop_node_count', 1)
        
        # 6. 多项式特征（二次项）- 捕捉非线性关系
        key_features = ['total_transaction_amount_log', 'max_ownership_percent', 'network_density']
        for feat in key_features:
            if feat in self.df.columns:
                self.df[f'{feat}_squared'] = self.df[feat] ** 2
                
        # 7. 交叉乘积特征
        if all(col in self.df.columns for col in ['max_degree_centrality', 'max_betweenness_centrality']):
            self.df['centrality_product'] = (
                self.df['max_degree_centrality'] * self.df['max_betweenness_centrality']
            )
        
        self.logger.info(f"创建了多个交互特征，当前特征总数: {len(self.df.columns)}")
    
    def create_anomaly_score_features(self):
        """创建基于统计的异常评分特征"""
        self.logger.info("创建异常评分特征...")
        
        # 1. 基于z-score的异常评分
        for col in self.numeric_features[:10]:  # 选择前10个重要特征
            if col in self.df.columns:
                z_scores = np.abs(stats.zscore(self.df[col].fillna(self.df[col].median())))
                self.df[f'{col}_zscore_anomaly'] = z_scores
        
        # 2. 基于IQR的异常评分
        for col in ['total_transaction_amount', 'max_ownership_percent']:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # 计算异常分数
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                self.df[f'{col}_iqr_anomaly'] = ((self.df[col] < lower_bound) | 
                                                  (self.df[col] > upper_bound)).astype(int)
                
                # 连续的异常程度
                self.df[f'{col}_iqr_distance'] = np.maximum(
                    (lower_bound - self.df[col]) / IQR,
                    (self.df[col] - upper_bound) / IQR
                ).clip(lower=0)
        
        # 3. 多变量异常评分（Mahalanobis距离的简化版）
        key_features = ['total_transaction_amount_log', 'max_ownership_percent', 
                       'network_density', 'loop_complexity_score']
        available_features = [f for f in key_features if f in self.df.columns]
        
        if len(available_features) >= 2:
            from sklearn.covariance import EllipticEnvelope
            detector = EllipticEnvelope(contamination=0.1, random_state=42)
            
            try:
                X = self.df[available_features].fillna(self.df[available_features].median())
                anomaly_labels = detector.fit_predict(X)
                anomaly_scores = detector.score_samples(X)
                
                self.df['multivariate_anomaly_label'] = (anomaly_labels == -1).astype(int)
                self.df['multivariate_anomaly_score'] = -anomaly_scores  # 负分数，越大越异常
            except:
                self.logger.warning("多变量异常检测失败，跳过")
        
        self.logger.info("异常评分特征创建完成")
    
    def select_features(self, variance_threshold=0.01, correlation_threshold=0.95):
        """
        特征选择 - 去除低方差和高相关特征
        
        参数:
            variance_threshold: 方差阈值，低于此值的特征将被删除
            correlation_threshold: 相关性阈值，高于此值的特征对将删除其中一个
        """
        self.logger.info("开始特征选择...")
        
        # 获取所有数值特征（排除ID列和已知字符串列）
        exclude_cols = ['loop_id', 'key_node_id', 'dominant_shareholder_type', 
                       'upstream_to_member_transaction_times', 'member_to_downstream_transaction_times']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # 严格检查数值特征，确保数据类型正确
        numeric_feature_cols = []
        for col in feature_cols:
            try:
                # 尝试转换为数值型，如果失败则跳过
                pd.to_numeric(self.df[col], errors='raise')
                numeric_feature_cols.append(col)
            except (ValueError, TypeError):
                self.logger.warning(f"跳过非数值列: {col}")
                continue
        
        self.logger.info(f"验证后的数值特征数量: {len(numeric_feature_cols)}")
        initial_feature_count = len(numeric_feature_cols)
        
        if len(numeric_feature_cols) == 0:
            self.logger.warning("没有有效的数值特征可供选择")
            return
        
        # 1. 去除低方差特征
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(self.df[numeric_feature_cols])
        low_variance_features = [numeric_feature_cols[i] for i in range(len(numeric_feature_cols)) 
                                if not selector.get_support()[i]]
        
        if low_variance_features:
            self.logger.info(f"删除 {len(low_variance_features)} 个低方差特征")
            self.df.drop(columns=low_variance_features, inplace=True)
            numeric_feature_cols = [col for col in numeric_feature_cols if col not in low_variance_features]
        
        # 2. 去除高相关特征
        if len(numeric_feature_cols) > 1:
            correlation_matrix = self.df[numeric_feature_cols].corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # 找到高相关特征对
            high_corr_features = []
            for column in upper_triangle.columns:
                high_corr_vars = list(upper_triangle.index[upper_triangle[column] > correlation_threshold])
                if high_corr_vars:
                    # 保留方差较大的特征
                    var_dict = {var: self.df[var].var() for var in [column] + high_corr_vars}
                    keep_feature = max(var_dict, key=var_dict.get)
                    drop_features = [f for f in var_dict.keys() if f != keep_feature]
                    high_corr_features.extend(drop_features)
            
            high_corr_features = list(set(high_corr_features))
            if high_corr_features:
                self.logger.info(f"删除 {len(high_corr_features)} 个高相关特征")
                self.df.drop(columns=[col for col in high_corr_features if col in self.df.columns], 
                            inplace=True)
                numeric_feature_cols = [col for col in numeric_feature_cols if col not in high_corr_features]
        
        # 3. 基于互信息的特征选择（选择信息量最大的特征）
        # 这里使用一个综合异常分数作为目标变量
        if 'multivariate_anomaly_score' in self.df.columns and len(numeric_feature_cols) > 1:
            remaining_features = [col for col in numeric_feature_cols 
                                if col != 'multivariate_anomaly_score']
            
            if len(remaining_features) > 0:
                try:
                    # 确保数据是纯数值型
                    X = self.df[remaining_features].select_dtypes(include=[np.number]).fillna(0)
                    y = self.df['multivariate_anomaly_score'].fillna(0)
                    
                    if X.shape[1] > 0:  # 确保有特征可以处理
                        mi_scores = mutual_info_regression(X, y, random_state=42)
                        mi_feature_importance = pd.DataFrame({
                            'feature': X.columns.tolist(),
                            'mi_score': mi_scores
                        }).sort_values('mi_score', ascending=False)
                        
                        # 保留前80%的高信息量特征
                        n_features_to_keep = max(1, int(len(X.columns) * 0.8))
                        top_features = mi_feature_importance.head(n_features_to_keep)['feature'].tolist()
                        
                        # 确保保留关键特征
                        essential_features = ['loop_id', 'multivariate_anomaly_score'] + [
                            col for col in self.numeric_features if col in self.df.columns
                        ][:20]  # 保留前20个原始特征
                        
                        features_to_keep = list(set(top_features + essential_features))
                        features_to_drop = [col for col in self.df.columns if col not in features_to_keep]
                        
                        if features_to_drop:
                            self.logger.info(f"基于互信息删除 {len(features_to_drop)} 个低信息量特征")
                            self.df.drop(columns=features_to_drop, inplace=True)
                    
                except Exception as e:
                    self.logger.warning(f"互信息特征选择失败，跳过该步骤: {e}")
        
        final_feature_count = len([col for col in self.df.columns if col != 'loop_id'])
        self.logger.info(f"特征选择完成: {initial_feature_count} -> {final_feature_count} 个特征")
        
        # 保存特征重要性报告
        self.save_feature_importance_report()
    
    def save_feature_importance_report(self):
        """保存特征重要性报告（简化版）"""
        # 只保存基本统计，不生成详细报告
        pass
    
    def save_engineered_features(self):
        """保存工程化后的特征"""
        output_file = f"{self.output_dir}/engineered_features.csv"
        self.df.to_csv(output_file, index=False)
        self.logger.info(f"工程化特征已保存至: {output_file}")
        
        # 简化输出，只保存核心数据
        feature_list = [col for col in self.df.columns if col != 'loop_id']
        
        self.logger.info(f"特征工程完成! 生成了 {len(feature_list)} 个特征")
        return self.df
    
    def run_feature_engineering(self):
        """执行完整的特征工程流程"""
        self.logger.info("=" * 50)
        self.logger.info("开始特征工程流程")
        self.logger.info("=" * 50)
        
        # 1. 加载数据
        if not self.load_data():
            return None
        
        # 2. 处理缺失值
        self.handle_missing_values()
        
        # 3. 特征变换
        self.transform_features()
        
        # 4. 创建比率特征
        self.create_ratio_features()
        
        # 5. 创建统计特征
        self.create_statistical_features()
        
        # 6. 创建交互特征
        self.create_interaction_features()
        
        # 7. 创建异常评分特征
        self.create_anomaly_score_features()
        
        # 8. 特征选择
        self.select_features()
        
        # 9. 保存结果
        result_df = self.save_engineered_features()
        
        self.logger.info("=" * 50)
        self.logger.info("特征工程流程完成!")
        self.logger.info("=" * 50)
        
        return result_df

def main():
    """主函数"""
    # 创建特征工程器
    engineer = FeatureEngineer()
    
    # 运行特征工程
    engineered_df = engineer.run_feature_engineering()
    
    if engineered_df is not None:
        print("\n特征工程成功完成!")
        print(f"生成特征矩阵维度: {engineered_df.shape}")
        print(f"输出文件位置: outputs/anomaly_detection/features/")
    else:
        print("\n特征工程失败，请检查日志")

if __name__ == "__main__":
    main()