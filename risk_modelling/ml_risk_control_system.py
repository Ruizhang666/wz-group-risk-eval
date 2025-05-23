#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机器学习风控系统 - 基于图神经网络和异常检测的智能风险识别

本系统包含三个核心模块：
1. 特征工程：从图结构中提取丰富的特征
2. 风险评分模型：基于历史案例学习风险模式
3. 异常检测：发现未知的可疑模式
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from datetime import datetime, timedelta
import logging
import pickle
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# 机器学习相关库
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 深度学习相关（用于图神经网络）
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告：PyTorch未安装，图神经网络功能将不可用")

# 配置
CONFIG = {
    "feature_version": "v1.0",
    "model_dir": "models/ml_risk",
    "data_dir": "data/ml_features",
    "log_file": "outputs/log/ml_risk_control.log",
    "random_seed": 42
}

def setup_logging():
    """设置日志"""
    os.makedirs(os.path.dirname(CONFIG["log_file"]), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(CONFIG["log_file"], 'w', 'utf-8'),
            logging.StreamHandler()
        ]
    )

class FeatureExtractor:
    """
    特征提取器 - 从图结构和闭环中提取机器学习特征
    
    这是整个系统的基础，好的特征是成功的一半
    """
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.feature_names = []
        
    def extract_loop_features(self, loop_nodes: List[str]) -> Dict[str, float]:
        """
        提取闭环的特征
        
        我们从多个维度提取特征：
        1. 结构特征：闭环的拓扑结构
        2. 节点特征：参与节点的属性
        3. 交易特征：交易行为模式
        4. 时序特征：时间分布特点
        """
        features = {}
        
        # 1. 基础结构特征
        features.update(self._extract_structural_features(loop_nodes))
        
        # 2. 节点角色特征
        features.update(self._extract_role_features(loop_nodes))
        
        # 3. 交易模式特征
        features.update(self._extract_transaction_features(loop_nodes))
        
        # 4. 股权结构特征
        features.update(self._extract_equity_features(loop_nodes))
        
        # 5. 时序行为特征
        features.update(self._extract_temporal_features(loop_nodes))
        
        # 6. 网络中心性特征
        features.update(self._extract_centrality_features(loop_nodes))
        
        return features
    
    def _extract_structural_features(self, loop_nodes: List[str]) -> Dict[str, float]:
        """提取结构特征"""
        features = {}
        
        # 闭环大小
        features['loop_size'] = len(set(loop_nodes))
        
        # 闭环内部的边密度
        subgraph = self.graph.subgraph(loop_nodes)
        if len(loop_nodes) > 1:
            max_edges = len(loop_nodes) * (len(loop_nodes) - 1)
            features['loop_density'] = subgraph.number_of_edges() / max_edges if max_edges > 0 else 0
        else:
            features['loop_density'] = 0
        
        # 闭环的聚类系数
        clustering_coeffs = []
        for node in loop_nodes:
            if node in self.graph:
                clustering_coeffs.append(nx.clustering(self.graph, node))
        
        features['avg_clustering'] = np.mean(clustering_coeffs) if clustering_coeffs else 0
        features['max_clustering'] = np.max(clustering_coeffs) if clustering_coeffs else 0
        
        # 闭环节点的平均度
        degrees = [self.graph.degree(node) for node in loop_nodes if node in self.graph]
        features['avg_degree'] = np.mean(degrees) if degrees else 0
        features['max_degree'] = np.max(degrees) if degrees else 0
        features['degree_variance'] = np.var(degrees) if degrees else 0
        
        return features
    
    def _extract_role_features(self, loop_nodes: List[str]) -> Dict[str, float]:
        """提取角色分布特征"""
        features = {}
        
        # 统计各种角色的数量
        role_counts = defaultdict(int)
        for node in loop_nodes:
            if node in self.graph.nodes:
                role = self.graph.nodes[node].get('label', 'unknown')
                role_counts[role] += 1
        
        # 各角色占比
        total_nodes = len(loop_nodes)
        features['shareholder_ratio'] = role_counts.get('股东', 0) / total_nodes if total_nodes > 0 else 0
        features['partner_ratio'] = role_counts.get('partner', 0) / total_nodes if total_nodes > 0 else 0
        features['member_ratio'] = role_counts.get('成员单位', 0) / total_nodes if total_nodes > 0 else 0
        
        # 角色多样性（熵）
        if total_nodes > 0:
            role_probs = [count/total_nodes for count in role_counts.values()]
            features['role_entropy'] = -sum(p * np.log(p + 1e-10) for p in role_probs)
        else:
            features['role_entropy'] = 0
        
        return features
    
    def _extract_transaction_features(self, loop_nodes: List[str]) -> Dict[str, float]:
        """提取交易特征"""
        features = {}
        
        # 收集闭环内的所有交易
        transactions = []
        transaction_amounts = []
        
        for i in range(len(loop_nodes)):
            u = loop_nodes[i]
            v = loop_nodes[(i + 1) % len(loop_nodes)]
            
            if self.graph.has_edge(u, v):
                edge_data = self.graph.get_edge_data(u, v)
                if edge_data.get('label') == '交易':
                    amount = edge_data.get('amount', 0)
                    if isinstance(amount, (int, float)) and amount > 0:
                        transaction_amounts.append(float(amount))
                        transactions.append(edge_data)
        
        # 交易金额统计
        if transaction_amounts:
            features['total_transaction_amount'] = sum(transaction_amounts)
            features['avg_transaction_amount'] = np.mean(transaction_amounts)
            features['max_transaction_amount'] = max(transaction_amounts)
            features['min_transaction_amount'] = min(transaction_amounts)
            features['transaction_amount_std'] = np.std(transaction_amounts)
            
            # 交易金额的偏度（是否有异常大的交易）
            mean_amount = np.mean(transaction_amounts)
            std_amount = np.std(transaction_amounts)
            if std_amount > 0:
                features['transaction_skewness'] = np.mean([(x - mean_amount)**3 for x in transaction_amounts]) / (std_amount**3)
            else:
                features['transaction_skewness'] = 0
        else:
            # 没有交易时的默认值
            for key in ['total_transaction_amount', 'avg_transaction_amount', 
                       'max_transaction_amount', 'min_transaction_amount', 
                       'transaction_amount_std', 'transaction_skewness']:
                features[key] = 0
        
        # 交易频率
        features['transaction_count'] = len(transactions)
        features['transaction_ratio'] = len(transactions) / len(loop_nodes) if loop_nodes else 0
        
        return features
    
    def _extract_equity_features(self, loop_nodes: List[str]) -> Dict[str, float]:
        """提取股权特征"""
        features = {}
        
        # 收集股权信息
        equity_percentages = []
        
        for node in loop_nodes:
            if node in self.graph:
                # 出边代表持股
                for _, target, data in self.graph.out_edges(node, data=True):
                    if data.get('label') == '控股':
                        percent = data.get('percent', 0)
                        if isinstance(percent, (int, float)) and percent > 0:
                            equity_percentages.append(float(percent))
        
        # 股权集中度指标
        if equity_percentages:
            features['avg_equity_percent'] = np.mean(equity_percentages)
            features['max_equity_percent'] = max(equity_percentages)
            features['min_equity_percent'] = min(equity_percentages)
            
            # HHI指数（赫芬达尔指数）
            total_equity = sum(equity_percentages)
            if total_equity > 0:
                normalized_equities = [e/total_equity for e in equity_percentages]
                features['equity_hhi'] = sum(e**2 for e in normalized_equities)
            else:
                features['equity_hhi'] = 0
                
            # 控制权特征（是否有绝对控股）
            features['has_absolute_control'] = 1.0 if any(e > 0.5 for e in equity_percentages) else 0.0
            features['has_relative_control'] = 1.0 if any(0.3 <= e <= 0.5 for e in equity_percentages) else 0.0
        else:
            for key in ['avg_equity_percent', 'max_equity_percent', 'min_equity_percent', 
                       'equity_hhi', 'has_absolute_control', 'has_relative_control']:
                features[key] = 0
        
        return features
    
    def _extract_temporal_features(self, loop_nodes: List[str]) -> Dict[str, float]:
        """提取时序特征"""
        features = {}
        
        # 收集交易时间
        transaction_dates = []
        
        for i in range(len(loop_nodes)):
            u = loop_nodes[i]
            v = loop_nodes[(i + 1) % len(loop_nodes)]
            
            if self.graph.has_edge(u, v):
                edge_data = self.graph.get_edge_data(u, v)
                if edge_data.get('label') == '交易':
                    year = edge_data.get('year')
                    month = edge_data.get('month')
                    if year and month:
                        try:
                            date = datetime(int(year), int(month), 1)
                            transaction_dates.append(date)
                        except:
                            pass
        
        if transaction_dates:
            transaction_dates.sort()
            
            # 时间跨度
            time_span = (transaction_dates[-1] - transaction_dates[0]).days
            features['transaction_time_span_days'] = time_span
            
            # 交易时间间隔统计
            if len(transaction_dates) > 1:
                intervals = [(transaction_dates[i+1] - transaction_dates[i]).days 
                           for i in range(len(transaction_dates)-1)]
                features['avg_transaction_interval'] = np.mean(intervals)
                features['min_transaction_interval'] = min(intervals)
                features['transaction_interval_std'] = np.std(intervals)
                
                # 交易聚集度（标准差越小越聚集）
                if features['avg_transaction_interval'] > 0:
                    features['transaction_clustering'] = features['transaction_interval_std'] / features['avg_transaction_interval']
                else:
                    features['transaction_clustering'] = 0
            else:
                for key in ['avg_transaction_interval', 'min_transaction_interval', 
                           'transaction_interval_std', 'transaction_clustering']:
                    features[key] = 0
                    
            # 季节性特征
            months = [d.month for d in transaction_dates]
            quarters = [(m-1)//3 + 1 for m in months]
            
            # 季度分布的熵（是否集中在特定季度）
            quarter_counts = pd.Series(quarters).value_counts()
            quarter_probs = quarter_counts / len(quarters)
            features['quarter_entropy'] = -sum(p * np.log(p + 1e-10) for p in quarter_probs)
            
            # 是否有年底突击交易
            q4_ratio = quarter_counts.get(4, 0) / len(quarters) if quarters else 0
            features['q4_transaction_ratio'] = q4_ratio
            
        else:
            # 没有时间信息时的默认值
            for key in ['transaction_time_span_days', 'avg_transaction_interval', 
                       'min_transaction_interval', 'transaction_interval_std', 
                       'transaction_clustering', 'quarter_entropy', 'q4_transaction_ratio']:
                features[key] = 0
        
        return features
    
    def _extract_centrality_features(self, loop_nodes: List[str]) -> Dict[str, float]:
        """提取中心性特征"""
        features = {}
        
        # 计算各种中心性指标
        centrality_metrics = {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph, k=min(100, len(self.graph))),
            'closeness': nx.closeness_centrality(self.graph)
        }
        
        # 闭环节点的中心性统计
        for metric_name, centrality_dict in centrality_metrics.items():
            values = [centrality_dict.get(node, 0) for node in loop_nodes if node in self.graph]
            if values:
                features[f'avg_{metric_name}_centrality'] = np.mean(values)
                features[f'max_{metric_name}_centrality'] = max(values)
                features[f'min_{metric_name}_centrality'] = min(values)
            else:
                features[f'avg_{metric_name}_centrality'] = 0
                features[f'max_{metric_name}_centrality'] = 0
                features[f'min_{metric_name}_centrality'] = 0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        if not self.feature_names:
            # 通过提取一个示例来获取特征名称
            sample_loop = list(self.graph.nodes())[:5]
            features = self.extract_loop_features(sample_loop)
            self.feature_names = list(features.keys())
        return self.feature_names


class RiskScoreModel:
    """
    风险评分模型 - 基于监督学习预测闭环风险
    
    这个模型从历史标注的风险案例中学习，能够自动识别新的风险模式
    """
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def build_model(self):
        """构建模型"""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='binary:logistic',
                random_state=CONFIG['random_seed']
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=CONFIG['random_seed']
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split=0.2):
        """
        训练模型
        
        参数:
            X: 特征矩阵
            y: 标签（0=正常, 1=风险）
            validation_split: 验证集比例
        """
        logging.info(f"开始训练{self.model_type}模型...")
        
        # 数据分割
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=CONFIG['random_seed'], stratify=y
        )
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 训练模型
        self.build_model()
        self.model.fit(X_train_scaled, y_train)
        
        # 评估模型
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)
        
        # 计算AUC
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        logging.info(f"训练完成！")
        logging.info(f"训练集准确率: {train_score:.3f}")
        logging.info(f"验证集准确率: {val_score:.3f}")
        logging.info(f"验证集AUC: {auc_score:.3f}")
        
        # 提取特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        return {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'val_auc': auc_score
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测风险概率"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save_model(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logging.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']
        logging.info(f"模型已从 {path} 加载")


class AnomalyDetector:
    """
    异常检测器 - 使用无监督学习发现未知风险模式
    
    这个模块不需要历史标签，能够自动发现偏离正常模式的异常闭环
    """
    
    def __init__(self, contamination=0.1):
        """
        参数:
            contamination: 预期的异常比例
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=CONFIG['random_seed'],
            n_estimators=100
        )
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray):
        """训练异常检测模型"""
        logging.info("训练异常检测模型...")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        logging.info(f"异常检测模型训练完成，预期异常率: {self.contamination*100:.1f}%")
        
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        计算异常分数
        
        返回:
            异常分数数组，分数越低越异常
        """
        X_scaled = self.scaler.transform(X)
        # 将隔离森林的分数转换为0-1之间，1表示最异常
        scores = self.model.score_samples(X_scaled)
        # 归一化到0-1
        normalized_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        return normalized_scores
    
    def find_anomalies(self, X: np.ndarray, threshold=None) -> np.ndarray:
        """
        找出异常样本
        
        返回:
            布尔数组，True表示异常
        """
        if threshold is None:
            # 使用模型的默认判断
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            return predictions == -1
        else:
            # 使用自定义阈值
            scores = self.predict_anomaly_score(X)
            return scores > threshold


class GraphNeuralNetwork(nn.Module):
    """
    图神经网络模型 - 用于学习闭环的图结构表示
    
    这是一个更高级的模型，能够直接从图结构中学习特征
    """
    
    def __init__(self, num_features, hidden_dim=64, num_classes=2):
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index, batch):
        # 图卷积层
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        # 全局池化
        x = global_mean_pool(x, batch)
        
        # 分类
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class MLRiskControlSystem:
    """
    机器学习风控系统主类 - 整合所有组件
    """
    
    def __init__(self, graph_path: str):
        setup_logging()
        logging.info("初始化机器学习风控系统...")
        
        # 加载图
        self.graph = nx.read_graphml(graph_path)
        logging.info(f"加载图完成: {len(self.graph.nodes)} 节点, {len(self.graph.edges)} 边")
        
        # 初始化组件
        self.feature_extractor = FeatureExtractor(self.graph)
        self.risk_model = RiskScoreModel()
        self.anomaly_detector = AnomalyDetector()
        
        # 创建必要的目录
        os.makedirs(CONFIG['model_dir'], exist_ok=True)
        os.makedirs(CONFIG['data_dir'], exist_ok=True)
        
    def prepare_training_data(self, labeled_loops: Dict[int, Dict], labels: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        参数:
            labeled_loops: 已标注的闭环数据 {loop_id: loop_info}
            labels: 标签 {loop_id: 0/1} (0=正常, 1=风险)
            
        返回:
            X: 特征矩阵
            y: 标签数组
        """
        logging.info("准备训练数据...")
        
        features_list = []
        labels_list = []
        
        for loop_id, loop_info in labeled_loops.items():
            if loop_id in labels:
                # 提取特征
                loop_nodes = loop_info['node_path']
                features = self.feature_extractor.extract_loop_features(loop_nodes)
                
                features_list.append(list(features.values()))
                labels_list.append(labels[loop_id])
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        logging.info(f"准备了 {len(X)} 个样本，其中风险样本 {sum(y)} 个")
        
        return X, y
    
    def train_risk_model(self, X: np.ndarray, y: np.ndarray):
        """训练风险评分模型"""
        logging.info("="*60)
        logging.info("训练风险评分模型")
        logging.info("="*60)
        
        metrics = self.risk_model.train(X, y)
        
        # 保存模型
        model_path = os.path.join(CONFIG['model_dir'], 'risk_score_model.pkl')
        self.risk_model.save_model(model_path)
        
        # 分析特征重要性
        if self.risk_model.feature_importance is not None:
            self._analyze_feature_importance()
        
        return metrics
    
    def train_anomaly_detector(self, X: np.ndarray):
        """训练异常检测器"""
        logging.info("="*60)
        logging.info("训练异常检测器")
        logging.info("="*60)
        
        self.anomaly_detector.fit(X)
        
        # 保存异常检测器
        detector_path = os.path.join(CONFIG['model_dir'], 'anomaly_detector.pkl')
        with open(detector_path, 'wb') as f:
            pickle.dump(self.anomaly_detector, f)
        logging.info(f"异常检测器已保存到: {detector_path}")
    
    def analyze_loops(self, loops: Dict[int, Dict]) -> pd.DataFrame:
        """
        分析给定的闭环
        
        返回包含风险分数和异常分数的DataFrame
        """
        logging.info(f"开始分析 {len(loops)} 个闭环...")
        
        results = []
        
        for loop_id, loop_info in loops.items():
            # 提取特征
            loop_nodes = loop_info['node_path']
            features = self.feature_extractor.extract_loop_features(loop_nodes)
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # 计算风险分数
            risk_score = self.risk_model.predict_proba(feature_vector)[0]
            
            # 计算异常分数
            anomaly_score = self.anomaly_detector.predict_anomaly_score(feature_vector)[0]
            
            # 综合评分 (可以根据需要调整权重)
            combined_score = 0.7 * risk_score + 0.3 * anomaly_score
            
            result = {
                'loop_id': loop_id,
                'risk_score': risk_score,
                'anomaly_score': anomaly_score,
                'combined_score': combined_score,
                'loop_type': loop_info.get('type', ''),
                'node_count': len(set(loop_nodes)),
                **features  # 包含所有特征
            }
            
            results.append(result)
        
        df_results = pd.DataFrame(results)
        
        # 按综合分数排序
        df_results = df_results.sort_values('combined_score', ascending=False)
        
        logging.info(f"分析完成！前10个高风险闭环的综合分数: {df_results['combined_score'].head(10).tolist()}")
        
        return df_results
    
    def _analyze_feature_importance(self):
        """分析并记录特征重要性"""
        feature_names = self.feature_extractor.get_feature_names()
        importance = self.risk_model.feature_importance
        
        if importance is not None and len(feature_names) == len(importance):
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            logging.info("\n特征重要性Top 10:")
            for idx, row in importance_df.head(10).iterrows():
                logging.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            # 保存特征重要性
            importance_path = os.path.join(CONFIG['model_dir'], 'feature_importance.csv')
            importance_df.to_csv(importance_path, index=False)
    
    def generate_risk_report(self, analysis_results: pd.DataFrame, output_path: str):
        """生成风险分析报告"""
        logging.info("生成风险分析报告...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 机器学习风险分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 风险概览\n\n")
            f.write(f"- 分析闭环总数: {len(analysis_results)}\n")
            f.write(f"- 高风险闭环数 (综合分数>0.7): {len(analysis_results[analysis_results['combined_score'] > 0.7])}\n")
            f.write(f"- 中风险闭环数 (0.4-0.7): {len(analysis_results[(analysis_results['combined_score'] >= 0.4) & (analysis_results['combined_score'] <= 0.7)])}\n")
            f.write(f"- 低风险闭环数 (<0.4): {len(analysis_results[analysis_results['combined_score'] < 0.4])}\n\n")
            
            f.write("## 风险指标分布\n\n")
            f.write(f"- 平均风险分数: {analysis_results['risk_score'].mean():.3f}\n")
            f.write(f"- 平均异常分数: {analysis_results['anomaly_score'].mean():.3f}\n")
            f.write(f"- 风险分数标准差: {analysis_results['risk_score'].std():.3f}\n\n")
            
            f.write("## 高风险闭环详情 (Top 20)\n\n")
            
            for idx, row in analysis_results.head(20).iterrows():
                f.write(f"### {idx+1}. 闭环ID: {row['loop_id']}\n\n")
                f.write(f"- **综合风险分数**: {row['combined_score']:.3f}\n")
                f.write(f"- **风险模型分数**: {row['risk_score']:.3f}\n")
                f.write(f"- **异常检测分数**: {row['anomaly_score']:.3f}\n")
                f.write(f"- **闭环类型**: {row['loop_type']}\n")
                f.write(f"- **节点数**: {row['node_count']}\n\n")
                
                f.write("**关键风险指标**:\n")
                # 显示最重要的几个特征
                risk_features = ['total_transaction_amount', 'transaction_count', 
                               'max_equity_percent', 'avg_degree_centrality']
                for feature in risk_features:
                    if feature in row:
                        f.write(f"- {feature}: {row[feature]:.3f}\n")
                
                f.write("\n---\n\n")
            
            f.write("## 模型解释\n\n")
            f.write("本报告使用了两种机器学习方法：\n\n")
            f.write("1. **监督学习风险模型**：基于历史标注数据训练，能够识别已知的风险模式\n")
            f.write("2. **无监督异常检测**：不依赖标注，自动发现偏离正常模式的异常闭环\n\n")
            f.write("综合分数 = 0.7 × 风险分数 + 0.3 × 异常分数\n")
        
        logging.info(f"风险报告已生成: {output_path}")


def demo_usage():
    """演示如何使用机器学习风控系统"""
    
    # 1. 初始化系统
    system = MLRiskControlSystem("model/final_heterogeneous_graph.graphml")
    
    # 2. 准备训练数据（这里使用模拟数据作为示例）
    # 在实际应用中，您需要：
    # - 从历史案例中收集已确认的风险闭环
    # - 标注每个闭环是否存在风险（0=正常, 1=风险）
    
    # 模拟一些训练数据
    from code.loop_filter import parse_loops  # 复用之前的解析函数
    
    loops = parse_loops("outputs/loop_results/equity_loops_optimized.txt")
    
    # 模拟标注（实际应用中需要人工标注或从历史案例获取）
    labels = {}
    for loop_id in list(loops.keys())[:100]:  # 假设标注了前100个
        # 这里随机标注，实际应该基于真实的风险判断
        # 例如：涉及特定公司、特定金额以上、特定时间模式等
        labels[loop_id] = 1 if np.random.random() > 0.8 else 0
    
    # 3. 准备特征
    X, y = system.prepare_training_data(loops, labels)
    
    # 4. 训练模型
    if len(X) > 20:  # 确保有足够的训练数据
        # 训练风险评分模型
        system.train_risk_model(X, y)
        
        # 训练异常检测器（使用所有数据，不需要标签）
        all_features = []
        for loop_id, loop_info in list(loops.items())[:500]:  # 使用更多数据
            features = system.feature_extractor.extract_loop_features(loop_info['node_path'])
            all_features.append(list(features.values()))
        
        X_all = np.array(all_features)
        system.train_anomaly_detector(X_all)
    
    # 5. 分析新的闭环
    test_loops = {k: v for k, v in loops.items() if k not in labels}  # 未标注的闭环
    if test_loops:
        analysis_results = system.analyze_loops(test_loops)
        
        # 6. 生成报告
        report_path = "outputs/ml_risk_analysis_report.md"
        system.generate_risk_report(analysis_results, report_path)
        
        # 7. 保存分析结果供进一步使用
        results_path = "outputs/ml_risk_scores.csv"
        analysis_results.to_csv(results_path, index=False)
        
        print(f"\n分析完成！")
        print(f"- 风险报告: {report_path}")
        print(f"- 详细结果: {results_path}")
        print(f"\n高风险闭环Top 5:")
        print(analysis_results[['loop_id', 'combined_score', 'risk_score', 'anomaly_score']].head())


if __name__ == "__main__":
    # 运行演示
    demo_usage()
