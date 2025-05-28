#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
化工图风控系统 - 综合建模脚本
一键运行完整的异常检测流程：特征工程 → 模型训练 → 集成优化 → 结果解释

功能流程：
1. 清理历史输出目录
2. 特征工程 - 提取和构建工程特征
3. 模型训练 - 训练多种异常检测模型
4. 集成优化 - 智能集成多模型结果
5. 结果解释 - 生成可解释性分析报告

版本: v1.0 - 综合建模脚本
作者: AI助手
"""

import os
import sys
import time
import shutil
import logging
from pathlib import Path
import warnings

# 设置NumExpr线程数，避免警告
os.environ['NUMEXPR_MAX_THREADS'] = '14'

# 添加modeling目录到Python路径
sys.path.append('modeling')

# 忽略警告
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clear_output_directory():
    """清理输出目录"""
    output_dir = Path('outputs/anomaly_detection')
    
    if output_dir.exists():
        logger.info("清理历史输出目录...")
        try:
            shutil.rmtree(output_dir)
            logger.info("✅ 输出目录清理完成")
        except Exception as e:
            logger.warning(f"清理目录时出现警告: {e}")
    
    # 重新创建必要的目录结构
    subdirs = [
        'features', 'models', 'ensemble', 'explanations', 
        'visualizations', 'reports', 'logs'
    ]
    
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ 输出目录结构重建完成")

def run_feature_engineering():
    """运行特征工程"""
    logger.info("=" * 70)
    logger.info("步骤 1/4: 特征工程")
    logger.info("=" * 70)
    
    try:
        # 导入特征工程模块
        from feature_engineering import FeatureEngineer
        
        # 初始化特征工程器（使用默认输入文件）
        engineer = FeatureEngineer()
        
        # 运行特征工程
        result = engineer.run_feature_engineering()
        
        if result is not None:
            logger.info("✅ 特征工程完成")
            return True
        else:
            logger.error("❌ 特征工程失败")
            return False
            
    except ImportError as e:
        logger.error(f"❌ 导入特征工程模块失败: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 特征工程执行失败: {e}")
        return False

def run_anomaly_detection():
    """运行异常检测模型训练"""
    logger.info("=" * 70)
    logger.info("步骤 2/4: 异常检测模型训练")
    logger.info("=" * 70)
    
    try:
        # 导入异常检测模块
        from optimized_anomaly_detection import OptimizedAnomalyDetection
        
        # 初始化异常检测器
        detector = OptimizedAnomalyDetection()
        
        # 运行异常检测
        success = detector.run_complete_pipeline()
        
        if success:
            logger.info("✅ 异常检测模型训练完成")
            return True
        else:
            logger.error("❌ 异常检测模型训练失败")
            return False
            
    except ImportError as e:
        logger.error(f"❌ 导入异常检测模块失败: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 异常检测模型训练失败: {e}")
        return False

def run_ensemble_integration():
    """运行集成优化"""
    logger.info("=" * 70)
    logger.info("步骤 3/4: 集成优化")
    logger.info("=" * 70)
    
    try:
        # 导入集成模块
        from optimized_ensemble_integration import OptimizedEnsembleIntegrator
        
        # 检查模型分数文件
        model_scores_file = 'outputs/anomaly_detection/optimized_model_scores.csv'
        if not Path(model_scores_file).exists():
            model_scores_file = 'outputs/anomaly_detection/model_scores.csv'
            if not Path(model_scores_file).exists():
                logger.error("❌ 找不到模型分数文件")
                return False
        
        # 初始化集成器
        integrator = OptimizedEnsembleIntegrator(
            model_scores_file=model_scores_file,
            n_jobs=-1,
            optimization_trials=50
        )
        
        # 运行集成优化
        results = integrator.run_optimized_ensemble()
        
        if results:
            # 为解释系统创建兼容的文件名
            optimized_results_file = 'outputs/anomaly_detection/ensemble/optimized_final_results.csv'
            explanation_results_file = 'outputs/anomaly_detection/ensemble/final_anomaly_results.csv'
            
            if Path(optimized_results_file).exists():
                import shutil
                shutil.copy2(optimized_results_file, explanation_results_file)
                logger.info("✅ 已为解释系统创建兼容文件")
            
            logger.info("✅ 集成优化完成")
            return True
        else:
            logger.error("❌ 集成优化失败")
            return False
            
    except ImportError as e:
        logger.error(f"❌ 导入集成模块失败: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 集成优化执行失败: {e}")
        return False

def run_explanation_system():
    """运行解释系统"""
    logger.info("=" * 70)
    logger.info("步骤 4/4: 结果解释分析")
    logger.info("=" * 70)
    
    try:
        # 导入简化解释系统模块
        from anomaly_explanation_system_clean import SimpleAnomalyExplainer
        
        # 初始化解释系统
        explainer = SimpleAnomalyExplainer()
        
        # 运行解释分析
        success = explainer.run_complete_explanation()
        
        if success:
            logger.info("✅ 结果解释分析完成")
            return True
        else:
            logger.error("❌ 结果解释分析失败")
            return False
            
    except ImportError as e:
        logger.error(f"❌ 导入解释系统模块失败: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 结果解释分析失败: {e}")
        return False

def check_prerequisites():
    """检查前置条件"""
    logger.info("检查前置条件...")
    
    # 检查modeling目录
    modeling_dir = Path('modeling')
    if not modeling_dir.exists():
        logger.error("❌ 找不到modeling目录")
        return False
    
    # 检查必需的模块文件
    required_modules = [
        'modeling/feature_engineering.py',
        'modeling/optimized_anomaly_detection.py',
        'modeling/optimized_ensemble_integration.py',
        'modeling/anomaly_explanation_system_clean.py'
    ]
    
    missing_modules = []
    for module_path in required_modules:
        if not Path(module_path).exists():
            missing_modules.append(module_path)
    
    if missing_modules:
        logger.error("❌ 缺少必需的模块文件:")
        for module_path in missing_modules:
            logger.error(f"   - {module_path}")
        return False
    
    logger.info("✅ 所有必需的模块文件都存在")
    logger.info("✅ 前置条件检查通过")
    return True

def generate_final_summary():
    """生成最终总结报告"""
    logger.info("生成最终总结报告...")
    
    try:
        # 检查输出文件
        output_files = {
            '特征数据': 'outputs/anomaly_detection/features/engineered_features.csv',
            '模型分数': 'outputs/anomaly_detection/optimized_model_scores.csv',
            '最终结果': 'outputs/anomaly_detection/ensemble/optimized_final_results.csv',
            '解释系统结果': 'outputs/anomaly_detection/ensemble/final_anomaly_results.csv',
            '综合解释报告': 'outputs/anomaly_detection/reports/comprehensive_explanation_report.json',
            'HTML报告': 'outputs/anomaly_detection/reports/explanation_report.html'
        }
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 化工图风控系统异常检测建模完成！")
        logger.info("=" * 70)
        
        logger.info("📊 生成的主要输出文件:")
        for desc, filepath in output_files.items():
            if Path(filepath).exists():
                file_size = Path(filepath).stat().st_size / (1024 * 1024)  # MB
                logger.info(f"   ✅ {desc}: {filepath} ({file_size:.1f}MB)")
            else:
                logger.info(f"   ❌ {desc}: {filepath} (未生成)")
        
        logger.info("\n📁 完整输出目录结构:")
        output_dir = Path('outputs/anomaly_detection')
        for subdir in ['features', 'models', 'ensemble', 'explanations', 'visualizations', 'reports']:
            subdir_path = output_dir / subdir
            if subdir_path.exists():
                file_count = len(list(subdir_path.glob('*')))
                logger.info(f"   📂 {subdir}/  ({file_count} 个文件)")
        
        # 读取风险统计
        try:
            import pandas as pd
            final_results = pd.read_csv('outputs/anomaly_detection/ensemble/optimized_final_results.csv')
            risk_distribution = final_results['anomaly_level'].value_counts()
            
            logger.info("\n🚨 风险等级统计:")
            for level, count in risk_distribution.items():
                percentage = count / len(final_results) * 100
                logger.info(f"   {level}: {count} 个环路 ({percentage:.1f}%)")
                
        except Exception as e:
            logger.warning(f"无法读取风险统计: {e}")
        
        logger.info("\n🔍 推荐后续步骤:")
        logger.info("   1. 查看 explanation_report.html 获取直观的分析结果")
        logger.info("   2. 检查 optimized_final_results.csv 中的高风险环路")
        logger.info("   3. 根据解释报告制定风险缓解策略")
        logger.info("   4. 建立持续监控机制")
        
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"生成总结报告时出现错误: {e}")

def main():
    """主函数 - 运行完整的建模流程"""
    print("=" * 80)
    print("🏭 化工图风控系统 - 综合建模脚本")
    print("🚀 一键运行完整异常检测流程")
    print("=" * 80)
    
    start_time = time.time()
    
    # 检查前置条件
    if not check_prerequisites():
        logger.error("前置条件检查失败，无法继续执行")
        
        # 提供帮助信息
        print("\n" + "=" * 60)
        print("📋 使用说明:")
        print("=" * 60)
        print("此脚本需要以下数据文件才能运行:")
        print("1. 环路数据文件 (loops_data.csv)")
        print("2. 节点数据文件 (nodes_data.csv)")
        print("3. 边数据文件 (edges_data.csv)")
        print("\n💡 如果您有其他格式的数据文件，请：")
        print("1. 将数据文件重命名或复制到 data/ 目录")
        print("2. 确保文件格式为 CSV")
        print("3. 或者先运行数据预处理脚本")
        print("\n📝 当前检测到的文件:")
        data_dir = Path('data')
        if data_dir.exists():
            for file in data_dir.glob('*.csv'):
                file_size = file.stat().st_size / (1024 * 1024)
                print(f"   - {file.name} ({file_size:.1f}MB)")
        print("=" * 60)
        return
    
    # 询问用户是否清理输出目录
    response = input("\n是否清理历史输出目录？(y/n, 默认y): ").strip().lower()
    if response in ['', 'y', 'yes']:
        clear_output_directory()
    
    # 执行四个步骤
    steps = [
        ("特征工程", run_feature_engineering),
        ("异常检测模型训练", run_anomaly_detection), 
        ("集成优化", run_ensemble_integration),
        ("结果解释分析", run_explanation_system)
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        try:
            if step_func():
                success_count += 1
                logger.info(f"✅ {step_name} 成功完成")
            else:
                logger.error(f"❌ {step_name} 执行失败")
                break
                
        except KeyboardInterrupt:
            logger.warning("用户中断执行")
            break
        except Exception as e:
            logger.error(f"❌ {step_name} 执行过程中出现异常: {e}")
            break
    
    # 计算总耗时
    total_time = time.time() - start_time
    
    # 生成最终报告
    if success_count == len(steps):
        generate_final_summary()
        logger.info(f"\n🎯 全部 {len(steps)} 个步骤成功完成！总耗时: {total_time/60:.1f} 分钟")
    else:
        logger.error(f"\n💥 只完成了 {success_count}/{len(steps)} 个步骤，总耗时: {total_time/60:.1f} 分钟")
        logger.error("请检查日志信息并解决问题后重新运行")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 用户中断执行，程序退出")
    except Exception as e:
        print(f"\n💥 程序执行出现异常: {e}")
        import traceback
        traceback.print_exc() 