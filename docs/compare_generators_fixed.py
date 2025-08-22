#!/usr/bin/env python
"""
量子vs古典生成器性能对比分析脚本 (FIXED VERSION)
用于比较不同生成器类型的性能指标

修复的问题:
1. 可视化数组索引错误
2. 添加MLflow错误处理
3. 增加指标验证
4. 改进错误处理和日志记录
"""

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
import logging
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_experiment_runs(experiment_name: str, max_runs: int = 1000) -> pd.DataFrame:
    """获取实验的所有运行记录
    
    Args:
        experiment_name: 实验名称
        max_runs: 最大返回运行数量，防止内存问题
    
    Returns:
        包含运行数据的DataFrame
    """
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            logger.warning(f"实验 '{experiment_name}' 不存在")
            return pd.DataFrame()
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attribute.start_time desc"],
            max_results=max_runs
        )
        
        if not runs:
            logger.warning(f"实验 '{experiment_name}' 中没有找到运行记录")
            return pd.DataFrame()
        
        # 提取运行数据
        data = []
        for run in runs:
            run_data = {
                'run_id': run.info.run_id,
                'generator_type': run.data.params.get('generator_type', 'unknown'),
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'duration_seconds': (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else None
            }
            
            # 添加关键指标
            metrics_to_track = [
                'ValidationStep_FakeData_KLDivergence',
                'ValidationStep_FakeData_LogLikelihood', 
                'ValidationStep_FakeData_IsPositive',
                'ValidationStep_FakeData_WassersteinDistance',
                'ValidationStep_FakeData_MMDDistance',
                'train_loss_step',
                'd_loss',
                'g_loss'
            ]
            
            for metric in metrics_to_track:
                run_data[metric] = run.data.metrics.get(metric, None)
            
            data.append(run_data)
        
        df = pd.DataFrame(data)
        logger.info(f"成功获取 {len(df)} 条运行记录")
        return df
        
    except Exception as e:
        logger.error(f"获取实验运行记录失败: {e}")
        return pd.DataFrame()

def validate_metrics(gen_runs: pd.DataFrame, gen_type: str) -> bool:
    """验证关键指标是否存在和有效
    
    Args:
        gen_runs: 生成器运行数据
        gen_type: 生成器类型
    
    Returns:
        验证是否通过
    """
    required_metrics = [
        'ValidationStep_FakeData_KLDivergence',
        'duration_seconds'
    ]
    
    issues = []
    for metric in required_metrics:
        if metric not in gen_runs.columns:
            issues.append(f"缺少指标列 {metric}")
        elif gen_runs[metric].isna().all():
            issues.append(f"指标 {metric} 全部为空值")
        elif gen_runs[metric].notna().sum() == 0:
            issues.append(f"指标 {metric} 没有有效数据")
    
    if issues:
        logger.warning(f"生成器 {gen_type} 数据验证失败: {'; '.join(issues)}")
        return False
    
    return True

def analyze_convergence(client, run_id: str) -> Dict:
    """分析单个运行的收敛特性
    
    Args:
        client: MLflow客户端
        run_id: 运行ID
    
    Returns:
        收敛分析结果字典
    """
    try:
        # 获取历史指标
        metric_history = client.get_metric_history(run_id, "ValidationStep_FakeData_KLDivergence")
        
        if not metric_history:
            logger.warning(f"运行 {run_id} 没有KL散度历史数据")
            return {}
        
        epochs = [m.step for m in metric_history]
        values = [m.value for m in metric_history]
        
        if not values:
            return {}
        
        # 计算收敛指标
        convergence_info = {
            'final_value': values[-1],
            'best_value': min(values),
            'epochs_to_best': epochs[np.argmin(values)],
            'improvement_rate': (values[0] - values[-1]) / len(values) if len(values) > 1 else 0,
            'stability': np.std(values[-5:]) if len(values) >= 5 else None
        }
        
        return convergence_info
        
    except Exception as e:
        logger.error(f"分析运行 {run_id} 收敛性失败: {e}")
        return {}

def safe_calculate_percentage_diff(val1: float, val2: float) -> Optional[float]:
    """安全计算百分比差异
    
    Args:
        val1: 基准值
        val2: 比较值
    
    Returns:
        百分比差异，如果计算失败返回None
    """
    try:
        if pd.isna(val1) or pd.isna(val2):
            return None
        if val1 == 0:
            return float('inf') if val2 != 0 else 0
        return ((val2 - val1) / val1 * 100)
    except Exception:
        return None

def compare_generators(experiment_name: str = "quantum_vs_classical_comparison", 
                      output_dir: str = ".") -> Optional[pd.DataFrame]:
    """主对比函数
    
    Args:
        experiment_name: 实验名称
        output_dir: 输出目录
    
    Returns:
        对比结果DataFrame，失败时返回None
    """
    print("=" * 80)
    print("量子 vs 古典生成器性能对比分析")
    print("=" * 80)
    
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取运行数据
    df = get_experiment_runs(experiment_name)
    
    if df.empty:
        logger.error("没有找到实验数据，请先运行实验")
        return None
    
    # 按生成器类型分组
    generator_types = df['generator_type'].unique()
    print(f"\n找到 {len(generator_types)} 种生成器类型: {list(generator_types)}")
    print(f"总共 {len(df)} 次运行\n")
    
    # 创建对比表格
    comparison_results = []
    
    try:
        client = mlflow.tracking.MlflowClient()
        
        for gen_type in generator_types:
            gen_runs = df[df['generator_type'] == gen_type]
            
            # 验证数据质量
            if not validate_metrics(gen_runs, gen_type):
                logger.warning(f"跳过生成器类型 {gen_type} 由于数据质量问题")
                continue
            
            # 计算平均指标（只使用有效数据）
            result = {
                '生成器类型': gen_type,
                '运行次数': len(gen_runs),
                '平均训练时间(秒)': gen_runs['duration_seconds'].mean(),
                'KL散度(平均)': gen_runs['ValidationStep_FakeData_KLDivergence'].mean(),
                'KL散度(最佳)': gen_runs['ValidationStep_FakeData_KLDivergence'].min(),
                '对数似然(平均)': gen_runs['ValidationStep_FakeData_LogLikelihood'].mean(),
                'Wasserstein距离': gen_runs['ValidationStep_FakeData_WassersteinDistance'].mean(),
                'MMD距离': gen_runs['ValidationStep_FakeData_MMDDistance'].mean(),
            }
            
            # 获取最佳运行的收敛信息
            valid_kl_runs = gen_runs.dropna(subset=['ValidationStep_FakeData_KLDivergence'])
            if not valid_kl_runs.empty:
                best_run = valid_kl_runs.nsmallest(1, 'ValidationStep_FakeData_KLDivergence').iloc[0]
                convergence = analyze_convergence(client, best_run['run_id'])
                result.update({
                    '收敛epochs': convergence.get('epochs_to_best', 'N/A'),
                    '改进率': convergence.get('improvement_rate', 'N/A'),
                    '最终稳定性': convergence.get('stability', 'N/A')
                })
            else:
                result.update({
                    '收敛epochs': 'N/A',
                    '改进率': 'N/A',
                    '最终稳定性': 'N/A'
                })
            
            comparison_results.append(result)
        
    except Exception as e:
        logger.error(f"生成器对比分析失败: {e}")
        return None
    
    if not comparison_results:
        logger.error("没有有效的对比结果")
        return None
    
    # 创建对比DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    # 打印详细对比
    print("\n" + "="*80)
    print("性能对比结果")
    print("="*80)
    
    # 训练效率对比
    print("\n📊 训练效率对比:")
    print("-" * 40)
    for _, row in comparison_df.iterrows():
        duration = row['平均训练时间(秒)']
        if pd.notna(duration):
            print(f"{row['生成器类型']:20s}: {duration:.2f} 秒")
        else:
            print(f"{row['生成器类型']:20s}: N/A")
    
    # 生成质量对比
    print("\n📈 生成质量对比 (越低越好):")
    print("-" * 40)
    print(f"{'指标':<20} {'古典生成器':<15} {'量子生成器':<15} {'差异':<15}")
    print("-" * 65)
    
    metrics_to_compare = ['KL散度(最佳)', 'Wasserstein距离', 'MMD距离']
    
    for metric in metrics_to_compare:
        classical_val = comparison_df[comparison_df['生成器类型'].str.contains('classical', na=False)][metric].values
        quantum_val = comparison_df[comparison_df['生成器类型'].str.contains('quantum', na=False)][metric].values
        
        if len(classical_val) > 0 and len(quantum_val) > 0:
            c_val = classical_val[0]
            q_val = quantum_val[0]
            diff = safe_calculate_percentage_diff(c_val, q_val)
            
            c_str = f"{c_val:.4f}" if pd.notna(c_val) else "N/A"
            q_str = f"{q_val:.4f}" if pd.notna(q_val) else "N/A"
            diff_str = f"{diff:+.1f}%" if diff is not None else "N/A"
            
            print(f"{metric:<20} {c_str:<15} {q_str:<15} {diff_str}")
    
    # 收敛速度对比
    print("\n⚡ 收敛速度对比:")
    print("-" * 40)
    for _, row in comparison_df.iterrows():
        print(f"{row['生成器类型']:20s}: {row['收敛epochs']} epochs")
    
    # 性能比率计算
    print("\n" + "="*80)
    print("性能比率分析")
    print("="*80)
    
    classical_time = comparison_df[comparison_df['生成器类型'].str.contains('classical', na=False)]['平均训练时间(秒)'].values
    quantum_time = comparison_df[comparison_df['生成器类型'].str.contains('quantum', na=False)]['平均训练时间(秒)'].values
    
    if len(classical_time) > 0 and len(quantum_time) > 0:
        c_time, q_time = classical_time[0], quantum_time[0]
        if pd.notna(c_time) and pd.notna(q_time) and c_time > 0:
            time_ratio = q_time / c_time
            print(f"\n⏱️  时间比率: 量子生成器比古典生成器慢 {time_ratio:.1f}x")
        else:
            print("\n⏱️  时间比率: 无法计算（数据不完整）")
    
    # 创建可视化
    try:
        create_visualization(comparison_df, output_path)
    except Exception as e:
        logger.error(f"可视化创建失败: {e}")
    
    # 保存结果
    try:
        csv_path = output_path / 'generator_comparison_results.csv'
        comparison_df.to_csv(csv_path, index=False)
        print(f"\n💾 结果已保存到: {csv_path}")
    except Exception as e:
        logger.error(f"保存CSV文件失败: {e}")
    
    return comparison_df

def create_visualization(comparison_df: pd.DataFrame, output_path: Path):
    """创建对比可视化图表
    
    Args:
        comparison_df: 对比数据DataFrame
        output_path: 输出路径
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 训练时间对比 - 修复数组索引
        ax = axes[0][0]  # 修复: 从 axes[0, 0] 改为 axes[0][0]
        valid_data = comparison_df.dropna(subset=['平均训练时间(秒)'])
        if not valid_data.empty:
            ax.bar(valid_data['生成器类型'], valid_data['平均训练时间(秒)'])
        ax.set_title('训练时间对比')
        ax.set_ylabel('时间 (秒)')
        ax.set_xlabel('生成器类型')
        ax.tick_params(axis='x', rotation=45)
        
        # KL散度对比 - 修复数组索引
        ax = axes[0][1]  # 修复: 从 axes[0, 1] 改为 axes[0][1]
        valid_data = comparison_df.dropna(subset=['KL散度(最佳)'])
        if not valid_data.empty:
            ax.bar(valid_data['生成器类型'], valid_data['KL散度(最佳)'])
        ax.set_title('KL散度对比 (越低越好)')
        ax.set_ylabel('KL散度')
        ax.set_xlabel('生成器类型')
        ax.tick_params(axis='x', rotation=45)
        
        # Wasserstein距离对比 - 修复数组索引
        ax = axes[1][0]  # 修复: 从 axes[1, 0] 改为 axes[1][0]
        valid_data = comparison_df.dropna(subset=['Wasserstein距离'])
        if not valid_data.empty:
            ax.bar(valid_data['生成器类型'], valid_data['Wasserstein距离'])
        ax.set_title('Wasserstein距离对比 (越低越好)')
        ax.set_ylabel('Wasserstein距离')
        ax.set_xlabel('生成器类型')
        ax.tick_params(axis='x', rotation=45)
        
        # MMD距离对比 - 修复数组索引
        ax = axes[1][1]  # 修复: 从 axes[1, 1] 改为 axes[1][1]
        valid_data = comparison_df.dropna(subset=['MMD距离'])
        if not valid_data.empty:
            ax.bar(valid_data['生成器类型'], valid_data['MMD距离'])
        ax.set_title('MMD距离对比 (越低越好)')
        ax.set_ylabel('MMD距离')
        ax.set_xlabel('生成器类型')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        png_path = output_path / 'generator_comparison_plots.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"📊 可视化图表已保存到: {png_path}")
        
        # 清理matplotlib资源
        plt.close()
        
    except Exception as e:
        logger.error(f"创建可视化图表失败: {e}")
        if 'fig' in locals():
            plt.close()

if __name__ == "__main__":
    # 运行对比分析
    try:
        results = compare_generators()
        
        if results is not None and not results.empty:
            print("\n" + "="*80)
            print("🎯 关键发现:")
            print("="*80)
            
            # 计算关键指标
            classical_rows = results[results['生成器类型'].str.contains('classical', na=False)]
            quantum_rows = results[results['生成器类型'].str.contains('quantum', na=False)]
            
            if not classical_rows.empty and not quantum_rows.empty:
                c_kl = classical_rows['KL散度(最佳)'].values[0]
                q_kl = quantum_rows['KL散度(最佳)'].values[0]
                c_time = classical_rows['平均训练时间(秒)'].values[0]
                q_time = quantum_rows['平均训练时间(秒)'].values[0]
                
                if pd.notna(c_time) and pd.notna(q_time) and c_time > 0:
                    print(f"\n1. 量子生成器训练时间是古典生成器的 {q_time/c_time:.1f} 倍")
                
                if pd.notna(c_kl) and pd.notna(q_kl):
                    if q_kl < c_kl:
                        print(f"2. 量子生成器的KL散度比古典生成器低 {(c_kl-q_kl)/c_kl*100:.1f}% (更好)")
                    else:
                        print(f"2. 量子生成器的KL散度比古典生成器高 {(q_kl-c_kl)/c_kl*100:.1f}% (更差)")
                
                print("\n这些数值化结果直接回答了Ale的问题：")
                print("✅ 我们现在可以精确测量古典和量子生成器的性能差异")
                print("✅ 不仅有可视化对比，还有具体的数值指标")
            else:
                print("未找到足够的古典和量子生成器数据进行对比")
        else:
            print("对比分析失败或没有有效数据")
            
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print("程序执行过程中发生错误，请检查日志")