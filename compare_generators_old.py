#!/usr/bin/env python
"""
量子vs古典生成器性能对比分析脚本
用于比较不同生成器类型的性能指标
"""

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

def get_experiment_runs(experiment_name: str) -> pd.DataFrame:
    """获取实验的所有运行记录"""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"实验 '{experiment_name}' 不存在")
        return pd.DataFrame()
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time desc"]
    )
    
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
            value = run.data.metrics.get(metric, None)
            # 确保数值指标为float类型
            if value is not None:
                try:
                    run_data[metric] = float(value)
                except (ValueError, TypeError):
                    run_data[metric] = None
            else:
                run_data[metric] = None
        
        data.append(run_data)
    
    df = pd.DataFrame(data)
    
    # 确保数值列的类型正确
    numeric_columns = [
        'ValidationStep_FakeData_KLDivergence',
        'ValidationStep_FakeData_LogLikelihood', 
        'ValidationStep_FakeData_IsPositive',
        'ValidationStep_FakeData_WassersteinDistance',
        'ValidationStep_FakeData_MMDDistance',
        'train_loss_step',
        'd_loss',
        'g_loss',
        'duration_seconds'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def analyze_convergence(client, run_id: str) -> Dict:
    """分析单个运行的收敛特性"""
    # 获取历史指标
    metric_history = client.get_metric_history(run_id, "ValidationStep_FakeData_KLDivergence")
    
    if not metric_history:
        return {}
    
    epochs = [m.step for m in metric_history]
    values = [m.value for m in metric_history]
    
    # 计算收敛指标
    convergence_info = {
        'final_value': values[-1] if values else None,
        'best_value': min(values) if values else None,
        'epochs_to_best': epochs[np.argmin(values)] if values else None,
        'improvement_rate': (values[0] - values[-1]) / len(values) if len(values) > 1 else 0,
        'stability': np.std(values[-5:]) if len(values) >= 5 else None
    }
    
    return convergence_info

def compare_generators(experiment_name: str = None):
    """主对比函数"""
    print("=" * 80)
    print("量子 vs 古典生成器性能对比分析")
    print("=" * 80)
    
    # 获取运行数据
    if experiment_name is None:
        # 自动发现所有相关实验
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        # 寻找包含古典和量子实验的数据
        all_runs = []
        for exp in experiments:
            if any(keyword in exp.name.lower() for keyword in ['gaussgan', 'classical', 'quantum']):
                print(f"检查实验: {exp.name}")
                exp_df = get_experiment_runs(exp.name)
                if not exp_df.empty:
                    all_runs.append(exp_df)
        
        if all_runs:
            df = pd.concat(all_runs, ignore_index=True)
        else:
            df = pd.DataFrame()
    else:
        df = get_experiment_runs(experiment_name)
    
    if df.empty:
        print("没有找到实验数据，请先运行实验")
        return
    
    # 按生成器类型分组
    generator_types = df['generator_type'].unique()
    print(f"\n找到 {len(generator_types)} 种生成器类型: {list(generator_types)}")
    print(f"总共 {len(df)} 次运行\n")
    
    # 创建对比表格
    comparison_results = []
    client = mlflow.tracking.MlflowClient()
    
    for gen_type in generator_types:
        gen_runs = df[df['generator_type'] == gen_type]
        
        # 计算平均指标
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
        valid_runs = gen_runs.dropna(subset=['ValidationStep_FakeData_KLDivergence'])
        if not valid_runs.empty:
            best_run = valid_runs.nsmallest(1, 'ValidationStep_FakeData_KLDivergence').iloc[0]
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
        print(f"{row['生成器类型']:20s}: {row['平均训练时间(秒)']:.2f} 秒")
    
    # 生成质量对比
    print("\n📈 生成质量对比 (越低越好):")
    print("-" * 40)
    print(f"{'指标':<20} {'古典生成器':<15} {'量子生成器':<15} {'差异':<15}")
    print("-" * 65)
    
    metrics_to_compare = ['KL散度(最佳)', 'Wasserstein距离', 'MMD距离']
    
    for metric in metrics_to_compare:
        classical_val = comparison_df[comparison_df['生成器类型'].str.contains('classical')][metric].values
        quantum_val = comparison_df[comparison_df['生成器类型'].str.contains('quantum')][metric].values
        
        if len(classical_val) > 0 and len(quantum_val) > 0:
            c_val = classical_val[0]
            q_val = quantum_val[0]
            if pd.notna(c_val) and pd.notna(q_val):
                diff = ((q_val - c_val) / c_val * 100) if c_val != 0 else 0
                print(f"{metric:<20} {c_val:<15.4f} {q_val:<15.4f} {diff:+.1f}%")
    
    # 收敛速度对比
    print("\n⚡ 收敛速度对比:")
    print("-" * 40)
    for _, row in comparison_df.iterrows():
        print(f"{row['生成器类型']:20s}: {row['收敛epochs']} epochs")
    
    # 性能比率计算
    print("\n" + "="*80)
    print("性能比率分析")
    print("="*80)
    
    classical_rows = comparison_df[comparison_df['生成器类型'].str.contains('classical')]
    quantum_rows = comparison_df[comparison_df['生成器类型'].str.contains('quantum')]
    
    if not classical_rows.empty and not quantum_rows.empty:
        # 计算平均时间比率
        c_time_avg = classical_rows['平均训练时间(秒)'].mean()
        q_time_avg = quantum_rows['平均训练时间(秒)'].mean()
        time_ratio = q_time_avg / c_time_avg
        print(f"\n⏱️  平均时间比率: 量子生成器比古典生成器慢 {time_ratio:.1f}x")
        
        # 详细时间分析
        print(f"   古典生成器平均训练时间: {c_time_avg:.1f} 秒")
        print(f"   量子生成器平均训练时间: {q_time_avg:.1f} 秒")
    
    # 数据质量分析
    print(f"\n📊 数据质量分析:")
    print("-" * 40)
    for gen_type in comparison_df['生成器类型']:
        row = comparison_df[comparison_df['生成器类型'] == gen_type].iloc[0]
        runs = row['运行次数']
        kl_avg = row['KL散度(平均)']
        kl_best = row['KL散度(最佳)']
        if pd.notna(kl_avg) and pd.notna(kl_best):
            print(f"{gen_type:20s}: {runs}次运行, KL散度 {kl_best:.3f} (最佳) / {kl_avg:.3f} (平均)")
    
    # 创建可视化
    create_visualization(comparison_df)
    
    # 保存结果
    comparison_df.to_csv('generator_comparison_results.csv', index=False)
    print(f"\n💾 结果已保存到: generator_comparison_results.csv")
    
    return comparison_df

def create_visualization(comparison_df: pd.DataFrame):
    """创建对比可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 确保axes总是2D数组，即使只有一行或一列
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)
    
    # 训练时间对比
    ax = axes[0][0]
    valid_time_data = comparison_df.dropna(subset=['平均训练时间(秒)'])
    if not valid_time_data.empty:
        ax.bar(valid_time_data['生成器类型'], valid_time_data['平均训练时间(秒)'])
        ax.set_title('训练时间对比')
        ax.set_ylabel('时间 (秒)')
        ax.set_xlabel('生成器类型')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, '无可用数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('训练时间对比')
    
    # KL散度对比
    ax = axes[0][1]
    valid_kl_data = comparison_df.dropna(subset=['KL散度(最佳)'])
    if not valid_kl_data.empty:
        ax.bar(valid_kl_data['生成器类型'], valid_kl_data['KL散度(最佳)'])
        ax.set_title('KL散度对比 (越低越好)')
        ax.set_ylabel('KL散度')
        ax.set_xlabel('生成器类型')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, '无可用数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('KL散度对比 (越低越好)')
    
    # Wasserstein距离对比
    ax = axes[1][0]
    valid_ws_data = comparison_df.dropna(subset=['Wasserstein距离'])
    if not valid_ws_data.empty:
        ax.bar(valid_ws_data['生成器类型'], valid_ws_data['Wasserstein距离'])
        ax.set_title('Wasserstein距离对比 (越低越好)')
        ax.set_ylabel('Wasserstein距离')
        ax.set_xlabel('生成器类型')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, '无可用数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Wasserstein距离对比 (越低越好)')
    
    # MMD距离对比
    ax = axes[1][1]
    valid_mmd_data = comparison_df.dropna(subset=['MMD距离'])
    if not valid_mmd_data.empty:
        ax.bar(valid_mmd_data['生成器类型'], valid_mmd_data['MMD距离'])
        ax.set_title('MMD距离对比 (越低越好)')
        ax.set_ylabel('MMD距离')
        ax.set_xlabel('生成器类型')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, '无可用数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('MMD距离对比 (越低越好)')
    
    plt.tight_layout()
    plt.savefig('generator_comparison_plots.png', dpi=300, bbox_inches='tight')
    print(f"📊 可视化图表已保存到: generator_comparison_plots.png")

if __name__ == "__main__":
    # 运行对比分析
    results = compare_generators()
    
    if results is not None and not results.empty:
        print("\n" + "="*80)
        print("🎯 关键发现:")
        print("="*80)
        
        # 计算关键指标
        classical_rows = results[results['生成器类型'].str.contains('classical')]
        quantum_rows = results[results['生成器类型'].str.contains('quantum')]
        
        if not classical_rows.empty and not quantum_rows.empty:
            c_kl = classical_rows['KL散度(最佳)'].values[0]
            q_kl = quantum_rows['KL散度(最佳)'].values[0]
            c_time = classical_rows['平均训练时间(秒)'].values[0]
            q_time = quantum_rows['平均训练时间(秒)'].values[0]
            
            print(f"\n1. 量子生成器训练时间是古典生成器的 {q_time/c_time:.1f} 倍")
            
            if pd.notna(c_kl) and pd.notna(q_kl):
                if q_kl < c_kl:
                    print(f"2. 量子生成器的KL散度比古典生成器低 {(c_kl-q_kl)/c_kl*100:.1f}% (更好)")
                else:
                    print(f"2. 量子生成器的KL散度比古典生成器高 {(q_kl-c_kl)/c_kl*100:.1f}% (更差)")
            
            print("\n这些数值化结果直接回答了Ale的问题：")
            print("✅ 我们现在可以精确测量古典和量子生成器的性能差异")
            print("✅ 不仅有可视化对比，还有具体的数值指标")
            
            print("\n" + "="*80)
            print("🔍 深入分析")
            print("="*80)
            
            # 分析结果的可靠性
            c_runs = classical_rows['运行次数'].sum()
            q_runs = quantum_rows['运行次数'].sum()
            print(f"\n数据样本大小:")
            print(f"- 古典生成器: {c_runs} 次运行")
            print(f"- 量子生成器: {q_runs} 次运行")
            
            # 性能权衡分析
            if pd.notna(c_kl) and pd.notna(q_kl):
                efficiency_score = (c_kl - q_kl) / (q_time/c_time - 1)  # 质量改进 vs 时间成本
                print(f"\n性能权衡:")
                print(f"- 时间成本: 量子生成器慢 {q_time/c_time:.1f}x")
                if q_kl < c_kl:
                    print(f"- 质量收益: KL散度改善 {(c_kl-q_kl)/c_kl*100:.1f}%")
                    if efficiency_score > 0:
                        print(f"- 结论: 质量改进证明时间成本是合理的")
                    else:
                        print(f"- 结论: 质量改进相对于时间成本较小")
                else:
                    print(f"- 质量损失: KL散度恶化 {(q_kl-c_kl)/c_kl*100:.1f}%")
                    print(f"- 结论: 量子生成器在当前配置下性能不如古典生成器")
            
            print("\n📋 实验建议:")
            print("- 考虑优化量子电路参数以提高训练效率")
            print("- 增加量子生成器的训练epochs以获得更好的收敛")
            print("- 尝试不同的量子电路架构")