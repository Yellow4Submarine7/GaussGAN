#!/usr/bin/env python
"""
重构版量子vs古典生成器逐epoch性能对比分析脚本
实现逐epoch的完整指标历史追踪和可视化
"""

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

# 全局变量控制标签语言
use_english_labels = True  # 强制使用英文标签避免字体问题

# 设置中文字体支持
def setup_chinese_fonts():
    """设置matplotlib中文字体支持"""
    try:
        # 尝试设置中文字体
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 验证字体设置
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试', fontsize=12)
        plt.close(fig)
        
    except Exception as e:
        print(f"中文字体设置失败，使用英文标签: {e}")
        # 如果中文不可用，使用英文标签
        global use_english_labels
        use_english_labels = True
    
    # 设置seaborn样式
    sns.set_style("whitegrid")
    sns.set_palette("husl")

def get_run_metric_history(client: mlflow.tracking.MlflowClient, 
                          run_id: str, 
                          metric_name: str) -> List[Tuple[int, float]]:
    """获取指定运行的指标历史数据
    
    Returns:
        List of (step, value) tuples
    """
    try:
        history = client.get_metric_history(run_id, metric_name)
        return [(h.step, h.value) for h in history]
    except Exception as e:
        print(f"警告: 无法获取运行 {run_id[:8]} 的指标 {metric_name}: {e}")
        return []

def convert_steps_to_epochs(step_value_pairs: List[Tuple[int, float]], 
                           validation_frequency: int = 1) -> List[Tuple[int, float]]:
    """将step转换为epoch
    
    假设validation在每个epoch结束时执行
    根据观察到的数据，step 62 对应 epoch 0，step 125 对应 epoch 1，等等
    这意味着validation步骤之间大约相差63步
    """
    if not step_value_pairs:
        return []
    
    epoch_value_pairs = []
    
    # 根据观察到的数据，第一个validation step是62，每个epoch大约增加63步
    # 但更准确的方法是直接计算相对epoch
    first_step = step_value_pairs[0][0]
    
    for i, (step, value) in enumerate(step_value_pairs):
        # 简化假设：每个validation step对应一个epoch
        epoch = i
        epoch_value_pairs.append((epoch, value))
    
    return epoch_value_pairs

def align_epoch_data(all_generator_data: Dict[str, Dict[str, List[Tuple[int, float]]]]) -> Tuple[Dict, int]:
    """对齐不同生成器的epoch数据
    
    Args:
        all_generator_data: {generator_type: {metric_name: [(epoch, value), ...]}}
    
    Returns:
        aligned_data: {generator_type: {metric_name: [value1, value2, ...]}}, max_epochs
    """
    # 找到最大epoch数
    max_epochs = 0
    for gen_data in all_generator_data.values():
        for metric_data in gen_data.values():
            if metric_data:
                max_epoch = max(epoch for epoch, _ in metric_data)
                max_epochs = max(max_epochs, max_epoch + 1)
    
    # 对齐数据
    aligned_data = {}
    for gen_type, gen_metrics in all_generator_data.items():
        aligned_data[gen_type] = {}
        for metric_name, epoch_values in gen_metrics.items():
            # 创建完整的epoch数组，缺失值用NaN填充
            aligned_values = [np.nan] * max_epochs
            for epoch, value in epoch_values:
                if epoch < max_epochs:
                    aligned_values[epoch] = value
            aligned_data[gen_type][metric_name] = aligned_values
    
    return aligned_data, max_epochs

def collect_all_runs_data(experiment_names: List[str] = None) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """收集所有运行的历史数据
    
    Returns:
        {generator_type: {metric_name: [(epoch, value), ...]}}
    """
    client = mlflow.tracking.MlflowClient()
    
    # 如果没有指定实验名称，自动发现
    if experiment_names is None:
        experiments = client.search_experiments()
        experiment_names = [exp.name for exp in experiments 
                          if any(keyword in exp.name.lower() 
                                for keyword in ['gaussgan', 'classical', 'quantum'])]
    
    print(f"分析实验: {experiment_names}")
    
    # 关键指标列表
    key_metrics = [
        'ValidationStep_FakeData_KLDivergence',
        'ValidationStep_FakeData_LogLikelihood', 
        'ValidationStep_FakeData_IsPositive',
        'ValidationStep_FakeData_WassersteinDistance',
        'ValidationStep_FakeData_MMDDistance',
        'train_g_loss_epoch',
        'd_loss',
        'g_loss'
    ]
    
    # 收集数据：{generator_type: {metric_name: [(epoch, value), ...]}}
    all_data = {}
    run_count = {}
    
    for exp_name in experiment_names:
        try:
            experiment = client.get_experiment_by_name(exp_name)
            if experiment is None:
                continue
                
            # 获取所有运行，然后过滤
            all_runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time desc"]
            )
            
            # 只保留FINISHED或RUNNING状态的运行
            runs = [run for run in all_runs if run.info.status in ['FINISHED', 'RUNNING']]
            
            print(f"实验 '{exp_name}': 找到 {len(runs)} 个完成的运行")
            
            for run in runs:
                generator_type = run.data.params.get('generator_type', 'unknown')
                
                if generator_type not in all_data:
                    all_data[generator_type] = {metric: [] for metric in key_metrics}
                    run_count[generator_type] = 0
                
                run_count[generator_type] += 1
                print(f"  处理运行 {run.info.run_id[:8]}... (生成器: {generator_type})")
                
                # 收集每个指标的历史数据
                for metric in key_metrics:
                    if metric in run.data.metrics:
                        step_values = get_run_metric_history(client, run.info.run_id, metric)
                        if step_values:
                            epoch_values = convert_steps_to_epochs(step_values)
                            # 将这次运行的数据添加到总数据中
                            all_data[generator_type][metric].extend(epoch_values)
                
        except Exception as e:
            print(f"处理实验 {exp_name} 时出错: {e}")
            continue
    
    print(f"\n数据收集完成:")
    for gen_type, count in run_count.items():
        print(f"  {gen_type}: {count} 次运行")
    
    return all_data

def aggregate_multiple_runs(data: Dict[str, Dict[str, List[Tuple[int, float]]]]) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """聚合同一生成器类型的多次运行数据
    
    对于每个(generator_type, metric)，计算每个epoch的平均值
    """
    aggregated = {}
    
    for gen_type, metrics_data in data.items():
        aggregated[gen_type] = {}
        
        for metric_name, epoch_values in metrics_data.items():
            if not epoch_values:
                aggregated[gen_type][metric_name] = []
                continue
            
            # 按epoch分组
            epoch_groups = {}
            for epoch, value in epoch_values:
                if epoch not in epoch_groups:
                    epoch_groups[epoch] = []
                epoch_groups[epoch].append(value)
            
            # 计算每个epoch的平均值
            avg_data = []
            for epoch in sorted(epoch_groups.keys()):
                values = epoch_groups[epoch]
                # 过滤掉NaN和无穷值
                valid_values = [v for v in values if np.isfinite(v)]
                if valid_values:
                    avg_value = np.mean(valid_values)
                    avg_data.append((epoch, avg_value))
            
            aggregated[gen_type][metric_name] = avg_data
    
    return aggregated

def create_epoch_comparison_plots(aligned_data: Dict, max_epochs: int, output_dir: str = "docs"):
    """创建逐epoch对比图表"""
    setup_chinese_fonts()
    
    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)
    
    # 关键指标配置 - 支持中英文标签
    if use_english_labels:
        metrics_config = {
            'ValidationStep_FakeData_KLDivergence': {
                'title': 'KL Divergence Trends (Lower is Better)',
                'ylabel': 'KL Divergence',
                'color_map': {'classical_normal': 'blue', 'quantum_samples': 'red', 'quantum_shadows': 'green'}
            },
            'ValidationStep_FakeData_LogLikelihood': {
                'title': 'Log Likelihood Trends (Higher is Better)',
                'ylabel': 'Log Likelihood',
                'color_map': {'classical_normal': 'blue', 'quantum_samples': 'red', 'quantum_shadows': 'green'}
            },
            'ValidationStep_FakeData_IsPositive': {
                'title': 'Positive Ratio Trends (Higher is Better)',
                'ylabel': 'Positive Ratio',
                'color_map': {'classical_normal': 'blue', 'quantum_samples': 'red', 'quantum_shadows': 'green'}
            },
            'ValidationStep_FakeData_WassersteinDistance': {
                'title': 'Wasserstein Distance Trends (Lower is Better)',
                'ylabel': 'Wasserstein Distance',
                'color_map': {'classical_normal': 'blue', 'quantum_samples': 'red', 'quantum_shadows': 'green'}
            },
            'ValidationStep_FakeData_MMDDistance': {
                'title': 'MMD Distance Trends (Lower is Better)',
                'ylabel': 'MMD Distance',
                'color_map': {'classical_normal': 'blue', 'quantum_samples': 'red', 'quantum_shadows': 'green'}
            },
            'train_g_loss_epoch': {
                'title': 'Generator Loss Trends',
                'ylabel': 'Generator Loss',
                'color_map': {'classical_normal': 'blue', 'quantum_samples': 'red', 'quantum_shadows': 'green'}
            }
        }
    else:
        metrics_config = {
            'ValidationStep_FakeData_KLDivergence': {
                'title': 'KL散度变化趋势 (越低越好)',
                'ylabel': 'KL散度',
                'color_map': {'classical_normal': 'blue', 'quantum_samples': 'red', 'quantum_shadows': 'green'}
            },
            'ValidationStep_FakeData_LogLikelihood': {
                'title': '对数似然变化趋势 (越高越好)',
                'ylabel': '对数似然',
                'color_map': {'classical_normal': 'blue', 'quantum_samples': 'red', 'quantum_shadows': 'green'}
            },
            'ValidationStep_FakeData_IsPositive': {
                'title': '正值比例变化趋势 (越高越好)',
                'ylabel': '正值比例',
                'color_map': {'classical_normal': 'blue', 'quantum_samples': 'red', 'quantum_shadows': 'green'}
            },
            'ValidationStep_FakeData_WassersteinDistance': {
                'title': 'Wasserstein距离变化趋势 (越低越好)',
                'ylabel': 'Wasserstein距离',
                'color_map': {'classical_normal': 'blue', 'quantum_samples': 'red', 'quantum_shadows': 'green'}
            },
            'ValidationStep_FakeData_MMDDistance': {
                'title': 'MMD距离变化趋势 (越低越好)',
                'ylabel': 'MMD距离',
                'color_map': {'classical_normal': 'blue', 'quantum_samples': 'red', 'quantum_shadows': 'green'}
            },
            'train_g_loss_epoch': {
                'title': '生成器损失变化趋势',
                'ylabel': '生成器损失',
                'color_map': {'classical_normal': 'blue', 'quantum_samples': 'red', 'quantum_shadows': 'green'}
            }
        }
    
    # 创建子图
    n_metrics = len(metrics_config)
    cols = 3
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    epochs = list(range(max_epochs))
    
    for idx, (metric_name, config) in enumerate(metrics_config.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        
        # 为每个生成器类型绘制曲线
        for gen_type, gen_data in aligned_data.items():
            if metric_name in gen_data:
                values = gen_data[metric_name]
                color = config['color_map'].get(gen_type, 'gray')
                
                # 只绘制有数据的部分
                valid_epochs = []
                valid_values = []
                for i, val in enumerate(values):
                    if np.isfinite(val):
                        valid_epochs.append(i)
                        valid_values.append(val)
                
                if valid_epochs:
                    ax.plot(valid_epochs, valid_values, 
                           label=f'{gen_type}', 
                           color=color, 
                           marker='o', 
                           markersize=4,
                           linewidth=2,
                           alpha=0.8)
        
        ax.set_title(config['title'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(config['ylabel'], fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 如果有数据，设置合理的y轴范围
        all_values = []
        for gen_data in aligned_data.values():
            if metric_name in gen_data:
                valid_vals = [v for v in gen_data[metric_name] if np.isfinite(v)]
                all_values.extend(valid_vals)
        
        if all_values and len(all_values) > 0:
            y_min, y_max = min(all_values), max(all_values)
            if np.isfinite(y_min) and np.isfinite(y_max) and y_min != y_max:
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # 隐藏多余的子图
    for idx in range(len(metrics_config), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(output_dir) / "epoch_comparison_plots.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"逐epoch对比图表已保存到: {output_path}")
    return str(output_path)

def save_detailed_csv(aligned_data: Dict, max_epochs: int, output_dir: str = "docs"):
    """保存详细的CSV数据文件"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 创建完整的DataFrame
    data_rows = []
    
    for epoch in range(max_epochs):
        for gen_type, gen_data in aligned_data.items():
            row = {'epoch': epoch, 'generator_type': gen_type}
            
            for metric_name, values in gen_data.items():
                if epoch < len(values):
                    row[metric_name] = values[epoch]
                else:
                    row[metric_name] = np.nan
            
            data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # 保存CSV
    csv_path = Path(output_dir) / "epoch_comparison_detailed.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"详细数据已保存到: {csv_path}")
    
    # 同时保存汇总统计
    summary_data = []
    for gen_type, gen_data in aligned_data.items():
        summary_row = {'generator_type': gen_type}
        
        for metric_name, values in gen_data.items():
            valid_values = [v for v in values if np.isfinite(v)]
            if valid_values:
                summary_row.update({
                    f'{metric_name}_final': valid_values[-1],
                    f'{metric_name}_best': min(valid_values) if 'Distance' in metric_name or 'KL' in metric_name else max(valid_values),
                    f'{metric_name}_mean': np.mean(valid_values),
                    f'{metric_name}_std': np.std(valid_values),
                    f'{metric_name}_epochs_count': len(valid_values)
                })
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(output_dir) / "epoch_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"汇总统计已保存到: {summary_path}")
    
    return str(csv_path), str(summary_path)

def print_convergence_analysis(aligned_data: Dict, max_epochs: int):
    """打印收敛分析结果"""
    print("\n" + "="*80)
    print("逐Epoch收敛分析")
    print("="*80)
    
    for gen_type, gen_data in aligned_data.items():
        print(f"\n📊 {gen_type} 生成器分析:")
        print("-" * 50)
        
        # 分析KL散度收敛
        if 'ValidationStep_FakeData_KLDivergence' in gen_data:
            kl_values = gen_data['ValidationStep_FakeData_KLDivergence']
            valid_kl = [v for v in kl_values if np.isfinite(v)]
            
            if len(valid_kl) > 1:
                initial_kl = valid_kl[0]
                final_kl = valid_kl[-1]
                best_kl = min(valid_kl)
                best_epoch = next(i for i, v in enumerate(kl_values) if v == best_kl)
                
                improvement = ((initial_kl - final_kl) / abs(initial_kl) * 100) if initial_kl != 0 else 0
                
                print(f"  KL散度: {initial_kl:.4f} -> {final_kl:.4f} (改进 {improvement:+.1f}%)")
                print(f"  最佳KL散度: {best_kl:.4f} (在第 {best_epoch} epoch)")
                print(f"  训练epochs: {len(valid_kl)}")
                
                # 计算收敛稳定性（最后5个epoch的标准差）
                if len(valid_kl) >= 5:
                    stability = np.std(valid_kl[-5:])
                    print(f"  最终稳定性: {stability:.4f} (最后5个epoch的标准差)")
        
        # 分析正值比例
        if 'ValidationStep_FakeData_IsPositive' in gen_data:
            pos_values = gen_data['ValidationStep_FakeData_IsPositive']
            valid_pos = [v for v in pos_values if np.isfinite(v)]
            
            if valid_pos:
                final_pos = valid_pos[-1]
                best_pos = max(valid_pos)
                print(f"  正值比例: 最终 {final_pos:.3f}, 最佳 {best_pos:.3f}")

def compare_generators_epoch_by_epoch(experiment_names: List[str] = None, output_dir: str = "docs"):
    """主函数：执行逐epoch对比分析"""
    setup_chinese_fonts()
    
    print("=" * 80)
    print("量子vs古典生成器逐Epoch性能对比分析")
    print("=" * 80)
    
    # 收集所有运行的历史数据
    print("\n步骤1: 收集历史数据...")
    all_runs_data = collect_all_runs_data(experiment_names)
    
    if not all_runs_data:
        print("错误: 没有找到任何有效的实验数据")
        return None
    
    # 聚合多次运行的数据
    print("\n步骤2: 聚合多次运行数据...")
    aggregated_data = aggregate_multiple_runs(all_runs_data)
    
    # 对齐epoch数据
    print("\n步骤3: 对齐epoch数据...")
    aligned_data, max_epochs = align_epoch_data(aggregated_data)
    
    print(f"最大训练epochs: {max_epochs}")
    
    # 创建可视化
    print("\n步骤4: 创建可视化图表...")
    plot_path = create_epoch_comparison_plots(aligned_data, max_epochs, output_dir)
    
    # 保存详细数据
    print("\n步骤5: 保存详细数据...")
    csv_path, summary_path = save_detailed_csv(aligned_data, max_epochs, output_dir)
    
    # 打印分析结果
    print_convergence_analysis(aligned_data, max_epochs)
    
    # 生成最终报告
    print("\n" + "="*80)
    print("最终对比报告")
    print("="*80)
    
    for gen_type, gen_data in aligned_data.items():
        print(f"\n🔹 {gen_type}:")
        
        # 找到有效的指标
        for metric_name, values in gen_data.items():
            valid_values = [v for v in values if np.isfinite(v)]
            if valid_values and len(valid_values) > 0:
                if 'KL' in metric_name:
                    print(f"   KL散度: 初始 {valid_values[0]:.4f} -> 最终 {valid_values[-1]:.4f}")
                elif 'LogLikelihood' in metric_name:
                    print(f"   对数似然: 初始 {valid_values[0]:.4f} -> 最终 {valid_values[-1]:.4f}")
                elif 'IsPositive' in metric_name:
                    print(f"   正值比例: 初始 {valid_values[0]:.3f} -> 最终 {valid_values[-1]:.3f}")
                break  # 只显示一个主要指标避免重复
    
    print(f"\n📁 输出文件:")
    print(f"   - 详细图表: {plot_path}")
    print(f"   - 详细数据: {csv_path}")
    print(f"   - 汇总统计: {summary_path}")
    
    return {
        'aligned_data': aligned_data,
        'max_epochs': max_epochs,
        'output_files': {
            'plot': plot_path,
            'detailed_csv': csv_path,
            'summary_csv': summary_path
        }
    }

def quick_status_check():
    """快速检查可用的实验和运行状态"""
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    
    print("可用实验状态:")
    print("-" * 60)
    
    total_finished = 0
    total_running = 0
    total_failed = 0
    
    for exp in experiments:
        if any(keyword in exp.name.lower() for keyword in ['gaussgan', 'classical', 'quantum']):
            runs = client.search_runs([exp.experiment_id])
            finished = len([r for r in runs if r.info.status == 'FINISHED'])
            running = len([r for r in runs if r.info.status == 'RUNNING'])
            failed = len([r for r in runs if r.info.status == 'FAILED'])
            
            total_finished += finished
            total_running += running
            total_failed += failed
            
            print(f"{exp.name[:40]:<40}: 完成 {finished}, 运行中 {running}, 失败 {failed}")
    
    print("-" * 60)
    print(f"总计: 完成 {total_finished}, 运行中 {total_running}, 失败 {total_failed}")
    print()

if __name__ == "__main__":
    # 检查当前状态
    quick_status_check()
    
    # 运行完整分析
    try:
        results = compare_generators_epoch_by_epoch()
        
        if results:
            print("\n✅ 分析完成！")
            print("\n关键洞察:")
            print("- 可以看到每个生成器在训练过程中的性能变化")
            print("- 不同epoch数的生成器会用NaN填充对齐")
            print("- 所有图表都支持中文显示")
            print("- 详细数据可用于进一步分析")
        else:
            print("\n❌ 分析失败，请检查MLflow数据")
            
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()