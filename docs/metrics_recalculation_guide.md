# 指标重新计算系统使用指南

## 概述

这个后处理系统可以从已保存的CSV文件中重新计算所有训练指标，而不需要在训练时进行复杂的指标计算。

### 优势

- **训练更快**：训练时不需要计算复杂指标（如KL散度、MMD等）
- **可扩展性**：随时添加新指标而无需重新训练
- **灵活性**：可以尝试不同的指标计算方法和参数
- **容错性**：即使训练中断，已保存的CSV仍可分析
- **重现性**：可以重复计算指标进行验证

## 快速开始

### 1. 基本用法 - 处理单个实验

```bash
# 使用快速模式处理单个实验
uv run python recalculate_metrics.py -r "mlruns/248720252569581412/7b3733330145425fb59df88f00376f45" --fast

# 只计算特定指标
uv run python recalculate_metrics.py -r "mlruns/248720252569581412/7b3733330145425fb59df88f00376f45" \
    --metrics IsPositive LogLikelihood WassersteinDistance --fast

# 计算所有指标（包括慢的KL散度）
uv run python recalculate_metrics.py -r "mlruns/248720252569581412/7b3733330145425fb59df88f00376f45"
```

### 2. 批量处理所有实验

```bash
# 快速处理所有实验
uv run python recalculate_metrics.py --fast -o "results/batch_processing"

# 只生成汇总报告（跳过单个处理）
uv run python recalculate_metrics.py --summary_only -o "results/batch_processing"
```

### 3. 使用示例脚本

```bash
cd docs
uv run python quick_metrics_example.py
```

## 可用指标

| 指标名称 | 描述 | 计算速度 | 建议使用 |
|---------|------|----------|----------|
| `IsPositive` | 简单位置指标(x>0 vs x<0) | ⚡ 很快 | 总是启用 |
| `LogLikelihood` | GMM对数似然 | 🚀 快 | 推荐 |
| `WassersteinDistance` | 地球移动距离 | 🚀 快 | 推荐 |
| `MMDDistance` | 最大均值差异 | ⏳ 中等 | 快速模式推荐 |
| `MMDivergenceFromGMM` | MMD (基于GMM生成) | ⏳ 中等 | 可选 |
| `KLDivergence` | KL散度 (KDE+GMM) | 🐌 很慢 | 仅在必要时使用 |

## 命令行参数详解

### 基本参数

- `-r, --run_path`: 指定单个MLflow运行目录路径
- `-m, --mlruns_dir`: MLflow运行目录（默认：mlruns）
- `-o, --output_dir`: 输出目录（默认：docs/recalculated_metrics）
- `-t, --target_data`: 目标分布数据文件（默认：data/normal.pickle）

### 性能优化

- `-f, --fast`: 快速模式，使用较少样本和跳过慢计算
- `--metrics`: 指定计算的指标列表

### 批量处理

- `-s, --summary_only`: 只生成汇总报告（跳过单个处理）

## 输出文件说明

### 单个实验结果

```csv
epoch,n_samples,samples_file,IsPositive,LogLikelihood,WassersteinDistance
0,500,gaussian_generated_epoch_0000.csv,-1.0,-28.291748046875,0.010086127556860447
1,500,gaussian_generated_epoch_0001.csv,-1.0,-33.72111511230469,0.011064448393881321
...
```

包含列：
- `epoch`: epoch编号
- `n_samples`: 该epoch的样本数量
- `samples_file`: 对应的CSV文件名
- 其他列为各指标的值

### 汇总报告

```csv
run_id,total_epochs,avg_samples_per_epoch,LogLikelihood_final,LogLikelihood_best,LogLikelihood_mean,LogLikelihood_std,...
```

包含每个运行的：
- 基本信息（总epoch数、平均样本数等）
- 每个指标的最终值、最佳值、均值、标准差

## 性能建议

### 快速分析（推荐用于日常分析）

```bash
uv run python recalculate_metrics.py -r "path/to/run" --fast \
    --metrics IsPositive LogLikelihood WassersteinDistance MMDDistance
```

- 使用 `--fast` 模式
- 选择核心指标
- 跳过KL散度计算

### 完整分析（用于最终报告）

```bash
uv run python recalculate_metrics.py -r "path/to/run" \
    --metrics IsPositive LogLikelihood KLDivergence WassersteinDistance MMDDistance MMDivergenceFromGMM
```

- 包含所有指标
- 不使用快速模式以获得最高精度

## 高级用法

### 1. 自定义配置

编辑 `docs/metrics_config.yaml` 来自定义指标设置：

```yaml
metrics:
  - "IsPositive"
  - "LogLikelihood" 
  - "WassersteinDistance"
  
fast_mode: true

metric_settings:
  WassersteinDistance:
    max_target_samples: 5000
    aggregation: "mean"
```

### 2. 编程接口

```python
from recalculate_metrics import MetricsRecalculator

# 初始化
recalculator = MetricsRecalculator("data/normal.pickle")

# 处理单个实验
results_df = recalculator.process_experiment(
    "mlruns/248720252569581412/7b3733330145425fb59df88f00376f45",
    selected_metrics=["IsPositive", "LogLikelihood"],
    fast_mode=True
)

# 处理所有实验
all_results = recalculator.process_all_experiments("mlruns", "output")
```

### 3. 数据分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# 加载结果
df = pd.read_csv("docs/recalculated_metrics/metrics_runid.csv")

# 绘制指标趋势
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(df['epoch'], df['LogLikelihood'])
plt.title('Log Likelihood over Epochs')

plt.subplot(2, 2, 2)
plt.plot(df['epoch'], df['WassersteinDistance'])
plt.title('Wasserstein Distance over Epochs')

plt.show()
```

## 故障排除

### 常见问题

1. **"No CSV files found"**
   - 检查MLflow运行是否包含artifacts目录
   - 确认CSV文件命名格式正确（`gaussian_generated_epoch_*.csv`）

2. **"Target distribution load failed"**
   - 确认目标数据文件路径正确
   - 检查pickle文件是否损坏

3. **计算过慢**
   - 使用 `--fast` 模式
   - 跳过KL散度计算
   - 选择特定指标

4. **内存不足**
   - 使用快速模式减少目标样本数量
   - 分批处理大量实验

### 性能优化建议

1. **对于大型数据集**：
   - 使用 `--fast` 模式
   - 避免KL散度
   - 减少目标样本数量

2. **对于批量处理**：
   - 先处理几个实验测试配置
   - 使用并行处理（待实现）

3. **对于精确分析**：
   - 不使用快速模式
   - 包含所有指标
   - 增加目标样本数量

## 扩展指南

### 添加新指标

1. 在 `source/metrics.py` 中实现新的指标类
2. 在 `recalculate_metrics.py` 的 `_initialize_metrics` 方法中添加
3. 更新文档和配置文件

### 自定义目标分布

1. 修改 `_load_target_distribution` 方法
2. 支持不同的数据格式
3. 添加分布参数验证

## 示例工作流程

### 1. 训练后快速分析

```bash
# 1. 快速检查最新训练结果
uv run python recalculate_metrics.py -r "mlruns/latest_run" --fast \
    --metrics IsPositive LogLikelihood WassersteinDistance

# 2. 查看结果
head docs/recalculated_metrics/metrics_latest_run.csv
```

### 2. 完整的实验分析

```bash
# 1. 处理所有实验
uv run python recalculate_metrics.py --fast -o "analysis/complete"

# 2. 生成汇总报告
uv run python recalculate_metrics.py --summary_only -o "analysis/complete"

# 3. 分析最佳模型
grep "LogLikelihood_best" analysis/complete/summary_report.csv | sort -k2 -n
```

### 3. 深度分析特定模型

```bash
# 1. 选择最佳运行
best_run="mlruns/experiment/best_run_id"

# 2. 计算所有指标
uv run python recalculate_metrics.py -r "$best_run" --metrics \
    IsPositive LogLikelihood KLDivergence WassersteinDistance MMDDistance

# 3. 生成可视化（需要自定义脚本）
python analysis/visualize_metrics.py docs/recalculated_metrics/metrics_best_run_id.csv
```

这个系统提供了强大且灵活的指标重新计算能力，可以显著提升训练效率和实验分析的灵活性！