# 如何使用重构版compare_generators.py

## 快速开始

### 1. 基本使用
```bash
# 运行完整的逐epoch对比分析
uv run python compare_generators.py
```

### 2. 输出文件
分析完成后会在`docs/`目录下生成：
- `epoch_comparison_plots.png` - 多指标对比图表
- `epoch_comparison_detailed.csv` - 详细的逐epoch数据
- `epoch_comparison_summary.csv` - 汇总统计信息

## 主要功能

### 1. 逐Epoch历史追踪
脚本会自动：
- 从MLflow获取所有实验的完整历史数据
- 将step单位转换为epoch单位
- 聚合同一生成器类型的多次运行
- 对齐不同长度的训练历史

### 2. 多指标可视化
创建6个子图展示：
- KL Divergence Trends (KL散度变化)
- Log Likelihood Trends (对数似然变化)
- Positive Ratio Trends (正值比例变化)
- Wasserstein Distance Trends (Wasserstein距离变化)
- MMD Distance Trends (MMD距离变化)
- Generator Loss Trends (生成器损失变化)

### 3. 统计分析
自动计算：
- 初始值 vs 最终值
- 最佳值及其对应的epoch
- 改进率百分比
- 训练稳定性（标准差）
- 有效数据点数量

## 高级使用

### 1. 指定特定实验
```python
from compare_generators import compare_generators_epoch_by_epoch

# 只分析特定实验
results = compare_generators_epoch_by_epoch(['GaussGAN-manual'])

# 分析多个实验
results = compare_generators_epoch_by_epoch([
    'GaussGAN-manual', 
    'WSL_Quantum_vs_Classical'
])
```

### 2. 自定义输出目录
```python
results = compare_generators_epoch_by_epoch(output_dir='custom_analysis')
```

### 3. 测试功能
```bash
# 运行测试验证各功能模块
uv run python docs/test_epoch_comparison.py
```

## 数据解读

### 1. KL散度分析
- **值越低越好**
- 显示生成分布与目标分布的差异
- 理想情况下应该随训练而降低

### 2. 对数似然分析
- **值越高越好**
- 反映生成样本的质量
- 通常随训练而提高

### 3. 正值比例分析
- **值越高越好**（对于特定应用）
- 显示生成样本在正半轴的比例
- 与"killer"功能相关

### 4. 距离指标分析
- **Wasserstein距离**: 分布间的最优传输距离
- **MMD距离**: 最大均值差异
- 两者都是越低越好

## 常见问题解决

### 1. 无数据或数据不足
**问题**: 显示"没有找到任何有效的实验数据"
**解决**: 
- 确保有完成状态的MLflow实验
- 检查实验名称是否包含'gaussgan', 'classical', 'quantum'关键词

### 2. 图表显示异常
**问题**: 图表空白或标签显示为方框
**解决**:
- 脚本已自动切换到英文标签
- 如需中文，修改`use_english_labels = False`

### 3. epoch数不匹配
**问题**: 不同生成器的epoch数差异很大
**解决**:
- 脚本自动用NaN填充较短的序列
- 在分析时只考虑有效数据点

### 4. 指标缺失
**问题**: 某些指标没有数据
**解决**:
- 检查实验是否启用了相应的metrics配置
- 某些指标（如Wasserstein、MMD）可能不是每次都记录

## 性能优化

### 1. 数据处理
- 使用pandas进行高效数据操作
- numpy处理数值计算
- 内存友好的数据结构

### 2. 可视化
- matplotlib子图并行渲染
- seaborn美化样式
- 高分辨率输出

## 扩展功能

### 1. 添加新指标
在`key_metrics`列表中添加新的MLflow指标名称：
```python
key_metrics = [
    'ValidationStep_FakeData_KLDivergence',
    'ValidationStep_FakeData_LogLikelihood', 
    # 添加新指标
    'your_new_metric_name'
]
```

### 2. 自定义图表样式
修改`metrics_config`字典：
```python
'your_metric_name': {
    'title': 'Your Metric Trends',
    'ylabel': 'Your Metric',
    'color_map': {'gen1': 'blue', 'gen2': 'red'}
}
```

### 3. 添加新的分析功能
可以扩展以下函数：
- `print_convergence_analysis()` - 添加新的统计分析
- `save_detailed_csv()` - 添加新的数据导出格式
- `create_epoch_comparison_plots()` - 添加新的可视化类型

## 文件结构

```
docs/
├── epoch_comparison_plots.png      # 主要可视化图表
├── epoch_comparison_detailed.csv   # 详细逐epoch数据
├── epoch_comparison_summary.csv    # 汇总统计
├── test_epoch_comparison.py        # 功能测试脚本
└── epoch_comparison_analysis_report.md  # 分析报告
```

## 使用建议

### 1. 定期运行
建议在每次重要实验完成后运行分析，跟踪性能变化

### 2. 结果备份
重要的分析结果建议备份，因为重新运行会覆盖之前的文件

### 3. 数据验证
使用测试脚本验证数据完整性：
```bash
uv run python docs/test_epoch_comparison.py
```

### 4. 自定义分析
利用生成的CSV文件进行自定义分析和可视化

## 注意事项

1. **数据依赖**: 需要完成状态的MLflow实验数据
2. **计算时间**: 大量运行数据可能需要几分钟处理时间
3. **内存使用**: 大规模数据可能占用较多内存
4. **字体支持**: 当前使用英文标签确保兼容性

## 未来改进方向

1. **实时监控**: 支持正在运行的实验的实时可视化
2. **交互式图表**: 使用plotly等工具创建交互式可视化
3. **统计检验**: 添加显著性检验和置信区间
4. **自动化报告**: 自动生成格式化的分析报告