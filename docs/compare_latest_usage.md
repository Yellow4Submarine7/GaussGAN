# Compare Latest Runs - 快速使用指南

## 概述

`compare_latest.py` 是一个自动化脚本，用于对比最新的量子GAN和经典GAN训练运行。

## 功能特点

- **自动发现**：自动找到最新的量子运行和经典运行
- **完整指标**：计算KL散度、Wasserstein距离、MMD、对数似然
- **可视化对比**：生成6个子图的对比图表
- **数据导出**：保存详细数据到CSV文件

## 使用方法

### 一键运行
```bash
uv run python compare_latest.py
```

就这么简单！脚本会自动：
1. 搜索最新的量子运行（quantum_samples 或 quantum_shadows）
2. 搜索最新的经典运行（classical_normal 或 classical_uniform）
3. 加载训练数据并计算指标
4. 生成6个子图的对比图表
5. 保存图片和数据文件

### 输出文件

脚本会生成两个文件：
- `latest_comparison_YYYYMMDD_HHMMSS.png` - 对比图表
- `latest_comparison_data_YYYYMMDD_HHMMSS.csv` - 详细数据

### 示例输出

```
🔍 GaussGAN Latest Runs Comparison
==================================================
1. Finding latest quantum and classical runs...
✅ Found quantum run: quantum_samples (4 epochs)
   Run ID: 76fbda7f
✅ Found classical run: classical_normal (30 epochs)  
   Run ID: f8d77c9c

2. Loading training data and computing metrics...
Loaded target data: (20000, 2) samples
Loading data from quantum_samples run: 76fbda7f...
Processed 4 epochs for quantum_samples
Loading data from classical_normal run: f8d77c9c...
Processed 30 epochs for classical_normal

3. Loading loss data from MLflow...
4. Generating comparison visualization...
✅ Comparison plot saved as: latest_comparison_20250822_235540.png

📊 Summary Statistics:
------------------------------
Quantum Run:
  - Type: quantum_samples
  - Epochs: 4
  - Final KL Divergence: -0.0381

Classical Run:
  - Type: classical_normal
  - Epochs: 30
  - Final KL Divergence: -0.0365

📄 Detailed data saved as: latest_comparison_data_20250822_235540.csv
```

## 生成的图表

图表包含6个子图：
1. **KL Divergence** - KL散度随训练进展
2. **Wasserstein Distance** - Wasserstein距离对比
3. **MMD** - 最大均值散度对比
4. **Generator Loss** - 生成器损失
5. **Discriminator Loss** - 判别器损失  
6. **Log Likelihood** - 对数似然

## 核心优势

### 全自动化
- 无需手动指定运行ID
- 自动处理不同epoch数量
- 智能错误处理

### 完整指标
- 使用项目自带的 `source/metrics.py`
- 支持所有主要GAN评估指标
- 与训练代码完全兼容

### 清晰可视化
- 专业的多子图布局
- 颜色编码区分量子/经典
- 包含运行类型和ID信息

## 技术细节

### 自动发现逻辑
- 扫描 `mlruns/` 目录下所有实验
- 按时间排序找到最新运行
- 支持所有生成器类型：
  - 量子：`quantum_samples`, `quantum_shadows`
  - 经典：`classical_normal`, `classical_uniform`

### 指标计算
- 从保存的CSV样本文件重新计算指标
- 使用与训练相同的target distribution
- 支持不同样本数量和epoch长度

### 容错设计
- 优雅处理缺失数据
- 自动跳过损坏的epoch文件
- 提供有意义的错误消息

## 依赖要求

脚本需要以下Python包（已在项目环境中）：
- torch
- numpy
- pandas  
- matplotlib
- seaborn
- pyyaml
- scikit-learn
- scipy

## 故障排除

### 常见问题

1. **找不到运行**
   - 确保 `mlruns/` 目录存在
   - 检查是否有有效的训练运行

2. **指标计算失败**
   - 检查CSV文件格式是否正确
   - 确保target data文件存在

3. **图表显示问题**
   - 脚本会自动保存PNG文件
   - 在headless环境中不会尝试显示图表

### 调试模式

如果遇到问题，可以查看详细的错误堆栈：
```bash
uv run python compare_latest.py 2>&1 | tee debug.log
```

## 扩展功能

脚本设计为模块化，可以轻松扩展：
- 添加新的指标类型
- 修改可视化样式
- 支持更多生成器类型
- 增加统计分析功能