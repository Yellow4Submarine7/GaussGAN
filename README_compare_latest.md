# 🔄 自动化运行对比工具

## 快速开始

```bash
uv run python compare_latest.py
```

一行命令完成所有操作！

## 功能

✅ **自动发现最新运行**
- 量子运行：`quantum_samples`, `quantum_shadows`  
- 经典运行：`classical_normal`, `classical_uniform`

✅ **完整指标计算**
- KL散度 (KL Divergence)
- Wasserstein距离
- 最大均值散度 (MMD)
- 对数似然 (Log Likelihood)

✅ **6子图可视化**
- 训练指标对比
- 损失函数曲线
- 专业图表布局

✅ **数据导出**
- PNG高分辨率图表
- CSV详细数据文件

## 输出示例

```
🔍 GaussGAN Latest Runs Comparison
==================================================
✅ Found quantum run: quantum_samples (4 epochs)
✅ Found classical run: classical_normal (30 epochs)
✅ Comparison plot saved as: latest_comparison_20250822_235724.png
📄 Detailed data saved as: latest_comparison_data_20250822_235724.csv
```

## 核心特性

- **零配置**：无需指定运行ID或参数
- **智能处理**：自动处理不同epoch数量
- **完全兼容**：使用项目自带的metrics.py
- **容错设计**：优雅处理异常情况

就这么简单！🚀