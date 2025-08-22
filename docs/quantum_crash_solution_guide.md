# GaussGAN Quantum_Samples 系统崩溃解决方案

## 🚨 崩溃原因总结

### 主要问题
1. **内存爆炸**: `batch_size=256` + 量子电路 = 系统级OOM
2. **设备泄漏**: 每个QuantumNoise实例创建独立量子设备
3. **线程竞争**: 非线程安全的随机数生成
4. **CUDA碎片化**: PennyLane与大批量CUDA内存管理冲突

### 崩溃证据
- MLflow日志显示quantum_samples运行仅完成1个epoch
- 系统内存充足(15GB)但GPU内存有限(8GB RTX 4060)
- 无系统级错误日志，说明是软件层面内存问题

## ✅ 立即可用的解决方案

### 方案1: 紧急配置修复 (最快速)

**使用安全配置文件:**
```bash
# 复制安全配置
cp /home/paperx/quantum/GaussGAN/docs/config_quantum_safe.yaml /home/paperx/quantum/GaussGAN/config_quantum_safe.yaml

# 使用安全配置运行
uv run python main.py --generator_type quantum_samples --max_epochs 20 --config config_quantum_safe.yaml
```

**关键参数调整:**
- `batch_size: 16` (从256降低16倍!)
- `quantum_qubits: 4` (从6降低，减少75%量子态空间)
- `quantum_shots: 50` (从100降低)
- `validation_samples: 200` (从500降低)

### 方案2: 代码级修复 (更彻底)

**修复QuantumNoise类:**
```python
# 使用修复后的量子噪声类
from docs.quantum_noise_fixed import QuantumNoiseFixed

# 在main.py中替换
if cfg['generator_type'] == 'quantum_samples':
    noise_generator = QuantumNoiseFixed(
        num_qubits=cfg['quantum_qubits'], 
        num_layers=cfg['quantum_layers']
    )
```

### 方案3: 环境兼容性检查

**运行兼容性检查:**
```bash
uv run python /home/paperx/quantum/GaussGAN/docs/check_pennylane_cuda_compatibility.py
```

这将检查:
- PyTorch CUDA环境
- PennyLane设备兼容性  
- 内存使用模式
- 批处理稳定性

## 🛡️ 预防措施

### 1. 内存监控
```python
def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU内存: 已分配={allocated:.2f}GB, 缓存={cached:.2f}GB")
        
        if allocated > 6.0:  # RTX 4060 8GB的75%
            torch.cuda.empty_cache()
            return True
    return False
```

### 2. 渐进式批大小测试
```bash
# 测试序列: 从小到大
for batch_size in 8 16 32 64; do
    echo "测试批大小: $batch_size"
    timeout 300 uv run python main.py --generator_type quantum_samples --max_epochs 5 --batch_size $batch_size
    if [ $? -eq 0 ]; then
        echo "批大小 $batch_size 成功"
    else
        echo "批大小 $batch_size 失败"
        break
    fi
done
```

### 3. 量子参数调优策略
```yaml
# 保守配置 (4GB GPU)
quantum_qubits: 3
quantum_layers: 1
batch_size: 8

# 中等配置 (8GB GPU) - 推荐
quantum_qubits: 4  
quantum_layers: 2
batch_size: 16

# 高性能配置 (16GB+ GPU)
quantum_qubits: 6
quantum_layers: 3
batch_size: 32
```

## 🔍 故障排除流程

### Step 1: 环境检查
```bash
# 检查GPU状态
nvidia-smi
free -h

# 检查Python环境
uv pip list | grep -E "(torch|pennylane)"
```

### Step 2: 最小化测试
```bash
# 使用最小配置测试
uv run python -c "
from source.nn import QuantumNoise
import torch
qn = QuantumNoise(num_qubits=2, num_layers=1)
print('创建成功')
samples = qn(4)  # 小批量
print(f'样本形状: {samples.shape}')
"
```

### Step 3: 逐步增加复杂度
1. 先测试 `batch_size=4, qubits=2`
2. 然后测试 `batch_size=8, qubits=3` 
3. 最后测试 `batch_size=16, qubits=4`

### Step 4: 监控资源使用
```bash
# 运行时监控
watch -n 1 'nvidia-smi; echo "---"; free -h'
```

## ⚠️ 已知限制和建议

### 硬件限制 (RTX 4060 8GB)
- **建议最大batch_size**: 16-32
- **建议最大qubits**: 4-5
- **建议最大layers**: 2-3

### PennyLane 0.42.2 注意事项
- `default.qubit`设备在大批量时内存效率较低
- 考虑升级到更新版本或使用Lightning
- 避免频繁创建/销毁量子设备

### 监控和调试
```python
# 在训练循环中添加
import psutil
import GPUtil

def log_resources():
    # CPU和内存
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    
    # GPU (如果可用)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"资源使用: CPU={cpu_percent}%, RAM={memory_info.percent}%, GPU_MEM={gpu_memory:.2f}GB")
```

## 📋 测试清单

在正式运行前，请确认:

- [ ] 使用安全配置文件 (`config_quantum_safe.yaml`)
- [ ] batch_size <= 32
- [ ] quantum_qubits <= 4  
- [ ] 运行了兼容性检查脚本
- [ ] 监控了前几个epoch的内存使用
- [ ] 准备了终止命令 (`Ctrl+C`)
- [ ] 系统有足够的虚拟内存 (4GB+)

## 🆘 紧急处理

如果再次发生崩溃:

1. **立即终止**: `Ctrl+C` 或 `kill -9 <pid>`
2. **清理GPU内存**: `nvidia-smi --gpu-reset` (如果需要)
3. **检查僵尸进程**: `ps aux | grep python`
4. **重启Python环境**: 关闭所有Python进程
5. **使用更保守的配置**: batch_size=8, qubits=3

## 📞 技术支持

如果问题持续存在:
1. 收集完整的错误日志
2. 运行兼容性检查脚本
3. 记录系统配置和资源使用情况
4. 考虑使用Classical generators作为备选方案

---

**最后更新**: 2025-08-22  
**测试环境**: WSL2 Ubuntu + RTX 4060 8GB  
**PennyLane版本**: 0.42.2  
**PyTorch版本**: 2.0+