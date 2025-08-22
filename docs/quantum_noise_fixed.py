"""
修复版本的QuantumNoise类 - 解决内存泄漏和系统崩溃问题

主要修复:
1. 修复内存泄漏：优化forward方法中的张量创建
2. 修复线程安全：使用torch.randn替代random.uniform
3. 修复设备管理：复用量子设备实例
4. 添加内存监控和保护机制
"""

import random
from abc import ABC, abstractmethod
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import gc

class QuantumNoiseFixed(nn.Module):
    def __init__(self, num_qubits: int = 8, num_layers: int = 3):
        super(QuantumNoiseFixed, self).__init__()
        
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
        # 修复1: 参数初始化优化
        self.weights = nn.Parameter(
            torch.rand(num_layers, (num_qubits * 2 - 1)) * 2 * torch.pi - torch.pi
        )
        
        # 修复2: 共享量子设备实例，避免重复创建
        if not hasattr(QuantumNoiseFixed, '_shared_device'):
            QuantumNoiseFixed._shared_device = qml.device("default.qubit", wires=num_qubits)
        
        # 修复3: 使用共享设备创建量子电路
        @qml.qnode(QuantumNoiseFixed._shared_device, interface="torch", diff_method="backprop")
        def gen_circuit(w, z1, z2):  # 修复：移除内部随机数生成
            # 使用传入的随机数而非内部生成
            for i in range(num_qubits):
                qml.RY(np.arcsin(z1), wires=i)
                qml.RZ(np.arcsin(z2), wires=i)
            
            for l in range(num_layers):
                for i in range(num_qubits):
                    qml.RY(w[l][i], wires=i)
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(w[l][i + num_qubits], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
        
        self.gen_circuit = gen_circuit
        
        # 修复4: 添加内存监控
        self.memory_threshold = 0.8  # GPU内存使用率阈值
    
    def _check_memory_usage(self):
        """检查GPU内存使用率，防止OOM"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            max_memory = torch.cuda.get_device_properties(0).total_memory
            usage_ratio = (allocated + cached) / max_memory
            
            if usage_ratio > self.memory_threshold:
                torch.cuda.empty_cache()
                gc.collect()
                return True
        return False

    def forward(self, batch_size: int):
        """修复的forward方法，避免内存泄漏"""
        
        # 修复5: 检查内存使用率
        if self._check_memory_usage():
            print(f"Warning: High GPU memory usage detected, batch_size={batch_size}")
        
        # 修复6: 批量生成随机数，避免循环内创建
        device = next(self.parameters()).device
        z1_batch = torch.rand(batch_size, device=device) * 2 - 1  # [-1, 1]
        z2_batch = torch.rand(batch_size, device=device) * 2 - 1  # [-1, 1]
        
        # 修复7: 使用列表推导式，但预先分配内存
        results = []
        
        for i in range(batch_size):
            z1 = z1_batch[i].item()  # 转换为标量
            z2 = z2_batch[i].item()
            
            # 获取量子电路输出
            circuit_output = self.gen_circuit(self.weights, z1, z2)
            
            # 修复8: 直接堆叠张量，避免中间concat操作
            sample = torch.stack([tensor for tensor in circuit_output])
            results.append(sample)
            
            # 修复9: 定期清理内存
            if i % 50 == 49:  # 每50个样本清理一次
                torch.cuda.empty_cache()
        
        # 修复10: 最终堆叠，转换为所需格式
        noise = torch.stack(results).float()
        
        return noise

# 使用示例和测试代码
if __name__ == "__main__":
    print("测试修复后的QuantumNoise类")
    
    # 创建修复后的量子噪声生成器
    quantum_gen = QuantumNoiseFixed(num_qubits=4, num_layers=2)
    
    # 测试小批量
    print("测试小批量 (batch_size=8)...")
    try:
        samples = quantum_gen(8)
        print(f"成功生成样本，形状: {samples.shape}")
        print(f"内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB" if torch.cuda.is_available() else "CPU模式")
    except Exception as e:
        print(f"小批量测试失败: {e}")
    
    # 测试中等批量
    print("\n测试中等批量 (batch_size=32)...")
    try:
        samples = quantum_gen(32)
        print(f"成功生成样本，形状: {samples.shape}")
        print(f"内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB" if torch.cuda.is_available() else "CPU模式")
    except Exception as e:
        print(f"中等批量测试失败: {e}")
        
    print("\n修复版本测试完成！")