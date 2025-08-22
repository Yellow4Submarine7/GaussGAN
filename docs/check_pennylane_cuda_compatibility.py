"""
PennyLane与CUDA兼容性检查脚本
检查量子模拟库与GPU环境的兼容性问题
"""

import torch
import pennylane as qml
import numpy as np
import sys
import gc
import traceback

def check_torch_cuda():
    """检查PyTorch CUDA环境"""
    print("=" * 50)
    print("PyTorch CUDA 环境检查")
    print("=" * 50)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name()}")
        
        # 内存信息
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        
        print(f"GPU总内存: {total_memory:.2f} GB")
        print(f"已分配内存: {allocated:.2f} GB")
        print(f"缓存内存: {cached:.2f} GB")
        print(f"可用内存: {total_memory - allocated - cached:.2f} GB")

def check_pennylane_info():
    """检查PennyLane信息"""
    print("\n" + "=" * 50)
    print("PennyLane 环境检查")
    print("=" * 50)
    
    print(f"PennyLane版本: {qml.__version__}")
    
    # 检查可用设备
    print("\n可用的量子设备:")
    try:
        available_devices = qml.device._get_device_entrypoints()
        for device_name in available_devices:
            print(f"  - {device_name}")
    except Exception as e:
        print(f"  无法获取设备列表: {e}")

def test_basic_quantum_circuit():
    """测试基本量子电路"""
    print("\n" + "=" * 50)
    print("基本量子电路测试")
    print("=" * 50)
    
    try:
        # 创建小型量子设备
        dev = qml.device("default.qubit", wires=2)
        
        @qml.qnode(dev, interface="torch")
        def simple_circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        # 测试单个调用
        x = torch.tensor(0.5, requires_grad=True)
        result = simple_circuit(x)
        print(f"单次电路调用成功: {result}")
        
        # 测试梯度计算
        loss = result ** 2
        loss.backward()
        print(f"梯度计算成功: {x.grad}")
        
    except Exception as e:
        print(f"基本量子电路测试失败: {e}")
        traceback.print_exc()

def test_batch_quantum_processing():
    """测试批处理量子计算"""
    print("\n" + "=" * 50)
    print("批处理量子计算测试")
    print("=" * 50)
    
    batch_sizes = [1, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        print(f"\n测试批大小: {batch_size}")
        
        try:
            # 创建量子设备
            dev = qml.device("default.qubit", wires=3)
            
            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def batch_circuit(params):
                qml.RY(params[0], wires=0)
                qml.RY(params[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RZ(params[2], wires=1)
                return [qml.expval(qml.PauliZ(i)) for i in range(3)]
            
            # 准备参数
            params = torch.rand(3, requires_grad=True)
            
            # 模拟批处理（类似QuantumNoise的做法）
            results = []
            for i in range(batch_size):
                # 添加少量随机性
                noise = torch.randn(3) * 0.1
                circuit_params = params + noise
                
                circuit_output = batch_circuit(circuit_params)
                result = torch.stack([tensor for tensor in circuit_output])
                results.append(result)
                
                # 检查内存使用
                if torch.cuda.is_available() and i % 8 == 7:
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    print(f"  步骤 {i+1}/{batch_size}: GPU内存 {allocated:.1f} MB")
            
            # 最终堆叠
            batch_output = torch.stack(results)
            print(f"  批处理成功! 输出形状: {batch_output.shape}")
            
            # 清理内存
            del results, batch_output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        except Exception as e:
            print(f"  批大小 {batch_size} 失败: {e}")
            # 尝试恢复
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

def test_memory_stress():
    """内存压力测试"""
    print("\n" + "=" * 50)
    print("内存压力测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("跳过GPU内存压力测试（无CUDA支持）")
        return
    
    try:
        # 记录初始内存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"初始GPU内存: {initial_memory:.1f} MB")
        
        # 创建大量量子设备（模拟原问题）
        devices = []
        circuits = []
        
        for i in range(10):  # 创建10个设备
            dev = qml.device("default.qubit", wires=6)  # 使用原配置
            devices.append(dev)
            
            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(w):
                for j in range(6):
                    qml.RY(w[j], wires=j)
                for j in range(5):
                    qml.CNOT(wires=[j, j+1])
                return [qml.expval(qml.PauliZ(k)) for k in range(6)]
            
            circuits.append(circuit)
            
            current_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"设备 {i+1}: GPU内存 {current_memory:.1f} MB (+{current_memory-initial_memory:.1f})")
            
            if current_memory > 1000:  # 如果超过1GB就停止
                print("内存使用过高，停止测试")
                break
        
        print(f"创建了 {len(devices)} 个量子设备")
        
    except Exception as e:
        print(f"内存压力测试失败: {e}")
    finally:
        # 清理
        torch.cuda.empty_cache()
        gc.collect()

def check_pennylane_lightning():
    """检查PennyLane Lightning加速器"""
    print("\n" + "=" * 50)
    print("PennyLane Lightning 检查")
    print("=" * 50)
    
    try:
        # 尝试使用Lightning设备
        dev_lightning = qml.device("lightning.qubit", wires=4)
        
        @qml.qnode(dev_lightning, interface="torch")
        def lightning_circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        x = torch.tensor(0.3)
        result = lightning_circuit(x)
        print(f"Lightning设备可用: {result}")
        
    except Exception as e:
        print(f"Lightning设备不可用: {e}")
        print("建议: 考虑使用Lightning来提升性能")

def main():
    """主检查函数"""
    print("GaussGAN 量子环境兼容性检查")
    print("=" * 80)
    
    check_torch_cuda()
    check_pennylane_info()
    test_basic_quantum_circuit()
    test_batch_quantum_processing()
    test_memory_stress()
    check_pennylane_lightning()
    
    print("\n" + "=" * 80)
    print("检查完成!")
    print("=" * 80)
    
    # 最终清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"最终GPU内存使用: {final_memory:.1f} MB")

if __name__ == "__main__":
    main()