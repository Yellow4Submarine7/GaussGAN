#!/usr/bin/env python
"""
最小化测试：隔离 Lightning callback 问题
"""

import sys
import traceback

def test_callback_import():
    """测试 callback 导入"""
    print("测试1: 导入 Lightning callbacks...")
    try:
        from lightning.pytorch.callbacks import Callback, ModelCheckpoint
        print("  ✅ Lightning callbacks 导入成功")
        return True
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False

def test_convergence_callback_import():
    """测试收敛回调导入"""
    print("\n测试2: 导入 ConvergenceCallback...")
    try:
        from source.training_integration import ConvergenceCallback
        print("  ✅ ConvergenceCallback 导入成功")
        return True
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        traceback.print_exc()
        return False

def test_convergence_callback_creation():
    """测试收敛回调创建"""
    print("\n测试3: 创建 ConvergenceCallback 实例...")
    try:
        from source.training_integration import ConvergenceCallback
        from source.metrics import ConvergenceTracker
        
        # 创建 tracker
        tracker = ConvergenceTracker(patience=5, min_delta=0.01)
        
        # 创建 callback
        callback = ConvergenceCallback(
            convergence_tracker=tracker,
            generator_type="test",
            save_frequency=10,
            plot_frequency=25
        )
        
        print(f"  ✅ ConvergenceCallback 创建成功: {type(callback)}")
        print(f"     Parent classes: {callback.__class__.__mro__}")
        return callback
    except Exception as e:
        print(f"  ❌ 创建失败: {e}")
        traceback.print_exc()
        return None

def test_callback_with_trainer():
    """测试回调与 Trainer 的兼容性"""
    print("\n测试4: 测试 callback 与 Trainer 兼容性...")
    try:
        from lightning import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint
        from source.training_integration import ConvergenceCallback
        from source.metrics import ConvergenceTracker
        
        # 创建 callbacks
        tracker = ConvergenceTracker(patience=5, min_delta=0.01)
        convergence_callback = ConvergenceCallback(
            convergence_tracker=tracker,
            generator_type="test",
            save_frequency=10,
            plot_frequency=25
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath="test_checkpoints",
            filename="test-{epoch:02d}",
            save_top_k=1,
            monitor="train_loss",
            mode="min"
        )
        
        # 创建 Trainer（这里应该会失败）
        trainer = Trainer(
            max_epochs=1,
            callbacks=[checkpoint_callback, convergence_callback],
            enable_progress_bar=False,
            logger=False
        )
        
        print("  ✅ Trainer 创建成功！")
        return True
        
    except Exception as e:
        print(f"  ❌ Trainer 创建失败: {e}")
        traceback.print_exc()
        return False

def test_simple_trainer():
    """测试最简单的 Trainer 创建"""
    print("\n测试5: 测试简单 Trainer...")
    try:
        from lightning import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint
        
        checkpoint_callback = ModelCheckpoint(
            dirpath="test_checkpoints",
            filename="test-{epoch:02d}",
            save_top_k=1,
            monitor="train_loss",
            mode="min"
        )
        
        # 只用内置回调
        trainer = Trainer(
            max_epochs=1,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            logger=False
        )
        
        print("  ✅ 简单 Trainer 创建成功！")
        return True
        
    except Exception as e:
        print(f"  ❌ 简单 Trainer 创建失败: {e}")
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("="*60)
    print("Lightning Callback 问题诊断测试")
    print("="*60)
    
    tests = [
        ("导入测试", test_callback_import),
        ("ConvergenceCallback导入", test_convergence_callback_import),
        ("ConvergenceCallback创建", test_convergence_callback_creation),
        ("简单Trainer测试", test_simple_trainer),
        ("Callback+Trainer兼容性", test_callback_with_trainer),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:25s}: {status}")

if __name__ == "__main__":
    main()