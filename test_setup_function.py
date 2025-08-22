#!/usr/bin/env python
"""
测试 setup_convergence_tracking 函数
"""

import sys
import traceback

def test_setup_function():
    """测试 setup_convergence_tracking 函数"""
    print("测试 setup_convergence_tracking 函数...")
    try:
        from source.training_integration import setup_convergence_tracking
        
        # 调用函数（这里应该是问题所在）
        tracker, callback = setup_convergence_tracking(
            config_path="config.yaml", 
            generator_type="classical_normal"
        )
        
        print(f"  ✅ setup_convergence_tracking 成功")
        print(f"     tracker type: {type(tracker)}")
        print(f"     callback type: {type(callback)}")
        print(f"     callback MRO: {callback.__class__.__mro__}")
        
        return tracker, callback
        
    except Exception as e:
        print(f"  ❌ setup_convergence_tracking 失败: {e}")
        traceback.print_exc()
        return None, None

def test_with_trainer():
    """测试与实际trainer的组合"""
    print("\n测试与实际trainer的组合...")
    try:
        from lightning import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint
        from lightning.pytorch.loggers import MLFlowLogger
        from source.training_integration import setup_convergence_tracking
        
        # 设置 MLflow logger
        mlflow_logger = MLFlowLogger(
            experiment_name="test_experiment",
            tracking_uri="file:./mlruns",
        )
        
        # 设置检查点回调
        checkpoint_callback = ModelCheckpoint(
            dirpath="test_checkpoints",
            filename="test-{epoch:02d}",
            save_top_k=3,
            monitor="train_loss",
            mode="min",
            every_n_epochs=5,
            save_last=True,
        )
        
        # 设置收敛跟踪
        convergence_tracker, convergence_callback = setup_convergence_tracking(
            config_path="config.yaml", 
            generator_type="classical_normal"
        )
        
        print(f"Convergence tracking enabled for classical_normal")
        print(f"Tracking metrics: {list(convergence_tracker.convergence_thresholds.keys())}")
        
        # 创建trainer（这里应该会失败）
        trainer = Trainer(
            max_epochs=5,
            accelerator="cpu",  # 使用CPU避免GPU问题
            logger=mlflow_logger,
            log_every_n_steps=5,
            limit_val_batches=2,
            callbacks=[checkpoint_callback, convergence_callback],
        )
        
        print("  ✅ Trainer 创建成功！")
        return True
        
    except Exception as e:
        print(f"  ❌ Trainer 创建失败: {e}")
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("setup_convergence_tracking 函数测试")
    print("="*60)
    
    tracker, callback = test_setup_function()
    
    if tracker and callback:
        test_with_trainer()

if __name__ == "__main__":
    main()