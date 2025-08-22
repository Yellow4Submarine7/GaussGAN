#!/usr/bin/env python
"""
完全复制 main.py 的配置进行测试
"""

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from source.training_integration import setup_convergence_tracking
from source.utils import return_parser, load_data, merge_args
from source.model import GaussGan

def test_exact_main_config():
    """完全复制main.py的配置"""
    print("测试完全复制main.py的配置...")
    
    try:
        # 设置参数（复制main.py的逻辑）
        parser = return_parser()
        cmd_args = parser.parse_args([
            "--generator_type", "classical_normal", 
            "--max_epochs", "5", 
            "--seed", "42"
        ])
        final_args = merge_args(cmd_args)
        
        print(f"Generator type: {final_args['generator_type']}")
        print(f"Max epochs: {final_args['max_epochs']}")
        
        # 设置设备和精度（复制main.py）
        torch.set_float32_matmul_precision('medium')
        
        # 加载数据（复制main.py）
        datamodule, gaussians = load_data(final_args)
        
        # 创建模型（复制main.py）
        model = GaussGan(
            target_data=gaussians,
            **final_args
        )
        print("Models created successfully")
        
        # 设置MLflow logger（复制main.py）
        mlflow_logger = MLFlowLogger(
            experiment_name=final_args["experiment_name"],
            tracking_uri="file:./mlruns",
        )
        
        # 设置检查点回调（复制main.py）
        checkpoint_callback = ModelCheckpoint(
            dirpath=final_args["checkpoint_path"],
            filename=f'{final_args["generator_type"]}-{{epoch:02d}}',
            save_top_k=3,
            monitor="train_loss",
            mode="min",
            every_n_epochs=5,
            save_last=True,
        )
        
        # 设置收敛跟踪（复制main.py）
        convergence_tracker, convergence_callback = setup_convergence_tracking(
            config_path="config.yaml", 
            generator_type=final_args["generator_type"]
        )
        
        print(f"Convergence tracking enabled for {final_args['generator_type']}")
        print(f"Tracking metrics: {list(convergence_tracker.convergence_thresholds.keys())}")
        
        # 记录超参数（复制main.py）
        hparams = final_args
        mlflow_logger.log_hyperparams(hparams)
        
        # 创建trainer（这里应该失败，复制main.py的确切配置）
        trainer = Trainer(
            max_epochs=final_args["max_epochs"],
            accelerator=final_args["accelerator"],
            logger=mlflow_logger,
            log_every_n_steps=5,
            limit_val_batches=2,
            callbacks=[checkpoint_callback, convergence_callback],
        )
        
        print("  ✅ Trainer 创建成功！")
        return True
        
    except Exception as e:
        print(f"  ❌ 配置失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("完全复制 main.py 配置测试")
    print("="*60)
    
    test_exact_main_config()