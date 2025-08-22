#!/usr/bin/env python
"""
简化版main.py - 禁用convergence callback测试训练
"""

import argparse
import os
import pprint
import warnings
import yaml
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from source.model import GaussGan
from source.utils import set_seed, load_data

def simple_main():
    """简化的main函数，禁用convergence callback"""
    
    # 设置简单的参数
    final_args = {
        'z_dim': 4,
        'generator_type': 'classical_normal', 
        'stage': 'train',
        'experiment_name': 'simple_test',
        'killer': False,
        'grad_penalty': 0.2,
        'n_critic': 5,
        'n_predictor': 5,
        'checkpoint_path': 'checkpoints/',
        'agg_method': 'prod',
        'max_epochs': 5,
        'batch_size': 256,
        'learning_rate': 0.001,
        'nn_gen': '[256,256]',
        'nn_disc': '[256,256]',
        'nn_validator': '[128,128]',
        'non_linearity': 'LeakyReLU',
        'std_scale': 1.1,
        'min_std': 0.5,
        'dataset_type': 'NORMAL',
        'metrics': ['IsPositive', 'LogLikelihood', 'KLDivergence', 'WassersteinDistance', 'MMDDistance'],
        'accelerator': 'gpu',
        'validation_samples': 500,
        'seed': 42,
        'rl_weight': 100
    }
    
    # 设置种子
    set_seed(final_args['seed'])
    
    # 设置精度
    torch.set_float32_matmul_precision('medium')
    
    # 加载数据
    datamodule, gaussians = load_data(final_args)
    print("Data loaded")
    
    # 创建模型
    model = GaussGan(
        target_data=gaussians,
        **final_args
    )
    print("Model created")
    
    # 设置logger
    mlflow_logger = MLFlowLogger(
        experiment_name=final_args["experiment_name"],
        tracking_uri="file:./mlruns",
    )
    
    # 设置检查点（只用这一个callback）
    checkpoint_callback = ModelCheckpoint(
        dirpath=final_args["checkpoint_path"],
        filename=f'{final_args["generator_type"]}-{{epoch:02d}}',
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        every_n_epochs=5,
        save_last=True,
    )
    
    # 记录超参数
    mlflow_logger.log_hyperparams(final_args)
    
    print("Creating trainer (without convergence callback)...")
    
    # 创建trainer（只用checkpoint callback）
    trainer = Trainer(
        max_epochs=final_args["max_epochs"],
        accelerator=final_args["accelerator"],
        logger=mlflow_logger,
        log_every_n_steps=5,
        limit_val_batches=2,
        callbacks=[checkpoint_callback],  # 只用checkpoint callback
    )
    
    print("Trainer created successfully!")
    print(f"Run ID: {mlflow_logger.run_id}")
    
    # 开始训练
    if final_args["stage"] == "train":
        print("Starting training...")
        trainer.fit(model=model, datamodule=datamodule)
        print("Training completed!")
    
    return True

if __name__ == "__main__":
    simple_main()