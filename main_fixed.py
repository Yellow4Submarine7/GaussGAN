#!/usr/bin/env python
"""
临时修复版的main.py - 增加convergence callback错误处理
"""

import argparse
import os
import pprint
import warnings
import yaml
from functools import partial
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import pdb

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

# from source.data import GaussianDataModule #, GaussianDataset
from source.model import GaussGan
from source.utils import set_seed, load_data


from source.nn import (
    MLPDiscriminator,
    MLPGenerator,
    ClassicalNoise,
    QuantumNoise,
    QuantumShadowNoise,
)

from source.training_integration import setup_convergence_tracking

def main():
    """主函数，增加错误处理"""
    
    # 加载配置
    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # 设置参数解析
    parser = argparse.ArgumentParser(description="Train the GaussGan model")
    
    # 添加所有命令行参数
    parser.add_argument("--z_dim", type=int, help="Dimension of the latent space")
    parser.add_argument("--generator_type", type=str, help="Type of generator")
    parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    
    args = parser.parse_args()
    
    # 合并配置
    final_args = config.copy()
    for key, value in vars(args).items():
        if value is not None:
            final_args[key] = value
    
    print(f"Generator type: {final_args.get('generator_type')}")
    print(f"Max epochs: {final_args.get('max_epochs')}")
    
    # 设置种子
    set_seed(final_args.get("seed", 42))
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('medium')
    
    # 加载数据
    datamodule, gaussians = load_data(final_args)
    
    # 设置生成器
    generator_type = final_args["generator_type"]
    z_dim = final_args["z_dim"]
    nn_gen = eval(final_args["nn_gen"])
    
    if generator_type == "classical_uniform":
        noise_fn = ClassicalNoise(z_dim, distribution="uniform")
    elif generator_type == "classical_normal":
        noise_fn = ClassicalNoise(z_dim, distribution="normal")
    elif generator_type == "quantum_samples":
        noise_fn = QuantumNoise(
            z_dim, 
            n_qubits=final_args.get("quantum_qubits", 6),
            n_layers=final_args.get("quantum_layers", 2),
            n_shots=final_args.get("quantum_shots", 100)
        )
    elif generator_type == "quantum_shadows":
        noise_fn = QuantumShadowNoise(
            z_dim,
            n_qubits=final_args.get("quantum_qubits", 6),
            n_layers=final_args.get("quantum_layers", 2),
            n_shots=final_args.get("quantum_shots", 100)
        )
    
    G = MLPGenerator(
        noise_fn=noise_fn, 
        nn_gen=nn_gen,
        non_linearity=final_args["non_linearity"],
        std_scale=final_args.get("std_scale", 1.1),
        min_std=final_args.get("min_std", 0.5)
    )
    
    # 设置判别器
    nn_disc = eval(final_args["nn_disc"])
    D = MLPDiscriminator(nn_disc, final_args["non_linearity"])
    
    # 设置预测器
    nn_validator = eval(final_args["nn_validator"])
    V = MLPDiscriminator(nn_validator, final_args["non_linearity"])
    
    # 移动到设备
    G.to(device)
    D.to(device)
    V.to(device)
    
    print("Nets created")
    
    # 目标数据
    target_data = torch.cat([gaussians["inputs"], gaussians["targets"]], dim=1)
    
    # 创建模型
    model = GaussGan(
        G, D, V,
        optimizer=partial(torch.optim.Adam, lr=final_args["learning_rate"]),
        killer=final_args["killer"],
        n_critic=final_args["n_critic"],
        grad_penalty=final_args["grad_penalty"],
        rl_weight=final_args["rl_weight"],
        n_predictor=final_args["n_predictor"],
        metrics=list(final_args["metrics"]),
        gaussians=gaussians,
        validation_samples=final_args["validation_samples"],
        non_linearity=final_args["non_linearity"],
        target_data=target_data,
        convergence_patience=final_args.get("convergence_patience", 10),
        convergence_min_delta=final_args.get("convergence_min_delta", 1e-4),
        convergence_monitor=final_args.get("convergence_monitor", "KLDivergence"),
        convergence_window=final_args.get("convergence_window", 5),
    )
    model.to(device)
    
    # 设置logger
    run_instance = f'{final_args["generator_type"]}'
    mlflow_logger = MLFlowLogger(
        experiment_name=final_args["experiment_name"], 
        run_name=run_instance
    )
    run_id = mlflow_logger.run_id
    print(f"---Run ID: {run_id}")
    
    # 设置检查点
    filename = f"run_id-{run_id}"
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=filename + "-{epoch:03d}",
        save_top_k=-1,
        every_n_epochs=5,
        save_last=True,
    )
    
    # 尝试设置convergence tracking，如果失败则继续
    callbacks_list = [checkpoint_callback]
    
    try:
        print("Attempting to setup convergence tracking...")
        convergence_tracker, convergence_callback = setup_convergence_tracking(
            config_path="config.yaml", 
            generator_type=final_args["generator_type"]
        )
        
        print(f"Convergence tracking enabled for {final_args['generator_type']}")
        print(f"Tracking metrics: {list(convergence_tracker.convergence_thresholds.keys())}")
        
        callbacks_list.append(convergence_callback)
        print("Convergence callback added successfully")
        
    except Exception as e:
        print(f"⚠️ Warning: Convergence tracking disabled due to error: {e}")
        print("Training will continue without convergence tracking")
    
    # 记录超参数
    mlflow_logger.log_hyperparams(final_args)
    
    # 创建trainer
    print("Creating trainer...")
    trainer = Trainer(
        max_epochs=final_args["max_epochs"],
        accelerator=final_args["accelerator"],
        logger=mlflow_logger,
        log_every_n_steps=5,
        limit_val_batches=2,
        callbacks=callbacks_list,
    )
    
    print("Trainer created successfully!")
    
    # 开始训练
    if final_args["stage"] == "train":
        print("Starting training...")
        trainer.fit(model=model, datamodule=datamodule)
        print("Training completed!")
    elif final_args["stage"] == "test":
        trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()