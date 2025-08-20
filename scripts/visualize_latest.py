import mlflow
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils_scripts

def plot_latest_training(csv_artifacts, run_id, filename):
    # 创建子图网格
    fig_rows = int((len(csv_artifacts) + 3) // 4)
    fig, axes = plt.subplots(nrows=fig_rows, ncols=4, figsize=(12, 3 * fig_rows))
    
    # 获取原始数据分布参数
    mean1 = torch.tensor([-5.0, 5.0])
    cov1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mean2 = torch.tensor([5.0, 5.0])
    cov2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    
    # 生成原始数据点
    inps1, inps2, _, _, _, _ = utils_scripts.generate_dataset(
        1000, mean1, cov1, mean2, cov2
    )

    # 获取MLflow客户端
    client = mlflow.tracking.MlflowClient()

    # 遍历每个时间点的数据
    for idx, csv_path in enumerate(csv_artifacts):
        # 下载并读取CSV数据
        data_path = client.download_artifacts(run_id, csv_path)
        df = pd.read_csv(data_path, header=None, sep=",")
        
        # 处理axes的1D和2D情况
        if axes.ndim == 1:
            ax = axes[idx % 4]
        else:
            ax = axes[idx // 4, idx % 4]

        # 数据处理
        df = df.drop(0).apply(pd.to_numeric).round(3)
        data = df.values
        x_values = data[:, 0]
        y_values = data[:, 1]

        # 跳过无效数据
        if np.all(np.isnan(x_values)) or np.all(np.isnan(y_values)):
            break

        # 绘制原始分布
        ax.scatter(inps1[:, 0], inps1[:, 1], color="blue", label="Class -1", alpha=0.5)
        ax.scatter(inps2[:, 0], inps2[:, 1], color="red", label="Class +1", alpha=0.5)
        
        # 绘制生成的数据
        ax.scatter(x_values, y_values, s=3, color='black', alpha=0.5, label='Generated')

        # 添加参考线
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.3)

        # 设置图表范围和标题
        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.set_title(f"Epoch {idx+1}")
        
        if idx == 0:  # 只在第一个子图显示图例
            ax.legend()

    # 设置总标题
    fig.suptitle(f"Latest Training Run (ID: {run_id})", fontsize=16)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # 确保输出目录存在
    os.makedirs("images", exist_ok=True)
    
    # 获取MLflow客户端
    client = mlflow.tracking.MlflowClient()
    
    # 获取实验
    experiment = client.get_experiment_by_name("GaussGAN-manual")
    
    # 获取最新的运行
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time desc"],
        max_results=1  # 只获取最新的一次运行
    )
    
    if not runs:
        print("No runs found in the experiment")
        return
        
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    print(f"Processing latest run: {run_id}")
    
    # 获取运行的artifacts
    artifacts = client.list_artifacts(run_id)
    csv_artifacts = [a.path for a in artifacts if a.path.endswith(".csv")]
    
    if not csv_artifacts:
        print("No CSV artifacts found in the run")
        return
        
    # 生成输出文件名
    filename = f"images/training_latest.png"
    
    # 绘制训练过程
    plot_latest_training(csv_artifacts, run_id, filename)
    print(f"Training visualization saved to {filename}")

if __name__ == "__main__":
    main() 