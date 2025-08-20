import torch
import matplotlib.pyplot as plt
import os

def plot_gaussian_data():
    # 定义两个高斯分布的参数
    mean1 = torch.tensor([-5.0, 5.0])
    cov1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mean2 = torch.tensor([5.0, 5.0])
    cov2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    # 创建MultivariateNormal分布
    dist1 = torch.distributions.MultivariateNormal(mean1, cov1)
    dist2 = torch.distributions.MultivariateNormal(mean2, cov2)

    # 从分布中采样点
    n_points = 1000
    samples1 = dist1.sample((n_points,))
    samples2 = dist2.sample((n_points,))

    # 创建图
    plt.figure(figsize=(8, 6))
    
    # 绘制两个高斯分布的点
    plt.scatter(samples1[:, 0], samples1[:, 1], color="blue", label="Class -1")
    plt.scatter(samples2[:, 0], samples2[:, 1], color="red", label="Class +1")

    # 添加参考线
    plt.axhline(y=0, color="r", linestyle="--")
    plt.axvline(x=0, color="r", linestyle="--")

    # 设置图的范围和刻度
    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    plt.xticks([-10, -5, 0, 5, 10])
    plt.yticks([-10, -5, 0, 5, 10])

    # 移除标题和标签（与训练图保持一致）
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Original Gaussian Distributions")
    plt.legend()

    # 确保images目录存在
    os.makedirs("images", exist_ok=True)
    
    # 保存图
    plt.savefig("images/original_data.png")
    plt.close()

if __name__ == "__main__":
    plot_gaussian_data()
