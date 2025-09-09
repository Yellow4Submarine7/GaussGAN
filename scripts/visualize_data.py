import torch
import matplotlib.pyplot as plt
import os

def plot_gaussian_data():
    mean1 = torch.tensor([-5.0, 5.0])
    cov1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mean2 = torch.tensor([5.0, 5.0])
    cov2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    dist1 = torch.distributions.MultivariateNormal(mean1, cov1)
    dist2 = torch.distributions.MultivariateNormal(mean2, cov2)

    n_points = 1000
    samples1 = dist1.sample((n_points,))
    samples2 = dist2.sample((n_points,))

    plt.figure(figsize=(8, 6))
    
    plt.scatter(samples1[:, 0], samples1[:, 1], color="blue", label="Class -1")
    plt.scatter(samples2[:, 0], samples2[:, 1], color="red", label="Class +1")

    plt.axhline(y=0, color="r", linestyle="--")
    plt.axvline(x=0, color="r", linestyle="--")

    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    plt.xticks([-10, -5, 0, 5, 10])
    plt.yticks([-10, -5, 0, 5, 10])

    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Original Gaussian Distributions")
    plt.legend()

    os.makedirs("images", exist_ok=True)
    
    plt.savefig("images/original_data.png")
    plt.close()

if __name__ == "__main__":
    plot_gaussian_data()
