from matplotlib import pyplot as plt
import torch


def plot_singular_value_cdf(matrix):
    _, S, _ = torch.svd(matrix, compute_uv=False)
    index = torch.arange(len(S) + 1)
    singular_value_cdf = [0.0] + list(torch.cumsum(S, dim=0) / torch.sum(S))
    plt.figure(figsize=(10, 8))
    plt.ylim(0, 1)
    plt.yticks(torch.linspace(0, 1, 11))
    plt.plot(index, singular_value_cdf, linewidth=2, color='red')
    plt.grid(True)
    plt.show()

def get_svd_decomposition(matrix, rank):
    return torch.svd_lowrank(matrix, q=rank)

def get_cur_decomposition(matrix):
    pass
