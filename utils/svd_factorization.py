import math
from matplotlib import pyplot as plt
import torch
from .common import CompressedLinear


def get_linear_svd_layers(model: torch.nn.Module, device: str = "cpu"):
    compressed_matrices = []
    model.to(device).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                U, S, Vh = torch.linalg.svd(module.weight.data)
                compressed_matrices.append((U.cpu(), S.cpu(), Vh.cpu(), module.bias.data.cpu()))
    return compressed_matrices

def get_compressed_classifier_by_rank(compressed_matrices, rank):
    classifier = torch.nn.Sequential()
    for i, (U, S, Vh, bias) in enumerate(compressed_matrices):
        if i == len(compressed_matrices) - 1:
            classifier.append(CompressedLinear(U, S.view(-1, 1) * Vh, bias))
            continue
        classifier.append(CompressedLinear(U[:, :rank], S[:rank].view(-1, 1) * Vh[:rank], bias))
        classifier.append(torch.nn.ReLU(inplace=True))
        classifier.append(torch.nn.Dropout())
    return classifier

def get_compressed_classifier_by_compression_rate(compressed_matrices, compression_rate):
    classifier = torch.nn.Sequential()
    for i, (U, S, Vh, bias) in enumerate(compressed_matrices):
        if i == len(compressed_matrices) - 1:
            classifier.append(CompressedLinear(U, S.view(-1, 1) * Vh, bias))
            continue
        m, n = U.shape[0], Vh.shape[1]
        rank = math.ceil(m * n / ((m + n) * compression_rate))
        classifier.append(CompressedLinear(U[:, :rank], S[:rank].view(-1, 1) * Vh[:rank], bias))
        classifier.append(torch.nn.ReLU(inplace=True))
        classifier.append(torch.nn.Dropout())
    return classifier

def plot_singular_values(S):
    sv_distribution = S / S[0]
    plt.rc("text", usetex=True)
    plt.semilogy(sv_distribution)
    plt.xlabel(r"Singular value index, $i$")
    plt.ylabel(r"$\sigma_i / \sigma_0$")
    plt.show()
