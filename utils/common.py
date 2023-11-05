import os
import torch
from torchvision import datasets, transforms
from tqdm import tqdm


def prepare_dataset(name: str ="mnist", path: str ="."):
    name = name.lower()
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(32, 32), antialias=True),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST
    elif name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        ])
        dataset = datasets.CIFAR10
    else:
        raise NotImplementedError(f"Dataset {name} can't be prepared")
    train_dataset = dataset(path, train=True, download=True, transform=transform)
    test_dataset = dataset(path, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def count_parameters(model: torch.nn.Module):
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
    return total_params

def freeze(module: torch.nn.Module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()

def unfreeze(module: torch.nn.Module):
    for param in module.parameters():
        param.requires_grad = True
    module.train()

def get_vgg11_classifier(num_classes: int = 10):
    return torch.nn.Sequential(
        torch.nn.Linear(25088, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, num_classes)
    )

def train_model(model, dataloader, loss_fn, optimizer, epochs=1, scheduler=None, device='cpu', log=True, path=None):
    if path is not None and os.path.exists(path):
        model.to('cpu')
        model.load_state_dict(torch.load(path))
        if log:
            print(f'Model successfully loaded from "{path}"')
        return
    model.to(device).train()
    logger = tqdm if log else iter
    for _ in range(epochs):
        for input, target in (pbar := logger(dataloader)):
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            if log:
                pbar.set_description(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    if path is not None:
        torch.save(model.state_dict(), path)

def test_model(model, dataloader, device='cpu', log=True):
    model.to(device).eval()
    correct, total = 0, 0
    logger = tqdm if log else iter
    with torch.no_grad():
        for input, target in logger(dataloader):
            input, target = input.to(device), target.to(device)
            output = model(input).argmax(dim=1)
            correct += (output == target).sum()
            total += target.numel()
    if log:
        print(f"Correctly classified targets: {correct} / {total}")
        print(f"Total accuracy: {100.0 * correct / total}%")
    return 100.0 * correct / total

class CompressedLinear(torch.nn.Module):
    """
    Линейный слой с факторизацией весов в виде W = A @ B
    
    bias из исходного представления Linear(x) = Wx + bias
    """
    def __init__(self, A, B, bias):
        super(CompressedLinear, self).__init__()
        self.A = torch.nn.Linear(in_features=A.shape[1], out_features=A.shape[0], bias=True)
        self.B = torch.nn.Linear(in_features=B.shape[1], out_features=B.shape[0], bias=False)
        self.A.weight.data = A
        self.A.bias.data = bias
        self.B.weight.data = B
    
    def forward(self, x):
        return self.A(self.B(x))