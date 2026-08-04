import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    if method == 'relu':
        return torch.maximum(torch.tensor(x), torch.tensor(0))
    if method == 'sigmoid':
        return torch.sigmoid(torch.tensor(x))
    if method == 'tanh':
        return torch.tanh(torch.tensor(x))
    if method == 'leaky_relu':
        return torch.where(torch.tensor(x) > 0, torch.tensor(x), torch.tensor(x) * 0.01)
        