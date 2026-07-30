import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    if op == 'add':
        # res = x + y
        res = torch.add(x, y)
        return res.tolist()
    if op == 'multiply':
        # res = x * y
        res = torch.mul(x, y)
        return res.tolist()
    if op == 'matmul':
        # res = x @ y
        res = torch.matmul(x, y)
        return res.tolist()
    if op == 'power':
        # res = x ** y
        res = torch.pow(x, y)
        return res.tolist()
    if op == 'max':
        res = torch.maximum(x, y)
        return res.tolist()