import torch

def inv_softplus(y, beta=1., eps=1E-9):
    # y = 1/beta * log(1 + exp(beta * x))
    # so
    # x = log(exp(beta * y) - 1)/beta
    # Y better be > 0, of course.
    if isinstance(y, float):
        y = torch.tensor(y)
    y = torch.clip(y, min=eps)
    return torch.log(torch.exp(beta * y) - 1.)/beta

def inv_sigmoid(y, eps=1E-16):
    # y = 1 / (1 + exp(-x))
    # so
    # x = -log((1 / y)-1)
    # Y better be in [0, 1]
    if isinstance(y, float):
        y = torch.tensor(y)
    y = torch.clip(y, min=eps, max=1.-eps)
    return -torch.log((1 / y) - 1)
