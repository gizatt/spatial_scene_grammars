import numpy as np
import torch
from torch.distributions import constraints, transform_to
import logging

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


class ConstrainedParameter(torch.nn.Module):
    # Based heavily on pyro's constrained param system, but detached
    # from a global param store.
    def __init__(self, init_value, constraint=constraints.real):
        super().__init__()
        if np.prod(init_value.shape) == 0 and constraint != constraints.real:
            logging.warning("Warning: overriding constraint for empty parameter to constraints.real.")
            constraint = constraints.real
        self.constraint = constraint
        self.unconstrained_value = None
        self.set(init_value)
    def set_unconstrained(self, unconstrained_value):
        if self.unconstrained_value is None:
            self.unconstrained_value = torch.nn.Parameter(unconstrained_value)
        else:
            assert unconstrained_value.shape == self.unconstrained_value.shape
            self.unconstrained_value.data = unconstrained_value
    def set(self, constrained_value):
        with torch.no_grad():
            unconstrained_value = transform_to(self.constraint).inv(constrained_value)
            unconstrained_value = unconstrained_value.contiguous()
        self.set_unconstrained(unconstrained_value)
    def get_value(self):
        constrained_value = transform_to(self.constraint)(self.unconstrained_value)
        return constrained_value
    def __call__(self):
        return self.get_value()
    def get_unconstrained_value(self):
        return self.unconstrained_value