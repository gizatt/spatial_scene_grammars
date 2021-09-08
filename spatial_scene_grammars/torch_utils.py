import numpy as np
import torch
from torch.distributions import constraints, transform_to
import logging

from pytorch3d.transforms.rotation_conversions import (
    matrix_to_quaternion, axis_angle_to_matrix,
    quaternion_to_axis_angle
)

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
        # Have to explicitly nuke the unconstrained value, as it might be
        # a registered parameter.
        del self.unconstrained_value
        self.unconstrained_value = unconstrained_value
    def set_unconstrained_as_param(self, unconstrained_value):
        if self.unconstrained_value is None:
            self.unconstrained_value = torch.nn.Parameter(unconstrained_value)
        else:
            unconstrained_value = unconstrained_value.reshape(self.unconstrained_value.shape)
            self.unconstrained_value.data = unconstrained_value
    def set(self, constrained_value):
        with torch.no_grad():
            unconstrained_value = transform_to(self.constraint).inv(constrained_value)
            unconstrained_value = unconstrained_value.contiguous()
        self.set_unconstrained_as_param(unconstrained_value)
    def get_value(self):
        constrained_value = transform_to(self.constraint)(self.unconstrained_value)
        return constrained_value
    def __call__(self):
        return self.get_value()
    def get_unconstrained_value(self):
        return self.unconstrained_value

def interp_translation(t1, t2, interp_factor):
    ''' Linearly interpolates between two translations t1 and t2
    according to interp_factor. Returns t1 for interp_factor=0
    and t2 for interp_factor=1. '''
    assert interp_factor >= 0. and interp_factor <= 1.
    return t1 * (1. - interp_factor) + t2 * (interp_factor)

def interp_rotation(r1, r2, interp_factor):
    ''' Given two rotation matrices r1 and r2, returns a rotation
    that is interp_factor between them; when factor=0, returns r1, and
    when factor=1, returns r2. Linearly interpolates along the geodesic
    between the rotations by converting the relative rotation to angle-axis
    and scaling the angle by interp_factor. If r1 and r2 pi radians apart,
    the returned rotation axis will be arbitrary. '''
    assert interp_factor >= 0. and interp_factor <= 1.
    # Convert to angle-axis, interpolate angle, convert back.
    # When interp_factor = 0, this return r1. When interp_factor = 1, this
    # returns 1.
    rel = torch.matmul(r2, r1.transpose(0, 1))
    rel_axis_angle = quaternion_to_axis_angle(matrix_to_quaternion(rel))
    # Scaling keeps axis the same, but changes angle.
    scaled_rel_axis_angle = interp_factor * rel_axis_angle
    return torch.matmul(axis_angle_to_matrix(scaled_rel_axis_angle), r1)