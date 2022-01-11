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

## MMD utilities modified from https://github.com/josipd/torch-two-sample/
def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances
        between two sample sets.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)
def se3_dist(sample_1, sample_2, beta=1., eps=1e-5):
    r"""Compute the matrix of all distances between
        two samples sets, where each sample is an element
        of SE(3) expressed as a 4x4 matrix. The distance
        is defined as
            || sample_1.t - sample_2.t ||_2
            + beta * acos( (tr(sample_1.R.T sample_2.R) - 1)/2 )
        i.e. the Euclidean translation distance, plus the
        angle between the poses in radians.

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, 4, 4)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, 4, 4)``.
    beta : float
        The weight multiplied by the angle distance to create the
        total distance.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        the SE(3) distance above between the ith and jth samples."""
    assert len(sample_1.shape) == 3 and sample_1.shape[1:] == (4, 4)
    assert len(sample_2.shape) == 3 and sample_2.shape[1:] == (4, 4)
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    beta = float(beta)

    # Compute translation distances.
    sample_1_t = sample_1[:, :3, 3]
    sample_2_t = sample_2[:, :3, 3]
    norms_1 = torch.sum(sample_1_t**2, dim=1, keepdim=True)
    norms_2 = torch.sum(sample_2_t**2, dim=1, keepdim=True)
    t_norms = (norms_1.expand(n_1, n_2) +
               norms_2.transpose(0, 1).expand(n_1, n_2))
    t_distances_squared = t_norms - 2 * sample_1_t.mm(sample_2_t.t())
    sample_t_distances = torch.sqrt(eps + torch.abs(t_distances_squared))
    
    # Compute rotation distances.
    sample_1_R = sample_1[:, :3, :3]
    sample_2_R = sample_2[:, :3, :3]

    # Prepare for a batch matrix multiply to get R1^T R2 terms.
    sample_1_R_expanded = sample_1_R.transpose(1, 2).unsqueeze(1).expand(n_1, n_2, 3, 3)
    sample_2_R_expanded = sample_2_R.unsqueeze(1).transpose(0, 1).expand(n_1, n_2, 3, 3)
    sample_R1tR2 = torch.matmul(sample_1_R_expanded, sample_2_R_expanded)
    sample_angle_distances = torch.abs(torch.acos(
        torch.clip(
            (sample_R1tR2.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1)/2,
            -1+eps, 1-eps
        )
    ))
    return sample_t_distances + sample_angle_distances * beta

def calculate_mmd(sample_1, sample_2, alphas, use_se3_metric=False, beta=1.0):
    r"""Evaluate the statistic.
    The kernel used is
    .. math::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j f(x, x')},
    for the provided ``alphas``.

    f(x, x') is the squared Euclidean norm, or
    an SE3 metric if use_se3_metric is True.

    By default, samples should be shaped [N, d] (with
    the same d between samples). If `use_se3_metric` is true,
    then the samples should be shaped [N, 4, 4].

    Arguments
    ---------
    sample_1: The first sample.
    sample_2: The second sample.
    alphas : list of :class:`float`
        The kernel parameters.
    Returns
    -------
    :class:`float`
        The test statistic."""
    if not use_se3_metric:
        assert len(sample_1.shape) == 2 and len(sample_2.shape) == 2
        assert sample_1.shape[1] == sample_2.shape[1]
    else:
        assert len(sample_1.shape) == 3 and len(sample_2.shape) == 3
        assert sample_1.shape[1:] == (4, 4) and sample_2.shape[1:] == (4, 4)

    n_1 = sample_1.shape[0]
    n_2 = sample_2.shape[0]

    a00 = 1. / (n_1 * (n_1 - 1))
    a11 = 1. / (n_2 * (n_2 - 1))
    a01 = - 2. / (n_1 * n_2)

    sample_12 = torch.cat((sample_1, sample_2), 0)
    if use_se3_metric:
        distances = se3_dist(sample_12, sample_12, beta=beta)
    else:
        distances = pdist(sample_12, sample_12, norm=2)

    kernels = None
    for alpha in alphas:
        kernels_a = torch.exp(- alpha * distances ** 2)
        if kernels is None:
            kernels = kernels_a
        else:
            kernels = kernels + kernels_a

    k_1 = kernels[:n_1, :n_1]
    k_2 = kernels[n_1:, n_1:]
    k_12 = kernels[:n_1, n_1:]

    mmd = (a01 * k_12.sum() +
           a00 * (k_1.sum() - torch.trace(k_1)) +
           a11 * (k_2.sum() - torch.trace(k_2)))
    return mmd
