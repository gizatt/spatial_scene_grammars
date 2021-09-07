import numpy as np
import logging

import torch
from numbers import Number
from torch.distributions import constraints
from torch.distributions.geometric import Geometric
from torch.distributions.categorical import Categorical
from torch.distributions.utils import broadcast_all
import pyro
from pyro.distributions.torch_distribution import TorchDistribution

class LeftSidedConstraint(constraints.Constraint):
    """
    Constrain to vectors that are 1s until they hit a 0, and then
    are all 0s.
    """
    is_discrete = True
    event_dim = 1

    def check(self, value):
        if len(value) == 0:
            return True
        is_boolean = (value == 0) | (value == 1)
        neighbor_diff = value[..., -1:] - value[..., :-1]
        return is_boolean.all(-1) & (neighbor_diff <= 0) .all(-1)


class UniformWithEqualityHandling(pyro.distributions.Uniform):
    ''' Uniform distribution, but if any of the lower bounds equal
    the upper bounds, those elements are replaced with Delta distributions,
    and modifies handling to allow upper bound to be inclusive. '''

    def __init__(self, low, high, validate_args=None, eps=1E-3):
        self.low, self.high = broadcast_all(low, high)

        # Should have been set by pyro Uniform __init__
        self._unbroadcasted_low = low
        self._unbroadcasted_high = high

        # Tolerance in bounds-checking. Very low because
        # I use SNOPT in this repo, which has 1E-6 default
        # tolerance and struggles to do better. TODO(gizatt) Need
        # a study of where I'm losing tolerance so I can tighten this.
        self.eps = eps

        if isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
            self.delta_mask = torch.isclose(torch.tensor(high) - torch.tensor(low), torch.tensor(0.))
        else:
            batch_shape = self.low.size()
            self.delta_mask = torch.isclose(high - low, high*0.)

        # Ensure we have at least slightly wide range on the inside
        # where we don't have equality.

        self.uniform_in_bounds_ll = -torch.log(self.high - self.low)
        # Have to be very selective with superclass constructors since we'll cause an
        # error in torch.distribution.Uniform.__init__
        # TorchDistributionMixin has no constructor, so we're OK not calling it
        super(torch.distributions.Uniform, self).__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            # TODO(gizatt) Can I turn this back on? It's throwing out
            # values that are within 1E-17 of the bounds
            #self._validate_sample(value)
            logging.warning("validate_sample disabled in UniformWithEqualityHandling")
        value = value.reshape(self.delta_mask.shape)
        # Handle uniform part
        in_bounds = torch.logical_and(
            value >= self.low,
            value <= self.high,
        )
        on_bounds = torch.logical_or(
            torch.isclose(value, self.low, atol=self.eps, rtol=self.eps),
            torch.isclose(value, self.high, atol=self.eps, rtol=self.eps)
        )
        ll = torch.log(torch.zeros_like(value))
        in_bounds_uniform = torch.logical_and(torch.logical_or(in_bounds, on_bounds), ~self.delta_mask)
        ll[in_bounds_uniform] = self.uniform_in_bounds_ll[in_bounds_uniform]
        ll[torch.logical_and(on_bounds, self.delta_mask)] = 0.
        return ll

    def cdf(self, value):
        print("WARN: NOT VALIDATED")
        if self._validate_args:
            self._validate_sample(value)
        result = (value - self.low) / (self.high - self.low)
        result[self.delta_mask] = (value >= self.low)[self.delta_mask]*1.
        return result.clamp(min=0, max=1)
    
    def entropy(self):
        print("WARN: NOT VALIDATED")
        diff = self.high - self.low
        diff[self.delta_mask] = 1.
        # Entropy of delta = 1. * log(1.) = 0.
        return torch.log(diff)


class LeftSidedRepeatingOnesDist(TorchDistribution):
    ''' Distribution with support over a binary vector x [1, ... 1, 0, ... 0],
    where the number of 1's is drawn from a Categorical distribution of specified
    parameters [p0 ... pN]. p0 is the prob of having 0 1's; pN is the prob of
    having all 1's.'''
    # TODO(gizatt): Implement enumerate_support if I ever want to rely on it
    # at inference time.
    arg_constraints = {}
    support = LeftSidedConstraint()

    def __init__(self, categorical_probs, validate_args=None):
        self._categorical = Categorical(categorical_probs)
        batch_shape = self._categorical.batch_shape
        event_shape = torch.Size((categorical_probs.shape[-1] - 1,))
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
    
    def sample(self, sample_shape=torch.Size()):
        ks = self._categorical.sample(sample_shape).int()
        out = torch.zeros(sample_shape + self.event_shape)
        out[..., :ks] = 1.
        return out

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        valid_entries = self.support.check(value)
        generating_k = value.sum(axis=-1)
        
        ll = self._categorical.log_prob(generating_k)
        ll[~valid_entries] = -np.inf
        return ll


class VectorCappedGeometricDist(TorchDistribution):
    ''' Distribution with support over a binary vector x of length max_repeats,
    where x is generated by taking a draw k from a geometric distribution,
    capping it at max_repeats, and activating entries up to k. '''
    # TODO(gizatt): Implement enumerate_support if I ever want to rely on it
    # at inference time.
    arg_constraints = {}
    support = LeftSidedConstraint()

    def __init__(self, geometric_prob, max_repeats, validate_args=None):
        # TODO: This could shell out to LeftSidedRepeatingOnesDist.
        self.max_repeats = max_repeats
        self._geometric = Geometric(geometric_prob)
        batch_shape = self._geometric.batch_shape
        event_shape = torch.Size((max_repeats,))
        super(VectorCappedGeometricDist, self).__init__(batch_shape, event_shape, validate_args=validate_args)
    
    def sample(self, sample_shape=torch.Size()):
        ks = self._geometric.sample(sample_shape).int()
        out = torch.zeros(sample_shape + self.event_shape)
        out[..., :ks] = 1.
        return out

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        valid_entries = self.support.check(value)
        generating_k = value.sum(axis=-1)
        
        geom_ll = self._geometric.log_prob(generating_k)
        # Fix those entries where generating k is the max size, which has extra prob
        # of geometric process generating any number above generating_k.
        geom_ll[generating_k == self.max_repeats] = torch.log((1.-self._geometric.probs)**((self.max_repeats - 1) + 1))
        # Also set infeasible entries to 0 prob.
        geom_ll[~valid_entries] = -np.inf
        return geom_ll
