import pytest

import numpy as np
import torch

from spatial_scene_grammars.distributions import *

torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=range(10))
def set_seed(request):
    torch.manual_seed(request.param)
    np.random.seed(request.param)

def test_UniformWithEqualityHandling(set_seed):
    lb = torch.tensor([0., 0., 0.])
    ub = torch.tensor([0., 1., 2.])
    expected_ll = torch.log(torch.tensor([1., 1., 0.5]))
    dist = UniformWithEqualityHandling(lb, ub, validate_args=True)
    val = dist.sample()
    assert all(val >= lb)
    assert all(val <= ub)
    val = dist.rsample()
    assert all(val >= lb)
    assert all(val <= ub)
    ll = dist.log_prob(val)
    assert all(torch.isclose(ll, expected_ll)), ll

    cdf = dist.cdf(torch.tensor([0., 0., 0.]))
    assert all(torch.isclose(cdf, torch.tensor([1., 0., 0.]))), cdf
    cdf = dist.cdf(torch.tensor([0., 1., 1.]))
    assert all(torch.isclose(cdf, torch.tensor([1., 1., 0.5]))), cdf
    cdf = dist.cdf(torch.tensor([0., 1., 2.]))
    assert all(torch.isclose(cdf, torch.tensor([1., 1., 1.]))), cdf

    entropy = dist.entropy()
    expected_entropy = torch.zeros(3)
    expected_entropy[1:] = torch.distributions.Uniform(lb[1:], ub[1:]).entropy()
    assert all(torch.isclose(entropy, expected_entropy)), entropy

    # Make sure it works with different input shapes
    lb = 0.
    ub = 1.
    dist = UniformWithEqualityHandling(lb, ub, validate_args=True)
    val = dist.rsample()
    dist.log_prob(val)
    dist.cdf(val)

    lb = 1.
    ub = 1.
    dist = UniformWithEqualityHandling(lb, ub, validate_args=True)
    val = dist.rsample()
    dist.log_prob(val)
    dist.cdf(val)

    lb = torch.zeros(3, 3)
    ub = torch.ones(3, 3)
    ub[1, 1] = 0.
    dist = UniformWithEqualityHandling(lb, ub, validate_args=True)
    val = dist.rsample()
    dist.log_prob(val)
    dist.cdf(val)




def test_left_sided_constraint():
    constraint = LeftSidedConstraint()
    assert constraint.check(torch.zeros(0))

    assert constraint.check(torch.zeros(1))
    assert constraint.check(torch.zeros(2))
    assert constraint.check(torch.zeros(10))

    assert constraint.check(torch.ones(1))
    assert constraint.check(torch.ones(2))
    assert constraint.check(torch.ones(10))


    assert constraint.check(torch.cat([torch.ones(1), torch.zeros(1)]))
    assert constraint.check(torch.cat([torch.ones(5), torch.zeros(5)]))
    assert constraint.check(torch.cat([torch.ones(9), torch.zeros(1)]))
    
    assert not constraint.check(torch.cat([torch.zeros(1), torch.ones(9)]))
    assert not constraint.check(torch.cat([torch.zeros(5), torch.ones(5)]))
    assert not constraint.check(torch.cat([torch.zeros(9), torch.ones(1)]))

def test_VectorCappedGeometricDist():
    p = 0.5
    k = 15
    dist = VectorCappedGeometricDist(geometric_prob=p, max_repeats=k, validate_args=True)
    vec = dist.sample()
    assert vec.shape == (k,)

    test_vec = torch.zeros(k).int()
    ll = dist.log_prob(test_vec).item()
    target_ll = np.log(p)
    assert np.allclose(ll, target_ll), "ll %f vs %f" % (ll, target_ll) 

    for stop_k in range(k+1):
        test_vec = torch.zeros(k).int()
        test_vec[:stop_k] = 1
        ll = dist.log_prob(test_vec).item()
        target_p = (1. - p)**stop_k * p
        if stop_k == k:
            print(np.log(target_p))
            target_p += (1. - p)**(k + 1)
            print(" changed to ", np.log(target_p))
        target_ll = np.log(target_p)
        assert np.allclose(ll, target_ll), "ll %f vs %f at %d" % (ll, target_ll, stop_k) 

def test_LeftSidedRepeatingOnesDist():
    N = 5
    weights = torch.arange(0, N+1) + 1.
    weights /= weights.sum()
    print(weights)
    dist = LeftSidedRepeatingOnesDist(categorical_probs=weights, validate_args=True)
    vec = dist.sample()
    assert vec.shape == (N,)

    for k in range(N+1):
        test_vec = torch.zeros(N).int()
        test_vec[:k] = 1
        ll = dist.log_prob(test_vec).item()
        target_ll = torch.log(weights[k])
        assert np.allclose(ll, target_ll), "ll %f vs %f" % (ll, target_ll)

def test_Bingham(set_seed):
    param_m = torch.eye(4)
    param_z = torch.tensor([-100., -10., -1., 0.])
    dist = BinghamDistribution(param_m, param_z)
    samples = dist.sample(sample_shape=(100,))
    assert samples.shape == (100, 4)
    sample_prob = dist.log_prob(samples)
    assert all(torch.isfinite(sample_prob)) and sample_prob.shape == (100,)

if __name__ == "__main__":
    pytest.main()
