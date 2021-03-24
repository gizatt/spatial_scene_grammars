import pytest

import numpy as np
import torch

from spatial_scene_grammars.distributions import *

torch.set_default_tensor_type(torch.DoubleTensor)

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


if __name__ == "__main__":
    pytest.main()