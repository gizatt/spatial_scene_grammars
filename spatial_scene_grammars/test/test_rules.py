import pytest

import torch
import pyro.poutine

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from torch.distributions import constraints

from .grammar import *

from pydrake.all import (
    UniformlyRandomRotationMatrix,
    RandomGenerator
)

torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=range(10))
def set_seed(request):
    torch.manual_seed(request.param)
    np.random.seed(request.param)
    return RandomGenerator(request.param)

def make_dummy_node():
    return NodeA(torch.eye(4))

## SamePositionRule
def test_SamePositionRule(set_seed):
    rule = SamePositionRule()

    params = rule.parameters
    assert isinstance(params, dict) and params == {}
    priors = rule.get_parameter_prior()

    parent = make_dummy_node()
    child = make_dummy_node()
    child.translation = rule.sample_xyz(parent)
    ll = rule.score_child(parent, child)
    assert torch.isclose(ll, torch.tensor(0.))
    assert torch.allclose(parent.translation, child.translation)

## WorldBBoxRule
def test_WorldBBoxRule(set_seed):
    lb = torch.zeros(3)
    ub = torch.ones(3)
    rule = WorldBBoxRule.from_bounds(lb, ub)

    params = rule.parameters
    assert isinstance(params, dict)
    priors = rule.get_parameter_prior()
    for k, params in params.items():
        assert all(torch.isfinite(priors[k].log_prob(params)))

    parent = make_dummy_node()

    xyz = rule.sample_xyz(parent)
    assert isinstance(xyz, torch.Tensor)
    assert all(xyz <= ub) and all(xyz >= lb)

    child = make_dummy_node()
    child.translation = xyz
    ll = rule.score_child(parent, child)
    # ll of uniform[0, 1] unit box is log(1) = 0
    assert torch.isclose(ll, torch.tensor(0.))

## AxisAlignedBBoxRule
def test_AxisAlignedBBoxRule(set_seed):
    lb = torch.zeros(3)
    ub = torch.ones(3)
    rule = AxisAlignedBBoxRule.from_bounds(lb, ub)

    params = rule.parameters
    assert isinstance(params, dict)
    priors = rule.get_parameter_prior()
    for k, params in params.items():
        assert all(torch.isfinite(priors[k].log_prob(params)))

    parent = make_dummy_node()
    parent.translation = torch.tensor([1., 1., 1.])

    xyz = rule.sample_xyz(parent)
    assert isinstance(xyz, torch.Tensor)
    assert all(xyz <= parent.translation + ub) and all(xyz >= parent.translation + lb), (xyz, parent.translation, lb, ub)

    child = make_dummy_node()
    child.translation = xyz
    ll = rule.score_child(parent, child)
    # ll of uniform[0, 1] unit box is log(1) = 0
    assert torch.isclose(ll, torch.tensor(0.))

## AxisAlignedBBoxRule
def test_AxisAlignedGaussianOffsetRule(set_seed):
    random_mean = torch.tensor(np.random.normal(0., 1., 3))
    random_covar = torch.tensor(np.random.uniform(np.zeros(3)*0.1, np.ones(3)))
    rule = AxisAlignedGaussianOffsetRule(mean=random_mean, variance=random_covar)

    params = rule.parameters
    assert isinstance(params, dict)
    priors = rule.get_parameter_prior()
    for k, params in params.items():
        assert all(torch.isfinite(priors[k].log_prob(params)))

    parent = make_dummy_node()
    parent.translation = torch.tensor([1., 1., 1.])

    xyz = rule.sample_xyz(parent)
    assert isinstance(xyz, torch.Tensor)

    child = make_dummy_node()
    child.translation = xyz
    ll = rule.score_child(parent, child)

## SameRotationRule
def test_SameRotationRule(set_seed):
    rule = SameRotationRule()

    params = rule.parameters
    assert isinstance(params, dict) and params == {}
    priors = rule.get_parameter_prior()

    parent = make_dummy_node()
    child = make_dummy_node()
    child.rotation = rule.sample_rotation(parent)

    ll = rule.score_child(parent, child)
    assert torch.isclose(ll, torch.tensor(0.))
    assert torch.allclose(parent.translation, child.translation)

## UnconstrainedRotationRule
def test_UnconstrainedRotationRule(set_seed):
    rule = UnconstrainedRotationRule()
    parent = make_dummy_node()

    params = rule.parameters
    assert params == {}
    assert rule.get_parameter_prior() == {}

    R = rule.sample_rotation(parent)
    assert isinstance(R, torch.Tensor)
    assert R.shape == (3, 3)
    # Is it a rotation?
    assert torch.allclose(torch.matmul(R, torch.transpose(R, 0, 1)), torch.eye(3))
    assert torch.isclose(torch.det(R), torch.ones(1))

    child = make_dummy_node()
    child.rotation = R
    ll = rule.score_child(parent, child)

## UniformBoundedRevoluteJointRule
def test_UniformBoundedRevoluteJointRule(set_seed):
    random_axis = np.random.normal(0., 1., 3)
    random_axis = torch.tensor(random_axis / np.linalg.norm(random_axis))
    rule = UniformBoundedRevoluteJointRule.from_bounds(axis=random_axis, lb=-np.pi+1E-3, ub=np.pi-1E-3)

    params = rule.parameters
    assert isinstance(params, dict)
    priors = rule.get_parameter_prior()
    for k, params in params.items():
        assert all(torch.isfinite(priors[k].log_prob(params)))

    parent = make_dummy_node()

    R = rule.sample_rotation(parent)
    assert isinstance(R, torch.Tensor)
    assert R.shape == (3, 3)
    # Is it a rotation?
    assert torch.allclose(torch.matmul(R, torch.transpose(R, 0, 1)), torch.eye(3))
    assert torch.isclose(torch.det(R), torch.ones(1))

    child = make_dummy_node()
    child.rotation = R
    ll = rule.score_child(parent, child)

## UniformBoundedRevoluteJointRule
def test_GaussianChordOffsetRule(set_seed):
    random_axis = np.random.normal(0., 1., 3)
    random_axis = torch.tensor(random_axis / np.linalg.norm(random_axis))
    random_loc = np.random.normal(0., 1.)
    random_concentration = np.random.uniform(0.01, 1.)
    rule = GaussianChordOffsetRule(axis=random_axis, loc=random_loc, concentration=random_concentration)

    params = rule.parameters
    assert isinstance(params, dict)
    priors = rule.get_parameter_prior()
    for k, params in params.items():
        assert all(torch.isfinite(priors[k].log_prob(params)))

    parent = make_dummy_node()

    R = rule.sample_rotation(parent)
    assert isinstance(R, torch.Tensor)
    assert R.shape == (3, 3)
    # Is it a rotation?
    assert torch.allclose(torch.matmul(R, torch.transpose(R, 0, 1)), torch.eye(3))
    assert torch.isclose(torch.det(R), torch.ones(1))

    child = make_dummy_node()
    child.rotation = R
    ll = rule.score_child(parent, child)

@pytest.mark.parametrize("angle", torch.arange(start=-np.pi+1E-3, end=np.pi-1E-3, step=np.pi/8.))
def test_AngleAxisInversion(set_seed, angle):
    # Test angle-axis inversion
    random_axis = np.random.normal(0., 1., 3)
    random_axis = torch.tensor(random_axis / np.linalg.norm(random_axis))
    rule = UniformBoundedRevoluteJointRule.from_bounds(axis=random_axis, lb=-np.pi+1E-3, ub=np.pi-1E-3)
    parent = make_dummy_node()
    parent.rotation = torch.tensor(UniformlyRandomRotationMatrix(set_seed).matrix())

    # Make child with known offset
    # TODO could grab this from inside the rule evaluation
    angle_axis = random_axis * angle
    R_offset = axis_angle_to_matrix(angle_axis.unsqueeze(0))[0, ...]
    R = torch.matmul(parent.rotation, R_offset)
    child = make_dummy_node()
    child.rotation = R

    recovered_angle, recovered_axis = recover_relative_angle_axis(parent, child, target_axis=rule.axis, zero_angle_width=1E-4)
    assert torch.allclose(recovered_angle, angle)
    if torch.abs(angle) > 0:
        assert torch.allclose(recovered_axis, random_axis, atol=1E-4, rtol=1E-4)

@pytest.mark.parametrize("xyz_rule", [
    WorldBBoxRule.from_bounds(lb=torch.zeros(3), ub=torch.ones(3)*3.),
    AxisAlignedBBoxRule.from_bounds(lb=torch.zeros(3), ub=torch.ones(3)*5.)
])
@pytest.mark.parametrize("rotation_rule", [
    UnconstrainedRotationRule(),
    UniformBoundedRevoluteJointRule.from_bounds(axis=torch.tensor([0., 1., 0.]), lb=-np.pi/2., ub=np.pi/2.)
])
def test_ProductionRule(set_seed, xyz_rule, rotation_rule):
    rule = ProductionRule(
        child_type=NodeB,
        xyz_rule=xyz_rule,
        rotation_rule=rotation_rule
    )
    parent = make_dummy_node()
    child = rule.sample_child(parent)
    rule.score_child(parent, child)

    priors = rule.get_parameter_prior()
    assert isinstance(priors, tuple) and len(priors) == 2
    params = rule.parameters
    assert isinstance(params, tuple) and len(params) == 2
    for prior_set, param_set in zip(priors, params):
        for k in prior_set.keys():
            assert all(torch.isfinite(prior_set[k].log_prob(param_set[k])))
    rule.parameters = params

    trace = pyro.poutine.trace(rule.sample_child).get_trace(parent)
    expected = trace.log_prob_sum()
    child = trace.nodes["_RETURN"]["value"]
    score = rule.score_child(parent, child, verbose=True).sum()
    assert torch.isclose(score, expected), "%f vs %f for rule types %s, %s" % (
        score, expected, xyz_rule, rotation_rule)


if __name__ == "__main__":
    pytest.main()