import pytest

import torch
import pyro.poutine

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from torch.distributions import constraints
import pyro.distributions as dist

from .grammar import *

from pydrake.all import (
    UniformlyRandomRotationMatrix,
    RandomGenerator,
    RigidTransform
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
    offset = dist.Normal(torch.zeros(3), torch.ones(3)).sample()
    rule = SamePositionRule(offset=offset)

    params = rule.parameters
    assert isinstance(params, dict) and params == {}
    priors = rule.get_parameter_prior()

    parent = make_dummy_node()
    child = make_dummy_node()
    child.translation = rule.sample_xyz(parent)
    ll = rule.score_child(parent, child)
    assert torch.isclose(ll, torch.tensor(0.))
    assert torch.allclose(parent.translation + offset, child.translation)
    assert torch.allclose(parent.rotation, child.rotation)

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

## WorldFramePlanarGaussianOffsetRule
def test_WorldFramePlanarGaussianOffsetRule(set_seed):
    random_mean = torch.tensor(np.random.normal(0., 1., 2))
    random_covar = torch.tensor(np.random.uniform(np.zeros(2)*0.1, np.ones(2)))
    random_plane_transform = RigidTransform(
        p=np.random.normal(0., 1., 3),
        R=UniformlyRandomRotationMatrix(set_seed)
    )
    rule = WorldFramePlanarGaussianOffsetRule(
        mean=random_mean, variance=random_covar,
        plane_transform=random_plane_transform
    )

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
    offset = torch.tensor(UniformlyRandomRotationMatrix(set_seed).matrix(), dtype=torch.double)
    rule = SameRotationRule(offset=offset)

    params = rule.parameters
    assert isinstance(params, dict) and params == {}
    priors = rule.get_parameter_prior()

    parent = make_dummy_node()
    child = make_dummy_node()
    child.rotation = rule.sample_rotation(parent)

    ll = rule.score_child(parent, child)
    assert torch.isclose(ll, torch.tensor(0.))
    assert torch.allclose(parent.translation, child.translation)
    assert torch.allclose(torch.matmul(parent.rotation, offset), child.rotation)

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
    random_loc = np.random.uniform(0., 2.*np.pi)
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


def test_WorldFrameBinghamRotationRule(set_seed):
    # Construct from random quaternion mode + tight RPY variances
    mode_R = UniformlyRandomRotationMatrix(set_seed)
    rpy_concentration = np.random.uniform([100, 100, 100], [1000, 1000., 1000.])
    rule = WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
        mode_R, rpy_concentration
    )

    parent = make_dummy_node()

    sampled_Rs = [
        RotationMatrix(
            rule.sample_rotation(parent).detach().numpy()
        ) for k in range(100)
    ]

    # Do a fast-and-dirty check that those rotations are close to the mean: rotate
    # the +x vector, and make sure the dist of rotated vecs has mean close to the
    # mode.
    px = np.array([1., 0., 0.])
    rotated_pxs = [R.multiply(px) for R in sampled_Rs]
    mean_rotated_px = np.mean(np.stack(rotated_pxs, axis=0), axis=0)
    expected_rotated_px = mode_R.multiply(px)
    err = np.abs(mean_rotated_px - expected_rotated_px)
    assert all(err < 0.25), (mean_rotated_px, expected_rotated_px)




@pytest.mark.parametrize("xyz_rule", [
    WorldBBoxRule.from_bounds(lb=torch.zeros(3), ub=torch.ones(3)*3.),
    AxisAlignedBBoxRule.from_bounds(lb=torch.zeros(3), ub=torch.ones(3)*5.),
    AxisAlignedGaussianOffsetRule(mean=torch.zeros(3), variance=torch.ones(3)),
    ParentFrameGaussianOffsetRule(mean=torch.zeros(3), variance=torch.ones(3)),
    WorldFramePlanarGaussianOffsetRule(
        mean=torch.zeros(2), variance=torch.ones(2), plane_transform=RigidTransform(p=np.array([1., 2., 3.]), rpy=RollPitchYaw(1., 2., 3.)))
])
@pytest.mark.parametrize("rotation_rule", [
    SameRotationRule(),
    UnconstrainedRotationRule(),
    UniformBoundedRevoluteJointRule.from_bounds(axis=torch.tensor([0., 1., 0.]), lb=-np.pi/2., ub=np.pi/2.),
    GaussianChordOffsetRule(axis=torch.tensor([0., 0., 1.]), loc=0.42, concentration=11.),
    WorldFrameBinghamRotationRule(M=torch.eye(4), Z=torch.tensor([-1., -1., -1., 0.])),
    ParentFrameBinghamRotationRule(M=torch.eye(4), Z=torch.tensor([-1., -1., -1., 0.]))
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
            assert all(torch.isfinite(prior_set[k].log_prob(param_set[k])).flatten())
    rule.parameters = params

    trace = pyro.poutine.trace(rule.sample_child).get_trace(parent)
    expected = trace.log_prob_sum()
    child = trace.nodes["_RETURN"]["value"]
    score = rule.score_child(parent, child, verbose=True).sum()
    assert torch.isclose(score, expected), "%f vs %f for rule types %s, %s" % (
        score, expected, xyz_rule, rotation_rule)

    # Reported site values should be exactly match with actual site values.
    reported_site_values = rule.get_site_values(parent, child)
    for key, site_value in reported_site_values.items():
        full_key = "%s/%s" % (child.name, key)
        assert full_key in trace.nodes.keys()
        trace_val = trace.nodes[full_key]["value"]
        assert torch.allclose(trace_val, site_value.value), (trace_val, site_value.value)
    for key, site in trace.nodes.items():
        if trace.nodes[key]["type"] == "sample":
            subkey = key.split("/")[-1]
            assert subkey in reported_site_values.keys()
            assert torch.allclose(site["value"], reported_site_values[subkey].value), (site["value"], reported_site_values[subkey].value)
            assert site["fn"] is reported_site_values[subkey].fn

if __name__ == "__main__":
    pytest.main()