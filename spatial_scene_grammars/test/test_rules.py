import pytest

import torch

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from torch.distributions import constraints

torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=range(10))
def set_seed(request):
    torch.manual_seed(request.param)
    np.random.seed(request.param)

def make_dummy_node():
    return Node(
        tf=torch.eye(4),
        observed=False,
        physics_geometry_info=None,
        do_sanity_checks=True
    )

## WorldBBoxRule
def test_WorldBBoxRule(set_seed):
    lb = torch.zeros(3)
    ub = torch.ones(3)
    rule = WorldBBoxRule(lb, ub)
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
    rule = AxisAlignedBBoxRule(lb, ub)
    parent = make_dummy_node()
    parent.translation = torch.tensor([1., 1., 1.])

    xyz = rule.sample_xyz(parent)
    assert isinstance(xyz, torch.Tensor)
    assert all(xyz <= parent.translation + ub) and all(xyz >= parent.translation + lb)

    child = make_dummy_node()
    child.translation = xyz
    ll = rule.score_child(parent, child)
    # ll of uniform[0, 1] unit box is log(1) = 0
    assert torch.isclose(ll, torch.tensor(0.))

## UnconstrainedRotationRule
def test_UnconstrainedRotationRule(set_seed):
    rule = UnconstrainedRotationRule()
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
def test_UniformBoundedRevoluteJointRule(set_seed):
    random_axis = np.random.normal(0., 1., 3)
    random_axis = torch.tensor(random_axis / np.linalg.norm(random_axis))
    rule = UniformBoundedRevoluteJointRule(axis=random_axis, lb=-np.pi+1E-3, ub=np.pi-1E-3)
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
    rule = UniformBoundedRevoluteJointRule(axis=random_axis, lb=-np.pi+1E-3, ub=np.pi-1E-3)
    parent = make_dummy_node()

    # Make child with known offset
    # TODO could grab this from inside the rule evaluation
    angle_axis = random_axis * angle
    R_offset = axis_angle_to_matrix(angle_axis.unsqueeze(0))[0, ...]
    R = torch.matmul(parent.rotation, R_offset)
    child = make_dummy_node()
    child.rotation = R

    recovered_angle, recovered_axis = rule._recover_relative_angle_axis(parent, child)
    assert torch.allclose(recovered_angle, angle)
    if torch.abs(angle) > 0:
        assert torch.allclose(recovered_axis, random_axis, atol=1E-4, rtol=1E-4)

def test_ProductionRule(set_seed):
    class DummyType(Node):
        def __init__(self, tf):
            super().__init__(observed=False, physics_geometry_info=None, tf=tf)

    rule = ProductionRule(
        child_type=DummyType,
        xyz_rule=WorldBBoxRule(lb=torch.zeros(3), ub=torch.ones(3)),
        rotation_rule=UnconstrainedRotationRule()
    )
    parent = make_dummy_node()
    child = rule.sample_child(parent)
    rule.score_child(parent, child)


if __name__ == "__main__":
    pytest.main()