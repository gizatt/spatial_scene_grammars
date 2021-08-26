import pytest

import torch

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from torch.distributions import constraints

from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix
)


torch.set_default_tensor_type(torch.DoubleTensor)

class DummyType(Node):
    def __init__(self, tf):
        super().__init__(observed=False, physics_geometry_info=None, tf=tf)
dummyRule = ProductionRule(
    child_type=DummyType,
    xyz_rule=WorldBBoxRule(lb=torch.zeros(3), ub=torch.ones(3)),
    rotation_rule=UnconstrainedRotationRule()
)


## Base Node type
def test_Node():
    node = Node(
        tf=torch.eye(4),
        observed=False,
        physics_geometry_info=None,
        do_sanity_checks=True
    )
    xyz = node.translation
    assert isinstance(xyz, torch.Tensor) and xyz.shape == (3,)
    R = node.rotation
    assert isinstance(R, torch.Tensor) and R.shape == (3, 3)
    new_xyz = torch.tensor([1., 2., 3.])
    new_R = axis_angle_to_matrix(torch.tensor([0.5, 0.5, 0.5]).unsqueeze(0))[0, ...]
    node.translation = new_xyz
    node.rotation = new_R
    xyz = node.translation
    R = node.rotation
    assert isinstance(xyz, torch.Tensor) and xyz.shape == (3,) and torch.allclose(new_xyz, xyz)
    R = node.rotation
    assert isinstance(R, torch.Tensor) and R.shape == (3, 3) and torch.allclose(R, new_R)

## TerminalNode
def test_TerminalNode():
    node = TerminalNode(
        tf=torch.eye(4),
        observed=False,
        physics_geometry_info=None,
        do_sanity_checks=True
    )
    assert node.sample_children() == []

## AndNode
def test_AndNode():
    node = AndNode(
        rules=[dummyRule, dummyRule, dummyRule],
        tf=torch.eye(4),
        observed=False,
        physics_geometry_info=None
    )
    children = node.sample_children()
    assert len(children) == 3
    assert all([isinstance(c, DummyType) for c in children])

## OrNode
def test_OrNode():
    node = OrNode(
        rules=[dummyRule, dummyRule, dummyRule],
        rule_probs=torch.tensor([0.75, 0.2, 0.05]),
        tf=torch.eye(4),
        observed=False,
        physics_geometry_info=None
    )
    children = node.sample_children()
    assert len(children) == 1
    assert all([isinstance(c, DummyType) for c in children])

## GeometricSetNode
def test_GeometricSetNode():
    node = GeometricSetNode(
        rule=dummyRule,
        p=0.2,
        max_children=5,
        tf=torch.eye(4),
        observed=False,
        physics_geometry_info=None
    )
    children = node.sample_children()
    assert len(children) <= 5 and len(children) >= 1
    assert all([isinstance(c, DummyType) for c in children])
