import pytest

import torch

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from torch.distributions import constraints

from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix
)


torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=range(10))
def set_seed(request):
    torch.manual_seed(request.param)


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
    trace = pyro.poutine.trace(node.sample_children).get_trace()
    children = trace.nodes["_RETURN"]["value"]
    assert len(children) == 3
    assert all([isinstance(c, DummyType) for c in children])
    score = node.score_child_set(children)
    expected_prob_hand = torch.zeros(1)
    # No sample nodes to get log prob from in this trace since AndNode
    # doesn't sample anything.
    assert torch.isclose(score, expected_prob_hand), "%s vs %s" % (expected_prob_hand, score)

## OrNode
def test_OrNode(set_seed):
    node = OrNode(
        rules=[dummyRule, dummyRule, dummyRule],
        rule_probs=torch.tensor([0.75, 0.2, 0.05]),
        tf=torch.eye(4),
        observed=False,
        physics_geometry_info=None
    )
    trace = pyro.poutine.trace(node.sample_children).get_trace()
    children = trace.nodes["_RETURN"]["value"]
    trace_node = trace.nodes["%s/OrNode_child" % node.name]
    expected_prob = trace_node["fn"].log_prob(trace_node["value"])
    assert len(children) == 1
    assert all([isinstance(c, DummyType) for c in children])
    expected_prob_hand = torch.log(node.rule_probs[children[0].rule_k])
    score = node.score_child_set(children)
    assert torch.isclose(score, expected_prob), "%s vs %s" % (expected_prob, score)
    assert torch.isclose(score, expected_prob_hand), "%s vs %s" % (expected_prob_hand, score)


## GeometricSetNode
def test_GeometricSetNode(set_seed):
    node = GeometricSetNode(
        rule=dummyRule,
        p=0.2,
        max_children=5,
        tf=torch.eye(4),
        observed=False,
        physics_geometry_info=None
    )
    trace = pyro.poutine.trace(node.sample_children).get_trace()
    children = trace.nodes["_RETURN"]["value"]
    trace_node = trace.nodes["%s/GeometricSetNode_n" % node.name]
    expected_prob = trace_node["fn"].log_prob(trace_node["value"])
    assert len(children) <= 5 and len(children) >= 1
    assert all([isinstance(c, DummyType) for c in children])
    score = node.score_child_set(children)
    assert torch.isclose(score, expected_prob), "%s vs %s" % (expected_prob, score)
