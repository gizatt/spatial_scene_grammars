import pytest

import torch

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from torch.distributions import constraints

from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix
)

from .grammar import *

torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=range(10))
def set_seed(request):
    torch.manual_seed(request.param)

## Base Node type
def test_Node():
    node = NodeA(tf=torch.eye(4))
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
    node = NodeA(tf=torch.eye(4))
    trace = pyro.poutine.trace(node.sample_children).get_trace()
    children = trace.nodes["_RETURN"]["value"]
    score = node.score_child_set(children)
    expected_prob_hand = torch.zeros(1)
    # No sample nodes to get log prob from in this trace since AndNode
    # doesn't sample anything.
    assert torch.isclose(score, expected_prob_hand), "%s vs %s" % (expected_prob_hand, score)
    assert len(node.parameters) == 0

## OrNode
def test_OrNode(set_seed):
    node = NodeB(tf=torch.eye(4))
    trace = pyro.poutine.trace(node.sample_children).get_trace()
    children = trace.nodes["_RETURN"]["value"]
    trace_node = trace.nodes["OrNode_child"]
    expected_prob = trace_node["fn"].log_prob(trace_node["value"])
    expected_prob_hand = torch.log(node.rule_probs[children[0].rule_k])
    score = node.score_child_set(children)
    assert torch.isclose(score, expected_prob), "%s vs %s" % (expected_prob, score)
    assert torch.isclose(score, expected_prob_hand), "%s vs %s" % (expected_prob_hand, score)
    assert torch.allclose(node.parameters, node.rule_probs)


## RepeatingSetNode
def test_RepeatingSetNode(set_seed):
    node = NodeC(tf=torch.eye(4))
    trace = pyro.poutine.trace(node.sample_children).get_trace()
    children = trace.nodes["_RETURN"]["value"]
    trace_node = trace.nodes["RepeatingSetNode_n"]
    expected_prob = trace_node["fn"].log_prob(trace_node["value"])
    assert len(children) <= node.max_children and len(children) >= 1
    score = node.score_child_set(children)
    assert torch.isclose(score, expected_prob), "%s vs %s" % (expected_prob, score)
    assert torch.allclose(node.parameters, node.rule_probs)

## IndependentSetNode
def test_IndependentSetNode(set_seed):
    node = NodeD(tf=torch.eye(4))
    trace = pyro.poutine.trace(node.sample_children).get_trace()
    children = trace.nodes["_RETURN"]["value"]
    trace_node = trace.nodes["IndependentSetNode_n"]
    expected_prob = trace_node["fn"].log_prob(trace_node["value"]).sum()
    score = node.score_child_set(children)
    assert torch.isclose(score, expected_prob), "%s vs %s" % (expected_prob, score)
    assert torch.allclose(node.parameters, node.rule_probs)
