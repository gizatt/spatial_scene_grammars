import pytest

import torch

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *

from .grammar import *

from torch.distributions import constraints

from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix
)

torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=range(10))
def set_seed(request):
    torch.manual_seed(request.param)


## Basic grammar and tree functionality
def test_grammar(set_seed):
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4)
    )
    tree = grammar.sample_tree()

    assert(torch.isfinite(tree.score()))

    assert isinstance(tree, SceneTree)

    root = tree.get_root()
    assert isinstance(root, NodeA)
    assert len(list(tree.predecessors(root))) == 0
    assert len(list(tree.successors(root))) == 2 # AND rule with 2 children

    obs = tree.get_observed_nodes()
    if len(obs) > 0:
        assert all([isinstance(c, (NodeD, NodeE, NodeF)) for c in obs])

    assert len(tree.find_nodes_by_type(NodeA)) == 1

def test_supertree():
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4)
    )
    # excessive recursion depth for this grammar
    super_tree = grammar.make_super_tree(max_recursion_depth=20)
    # Some basic, necessary-but-not-sufficient checks that the
    # tree isn't obviously wrong.
    assert isinstance(super_tree.get_root(), NodeA)
    C = super_tree.find_nodes_by_type(NodeC)[0]
    assert len(super_tree.find_nodes_by_type(NodeF)) == C.max_children
    Bs = super_tree.find_nodes_by_type(NodeB)
    assert len(Bs) == 1
    B = Bs[0]
    assert len(list(super_tree.successors(B))) == 2 # Takes both options

    # Test that recursion depth takes effect by making it too small
    super_tree = grammar.make_super_tree(max_recursion_depth=1)
    assert len(super_tree.find_nodes_by_type(NodeA)) == 1
    assert len(super_tree.find_nodes_by_type(NodeC)) == 1
    assert len(super_tree.find_nodes_by_type(NodeF)) == 0
    