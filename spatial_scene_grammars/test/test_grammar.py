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

    assert isinstance(tree, SceneTree)

    root = tree.get_root()
    assert isinstance(root, NodeA)
    assert len(list(tree.predecessors(root))) == 0
    assert len(list(tree.successors(root))) == 2 # AND rule with 2 children

    obs = tree.get_observed_nodes()
    if len(obs) > 0:
        assert all([isinstance(c, (NodeD, NodeE, NodeF)) for c in obs])

    assert len(tree.find_nodes_by_type(NodeA)) == 1