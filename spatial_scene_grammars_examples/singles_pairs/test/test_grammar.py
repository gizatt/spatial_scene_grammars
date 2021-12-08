import pytest

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import time

import torch
torch.set_default_tensor_type(torch.DoubleTensor)

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars.parsing import *

import meshcat
import meshcat.geometry as meshcat_geom

from pydrake.all import SnoptSolver

import spatial_scene_grammars_examples.singles_pairs.grammar_dependency as grammar_dependency
import spatial_scene_grammars_examples.singles_pairs.grammar_constituency as grammar_constituency

@pytest.fixture(params=range(3))
def set_seed(request):
    torch.manual_seed(request.param)

@pytest.mark.parametrize("grammar_library",
    [grammar_dependency, grammar_constituency]
)
def test_sampling(set_seed, grammar_library):
    # Draw a random sample from the grammar and visualize it.
    grammar = SpatialSceneGrammar(
        root_node_type = grammar_library.Root,
        root_node_tf = torch.eye(4)
    )
    torch.random.manual_seed(42)
    tree = grammar.sample_tree()

    assert torch.isfinite(tree.score(verbose=True)), "Sampled tree was infeasible."

if __name__ == "__main__":
    pytest.main()
