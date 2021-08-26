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
from spatial_scene_grammars_examples.oriented_clusters.grammar import *

import meshcat
import meshcat.geometry as meshcat_geom

def test_sampling():
    vis = meshcat.Visualizer()

    # Draw a random sample from the grammar and visualize it.
    grammar = SpatialSceneGrammar(
        root_node_type = OrientedCluster,
        root_node_tf = torch.eye(4)
    )
    torch.random.manual_seed(42)
    tree = grammar.sample_tree()
    draw_scene_tree_contents_meshcat(tree, zmq_url=vis.window.zmq_url)
    draw_scene_tree_structure_meshcat(tree, zmq_url=vis.window.zmq_url)


if __name__ == "__main__":
    pytest.main()
