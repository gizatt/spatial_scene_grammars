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
from spatial_scene_grammars_examples.planar_clusters_gaussians.grammar import *
from spatial_scene_grammars.parsing import *

import meshcat
import meshcat.geometry as meshcat_geom

from pydrake.all import SnoptSolver

def test_sampling():
    vis = meshcat.Visualizer()

    # Draw a random sample from the grammar and visualize it.
    grammar = SpatialSceneGrammar(
        root_node_type = Desk,
        root_node_tf = torch.eye(4)
    )
    torch.random.manual_seed(42)
    tree = grammar.sample_tree()

    assert torch.isfinite(tree.score(verbose=True)), "Sampled tree was infeasible."

    draw_scene_tree_contents_meshcat(tree, zmq_url=vis.window.zmq_url)
    draw_scene_tree_structure_meshcat(tree, zmq_url=vis.window.zmq_url)



@pytest.mark.skipif(os.environ.get('GUROBI_PATH') is None or not SnoptSolver().available(),
                    reason='This test relies on Gurobi and SNOPT.')
def test_parsing():
    # Try to parse an example of this grammar.
    grammar = SpatialSceneGrammar(
        root_node_type = Desk,
        root_node_tf = torch.eye(4)
    )
    torch.random.manual_seed(42)
    observed_tree = grammar.sample_tree(detach=True)
    observed_nodes = observed_tree.get_observed_nodes()

    inference_results = infer_mle_tree_with_mip(
        grammar, observed_nodes, verbose=True,
        max_scene_extent_in_any_dir=10.
    )
    assert inference_results.optim_result.is_success()
    mip_optimized_tree = get_optimized_tree_from_mip_results(inference_results)
    refinement_results = optimize_scene_tree_with_nlp(grammar, mip_optimized_tree, verbose=True,
        max_scene_extent_in_any_dir=10.)
    assert refinement_results.is_success()
    refined_tree = refinement_results.refined_tree
    score = refined_tree.score(verbose=True)
    assert torch.isfinite(score), "Refined tree was infeasible."

if __name__ == "__main__":
    pytest.main()
