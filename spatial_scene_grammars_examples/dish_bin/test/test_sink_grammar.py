import pytest

import networkx as nx
import numpy as np
import os
import pickle
import time

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import torch
torch.set_default_tensor_type(torch.DoubleTensor)

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars_examples.dish_bin.grammar import *
from spatial_scene_grammars_examples.dish_bin.grammar_baseline import *
from spatial_scene_grammars_examples.dish_bin.utils import get_observed_node_sets
from spatial_scene_grammars.parsing import *
from spatial_scene_grammars.sampling import *
from spatial_scene_grammars.parameter_estimation import *
from spatial_scene_grammars.dataset import *

import meshcat
import meshcat.geometry as meshcat_geom

from pydrake.all import SnoptSolver

@pytest.fixture(params=range(3))
def set_seed(request):
    pyro.set_rng_seed(request.param)

def test_sampling(set_seed):
    vis = meshcat.Visualizer()

    # Draw a random sample from the grammar and visualize it.
    grammar = SpatialSceneGrammar(
        root_node_type = DishBin,
        root_node_tf = torch.eye(4)
    )
    torch.random.manual_seed(42)
    tree = grammar.sample_tree()

    assert torch.isfinite(tree.score(verbose=True)), "Sampled tree was infeasible."

    draw_scene_tree_contents_meshcat(tree, zmq_url=vis.window.zmq_url)
    draw_scene_tree_structure_meshcat(tree, zmq_url=vis.window.zmq_url)


def test_sampling_baseline(set_seed):
    vis = meshcat.Visualizer()

    # Draw a random sample from the grammar and visualize it.
    grammar = SpatialSceneGrammar(
        root_node_type = DishBinBaseline,
        root_node_tf = torch.eye(4)
    )
    torch.random.manual_seed(42)
    tree = grammar.sample_tree()

    assert torch.isfinite(tree.score(verbose=True)), "Sampled tree was infeasible."


@pytest.mark.skipif(os.environ.get('GUROBI_PATH') is None or not SnoptSolver().available(),
                    reason='This test relies on Gurobi and SNOPT.')
def test_parsing_mip(set_seed):
    # Try to parse an example of this grammar.
    grammar = SpatialSceneGrammar(
        root_node_type = DishBin,
        root_node_tf = torch.eye(4)
    )
    torch.random.manual_seed(42)
    observed_tree = grammar.sample_tree(detach=True)
    observed_nodes = observed_tree.get_observed_nodes()

    inference_results = infer_mle_tree_with_mip(
        grammar, observed_nodes, verbose=True,
        max_scene_extent_in_any_dir=10.
    )
    mip_optimized_tree = get_optimized_tree_from_mip_results(inference_results)
    refinement_results = optimize_scene_tree_with_nlp(grammar, mip_optimized_tree, verbose=True,
        max_scene_extent_in_any_dir=10.)
    refined_tree = refinement_results.refined_tree
    score = refined_tree.score(verbose=True)
    assert torch.isfinite(score), "Refined tree was infeasible."


@pytest.mark.skipif(os.environ.get('GUROBI_PATH') is None or not SnoptSolver().available(),
                    reason='This test relies on Gurobi and SNOPT.')
def test_parsing_ip(set_seed):
    # Try to parse an example of this grammar.
    grammar = SpatialSceneGrammar(
        root_node_type = DishBin,
        root_node_tf = torch.eye(4)
    )
    torch.random.manual_seed(42)
    observed_tree = grammar.sample_tree(detach=True)
    observed_nodes = observed_tree.get_observed_nodes()

    N_solutions = 3
    parse_trees = infer_mle_tree_with_mip_from_proposals(
        grammar, observed_nodes, {}, verbose=False, N_solutions=N_solutions,
        min_ll_for_consideration=-1000.
    )
    assert len(parse_trees) > 0 and len(parse_trees) <= N_solutions
    refinement_results = optimize_scene_tree_with_nlp(grammar, parse_trees[0], verbose=True,
        max_scene_extent_in_any_dir=10.)
    refined_tree = refinement_results.refined_tree
    score = refined_tree.score(verbose=True)
    assert torch.isfinite(score), "Refined tree was infeasible."


if __name__ == "__main__":
    pytest.main()