import pytest

import torch
import pyro.poutine
from pydrake.all import SnoptSolver

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.sampling import *

from .grammar import *

from torch.distributions import constraints


torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=[0, 1])
def set_seed(request):
    torch.manual_seed(request.param)

# Proof-of-life of sampling routines
@pytest.mark.skipif(not SnoptSolver().available(),
                    reason='This test relies on Gurobi and SNOPT.')
@pytest.mark.parametrize("perturb_in_config_space", [True, False])
@pytest.mark.parametrize("do_hit_and_run_postprocess", [True, False])
def test_grammar(set_seed, perturb_in_config_space, do_hit_and_run_postprocess):
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4)
    )
    tree = grammar.sample_tree()

    sampled_trees = do_fixed_structure_mcmc(
        grammar, tree, num_samples=5,
        perturb_in_config_space=perturb_in_config_space, verbose=2,
        vis_callback=None,
        translation_variance=0.1,
        rotation_variance=0.1,
        do_hit_and_run_postprocess=do_hit_and_run_postprocess)

    assert len(sampled_trees) == 5
    new_tree = sampled_trees[-1]
    # Should be true that the last tree isn't exactly the same as the
    # first tree.
    has_difference = False
    for new_node, old_node in zip(new_tree.nodes, tree.nodes):
        if not (torch.allclose(new_node.translation, old_node.translation) and
                torch.allclose(new_node.rotation, old_node.rotation)):
            has_difference = True
    assert has_difference, "Sampling did not produce a unique / different tree."
