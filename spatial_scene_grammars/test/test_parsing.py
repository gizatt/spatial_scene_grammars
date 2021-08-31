import pytest

import torch

from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.parsing import *

from .grammar import *

from pydrake.all import SnoptSolver

torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=range(3))
def set_seed(request):
    torch.manual_seed(request.param)

def assert_feasible(optimized_tree):
    # Assert optimized tree structure is feasible.
    score = optimized_tree.score(verbose=True)
    assert torch.isfinite(score), "Infeasible recovered score: %f" % score
    for opt_node in optimized_tree.nodes:
        # Expect failure due to imprecise constraint, but how close?
        R = opt_node.rotation
        t = opt_node.translation
        RtR = torch.matmul(R, R.transpose(0, 1))
        assert torch.allclose(RtR, torch.eye(3), rtol=1E-4, atol=1E-4), RtR
        det = torch.det(R)
        assert torch.isclose(det, torch.tensor(1.)), det

def assert_explains_observeds(observed_tree, optimized_tree):
    # Assert optimized tree structure explains observed nodes.
    for obs_node in observed_tree.get_observed_nodes():
        closest_dist = torch.tensor(1.)
        for matching_node in optimized_tree.find_nodes_by_type(type(obs_node)):
            R_dist = (matching_node.rotation - obs_node.rotation).abs().sum()
            t_dist = (matching_node.translation - obs_node.translation).abs().sum()
            closest_dist = min(closest_dist, R_dist + t_dist)
        assert torch.isclose(closest_dist, torch.zeros(1), atol=1E-4, rtol=1E-4), "Didn't match observation well enough: err %f" % closest_dist

# Try to parse a trivial subset of our grammar with a
# simple solver. (The solver is very slow with additional
# complexity.)
@pytest.mark.skipif(not SnoptSolver().available(),
                    reason='This test relies on Gurobi and SNOPT.')
def test_parsing_simple(set_seed):
    grammar = SpatialSceneGrammar(
        root_node_type = NodeC,
        root_node_tf = torch.eye(4)
    )
    tree = grammar.sample_tree()
    observed_nodes = tree.get_observed_nodes()

    start_time = time.time()
    inference_results = infer_mle_tree_with_mip(
        grammar, observed_nodes, verbose=True,
        solver="branch_and_bound"
    )
    elapsed = time.time() - start_time
    print("Parsing took %f secs." % elapsed)
    assert inference_results.optim_result[1]

    mip_optimized_tree = get_optimized_tree_from_mip_results(inference_results)
    assert_explains_observeds(tree, mip_optimized_tree), "MIP parsing failed."
    
    start_time = time.time()
    refinement_results = optimize_scene_tree_with_nlp(mip_optimized_tree, verbose=True)
    elapsed = time.time() - start_time
    print("Refinement took %f secs." % elapsed)
    assert refinement_results.optim_result.is_success()
    refined_tree = refinement_results.refined_tree
    assert_explains_observeds(tree, refined_tree), "Refinement failed."
    assert_feasible(refined_tree)


## Try to parse samples from our very simple (but
# still nontrivial, due to rotations and hidden
# nodes) grammar.
@pytest.mark.skipif(os.environ.get('GUROBI_PATH') is None or not SnoptSolver().available(),
                    reason='This test relies on Gurobi and SNOPT.')
def test_parsing_complex(set_seed):
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4)
    )
    tree = grammar.sample_tree()
    observed_nodes = tree.get_observed_nodes()

    start_time = time.time()
    inference_results = infer_mle_tree_with_mip(
        grammar, observed_nodes, verbose=True,
    )
    elapsed = time.time() - start_time
    print("Parsing took %f secs." % elapsed)
    assert inference_results.optim_result.is_success(), "MIP parsing failed."

    mip_optimized_tree = get_optimized_tree_from_mip_results(inference_results)
    assert_explains_observeds(tree, mip_optimized_tree)

    start_time = time.time()
    refinement_results = optimize_scene_tree_with_nlp(mip_optimized_tree, objective="mle", verbose=True)
    elapsed = time.time() - start_time
    print("Refinement took %f secs." % elapsed)
    assert refinement_results.optim_result.is_success(), "Refinement failed."
    refined_tree = refinement_results.refined_tree
    assert_explains_observeds(tree, refined_tree)
    assert_feasible(refined_tree)

    # Check out of get-tree-as-close-to-this-one works too
    projection_Results = optimize_scene_tree_with_nlp(mip_optimized_tree, objective="projection", verbose=True)
    elapsed = time.time() - start_time
    print("Refinement took %f secs." % elapsed)
    assert refinement_results.optim_result.is_success(), "Refinement failed."
    refined_tree = refinement_results.refined_tree
    assert_explains_observeds(tree, refined_tree)
    assert_feasible(refined_tree)