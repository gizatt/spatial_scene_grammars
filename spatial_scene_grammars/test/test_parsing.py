import pytest

import torch

from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.parsing import *

from .grammar import *

torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=range(3))
def set_seed(request):
    torch.manual_seed(request.param)

# Try to parse a trivial subset of our grammar with a
# simple solver. (The solver is very slow with additional
# complexity.)
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

## Try to parse samples from our very simple (but
# still nontrivial, due to rotations and hidden
# nodes) grammar.
@pytest.mark.skipif(os.environ.get('TRAVIS') == 'true', reason='This test relies on Gurobi.')
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
    assert inference_results.optim_result.is_success()