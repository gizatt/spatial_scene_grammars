import unittest
import pytest

from pydrake.all import (
    PackageMap
)

import numpy as np
import torch
import pyro
import pyro.distributions as dist

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.tree import *
from spatial_scene_grammars.neural_grammar_proposal import *

from spatial_scene_grammars.test.grammar import *

torch.set_default_tensor_type(torch.DoubleTensor)
pyro.enable_validation(True)

root_node_type = Building
inst_dict = {"xy": dist.Delta(torch.zeros(2))}

@pytest.fixture(params=range(10))
def set_seed(request):
    pyro.clear_param_store()
    torch.manual_seed(request.param)


def test_estimate_observation_likelihood(set_seed):
    grammar = SceneGrammar(root_node_type)
    generated_tree = grammar(inst_dict)
    observed_nodes = [n for n in generated_tree.nodes() if isinstance(n, TerminalNode)]

    # Error with itself should be zero. (Error is log-prob sum of likelihood
    # functions, which with this variance peak at 1. -> ll = 0.)
    error = estimate_observation_likelihood(observed_nodes, observed_nodes, 1./(np.sqrt(2 * np.pi)))
    assert np.allclose(error.item(), 0.)

def test_node_embedding():
    test_embedding = NodeEmbedding(root_node_type.init_with_default_parameters(), output_size=64)
    test_building = root_node_type.init_with_default_parameters()
    test_building.instantiate({"xy": dist.Delta(torch.zeros(2))})
    out = test_embedding(test_building.get_all_continuous_variables_as_vector())
    assert torch.all(torch.isfinite(out))

def test_grammar_encoder(set_seed):
    grammar = SceneGrammar(root_node_type)
    generated_tree = grammar(inst_dict)
    observed_nodes = [n for n in generated_tree.nodes() if isinstance(n, TerminalNode)]
    meta_tree = SceneGrammar.make_meta_scene_tree(root_node_type)

    # Can we encode?
    encoder = GrammarEncoder(meta_tree, embedding_size=64)
    x = encoder(observed_nodes)
    assert torch.all(torch.isfinite(x))

    # Can we recover parameters that would reproduce a given tree?
    # Make sure we can directly encode trees for supervision correctly.
    x_enc = encoder.get_grammar_parameters_from_actual_tree(meta_tree, generated_tree, assign_var=0.00001)
    sampled_tree, ll, _ = encoder.sample_tree_from_grammar_vector(meta_tree, x_enc, inst_dict)

    # While I can't guarantee the trees have the same variable ordering, some high-order
    # stats should agree: the value of each continuous variable in the original tree
    # should show up in the sampled tree, and the # of nodes should match.
    assert len(sampled_tree.nodes) == len(generated_tree.nodes)
    v1 = torch.cat([node.get_all_continuous_variables_as_vector() for node in generated_tree])
    v2 = torch.cat([node.get_all_continuous_variables_as_vector() for node in sampled_tree])
    print(v1, v2)
    assert v1.shape == v2.shape
    v1_batch = v1.repeat(len(v2), 1)
    v2_batch = v2.repeat(len(v1), 1).T
    all_to_all = torch.abs(v1_batch - v2_batch)
    # V2 will be normally distributed very close to the value of v1, but not exactly.
    assert torch.isclose(min(torch.min(all_to_all, dim=0)[0]), torch.Tensor([0.]), atol=1E-3), min(torch.min(all_to_all, dim=0)[0])
    assert torch.isclose(min(torch.min(all_to_all, dim=1)[0]), torch.Tensor([0.]), atol=1E-3), min(torch.min(all_to_all, dim=1)[0])


if __name__ == "__main__":
    pytest.main()