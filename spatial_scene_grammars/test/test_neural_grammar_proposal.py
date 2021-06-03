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
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.neural_grammar_proposal import *

from spatial_scene_grammars.test.grammar import *

torch.set_default_tensor_type(torch.DoubleTensor)
pyro.enable_validation(True)

root_node_type = Building
inst_dict = {"xy": dist.Delta(torch.zeros(2))}
grammar_types = [FullyParameterizedGrammar, SceneGrammar, FullyParameterizedSuperTreeGrammar]
configs = [GrammarEncoder.Config(rnn_type="GCN"), GrammarEncoder.Config(rnn_type="GRU")]

@pytest.fixture(params=range(3))
def set_seed(request):
    pyro.clear_param_store()
    torch.manual_seed(request.param)

def test_estimate_observation_likelihood(set_seed):
    grammar = SceneGrammar(root_node_type, inst_dict)
    generated_tree = grammar()
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

@pytest.mark.parametrize('grammar_type', grammar_types)
@pytest.mark.parametrize('config', configs)
def test_grammar_encoder(set_seed, grammar_type, config):
    grammar = grammar_type(root_node_type, inst_dict)
    generated_tree = grammar()
    observed_nodes = [n for n in generated_tree.nodes() if isinstance(n, TerminalNode)]
    
    # Can we encode and decode?
    encoder = GrammarEncoder(grammar, embedding_size=64, config=config)
    x = encoder(observed_nodes)
    assert torch.all(torch.isfinite(x))
    sampled_tree, ll, reparam_ll = encoder.sample_tree_from_grammar_vector(x)
    assert torch.isfinite(ll)


if __name__ == "__main__":
    pytest.main()