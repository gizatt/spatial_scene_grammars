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


@pytest.fixture(params=range(10))
def set_seed(request):
    pyro.clear_param_store()
    torch.manual_seed(request.param)


def test_estimate_observation_likelihood(set_seed):
    generated_tree = SceneTree.forward_sample_from_root_type(Building, {"xy": dist.Delta(torch.zeros(2))})
    observed_nodes = [n for n in generated_tree.nodes() if isinstance(n, TerminalNode)]

    # Error with itself should be zero. (Error is log-prob sum of likelihood
    # functions, which with this variance peak at 1. -> ll = 0.)
    error = estimate_observation_likelihood(observed_nodes, observed_nodes, 1./(np.sqrt(2 * np.pi)))
    assert np.allclose(error.item(), 0.)

def test_node_embedding():
    test_embedding = NodeEmbedding(Building(), output_size=64)
    test_building = Building()
    test_building.instantiate({"xy": dist.Delta(torch.zeros(2))})
    out = test_embedding(test_building.get_all_continuous_variables_as_vector())
    assert torch.all(torch.isfinite(out))

def test_grammar_encoder(set_seed):
    root_node_type = Building
    generated_tree = SceneTree.forward_sample_from_root_type(
        root_node_type, {"xy": dist.Delta(torch.zeros(2))}
    )
    observed_nodes = [n for n in generated_tree.nodes() if isinstance(n, TerminalNode)]
    meta_tree = SceneTree.make_meta_scene_tree(root_node_type())
    encoder = GrammarEncoder(meta_tree, embedding_size=64)
    x = encoder(observed_nodes)
    assert torch.all(torch.isfinite(x))


if __name__ == "__main__":
    pytest.main()