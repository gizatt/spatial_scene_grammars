import numpy as np
import unittest
import pytest

from pydrake.all import (
    PackageMap
)

import pyro
import pyro.distributions as dist
import torch

from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *

from spatial_scene_grammars.test.grammar import *

torch.set_default_tensor_type(torch.DoubleTensor)
pyro.enable_validation(True)

root_node_type = Building
inst_dict = {"xy": dist.Delta(torch.zeros(2))}

def assert_identical_dicts_of_tensors(d1, d2):
    for key, value in d1.items():
        assert key in d2.keys()
        assert torch.allclose(value, d2[key])
    for key, value in d2.items():
        assert key in d1.keys()
        assert torch.allclose(value, d1[key])

@pytest.fixture(params=range(10))
def set_seed(request):
    pyro.clear_param_store()
    torch.manual_seed(request.param)

def test_get_all_types():
    all_types = SceneGrammar.get_all_types_in_grammar(root_node_type)
    expected_types = set([root_node_type, NorthRoom, SouthRoom, EastRoom, WestRoom, Table, ColoredObject, StackedObject, ObjectStack])
    for type in expected_types:
        assert type in all_types
    for type in all_types:
        assert type in expected_types

def test_get_params():
    building = root_node_type.init_with_default_parameters()
    params = building.get_parameters()
    for k in ["room_spacing", "child_probs"]:
        assert k in params.keys()
        assert isinstance(params[k], NodeParameter)

def test_grammar_params(set_seed):
    grammar = SceneGrammar(root_node_type, inst_dict)
    scene_tree = grammar()
    assert isinstance(scene_tree, SceneTree)

    torch_param_dict = {k: v for k, v in grammar.named_parameters()}
    for k in ["Building:room_spacing", "Building:child_probs"]:
        assert k in torch_param_dict.keys()

def test_forward_sampling(set_seed):
    # Sanity checks forward sampling and scoring functions.
    grammar = SceneGrammar(root_node_type, inst_dict)
    trace = pyro.poutine.trace(grammar.forward).get_trace()
    tree = trace.nodes["_RETURN"]["value"]

    root_node = get_tree_root(tree)
    tree_ll = tree.get_subtree_log_prob(root_node).detach().numpy()
    expected_ll = trace.log_prob_sum().detach().numpy()
    assert np.allclose(tree_ll, expected_ll), "%s vs %s" % (tree_ll, expected_ll)

    # Make sure these calls are identical
    tree_ll = tree.get_subtree_log_prob(root_node).detach().numpy()
    tree_ll_shorthand = tree.get_log_prob().detach().numpy()
    assert np.allclose(tree_ll, tree_ll_shorthand), "%s vs %s" % (tree_ll, tree_ll_shorthand)

    # Make sure gradients go backwards as we expect.
    rooms = list(tree.successors(root_node))
    if len(rooms) > 0:
        tree_ll = tree.get_log_prob()
        tree_ll.backward()
        # Collect into a dict so our failure message is more
        # informative about which parameters don't have grads.
        parameters_have_grad = {}
        for name, parameter in grammar.named_parameters():
            parameters_have_grad[name] = parameter.grad is not None
        assert all(parameters_have_grad.values()), parameters_have_grad
    

def test_variable_getters(set_seed):
    grammar = SceneGrammar(root_node_type, inst_dict)
    generated_tree = grammar.forward()

    # Not a detailed test, but make sure these calls don't fail,
    # and agree with each other.
    for node in generated_tree.nodes:
        all_cvars_dict = node.get_all_continuous_variable_values()
        all_derived_vars_vec = node.get_derived_variables_as_vector()
        all_local_vars_vec = node.get_local_variables_as_vector()
        all_cvars_vec = node.get_all_continuous_variables_as_vector()

        total_len = sum([np.prod(v.shape) for v in all_cvars_dict.values()])
        assert total_len == len(all_cvars_vec)
        assert len(all_derived_vars_vec) + len(all_local_vars_vec) == len(all_cvars_vec)

def test_node_instantiate(set_seed):
    building = root_node_type.init_with_default_parameters()
    assert building.instantiated is False
    trace = pyro.poutine.trace(building.instantiate).get_trace(inst_dict)
    assert building.instantiated
    building_ll = building.get_continuous_variable_ll().detach().numpy()
    expected_ll = trace.log_prob_sum().detach().numpy()
    assert np.allclose(building_ll, expected_ll)

def test_meta_scene_tree(set_seed):
    meta_tree = SceneGrammar.make_meta_scene_tree(root_node_type)

def test_grammar_parameter_update(set_seed):
    grammar = SceneGrammar(root_node_type, inst_dict)
    default_tree = grammar()

    # Scoring tree should be the same as the tree score.
    orig_tree_rerun_prob = grammar.score(default_tree)
    orig_tree_prob = default_tree.get_log_prob()
    assert torch.isclose(orig_tree_prob, orig_tree_rerun_prob)
    
    # Change parameters of root node children
    grammar.params_by_node_type[root_node_type]["child_probs"].set(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    new_tree = grammar()
    assert len(new_tree.find_nodes_by_type(NorthRoom)) == 1
    assert len(new_tree.find_nodes_by_type(SouthRoom)) == 0
    assert len(new_tree.find_nodes_by_type(WestRoom)) == 0
    assert len(new_tree.find_nodes_by_type(EastRoom)) == 0

    # See that we can re-score both trees under new params, and that the
    # orig tree's score has changed.
    orig_tree_rerun_prob = grammar.score(default_tree)
    orig_tree_prob = default_tree.get_log_prob()
    assert not torch.isclose(orig_tree_prob, orig_tree_rerun_prob)


if __name__ == "__main__":
    pytest.main()