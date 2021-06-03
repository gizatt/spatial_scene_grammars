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
from spatial_scene_grammars.torch_utils import ConstrainedParameter

from spatial_scene_grammars.test.grammar import *

torch.set_default_tensor_type(torch.DoubleTensor)
pyro.enable_validation(True)

root_node_type = Building
inst_dict = {"xy": dist.Normal(torch.zeros(2), torch.ones(2)*0.001)}
grammar_types = [FullyParameterizedGrammar, SceneGrammar, FullyParameterizedSuperTreeGrammar]

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
    all_types = SceneGrammarBase.get_all_types_in_grammar_given_root(root_node_type)
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
        assert isinstance(params[k], ConstrainedParameter)

def test_grammar_params(set_seed):
    grammar = SceneGrammar(root_node_type, inst_dict)
    scene_tree = grammar()
    assert isinstance(scene_tree, SceneTree)

    for k in ["Building:room_spacing", "Building:child_probs"]:
        assert k in grammar.default_params.keys()

def test_node_sample_overrides():
    # Forced choice of children
    for k in range(10):
        test_node = Building.init_with_default_parameters()
        override_weights = torch.tensor([1., 0., 0., 0.])
        override_child_dist = dist.Bernoulli(override_weights).to_event(1)
        forced_children = test_node.sample_children(
            child_inclusion_dist_override=override_child_dist
        )
        assert len(forced_children) == 1 and forced_children[0] == NorthRoom
        assert torch.isclose(test_node.get_children_ll(), torch.zeros(1))

        forced_child_types = [NorthRoom]
        forced_children = test_node.sample_children(
            observed_child_types=forced_child_types,
            child_inclusion_dist_override=override_child_dist
        )
        assert len(forced_children) == 1 and forced_children[0] == NorthRoom
        assert torch.isclose(test_node.get_children_ll(), torch.zeros(1))

    # Forced local attributes
    for k in range(10):
        test_node = ColoredObject.init_with_default_parameters()
        target_xy = torch.rand((2,))
        derived_xy = {"xy": dist.Delta(target_xy)}
        target_color = torch.rand((3,))
        override_color = {"color": dist.Delta(target_color)}
        test_node.instantiate(
            derived_variable_distributions=derived_xy,
            observed_derived_variables=None,
            observed_local_variables=None,
            local_variable_distributions_override=override_color
        )
        assert torch.allclose(target_color, test_node.color)


@pytest.mark.parametrize('grammar_type', grammar_types)
def test_grammar_param_override(set_seed, grammar_type):
    grammar = grammar_type(root_node_type, inst_dict)

    # Forward sample under default grammar params should work
    # normally.
    scene_tree = grammar()

    # Forward sample under new grammar params
    default_param_dict = grammar.get_default_param_dict()
    new_param_dict = deepcopy(default_param_dict)
    for v in new_param_dict.values():
        x = torch.nn.Parameter(torch.zeros(v().shape))
        y = x ** 2
        v.set_unconstrained(y)
    new_tree = grammar(params=new_param_dict)
    tree_ll = new_tree.get_log_prob()
    matching_ll = grammar.score(new_tree, params=new_param_dict)
    assert torch.allclose(tree_ll, matching_ll)
    # This should work but I can't guarantee what the output is.
    diff_ll = grammar.score(new_tree, params=None)

@pytest.mark.parametrize('grammar_type', grammar_types)
def test_forward_sampling(set_seed, grammar_type):
    # Sanity checks forward sampling and scoring functions.
    grammar = grammar_type(root_node_type, inst_dict)
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
        # For this particular test grammar, the parameters are all
        # going to always appear in any tree (since they belong to
        # the root node). So they all should have gradients.
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
    meta_tree = SceneGrammar.make_super_scene_tree(root_node_type)

def test_grammar_parameter_update(set_seed):
    grammar = SceneGrammar(root_node_type, inst_dict)
    default_tree = grammar()

    # Scoring tree should be the same as the tree score.
    orig_tree_rerun_prob = grammar.score(default_tree)
    orig_tree_prob = default_tree.get_log_prob()
    assert torch.isclose(orig_tree_prob, orig_tree_rerun_prob)
    
    # Change parameters of root node children
    full_name = grammar._get_name_for_param(root_node_type, "child_probs")
    grammar.default_params[full_name].set(torch.tensor([1.0, 0.0, 0.0, 0.0]))
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