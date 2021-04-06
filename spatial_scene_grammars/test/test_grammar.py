import numpy as np
import unittest
import pytest

from pydrake.all import (
    PackageMap
)

import pyro
import pyro.distributions as dist
import torch

from spatial_scene_grammars.tree import *
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
    grammar = SceneGrammar(root_node_type)
    scene_tree = grammar({"xy": dist.Delta(torch.zeros(2))})
    assert isinstance(scene_tree, SceneTree)

    torch_param_dict = {k: v for k, v in grammar.named_parameters()}
    for k in ["Building:room_spacing", "Building:child_probs"]:
        assert k in torch_param_dict.keys()

def test_forward_sampling(set_seed):
    # Sanity checks forward sampling and scoring functions.
    grammar = SceneGrammar(root_node_type)
    trace = pyro.poutine.trace(grammar.forward).get_trace(inst_dict)
    tree = trace.nodes["_RETURN"]["value"]

    root_node = get_tree_root(tree)
    tree_ll = tree.get_subtree_log_prob(root_node).detach().numpy()
    expected_ll = trace.log_prob_sum().detach().numpy()
    assert np.allclose(tree_ll, expected_ll), "%s vs %s" % (tree_ll, expected_ll)

    # Make sure these calls are identical
    tree_ll = tree.get_subtree_log_prob(root_node).detach().numpy()
    tree_ll_shorthand = tree.get_log_prob().detach().numpy()
    assert np.allclose(tree_ll, tree_ll_shorthand), "%s vs %s" % (tree_ll, tree_ll_shorthand)

def test_variable_getters(set_seed):
    grammar = SceneGrammar(root_node_type)
    generated_tree = grammar.forward(inst_dict)

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

def test_conditioned_instantiate(set_seed):
    target_object = ColoredObject.init_with_default_parameters()
    target_object.instantiate({"xy": dist.Delta(torch.tensor([1., 2.]))})

    new_object = ColoredObject.init_with_default_parameters()
    conditioned_trace = pyro.poutine.trace(
        new_object.conditioned_instantiate
    ).get_trace(
        target_object.derived_variable_values,
        target_object.local_variable_values
    )
    assert np.allclose(conditioned_trace.log_prob_sum().item(), 0.)
    assert_identical_dicts_of_tensors(new_object.derived_variable_values, target_object.derived_variable_values)
    assert_identical_dicts_of_tensors(new_object.local_variable_values, target_object.local_variable_values)

def test_conditioned_sample_children(set_seed):
    grammar = SceneGrammar(root_node_type)
    generated_tree = grammar.forward(inst_dict)
    
    for target_object_type in [root_node_type, Room, Table]:
        target_object = generated_tree.find_nodes_by_type(target_object_type)[0]
        children = list(generated_tree.successors(target_object))
        child_types = [type(c) for c in children]

        new_object = target_object_type.init_with_default_parameters()
        conditioned_trace = pyro.poutine.trace(
            new_object.conditioned_sample_children
        ).get_trace(child_types)
        assert torch.isclose(conditioned_trace.log_prob_sum(), torch.Tensor([0.]))
        assert torch.allclose(
            new_object.get_child_indicator_vector(child_types),
            new_object.child_inclusion_values
        )

def test_meta_scene_tree(set_seed):
    meta_tree = SceneGrammar.make_meta_scene_tree(root_node_type)

def test_grammar_parameter_update(set_seed):
    grammar = SceneGrammar(root_node_type)
    default_tree = grammar(inst_dict)
    # Change parameters of root node children
    grammar.params_by_node_type[root_node_type]["child_probs"].set(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    new_tree = grammar(inst_dict)
    assert len(new_tree.find_nodes_by_type(NorthRoom)) == 1
    assert len(new_tree.find_nodes_by_type(SouthRoom)) == 0
    assert len(new_tree.find_nodes_by_type(WestRoom)) == 0
    assert len(new_tree.find_nodes_by_type(EastRoom)) == 0

    # See that we can re-score both trees under new params, and that the
    # orig tree's score has changed.
    orig_tree_rerun_prob = grammar.get_tree_generation_log_prob(default_tree, inst_dict)
    orig_tree_prob = default_tree.get_log_prob()
    assert not torch.isclose(orig_tree_prob, orig_tree_rerun_prob)


if __name__ == "__main__":
    pytest.main()