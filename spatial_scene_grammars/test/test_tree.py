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

def test_forward_sampling(set_seed):
    # Sanity checks forward sampling and scoring functions.
    building = Building()
    assert building.instantiated is False
    trace = pyro.poutine.trace(building.instantiate).get_trace({"xy": dist.Delta(torch.zeros(2))})
    assert building.instantiated
    building_ll = building.get_continuous_variable_ll()
    expected_ll = trace.log_prob_sum()
    assert np.allclose(building_ll, expected_ll)

    trace = pyro.poutine.trace(SceneTree.forward_sample_from_root).get_trace(building)
    tree = trace.nodes["_RETURN"]["value"]
    tree_ll = tree.get_subtree_log_prob(building)
    expected_ll = trace.log_prob_sum()
    assert np.allclose(tree_ll, expected_ll), "%s vs %s" % (tree_ll, expected_ll)

    # Make sure these calls are identical
    tree_ll = tree.get_subtree_log_prob(building)
    tree_ll_shorthand = tree.get_log_prob()
    assert np.allclose(tree_ll, tree_ll_shorthand), "%s vs %s" % (tree_ll, tree_ll_shorthand)

def test_variable_getters(set_seed):
    generated_tree = SceneTree.forward_sample_from_root_type(Building, {"xy": dist.Delta(torch.zeros(2))})

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


@pytest.mark.skip(reason="feature (temporarily?) removed")
def test_conditioned_instantiate(set_seed):
    target_object = ColoredObject()
    target_object.instantiate({"xy": dist.Delta(torch.tensor([1., 2.]))})

    new_object = ColoredObject()
    new_object.conditioned_instantiate(
        target_object.derived_variables,
        target_object.local_variables
    )
    assert torch.isclose(
        torch.tensor(new_object.instantiate_trace.log_prob_sum()),
        torch.tensor(target_object.instantiate_trace.log_prob_sum())
    )
    assert_identical_dicts_of_tensors(new_object.derived_variables, target_object.derived_variables)
    assert_identical_dicts_of_tensors(new_object.local_variables, target_object.local_variables)

@pytest.mark.skip(reason="feature (temporarily?) removed")
def test_conditioned_instantiate_children(set_seed):
    instantiate_dists = {"xy": dist.Delta(torch.zeros(2))}
    tree = SceneTree.forward_sample_from_root_type(Building, instantiate_dists)
    
    for target_object_type in [Building, Room, Table]:
        target_object = tree.find_nodes_by_type(target_object_type)[0]
        children = list(tree.successors(target_object))
    
        new_object = target_object_type()
        new_object.instantiate(target_object.derived_variables)
        new_object.conditioned_instantiate_children(children)
        assert torch.isclose(
            torch.tensor(new_object.instantiate_children_trace.log_prob_sum()),
            torch.tensor(target_object.instantiate_children_trace.log_prob_sum())
        )

@pytest.mark.skip(reason="feature (temporarily?) removed")
def test_conditioned_sample_children(set_seed):
    instantiate_dists = {"xy": dist.Delta(torch.zeros(2))}
    tree = SceneTree.forward_sample_from_root_type(Building, instantiate_dists)
    
    for target_object_type in [Building, Room, Table]:
        target_object = tree.find_nodes_by_type(target_object_type)[0]
        children = list(tree.successors(target_object))
    
        new_object = target_object_type()
        new_object.conditioned_sample_children(children)
        assert torch.isclose(
            torch.tensor(new_object.sample_children_trace.log_prob_sum()),
            torch.tensor(target_object.sample_children_trace.log_prob_sum())
        )

@pytest.mark.skip(reason="feature (temporarily?) removed")
def test_rebuild_trace(set_seed):
    # TODO: Re-enable when this feature is used again / fixed.
    trace = pyro.poutine.trace(SceneTree.forward_sample_from_root_type).get_trace(
        Building, {"xy": torch.zeros(2)}
    )
    scene_tree = trace.nodes["_RETURN"]["value"]
    # Make sure we can rebuild the trace of the full tree
    rebuilt_trace = scene_tree.get_trace()
    print("Rebuilt trace with keys ", list(rebuilt_trace.nodes.keys()))
    assert rebuilt_trace.nodes["_RETURN"]["value"] == scene_tree
    for key, site in trace.nodes.items():
        assert key in rebuilt_trace.nodes.keys()
        if site["type"] is "sample":
            assert torch.allclose(site["value"], trace.nodes[key]["value"]), str(site["value"]) + " vs " + str(trace.nodes[key]["value"])
    for key, site in rebuilt_trace.nodes.items():
        assert key in trace.nodes.keys()
        if site["type"] is "sample":
            assert torch.allclose(site["value"], trace.nodes[key]["value"]), str(site["value"]) + " vs " + str(trace.nodes[key]["value"])

def test_meta_scene_tree(set_seed):
    meta_tree = SceneTree.make_meta_scene_tree(Building())

if __name__ == "__main__":
    pytest.main()