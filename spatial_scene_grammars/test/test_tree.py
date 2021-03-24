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

torch.set_default_tensor_type(torch.DoubleTensor)
pyro.enable_validation(True)

def assert_identical_dicts_of_tensors(d1, d2):
    for key, value in d1.items():
        assert key in d2.keys()
        assert torch.allclose(value, d2[key])
    for key, value in d2.items():
        assert key in d1.keys()
        assert torch.allclose(value, d1[key])

class HasOnlyXyDerivedVariablesMixin():
    @classmethod
    def get_derived_variable_info(cls):
        return {"xy": (2,)}

# Mixin must come first, since it's overriding a class method
# also provided by the base node type.
class Building(HasOnlyXyDerivedVariablesMixin, IndependentSetNode):
    def __init__(self):
        # TODO(gizatt): Are metaclasses *really* the answer here?
        # The problem is that instantiate_children doesn't have visibility
        # of which child rules (in terms of their index) were chosen; packing
        # that info into the class chosen is appealing...
        super().__init__(child_types=[type("North_Room", (Room,), {}),
                                      type("South_Room", (Room,), {}),
                                      type("East_Room", (Room,), {}),
                                      type("West_Room", (Room,), {})],
                         production_probs=torch.tensor([0.5, 0.5, 0.5, 0.5]))
    def _instantiate_children_impl(self, children):
        all_dist_sets = []
        for child in children:
            child_type = child.__class__.__name__
            if child_type == "North_Room":
                dists = {"xy": dist.Delta(self.xy + torch.tensor([0., 5.]))}
            elif child_type == "South_Room":
                dists = {"xy": dist.Delta(self.xy + torch.tensor([0., -5.]))}
            elif child_type == "East_Room":
                dists = {"xy": dist.Delta(self.xy + torch.tensor([5., 0.]))}
            elif child_type == "West_Room":
                dists = {"xy": dist.Delta(self.xy + torch.tensor([-5., 0.]))}
            else:
                raise ValueError(child_type)
            all_dist_sets.append(dists)
        return all_dist_sets

class Room(HasOnlyXyDerivedVariablesMixin, AndNode):
    def __init__(self):
        super().__init__(
            child_types=[Table, Table]
        )
    def _instantiate_children_impl(self, children):
        assert len(children) == 2
        all_dist_sets = []
        for k in range(2):
            new_xy = dist.Normal(self.xy, torch.ones(2))
            all_dist_sets.append({"xy": new_xy})
        return all_dist_sets

class Table(HasOnlyXyDerivedVariablesMixin, OrNode):
    def __init__(self):
        super().__init__(
            child_types=[ColoredObject, ObjectStack],
            production_weights=torch.ones(2)
        )
    def _instantiate_children_impl(self, children):
        assert len(children) == 1
        new_xy = dist.Normal(self.xy, torch.ones(2))
        return [{"xy": new_xy}]

class ColoredObject(HasOnlyXyDerivedVariablesMixin, TerminalNode):
    @classmethod
    def get_local_variable_info(cls):
        return {"color": (3,)}
    def _instantiate_impl(self, derived_variable_values):
        return {
            "color": dist.Normal(torch.zeros(3), torch.ones(3))
        }

class ObjectStack(HasOnlyXyDerivedVariablesMixin, GeometricSetNode):
    def __init__(self):
        super().__init__(
            child_type=StackedObject,
            geometric_prob=0.5,
            max_repeats=3
        )
    def _instantiate_children_impl(self, children):
        dists = []
        for k, child in enumerate(children):
            offset = torch.tensor([0., 1.])
            var = torch.tensor([0.01, 0.0001])
            new_xy = dist.Normal(self.xy + offset, var)
            dists.append({"xy": new_xy})
        return dists

class StackedObject(HasOnlyXyDerivedVariablesMixin, TerminalNode):
    pass


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