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

class Building(IndependentSetNode):
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
        all_args = []
        for child in children:
            child_type = child.__class__.__name__
            if child_type == "North_Room":
                args = {"xy": self.xy + torch.tensor([0., 5.])}
            elif child_type == "South_Room":
                args = {"xy": self.xy + torch.tensor([0., -5.])}
            elif child_type == "East_Room":
                args = {"xy": self.xy + torch.tensor([5., 0.])}
            elif child_type == "West_Room":
                args = {"xy": self.xy + torch.tensor([-5., 0.])}
            else:
                raise ValueError(child_type)
            all_args.append(args)
        return all_args

    def _instantiate_impl(self, derived_attributes):
        assert "xy" in derived_attributes.keys()
        self.xy = derived_attributes["xy"]


class Room(AndNode):
    def __init__(self):
        super().__init__(
            child_types=[Table, Table]
        )
    def _instantiate_children_impl(self, children):
        assert len(children) == 2
        all_attrs = []
        for k in range(2):
            new_xy = self.xy + pyro.sample("offset_%d" % k, dist.Normal(torch.zeros(2), torch.ones(2)))
            all_attrs.append({"xy": new_xy})
        return all_attrs

    def _instantiate_impl(self, derived_attributes):
        assert "xy" in derived_attributes.keys()
        self.xy = derived_attributes["xy"]

class Table(OrNode):
    def __init__(self):
        super().__init__(
            child_types=[Object1, Object2],
            production_weights=torch.ones(2)
        )
    def _instantiate_children_impl(self, children):
        assert len(children) == 1
        new_xy = self.xy + pyro.sample("offset", dist.Normal(torch.zeros(2), torch.ones(2)))
        return [{"xy": new_xy}]

    def _instantiate_impl(self, derived_attributes):
        assert "xy" in derived_attributes.keys()
        self.xy = derived_attributes["xy"]

class Object1(TerminalNode):
    def _instantiate_impl(self, derived_attributes):
        assert "xy" in derived_attributes.keys()
        self.xy = derived_attributes["xy"]

class Object2(TerminalNode):
    def _instantiate_impl(self, derived_attributes):
        assert "xy" in derived_attributes.keys()
        self.xy = derived_attributes["xy"]


@pytest.fixture(params=range(10))
def set_seed(request):
    pyro.clear_param_store()
    torch.manual_seed(request.param)

def test_forward_sampling(set_seed):
    # Sanity checks forward sampling and scoring functions.
    building = Building()
    assert building.instantiated is False
    trace = pyro.poutine.trace(building.instantiate).get_trace({"xy": torch.zeros(2)})
    assert building.instantiated
    building_ll = building.get_instantiate_ll()
    expected_ll = trace.log_prob_sum()
    assert np.allclose(building_ll, expected_ll)

    trace = pyro.poutine.trace(SceneTree.forward_sample_from_root).get_trace(building)
    tree = trace.nodes["_RETURN"]["value"]
    tree_ll = tree.get_subtree_log_prob(building)
    expected_ll = trace.log_prob_sum()
    assert np.allclose(tree_ll, expected_ll)

    # Make sure these calls are identical
    tree_ll = tree.get_subtree_log_prob(building)
    tree_ll_shorthand = tree.get_log_prob()
    assert np.allclose(tree_ll, tree_ll_shorthand)

def test_rebuild_trace(set_seed):
    trace = pyro.poutine.trace(SceneTree.forward_sample_from_root_type).get_trace(
        Building, {"xy": torch.zeros(2)}
    )
    scene_tree = trace.nodes["_RETURN"]["value"]
    # Make sure we can rebuild teh trace of the full tree
    rebuilt_trace = scene_tree.get_trace()
    print("Rebuilt trace with keys ", list(rebuilt_trace.nodes.keys()))
    assert rebuilt_trace.nodes["_RETURN"]["value"] == scene_tree
    for key, site in trace.nodes.items():
        assert key in rebuilt_trace.nodes.keys()
        if site["type"] is "sample":
            assert torch.allclose(site["value"], trace.nodes[key]["value"])
    for key, site in rebuilt_trace.nodes.items():
        assert key in trace.nodes.keys()
        if site["type"] is "sample":
            assert torch.allclose(site["value"], trace.nodes[key]["value"])

def test_meta_scene_tree(set_seed):
    meta_tree = SceneTree.make_meta_scene_tree(Building())

if __name__ == "__main__":
    pytest.main()