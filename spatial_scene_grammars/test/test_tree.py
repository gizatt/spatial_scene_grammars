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


class Room(AndNode):
    def __init__(self):
        super().__init__(
            child_types=[Table, Table]
        )
    def _instantiate_children_impl(self, children):
        assert len(children) == 2
        all_attrs = []
        for k in range(2):
            new_xy = self.xy + pyro.sample("offset", dist.Normal(torch.zeros(2), torch.ones(2)))
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
    room = Room()
    assert room.instantiated is False
    room.instantiate({"xy": torch.zeros(2)})
    assert room.instantiated

    tree = SceneTree.forward_sample_from_root_type(room)

def test_meta_scene_tree(set_seed):
    meta_tree = SceneTree.make_meta_scene_tree(Room())

if __name__ == "__main__":
    pytest.main()