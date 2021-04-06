import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from torch.nn.parameter import Parameter

from spatial_scene_grammars.tree import *
from spatial_scene_grammars.nodes import *


torch.set_default_tensor_type(torch.DoubleTensor)
pyro.enable_validation(True)


class HasOnlyXyDerivedVariablesMixin():
    @classmethod
    def get_derived_variable_info(cls):
        return {"xy": (2,)}

# Mixin must come first, since it's overriding a class method
# also provided by the base node type.
class Building(HasOnlyXyDerivedVariablesMixin, IndependentSetNode):
    def __init__(self):
        self.room_spacing = NodeParameter(torch.tensor([5.]))
        self.child_probs = NodeParameter(torch.tensor([0.5, 0.5, 0.5, 0.5]), constraint=constraints.unit_interval)
        super().__init__(child_types=[NorthRoom, WestRoom, EastRoom, SouthRoom],
                         production_probs=self.child_probs.get_value())
    def _instantiate_children_impl(self, children):
        all_dist_sets = []
        offset = self.room_spacing.get_value()[0]
        for child in children:
            child_type = child.__class__.__name__
            if child_type == "North_Room":
                dists = {"xy": dist.Delta(self.xy + torch.tensor([0., offset]))}
            elif child_type == "South_Room":
                dists = {"xy": dist.Delta(self.xy + torch.tensor([0., -offset]))}
            elif child_type == "East_Room":
                dists = {"xy": dist.Delta(self.xy + torch.tensor([offset, 0.]))}
            elif child_type == "West_Room":
                dists = {"xy": dist.Delta(self.xy + torch.tensor([-offset, 0.]))}
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
NorthRoom = type("North_Room", (Room,), {})
WestRoom = type("West_Room", (Room,), {})
EastRoom = type("East_Room", (Room,), {})
SouthRoom = type("South_Room", (Room,), {})
    
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
            geometric_prob=0.0,
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
