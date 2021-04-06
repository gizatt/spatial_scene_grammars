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
    def __init__(self, parameters):
        super().__init__(child_types=[NorthRoom, WestRoom, EastRoom, SouthRoom],
                         production_probs=parameters["child_probs"].get_value(),
                         parameters=parameters)
    @classmethod
    def get_default_parameters(cls):
        return {
            "room_spacing": NodeParameter(torch.tensor([5.])),
            "child_probs": NodeParameter(torch.tensor([0.5, 0.5, 0.5, 0.5]), constraint=constraints.unit_interval)
        }
    def get_derived_variable_dists_for_children(self, child_types):
        all_dist_sets = []
        offset = self.room_spacing.get_value()[0]
        for child_type in child_types:
            child_name = child_type.__name__
            if child_name == "North_Room":
                dists = {"xy": dist.Delta(self.xy + torch.tensor([0., offset]))}
            elif child_name == "South_Room":
                dists = {"xy": dist.Delta(self.xy + torch.tensor([0., -offset]))}
            elif child_name == "East_Room":
                dists = {"xy": dist.Delta(self.xy + torch.tensor([offset, 0.]))}
            elif child_name == "West_Room":
                dists = {"xy": dist.Delta(self.xy + torch.tensor([-offset, 0.]))}
            else:
                raise ValueError(child_name)
            all_dist_sets.append(dists)
        return all_dist_sets

class Room(HasOnlyXyDerivedVariablesMixin, AndNode):
    def __init__(self, **kwargs):
        super().__init__(
            child_types=[Table, Table],
            **kwargs
        )
    def get_derived_variable_dists_for_children(self, child_types):
        assert len(child_types) == 2
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
    def __init__(self, **kwargs):
        super().__init__(
            child_types=[ColoredObject, ObjectStack],
            production_weights=torch.ones(2),
            **kwargs
        )
    def get_derived_variable_dists_for_children(self, child_types):
        assert len(child_types) == 1
        new_xy = dist.Normal(self.xy, torch.ones(2))
        return [{"xy": new_xy}]

class ColoredObject(HasOnlyXyDerivedVariablesMixin, TerminalNode):
    @classmethod
    def get_local_variable_info(cls):
        return {"color": (3,)}
    def get_local_variable_dists(self, derived_variable_values):
        return {
            "color": dist.Normal(torch.zeros(3), torch.ones(3))
        }

class ObjectStack(HasOnlyXyDerivedVariablesMixin, GeometricSetNode):
    def __init__(self, **kwargs):
        super().__init__(
            child_type=StackedObject,
            geometric_prob=0.0,
            max_repeats=3,
            **kwargs
        )
    def get_derived_variable_dists_for_children(self, child_types):
        dists = []
        for k, _ in enumerate(child_types):
            offset = torch.tensor([0., 1.])
            var = torch.tensor([0.01, 0.0001])
            new_xy = dist.Normal(self.xy + offset, var)
            dists.append({"xy": new_xy})
        return dists

class StackedObject(HasOnlyXyDerivedVariablesMixin, TerminalNode):
    pass
