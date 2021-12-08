import glob
import os
from functools import lru_cache

import torch
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.drake_interop import *

import pydrake
import pydrake.geometry as pydrake_geom
from pydrake.all import (
    RollPitchYaw,
    RigidTransform
)

''' Grammar describing how mugs, plates, and cups appear
in a dish bin.

The dish bin places a random number of indeterminate objects
within the bin. Each indeterminate object specializes into
one of the three object types.

A concrete object type is observed, and can produce one
additional indeterminant object at an offset distribution, or nothing.
'''

eps = 1E-2

## Primitive types
class ObjectModel(TerminalNode):
    sdf = None
    def __init__(self, tf):
        assert self.sdf is not None, "Don't instantiate ObjectModel itself; use a reified version."
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            self.sdf
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

# Use caching to try to avoid re-generating objects, which messes with pickling
# and node identity checking.
@lru_cache(maxsize=None)
def reify_models_from_folder_to_object_types(folder):
    print("Generating from folder ", folder)
    sdfs = glob.glob(os.path.join(folder, "*/model_simplified.sdf"))
    new_types = []
    for sdf in sdfs:
        new_types.append(
            type(
                "%s" % os.path.split(sdf)[0].replace("/", ":"),
                (ObjectModel,),
                {"sdf": sdf}
            )
        )
    # Make these dynamically generated types pickle-able
    # by registering them globally. Dangerous -- make sure
    # their names are unique!
    # https://stackoverflow.com/questions/11658511/pickling-dynamically-generated-classes
    for new_type in new_types:
        print(new_type.__name__, ": ", new_type)
        globals()[new_type.__name__] = new_type

    return new_types


PlateModels = reify_models_from_folder_to_object_types(
    "sink/plates_cups_and_bowls/plates"
)
class TerminalPlate(OrNode):
    # One of any available plate model.
    def __init__(self, tf):
        super().__init__(
            rule_probs=torch.ones(len(PlateModels)),
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        ModelRules = [
            ProductionRule(
                child_type=model_type,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ) for model_type in PlateModels
        ]
        return ModelRules

CupModels = reify_models_from_folder_to_object_types(
    "sink/plates_cups_and_bowls/cups"
)
class TerminalCup(OrNode):
    # One of any available plate model.
    def __init__(self, tf):
        super().__init__(
            rule_probs=torch.ones(len(CupModels)),
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        ModelRules = [
            ProductionRule(
                child_type=model_type,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ) for model_type in CupModels
        ]
        return ModelRules
class TerminalObject(OrNode):
    def __init__(self, tf):
        super().__init__(
            rule_probs=torch.tensor([1., 1., 1.]),
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )

    @classmethod
    def generate_rules(cls):
        ObjectTypes = [TerminalPlate, TerminalBowl, TerminalCup]
        ObjectRules = [
            ProductionRule(
                child_type=object_type,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ) for object_type in ObjectTypes
        ]
        return ObjectRules

BowlModels = reify_models_from_folder_to_object_types(
    "sink/plates_cups_and_bowls/bowls"
)
class TerminalBowl(OrNode):
    def __init__(self, tf):
        super().__init__(
            rule_probs=torch.ones(len(BowlModels)),
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        ModelRules = [
            ProductionRule(
                child_type=model_type,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ) for model_type in BowlModels
        ]
        return ModelRules

class ObjectsOnPlate(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.5, max_children=3),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        rule = ProductionRule(
            child_type=TerminalObject,
            xyz_rule=ParentFrameGaussianOffsetRule(
                mean=torch.tensor([0.0, 0.0, 0.01]),
                variance=torch.tensor([0.0025, 0.0025, 0.0025])),
            rotation_rule=WorldFrameBinghamRotationRule(torch.eye(4), torch.tensor([-0.1, -0.1, -0.1, 0.]))
        )
        return [rule]
class MaybeObjectsOnPlate(IndependentSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.8]),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=ObjectsOnPlate,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
        ]
class Plate(AndNode):
    # Always make a plate; sometimes produces
    # additional stuff on top of it.
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=TerminalPlate,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=MaybeObjectsOnPlate,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
        ]
        return rules

class ObjectsInBowl(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.5, max_children=3),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        rule = ProductionRule(
            child_type=TerminalObject,
            xyz_rule=ParentFrameGaussianOffsetRule(
                mean=torch.tensor([0.0, 0.0, 0.05]),
                variance=torch.tensor([0.005, 0.005, 0.005])),
            rotation_rule=WorldFrameBinghamRotationRule(torch.eye(4), torch.tensor([-0.01, -0.01, -0.01, 0.]))
        )
        return [rule]
class MaybeObjectsInBowl(IndependentSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.8]),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=ObjectsInBowl,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
        ]
class Bowl(AndNode):
    # Always make a bowl; sometimes produces
    # additional stuff inside of it.
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=TerminalBowl,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=MaybeObjectsInBowl,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
        ]
        return rules

class Object(OrNode):
    def __init__(self, tf):
        super().__init__(
            rule_probs=torch.tensor([1., 1., 1.]),
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )

    @classmethod
    def generate_rules(cls):
        ObjectTypes = [Plate, Bowl, TerminalCup]
        ObjectRules = [
            ProductionRule(
                child_type=object_type,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ) for object_type in ObjectTypes
        ]
        return ObjectRules
    
class DishBin(RepeatingSetNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom.register_model_file(torch.eye(4), "sink/bin.sdf")
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.4, max_children=6),
            physics_geometry_info=geom,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Object,
                xyz_rule=WorldFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.1]),
                    variance=torch.tensor([0.01, 0.02, 0.005])),
                rotation_rule=WorldFrameBinghamRotationRule(torch.eye(4), torch.tensor([-1, -1, -1, 0.]))
            )
        ]