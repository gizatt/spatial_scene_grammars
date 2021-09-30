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

class Null(TerminalNode):
    def __init__(self, tf):
        super().__init__(tf=tf, observed=False, physics_geometry_info=None)
    
PlateModels = reify_models_from_folder_to_object_types(
    "sink/plates_cups_and_bowls/plates"
)
class Plate(OrNode):
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

    
class PlateContents(OrNode):
    def __init__(self, tf):
        super().__init__(
            rule_probs=torch.tensor([0.4, 0.4, 0.2]),
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule( # Do nothing
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule( # Another plate on top
                child_type=PlateAndContents,
                xyz_rule=AxisAlignedGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.05]),
                    variance=torch.tensor([0.001, 0.001, 0.01])),
                rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), [100., 100., 0.1]) # Aligned vertically
            ),
            ProductionRule( # Stuff on top
                child_type=Object,
                xyz_rule=AxisAlignedGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.05]),
                    variance=torch.tensor([0.01, 0.01, 0.01])),
                rotation_rule=WorldFrameBinghamRotationRule(torch.eye(4), torch.tensor([-1, -1, -1, 0.]))
           )
        ]
    
class PlateAndContents(AndNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Plate,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=PlateContents,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
        ]

CupModels = reify_models_from_folder_to_object_types(
    "sink/plates_cups_and_bowls/cups"
)
class Cup(OrNode):
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

BowlModels = reify_models_from_folder_to_object_types(
    "sink/plates_cups_and_bowls/bowls"
)
class Bowl(OrNode):
    # One of any available plate model.
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

class BowlContents(GeometricSetNode):
    # Maybe make stuff inside the bowl.
    def __init__(self, tf):
        super().__init__(
            p=0.5,
            max_children=3,
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        rule = ProductionRule(
            child_type=Object,
            xyz_rule=AxisAlignedGaussianOffsetRule(
                mean=torch.tensor([0.0, 0.0, 0.05]),
                variance=torch.tensor([0.005, 0.005, 0.01])),
            rotation_rule=WorldFrameBinghamRotationRule(torch.eye(4), torch.tensor([-1, -1, -1, 0.]))
        )
        return [rule]

class MaybeBowlContents(IndependentSetNode):
    # Maybe make stuff inside the bowl.
    def __init__(self, tf):
        super().__init__(
            rule_probs=torch.tensor([0.5]),
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=BowlContents,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
        ]

class BowlAndContents(AndNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Bowl,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=MaybeBowlContents,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
        ]
    
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
        ObjectTypes = [PlateAndContents, BowlAndContents, Cup]
        ObjectRules = [
            ProductionRule(
                child_type=object_type,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ) for object_type in ObjectTypes
        ]
        return ObjectRules

class DishBin(GeometricSetNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom.register_model_file(torch.eye(4), "sink/bin.sdf")
        super().__init__(
            tf=tf,
            p=0.2,
            max_children=6,
            physics_geometry_info=geom,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        rule = ProductionRule(
            child_type=Object,
            xyz_rule=AxisAlignedGaussianOffsetRule(
                mean=torch.tensor([0.0, 0.0, 0.]),
                variance=torch.tensor([0.1, 0.1, 0.05])),
            rotation_rule=WorldFrameBinghamRotationRule(torch.eye(4), torch.tensor([-1, -1, -1, 0.]))
        )
        return [rule]