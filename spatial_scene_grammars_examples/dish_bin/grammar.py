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

class PlateStack(GeometricSetNode):
    def __init__(self, tf):
        super().__init__(
            p=torch.tensor(0.3),
            max_children=3,
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Plate,
                xyz_rule=AxisAlignedGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.025]),
                    variance=torch.tensor([0.001, 0.001, 0.01])),
                # Assume world-frame vertically-oriented plate stacks
                rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), [100., 100., 0.1]
                )
            )
        ]
        return rules

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
    def __init__(self, tf):
        super().__init__(
            p=torch.tensor(0.5),
            max_children=3,
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Object,
                xyz_rule=AxisAlignedGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.02]),
                    variance=torch.tensor([0.001, 0.001, 0.01])),
                # Assume world-frame vertically-oriented plate stacks
                rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), [100., 100., 0.1]
                )
            )
        ]
        return rules

class FullBowl(AndNode):
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
                child_type=Bowl,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=BowlContents,
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
        ObjectTypes = [Plate, Bowl, Cup]
        ObjectRules = [
            ProductionRule(
                child_type=object_type,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ) for object_type in ObjectTypes
        ]
        return ObjectRules

class AssortedObjects(GeometricSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            p=0.2,
            max_children=6,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        rule = ProductionRule(
            child_type=Object,
            xyz_rule=AxisAlignedGaussianOffsetRule(
                mean=torch.tensor([0.0, 0.0, 0.025]),
                variance=torch.tensor([0.05, 0.05, 0.05])),
            rotation_rule=WorldFrameBinghamRotationRule(torch.eye(4), torch.tensor([-1, -1, -1, 0.]))
        )
        return [rule]

class AssortedPlateStacks(GeometricSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            p=0.5,
            max_children=2,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        rule = ProductionRule(
            child_type=PlateStack,
            xyz_rule=AxisAlignedGaussianOffsetRule(
                mean=torch.tensor([0.0, 0.0, 0.01]),
                variance=torch.tensor([0.05, 0.50, 0.005])),
            rotation_rule=SameRotationRule() # Upright
        )
        return [rule]

class AssortedFullBowls(GeometricSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            p=0.5,
            max_children=2,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        rule = ProductionRule(
            child_type=FullBowl,
            xyz_rule=AxisAlignedGaussianOffsetRule(
                mean=torch.tensor([0.0, 0.0, 0.01]),
                variance=torch.tensor([0.05, 0.50, 0.005])),
            # Assume world-frame vertically-oriented plate stacks
            rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                RotationMatrix(), [100., 100., 0.1]
            )
        )
        return [rule]
    
class DishBin(IndependentSetNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom.register_model_file(torch.eye(4), "sink/bin.sdf")
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.5, 0.5, 0.5]),
            physics_geometry_info=geom,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=AssortedObjects,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=AssortedPlateStacks,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=AssortedFullBowls,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
        ]