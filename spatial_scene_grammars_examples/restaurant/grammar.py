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

''' 
Restaurant -> tables in bounds (with clearance constraints)
Tables -> place settings and misc dishware
Place settings - > dishware
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
    "models/plates_cups_and_bowls/plates"
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
class Plates(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            p=torch.tensor([0.5]),
            tf=tf,
            max_children=2,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Plate,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.025]),
                    variance=torch.tensor([0.05, 0.05, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), torch.tensor([1000., 1000., 1.])
                )
            )
        ]

CupModels = reify_models_from_folder_to_object_types(
    "models/plates_cups_and_bowls/cups"
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
class Cups(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            p=torch.tensor([0.5]),
            tf=tf,
            max_children=2,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Cup,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.025]),
                    variance=torch.tensor([0.05, 0.05, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), torch.tensor([1000., 1000., 1.])
                )
            )
        ]


UtensilModels = reify_models_from_folder_to_object_types(
    "models/plates_cups_and_bowls/utensils"
)
class Utensil(OrNode):
    # One of any available model.
    def __init__(self, tf):
        super().__init__(
            rule_probs=torch.ones(len(UtensilModels)),
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
            ) for model_type in UtensilModels
        ]
        return ModelRules
class Utensils(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            p=torch.tensor([0.5]),
            tf=tf,
            max_children=4,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Utensil,
                xyz_rule=AxisAlignedGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.025]),
                    variance=torch.tensor([0.05, 0.05, 0.0001])),
                rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), torch.tensor([1000., 1000., 1.])
                )
            )
        ]

class PlaceSetting(IndependentSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.ones(3)*0.5,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Utensils,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Plates,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Cups,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            )
        ]
        return rules


class Table(IndependentSetNode):
    # Maybe one person at each cardinal direction
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom.register_model_file(torch.eye(4), "models/table/table.sdf")
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.5, 0.5, 0.5, 0.5]),
            physics_geometry_info=geom,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=SamePositionRule(offset=torch.tensor([-0.33, 0., 0.8])),
                rotation_rule=SameRotationRule(offset=torch.tensor(RotationMatrix(RollPitchYaw(0., 0., 0.)).matrix()))
            ),
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=SamePositionRule(offset=torch.tensor([0.33, 0., 0.8])),
                rotation_rule=SameRotationRule(offset=torch.tensor(RotationMatrix(RollPitchYaw(0., 0., np.pi)).matrix()))
            ),
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=SamePositionRule(offset=torch.tensor([0., 0.33, 0.8])),
                rotation_rule=SameRotationRule(offset=torch.tensor(RotationMatrix(RollPitchYaw(0., 0., -np.pi/2.)).matrix()))
            ),
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=SamePositionRule(offset=torch.tensor([0., -0.33, 0.8])),
                rotation_rule=SameRotationRule(offset=torch.tensor(RotationMatrix(RollPitchYaw(0., 0., np.pi/2.)).matrix()))
            )
        ]


class Restaurant(RepeatingSetNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom.register_model_file(torch.eye(4), "models/restaurant/room.sdf")
        super().__init__(
            tf=tf,
            p=torch.tensor([0.25]),
            max_children=10,
            physics_geometry_info=geom,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        return [ProductionRule(
            child_type=Table,
            xyz_rule=AxisAlignedBBoxRule.from_bounds(
                torch.tensor([-4., -4., 0.]),
                torch.tensor([4., 4., 0.])
            ),
            rotation_rule=UniformBoundedRevoluteJointRule(
                axis=torch.tensor([0., 0., 1.]),
                center=torch.tensor([0.]),
                width=torch.tensor([2.*np.pi])
            )
        )]