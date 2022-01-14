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
Table -> place settings and shared dishware
Shared dishware -> Tea kettle, food plates, bamboo steamer towers
Place settings - > cup, plate, chopsticks, chair?
'''

class PersonalPlate(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "models/plates_cups_and_bowls/plates/Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring/model_simplified.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class Teacup(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "models/plates_cups_and_bowls/cups/coffee_cup_white/model_simplified.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class Teapot(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "models/plates_cups_and_bowls/cups/Threshold_Porcelain_Teapot_White/model_simplified.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class ServingDish(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "models/plates_cups_and_bowls/plates/Threshold_Dinner_Plate_Square_Rim_White_Porcelain/model_simplified.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class Null(TerminalNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
class SteamerBottom(OrNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "models/misc/steamer_bottom/model.sdf"
        )
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.2, 0.5, 0.3]),
            physics_geometry_info=geom,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=SteamerBottom,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0., 0.0, 0.1]),
                    variance=torch.tensor([0.0001, 0.0001, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([10000, 10000, 1])
                )
            ),
            ProductionRule(
                child_type=SteamerTop,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0., 0.0, 0.1]),
                    variance=torch.tensor([0.0001, 0.0001, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([10000, 10000, 1])
                )
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
        ]
        return rules

class SteamerTop(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "models/misc/steamer_top/model.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )


class PlaceSetting(IndependentSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.ones(2)*0.9,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=PersonalPlate,
                xyz_rule=SamePositionRule(),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            ),
            ProductionRule(
                child_type=Teacup,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([-0.1, 0.0, 0.00]),
                    variance=torch.tensor([0.005, 0.01, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            )
        ]
        return rules

class PlaceSettings(IndependentSetNode):
    DISTANCE_FROM_CENTER = 0.6
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.5, 0.5, 0.5, 0.5]),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([cls.DISTANCE_FROM_CENTER, 0., 0.]),
                    variance=torch.tensor([0.01, 0.01, 0.0001])
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([10000, 10000, 1])
                )
            ),
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([-cls.DISTANCE_FROM_CENTER, 0., 0.]),
                    variance=torch.tensor([0.01, 0.01, 0.0001])
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0., 0., np.pi)), np.array([10000, 10000, 1])
                )
            ),
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0., cls.DISTANCE_FROM_CENTER, 0.]),
                    variance=torch.tensor([0.01, 0.01, 0.0001])
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0., 0., -np.pi/2.)), np.array([10000, 10000, 1])
                )
            ),
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0., -cls.DISTANCE_FROM_CENTER, 0.]),
                    variance=torch.tensor([0.01, 0.01, 0.0001])
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0., 0., np.pi/2.)), np.array([10000, 10000, 1])
                )
            )
        ]


class SharedDishes(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.5, max_children=2, start_at_one=True),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=ServingDish,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.025, 0.025, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            )
        ]

class SharedTeapots(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.5, max_children=2, start_at_one=True),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Teapot,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.025, 0.025, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            )
        ]

class SharedSteamers(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.5, max_children=3, start_at_one=True),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=SteamerBottom,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.025, 0.025, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            )
        ]

class SharedStuff(IndependentSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.95, 0.5, 0.9]),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=SharedTeapots,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=SharedDishes,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=SharedSteamers,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
        ]


class Table(AndNode):
    # Place settings + misc common dishware
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom_tf = torch.eye(4)
        geom_tf[2, 3] = -0.8
        geom.register_model_file(geom_tf, "models/misc/cafe_table/model.sdf")
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=PlaceSettings,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=SharedStuff,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
        ]

    