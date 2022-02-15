import glob
import os
from functools import lru_cache

import torch
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.drake_interop import *
from spatial_scene_grammars.constraints import *


import pydrake
import pydrake.geometry as pydrake_geom
from pydrake.all import (
    RollPitchYaw,
    RigidTransform
)

from spatial_scene_grammars_examples.table.grammar import *

class DumplingDecoration(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom_tf = torch.eye(4)
        geom.register_model_file(geom_tf, "models/misc/dumplings/model.sdf")
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )
class EggBunDecoration(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom_tf = torch.eye(4)
        geom.register_model_file(geom_tf, "models/misc/egg_buns/model.sdf")
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )
class RiceWrapDecoration(TerminalNode):
    # Lo mai gai
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom_tf = torch.eye(4)
        geom.register_model_file(geom_tf, "models/misc/rice_wrap/model.sdf")
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )
class ShrimpDumplingsDecoration(TerminalNode):
    # Har gow
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom_tf = torch.eye(4)
        geom.register_model_file(geom_tf, "models/misc/shrimp_dumplings/model.sdf")
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )
class SteamerDecoration(OrNode):
    # Various foods that could be inside.
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            rule_probs=torch.ones(len(self.generate_rules())),
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=DumplingDecoration,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=EggBunDecoration,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=RiceWrapDecoration,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=ShrimpDumplingsDecoration,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
        ]


class EggTartsDecoration(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom_tf = torch.eye(4)
        geom.register_model_file(geom_tf, "models/misc/egg_tarts/model.sdf")
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class ChairDecoration(TerminalNode):
    # Place settings + misc common dishware
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom_tf = torch.eye(4)
        geom.register_model_file(geom_tf, "models/misc/chair/model.sdf")
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class PersonalPlateDecoration(IndependentSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False,
            rule_probs=torch.tensor([0.8])
    )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=FirstChopstick,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.05]),
                    variance=torch.tensor([0.0005, 0.0005, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0., np.pi/2., 0.)), np.array([1000, 1000, 1])
                )
            )
        ]

class PlaceSettingDecoration(AndNode):
    TABLE_HEIGHT = 0.8
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False,
    )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=ChairDecoration,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([-0.4, 0.0, -cls.TABLE_HEIGHT]),
                    variance=torch.tensor([0.01, 0.01, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0., 0., -np.pi/2.)), np.array([1000, 1000, 100])
                )
            )
        ]


decoration_mapping = {
    PlaceSetting: PlaceSettingDecoration,
    PersonalPlate: PersonalPlateDecoration,
    SteamerBottom: SteamerDecoration,
    ServingDish: EggTartsDecoration
}