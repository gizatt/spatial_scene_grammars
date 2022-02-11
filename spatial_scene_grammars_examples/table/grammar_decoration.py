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

class PlaceSettingDecoration(IndependentSetNode):
    TABLE_HEIGHT = 0.8
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False,
            rule_probs=torch.tensor([0.75, 1.0])
    )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=FirstChopstick,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.02]),
                    variance=torch.tensor([0.005, 0.005, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0., np.pi/2., 0.)), np.array([1000, 1000, 1])
                )
            ),
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
    PlaceSetting: PlaceSettingDecoration
}