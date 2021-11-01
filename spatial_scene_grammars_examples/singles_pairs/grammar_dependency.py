import torch
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.drake_interop import *

import pydrake
import pydrake.geometry as pydrake_geom

''' Scene contains a number of oriented objects
(represented by airplanes)  that can appear either
on their own, or in pairs with tightly coupled position
and orientation. '''

class Object(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        geom.register_model_file(torch.eye(4), "models/xwing.sdf")
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class Pair(AndNode):
    PAIR_XYZ_VAR = 0.01
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
                child_type=Object,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=Object,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.zeros(3),
                    variance=torch.ones(3)*2.*cls.PAIR_XYZ_VAR
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(),
                    [100., 100., 100.]
                )
            )
        ]

class Pairs(GeometricSetNode):
    MAX_N_PAIRS = 3
    P = 0.3
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            p=self.P,
            max_children=self.MAX_N_PAIRS,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [ProductionRule(
            child_type=Pair,
            xyz_rule=WorldFrameGaussianOffsetRule(
                mean=torch.zeros(3),
                variance=torch.tensor([1.0, 1.0, 1.0])
            ),
            rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                RotationMatrix(),
                [0.1, 0.1, 0.1]
            )
        )]

class Singles(GeometricSetNode):
    MAX_N_SINGLES = 3
    P = 0.3
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            p=self.P,
            max_children=self.MAX_N_SINGLES,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [ProductionRule(
            child_type=Object,
            xyz_rule=WorldFrameGaussianOffsetRule(
                mean=torch.zeros(3),
                variance=torch.tensor([1.0, 1.0, 1.0])
            ),
            rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                RotationMatrix(),
                [0.1, 0.1, 0.1]
            )
        )]

class Root(IndependentSetNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        geom.register_model_file(torch.eye(4), "models/deathstar.sdf")
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.8, 0.8]),
            physics_geometry_info=geom,
            observed=True,

        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Singles,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=Pairs,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
        ]