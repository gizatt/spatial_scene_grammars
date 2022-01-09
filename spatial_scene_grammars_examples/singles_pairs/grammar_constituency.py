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
    # TODO(gizatt) This node adds some parsing ambiguity: the children
    # are exchangeable, so both orderings are equivalent parses. Changing
    # this to a RepeatingSetNode with rule_probs [0, 0, 1] resolves this issue,
    # but I'm not making the change now as it makes for a more confusing-looking
    # grammar...
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
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.zeros(3),
                    variance=torch.ones(3)*cls.PAIR_XYZ_VAR
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0., 0., 1.)), # This 1 is weird, but I'm keeping it until I finish figure gen
                    [100., 100., 100.]
                )
            ),
            ProductionRule(
                child_type=Object,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.zeros(3),
                    variance=torch.ones(3)*cls.PAIR_XYZ_VAR
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0., 0., 1.)), # Likewise
                    [100., 100., 100.]
                )
            )
        ]

class Pairs(RepeatingSetNode):
    MAX_N_PAIRS = 3
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.3, max_children=self.MAX_N_PAIRS),
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
                RotationMatrix(RollPitchYaw(1., 0., 0.)), # Likewise
                [0.1, 0.1, 0.1]
            )
        )]

class Singles(RepeatingSetNode):
    MAX_N_SINGLES = 3
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.3, max_children=self.MAX_N_SINGLES),
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
                RotationMatrix(RollPitchYaw(1., 0., 0.)), # Likewise
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