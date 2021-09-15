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
Minimal grammar that always comes out with a tree with the same structure,
but with unobservable intermediate node pose.

ROOT -> HIDDEN NODE -> VISIBLE NODE

where both hidden node and visible node are offset with a unit normal
from their parent in xyz, but have identical rotation.
'''

# Deviation from planar epsilon
eps = 1E-3

## Food waste
class VisibleNode(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        geom.register_geometry(
            tf=drake_tf_to_torch_tf(RigidTransform(p=[0., 0., 0.05])),
            geometry=pydrake_geom.Box(0.1, 0.1, 0.1),
            color=np.array([0.8, 0.5, 0.2, 1.0])
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )


class HiddenNode(AndNode):
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
                child_type=VisibleNode,
                xyz_rule=AxisAlignedGaussianOffsetRule(
                    mean=torch.zeros(3),
                    variance=torch.tensor([1.0, 1.0, eps])),
                rotation_rule=SameRotationRule() 
            )
        ]


class RootNode(AndNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        geom.register_geometry(
            tf=drake_tf_to_torch_tf(RigidTransform(p=[0., 0., -0.05])),
            geometry=pydrake_geom.Box(0.1, 0.1, 0.1),
            color=np.array([0.2, 0.5, 0.8, 1.0])
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=HiddenNode,
                xyz_rule=AxisAlignedGaussianOffsetRule(
                    mean=torch.zeros(3),
                    variance=torch.tensor([1.0, 2.0, eps])),
                rotation_rule=SameRotationRule()
            )
        ]