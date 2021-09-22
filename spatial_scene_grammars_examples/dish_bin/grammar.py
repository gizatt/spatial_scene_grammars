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

class Plate(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "sink/plates_cups_and_bowls/plates/Ecoforms_Plant_Plate_S11Turquoise/model_simplified.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class Cup(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "sink/plates_cups_and_bowls/cups/Cole_Hardware_Mug_Classic_Blue/model_simplified.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class Bowl(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "sink/plates_cups_and_bowls/bowls/Room_Essentials_Bowl_Turquiose/model_simplified.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

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
        ClusterRules = [
            ProductionRule(
                child_type=cluster_type,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ) for cluster_type in ObjectTypes
        ]
        return ClusterRules

class DishBin(GeometricSetNode):
    bin_size = []
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
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
                variance=torch.tensor([0.2, 0.2, 0.2])),
            rotation_rule=UnconstrainedRotationRule()
        )
        return [rule]