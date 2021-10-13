import torch
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.drake_interop import *

import pydrake
import pydrake.geometry as pydrake_geom

''' Root creates geometric number of oriented
clusters with random rotations, with xyz distributed
as a random normal.
Each cluster produces a geometric number of long boxes with
minor rotations around their non-long axes. '''

class LongBox(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        geom.register_geometry(
            tf=torch.eye(4),
            geometry=pydrake_geom.Box(0.025, 0.025, 0.1),
            color=np.array([0.8, 0.5, 0.2, 1.0])
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class OrientedCluster(GeometricSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            p=0.3,
            max_children=5,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [ProductionRule(
            child_type=LongBox,
            xyz_rule=ParentFrameGaussianOffsetRule(
                mean=torch.zeros(3),
                variance=torch.tensor([0.05, 0.05, 0.01])
            ),
            rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                RotationMatrix(RollPitchYaw(0., 0., 1.)),
                [100., 100., 10.]
            )
        )]

class OrientedClusterRoot(GeometricSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            p=0.3,
            max_children=3,
            physics_geometry_info=None,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        return [ProductionRule(
            child_type=OrientedCluster,
            xyz_rule=ParentFrameGaussianOffsetRule(
                mean=torch.zeros(3),
                variance=torch.tensor([1.0, 1.0, 0.01])
            ),
            rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                RotationMatrix(RollPitchYaw(1., 0., 0.)),
                [0.1, 0.1, 0.1]
            )
        )]