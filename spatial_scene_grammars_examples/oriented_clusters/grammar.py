import torch
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.drake_interop import *

import pydrake
import pydrake.geometry as pydrake_geom

''' Root creates geometric number of oriented
clusters with random rotations in the unit box.
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


class BoxPreYRotation(AndNode):
    LongBoxRule = ProductionRule(
        child_type=LongBox,
        xyz_rule=AxisAlignedBBoxRule.from_bounds(
            lb=torch.zeros(3),
            ub=torch.zeros(3)
        ),
        rotation_rule=UniformBoundedRevoluteJointRule.from_bounds(
            axis=torch.tensor([0., 1., 0.]),
            lb=-np.pi/8., ub=np.pi/8.
        )
    )
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [BoxPreYRotation.LongBoxRule]


class OrientedCluster(GeometricSetNode):
    BoxPreYRotationRule = ProductionRule(
        child_type=LongBox,
        xyz_rule=AxisAlignedBBoxRule.from_bounds(
            lb=-torch.ones(3)*0.1,
            ub=torch.ones(3)*0.1
        ),
        rotation_rule=UniformBoundedRevoluteJointRule.from_bounds(
            axis=torch.tensor([1., 0., 0.]),
            lb=-np.pi/8., ub=np.pi/8.
        )
    )
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
        return [OrientedCluster.BoxPreYRotationRule]

class OrientedClusterRoot(GeometricSetNode):
    OrientedClusterRule = ProductionRule(
        child_type=OrientedCluster,
        xyz_rule=AxisAlignedBBoxRule.from_bounds(lb=torch.zeros(3), ub=torch.ones(3)),
        rotation_rule=UnconstrainedRotationRule()
    )
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            p=0.3,
            max_children=2,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [OrientedClusterRoot.OrientedClusterRule]