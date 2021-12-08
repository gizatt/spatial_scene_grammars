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
Root can also sometimes produce boxes on their own, to account
for outliers.
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

class OrientedCluster(RepeatingSetNode):
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
            xyz_rule=SamePositionRule(),#WorldFrameGaussianOffsetRule(
                #mean=torch.zeros(3),
                #variance=torch.tensor([0.05, 0.05, 0.001])
            #),
            rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                RotationMatrix(RollPitchYaw(0., 0., 1.)),
                [1000., 1000., 1.]
            )
        )]

class AssortedOrientedClusters(RepeatingSetNode):
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
            xyz_rule=SamePositionRule(),#WorldFrameGaussianOffsetRule(
                #mean=torch.zeros(3),
                #variance=torch.tensor([1.0, 1.0, 1.0])
            #),
            rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                RotationMatrix(RollPitchYaw(1., 0., 0.)),
                [0.1, 0.1, 0.1]
            )
        )]


class AssortedOutliers(RepeatingSetNode):
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
            child_type=LongBox,
            xyz_rule=SamePositionRule(),#WorldFrameGaussianOffsetRule(
                #mean=torch.zeros(3),
                #variance=torch.tensor([1.0, 1.0, 1.0])
            #),
            rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                RotationMatrix(RollPitchYaw(1., 0., 0.)),
                [0.1, 0.1, 0.1]
            )
        )]

class OrientedClusterRoot(IndependentSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.8, 0.8]),
            physics_geometry_info=None,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=AssortedOutliers,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=AssortedOrientedClusters,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
        ]