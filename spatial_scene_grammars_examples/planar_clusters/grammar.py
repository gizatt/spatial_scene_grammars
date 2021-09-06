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
''' Simple grammar that flexes most of the rule
types while keeping all objects in the XY plane for
easy visualization.

Themed like clutter on a desk: a desk produces
clusters of stuff on it. Clusters can be clusters
of food waste, papers, or pencils. Each cluster
has a geometric number of stuff distributed
locally.
'''

## Food waste
class Plate(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        geom.register_geometry(
            tf=drake_tf_to_torch_tf(RigidTransform(p=[0., 0., .025/2.])),
            geometry=pydrake_geom.Cylinder(radius=0.1, length=0.025),
            color=np.array([0.8, 0.5, 0.2, 1.0])
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class Drink(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        geom.register_geometry(
            tf=drake_tf_to_torch_tf(RigidTransform(p=[0., 0., .1/2.])),
            geometry=pydrake_geom.Cylinder(radius=0.05, length=0.1),
            color=np.array([0.3, 0.8, 0.5, 1.0])
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )


class FoodWasteCluster(IndependentSetNode):
    # Might make either a plate or soda can,
    # the plate might make more stuff on it.
    Stuff = [Plate, Drink]
    Rules = [
        ProductionRule(
            child_type=stuff,
            xyz_rule=AxisAlignedBBoxRule.from_bounds(
                lb=torch.tensor([-0.2, -0.2, 0.0]),
                ub=torch.tensor([0.2, 0.2, 0.0])
            ),
            rotation_rule=UniformBoundedRevoluteJointRule.from_bounds(
                axis=torch.tensor([0., 0., 1.]),
                lb=-np.pi, ub=np.pi
            )   
        ) for stuff in Stuff
    ]
    Rule_Probs = torch.tensor([0.5, 0.8])

    def __init__(self, tf):
        super().__init__(
            rule_probs=self.Rule_Probs,
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return cls.Rules

## Paper stack
class Paper(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        rgba = np.random.uniform([0.85, 0.85, 0.85, 1.0], [0.95, 0.95, 0.95, 1.0])
        geom.register_geometry(
            tf=drake_tf_to_torch_tf(RigidTransform(p=[0., 0., .01/2.])),
            geometry=pydrake_geom.Box(0.2159, 0.2794, 0.01), # 8.5" x 11"
            color=rgba
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class PaperCluster(GeometricSetNode):
    # Make a stack of papers
    PaperRule = ProductionRule(
        child_type=Paper,
        xyz_rule=AxisAlignedBBoxRule.from_bounds(
            lb=torch.tensor([-0.05, -0.05, 0.0]),
            ub=torch.tensor([0.05, 0.05, 0.0])
        ),
        rotation_rule=UniformBoundedRevoluteJointRule.from_bounds(
            axis=torch.tensor([0., 0., 1.]),
            lb=-np.pi/8., ub=np.pi/8.
        )   
    )

    def __init__(self, tf):
        super().__init__(
            p=0.3,
            max_children=3,
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )

    @classmethod
    def generate_rules(cls):
        return [cls.PaperRule]

## Pencils
class Pencil(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        rgba = np.random.uniform([0.85, 0.75, 0.45, 1.0], [0.95, 0.85, 0.55, 1.0])
        geom.register_geometry(
            tf=drake_tf_to_torch_tf(RigidTransform(p=[0., 0., .01/2.], rpy=RollPitchYaw(0., np.pi/2., 0.))),
            geometry=pydrake_geom.Cylinder(radius=0.01, length=0.15),
            color=rgba,
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class PencilCluster(GeometricSetNode):
    # Make a geometric cluster of roughly-aligned pencils
    PencilRule = ProductionRule(
        child_type=Pencil,
        xyz_rule=AxisAlignedBBoxRule.from_bounds(
            lb=torch.tensor([-0.05, -0.05, 0.0]),
            ub=torch.tensor([0.05, 0.05, 0.0])
        ),
        rotation_rule=UniformBoundedRevoluteJointRule.from_bounds(
            axis=torch.tensor([0., 0., 1.]),
            lb=-np.pi/8., ub=np.pi/8.
        )   
    )

    def __init__(self, tf):
        super().__init__(
            p=0.5,
            max_children=3,
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [cls.PencilRule]

## Desk and abstract cluster
class ObjectCluster(OrNode):
    # Specialize into a type of cluster
    ClusterTypes = [FoodWasteCluster, PaperCluster, PencilCluster]
    ClusterRules = [
        ProductionRule(
            child_type=cluster_type,
            xyz_rule=AxisAlignedBBoxRule.from_bounds(
                lb=torch.zeros(3),
                ub=torch.zeros(3)
            ),
            rotation_rule=UniformBoundedRevoluteJointRule.from_bounds(
                axis=torch.tensor([0., 0., 1.]),
                lb=0., ub=0.
            )
        ) for cluster_type in ClusterTypes
    ]
    ClusterTypeWeights = torch.tensor([1.0, 1.0, 1.0])
    def __init__(self, tf):
        super().__init__(
            rule_probs=self.ClusterTypeWeights,
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )

    @classmethod
    def generate_rules(cls):
        return cls.ClusterRules

class Desk(GeometricSetNode):
    # Make geometric # of object clusters
    desk_size=[1., 1.]
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        geom.register_geometry(
            tf=drake_tf_to_torch_tf(RigidTransform(p=[self.desk_size[0]/2., self.desk_size[1]/2., -0.5])),
            geometry=pydrake_geom.Box(self.desk_size[0], self.desk_size[1], 1.0),
            color=np.array([0.3, 0.2, 0.2, 1.0])
        )
        super().__init__(
            tf=tf,
            p=0.2,
            max_children=6,
            physics_geometry_info=geom,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        lb = torch.tensor([0.2, 0.2, 0.0])
        ub = torch.tensor([cls.desk_size[0] - 0.2, cls.desk_size[1] - 0.2, 0.0])
        rule = ProductionRule(
            child_type=ObjectCluster,
            xyz_rule=AxisAlignedBBoxRule.from_bounds(lb=lb, ub=ub),
            rotation_rule=UniformBoundedRevoluteJointRule.from_bounds(
                axis=torch.tensor([0., 0., 1.]),
                lb=-np.pi, ub=np.pi
            )
        )
        return [rule]