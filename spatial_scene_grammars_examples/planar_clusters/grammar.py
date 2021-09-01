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

Themed sort of like the restaurant grammar in
"Synthesizing Open Worlds with Constraints using
Locally Annealed Reversible Jump MCMC": geometric
production of many tables, each of which might produce
chairs at each cardinal direction.

Root -> Geometric # of (square, axis-aligned) Tables
Table -> MaybeChairs at 4 offsets
MaybeChairs -> Chair or Null
'''

class Chair(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo()
        geom.register_geometry(
            tf=drake_tf_to_torch_tf(RigidTransform(p=[0., 0., .25])),
            geometry=pydrake_geom.Box(0.5, 0.5, 0.5),
            color=np.array([0.8, 0.5, 0.2, 1.0])
        )
        geom.register_geometry(
            tf=drake_tf_to_torch_tf(RigidTransform(p=[0.25, 0., .5])),
            geometry=pydrake_geom.Box(0.1, 0.5, 1.0),
            color=np.array([0.8, 0.5, 0.2, 1.0])
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )


class Table(IndependentSetNode):
    def __init__(self, tf):
        chair_offset = RigidTransform(p=[1.0, 0., 0.])
        rules = []
        for k in range(4):
            angle = k*np.pi/2.
            rotation = RigidTransform(p=np.zeros(3), rpy=RollPitchYaw(0., 0., angle))
            chair_centroid = rotation.multiply(chair_offset)
            xyz_center = torch.tensor(chair_centroid.translation().copy())
            rules.append(
                ProductionRule(
                    child_type=Chair,
                    xyz_rule=AxisAlignedBBoxRule(
                        lb=xyz_center-torch.ones(3)*0.1,
                        ub=xyz_center+torch.ones(3)*0.1
                    ),
                    rotation_rule=UniformBoundedRevoluteJointRule(
                        axis=torch.tensor([0., 0., 1.]),
                        lb=angle-np.pi/8., ub=angle+np.pi/8.
                    )
                )
            )
        geom = PhysicsGeometryInfo()
        geom.register_geometry(
            tf=drake_tf_to_torch_tf(RigidTransform(p=[0., 0., .5])),
            geometry=pydrake_geom.Box(1.0, 1.0, 1.0),
            color=np.array([0.8, 0.8, 0.85, 1.0])
        )
        super().__init__(
            rules=rules,
            rule_probs=torch.ones(4)*0.5,
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )


class RestaurantRoom(GeometricSetNode):
    def __init__(self, tf, room_size=[10., 10.]):
        rule = ProductionRule(
            child_type=Table,
            xyz_rule=AxisAlignedBBoxRule(lb=torch.zeros(3), ub=torch.tensor([room_size[0], room_size[1], 0.])),
            rotation_rule=UniformBoundedRevoluteJointRule(
                axis=torch.tensor([0., 0., 1.]),
                lb=0., ub=0.
            )
        )
        geom = PhysicsGeometryInfo()
        geom.register_geometry(
            tf=drake_tf_to_torch_tf(RigidTransform(p=[room_size[0]/2., room_size[1]/2., -0.5])),
            geometry=pydrake_geom.Box(room_size[0], room_size[1], 1.0),
            color=np.array([0.3, 0.2, 0.2, 1.0])
        )
        super().__init__(
            rule=rule,
            tf=tf,
            p=0.2,
            max_children=5,
            physics_geometry_info=geom,
            observed=True
        )