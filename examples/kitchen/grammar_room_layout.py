from functools import partial
import numpy as np
import os
import time

import pydrake
from pydrake.all import (
    Box, RollPitchYaw, RigidTransform, Parser
)

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoname import name_count

from scene_grammar.src.nodes import *
from scene_grammar.src.rules import *
from scene_grammar.src.tree import *
from scene_grammar.src.transform_utils import *
from scene_grammar.src.drake_interop import *

from .grammar_objects import *
from .grammar_table import *
from .grammar_cabinet import *

class Kitchen(AndNode, SpatialNode):
    ''' Implements a square-footprint kitchen of varying length/width/height,
    with four walls and a floor. '''
    def __init__(self, name, tf):
        super().__init__(name=name, tf=tf)

    def _setup(self):
        kitchen_height = pyro.sample("kitchen_height", dist.Uniform(2.0, 3.0))
        kitchen_width = pyro.sample("kitchen_width", dist.Uniform(2.0, 4.0)) # x axis
        kitchen_length = pyro.sample("kitchen_length", dist.Uniform(2.0, 4.0)) # y axis
        # North is +y
        # East is +x
        n_wall_rule = DeterministicRelativePoseProductionRule(
            child_constructor=Wall,
            child_name="north_wall",
            relative_tf=pose_to_tf_matrix(torch.tensor([0., kitchen_length/2., 0., 0., 0., 0.])),
            height=kitchen_height,
            width=kitchen_width
        )
        e_wall_rule = DeterministicRelativePoseProductionRule(
            child_constructor=Wall,
            child_name="east_wall",
            relative_tf=pose_to_tf_matrix(torch.tensor([kitchen_width/2., 0., 0., 0., 0., -np.pi/2.])),
            height=kitchen_height,
            width=kitchen_length
        )
        w_wall_rule = DeterministicRelativePoseProductionRule(
            child_constructor=Wall,
            child_name="west_wall",
            relative_tf=pose_to_tf_matrix(torch.tensor([-kitchen_width/2., 0., 0., 0., 0., np.pi/2.])),
            height=kitchen_height,
            width=kitchen_length
        )
        s_wall_rule = DeterministicRelativePoseProductionRule(
            child_constructor=Wall,
            child_name="south_wall",
            relative_tf=pose_to_tf_matrix(torch.tensor([0., -kitchen_length/2., 0., 0., 0., np.pi])),
            height=kitchen_height,
            width=kitchen_width
        )
        floor_rule = DeterministicRelativePoseProductionRule(
            child_constructor=Floor,
            child_name="floor",
            relative_tf=torch.eye(4),
            width=kitchen_width,
            length=kitchen_length
        )
        
        self.register_production_rules(
            production_rules=[
                n_wall_rule,
                e_wall_rule,
                #w_wall_rule,
                #s_wall_rule,
                floor_rule])

class Wall(GeometricSetNode, PhysicsGeometryNode):
    ''' Each wall can produce some number of cabinets on its surface and
    random (feasible) positions. '''

    def __init__(self, name, tf, height, width):
        super().__init__(name=name, tf=tf, height=height, width=width, fixed=True)

    def _setup(self, height, width):
        # Handle geometry and physics.
        self.wall_thickness = 0.1
        self.width = width
        self.height = height
        # Move the collision geometry so the wall surface is at y=0 (local frame),
        # and is open in the -y direction, and the base of the wall is at z=0.
        geom_tf = pose_to_tf_matrix(torch.tensor([0., self.wall_thickness/2., height/2., 0., 0., 0.]))
        geometry = Box(width=width, depth=self.wall_thickness, height=height)
        self.register_geometry(geom_tf, geometry, color=np.array([1., 0.898, 0.706, 1.0]))
        # Use the same geom as clearance geometry
        self.register_clearance_geometry(geom_tf, geometry)

        # This node produces a geometric number of cabinets on its surface.
        self.register_production_rules(
            production_rule_type=RandomRelativePoseProductionRule,
            production_rule_kwargs={
                "child_constructor": Cabinet,
                "child_name": "cabinet",
                "relative_tf_sampler": self._sample_cabinet_pose_on_wall
            },
            geometric_prob=0.5
        )

    def _sample_cabinet_pose_on_wall(self):
        # For now, hard-code cabinet size to help it not intersect the other walls...
        min_cab_height = 0.5
        max_cab_height = 1.5
        cabinet_width = 0.6
        x_on_wall = pyro.sample("cabinet_x",
                                dist.Uniform(-self.width/2. + cabinet_width/2.,
                                              self.width/2. - cabinet_width/2.))
        z_on_wall = pyro.sample("cabinet_z",
                                dist.Uniform(min_cab_height, max_cab_height))
        return pose_to_tf_matrix(torch.tensor([x_on_wall, 0., z_on_wall, 0., 0., -np.pi/2.]))


class Floor(AndNode, PhysicsGeometryNode):
    def __init__(self, name, tf, width, length):
        super().__init__(name=name, tf=tf, width=width, length=length)

    def _setup(self, width, length):
        floor_depth = 0.1
        # Move the collision geometry so the surface is at z=0
        geom_tf = pose_to_tf_matrix(torch.tensor([0., 0., -floor_depth/2., 0., 0., 0.]))
        geometry = Box(width=width, depth=length, height=floor_depth)
        self.register_geometry(geom_tf, geometry, color=np.array([0.8, 0.8, 0.8, 1.0]))

        # Spawn a table at a determined location.
        # (Currently just for testing item placement.)
        table_spawn_rule = DeterministicRelativePoseProductionRule(
            child_constructor=Table,
            child_name="table",
            relative_tf=pose_to_tf_matrix(torch.tensor([1., 0., 0., 0., 0., 0.]))
        )
        robot_spawn_rule = DeterministicRelativePoseProductionRule(
            child_constructor=RobotSpawnLocation,
            child_name="robot_spawn",
            relative_tf=pose_to_tf_matrix(torch.tensor([-1, 0., 0., 0., 0., 0.]))
        )
        self.register_production_rules(
            production_rules=[table_spawn_rule,
                              robot_spawn_rule]
        )


class RobotSpawnLocation(TerminalNode, PhysicsGeometryNode):
    ''' Node with clearance geometry to indicate where the robot should
    spawn in. '''
    def __init__(self, name, tf):
        super().__init__(name=name, tf=tf)

    def _setup(self):
        # Only has clearance geometry: indicates robot start location, and
        # ensures that there's space to put a robot into this scene.
        geom_tf = pose_to_tf_matrix(torch.tensor([0., 0., 1., 0., 0., 0.]))
        geometry = Box(width=1., depth=1., height=2., )
        self.register_clearance_geometry(geom_tf, geometry)