from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import glob
import trimesh

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

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.tree import *
from spatial_scene_grammars.transform_utils import *
from spatial_scene_grammars.drake_interop import *


class BoxNode(TerminalNode, PhysicsGeometryNode):
    ''' Terminal box type that grounds into real geometry. '''
    def __init__(self, name, tf, dimensions):
        super().__init__(name=name, tf=tf, fixed=False, dimensions=dimensions)
    
    def _setup(self, dimensions):
        # Generate geometry from specified dimensions.
        geom_tf = torch.eye(4)
        self.width, self.depth, self.height = dimensions

        # Color gets colormapped through self.cmap
        self.cmap = plt.cm.get_cmap("viridis")
        self.color = pyro.sample("color", dist.Uniform(0., 1.))

        geometry = Box(width=self.width, depth=self.depth, height=self.height)
        self.register_geometry(geom_tf, geometry,
                               color=self.cmap(self.color.item()))
        
class RandomlySizedBox(OrNode, SpatialNode):
    ''' Intermediate single box that samples box shape and chooses between
    the different static configurations of the box. The TF of this
    object is precisely on the (assumed infinite) flat support surface
    that the box will rest on. '''
    
    def _setup(self, **kwargs):
        # First pick actual box size.
        size_dist = dist.Uniform(torch.ones(3)*0.025,
                                 torch.ones(3)*0.2)
        self.dimensions = pyro.sample("dimensions", size_dist)

        # Get the valid poses for the box
        valid_poses = self.get_valid_poses_of_box(self.dimensions)

        production_weights = torch.ones(len(valid_poses))
        production_rules = []
        for valid_pose in valid_poses:
            kwargs["dimensions"] = self.dimensions
            kwargs["relative_tf_sampler"] = partial(
                self._sample_box_pose,
                base_pose=valid_pose
            )
            production_rules.append(
                RandomRelativePoseProductionRule(BoxNode, **kwargs)
            )
        self.register_production_rules(production_rules=production_rules,
                                       production_weights=production_weights)

    def _sample_box_pose(self, base_pose):
        # Base pose can be yawed into any desired config.
        yaw_around_support = pyro.sample(
            "object_support_yaw", dist.Uniform(0., 2*np.pi))
        tf = pose_to_tf_matrix(
            torch.tensor([0., 0., 0., 0., 0., yaw_around_support]))
        return torch.matmul(tf, base_pose)

    @staticmethod
    def get_valid_poses_of_box(dimensions):
        # This set, I think, puts each face of the box
        # onto the ground.
        rpys = np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [2., 0., 0.],
            [3., 0., 0.],
            [0., -1., 0.],
            [0., 1., 0.],
        ])*np.pi/2.
        valid_poses = []
        verts = trimesh.creation.box(
            extents=dimensions.detach().cpu().numpy()).vertices
        for rpy in rpys:
            # Figure out z-offset to put the lowest verts at the
            # ground.
            tf = RigidTransform(rpy=RollPitchYaw(rpy), p=np.zeros(3))
            verts_tf = tf.multiply(verts.T)
            tf_mat = tf.GetAsMatrix4()
            tf_mat[2, 3] = -np.min(verts_tf[2, :])
            valid_poses.append(torch.tensor(tf_mat))
        return valid_poses

class Table(GeometricSetNode, PhysicsGeometryNode):
    '''
        Produces a geometric number of RandomlySizedBox in a bounded volume
        by randomly sampling their placement on the surface.

        Args:
            object_production_rate: Control parameter for the geometric distribution
                over object count.
            table_size: (x_len, y_len) tuple
    '''
    def __init__(self, name, tf, object_production_rate, table_size):
        super().__init__(
            name=name, tf=tf, object_production_rate=object_production_rate,
            table_size=table_size, fixed=True)
        
    def _setup(self, object_production_rate, table_size):
        # Put surface of 1-high tabletop at z=0
        self.table_size = table_size
        self.eps = 1E-2 # Object centers won't be within this margin of edges
        geom_tf = pose_to_tf_matrix(torch.tensor([0., 0., -0.5, 0., 0., 0.]))
        geometry = Box(width=table_size[0],
                       depth=table_size[1],
                       height=1.0)
        self.register_geometry(geom_tf, geometry, color=np.array([0.9, 0.9, 0.9, 1.0]))

        # Produce a geometric number of objects, using a custom
        # pose sampling rule to generate stable poses.
        self.register_production_rules(
            production_rule_type=RandomRelativePoseProductionRule,
            production_rule_kwargs={
                "child_type": RandomlySizedBox,
                "relative_tf_sampler": self._sample_box_pose,
            },
            geometric_prob=object_production_rate
        )

    def _sample_box_pose(self):
        # Put XY anywhere inside of table, with a tiny margin.
        tabletop_dist = dist.Uniform(
            -self.table_size/2. + self.eps,
            self.table_size/2. - self.eps
        )
        xy = pyro.sample("object_xy_on_table", tabletop_dist)
        tf = pose_to_tf_matrix(
            torch.tensor([xy[0], xy[1], 0., 0., 0., 0.]))
        return tf