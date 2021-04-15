from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import glob
import trimesh

import pydrake
from pydrake.all import (
    Box, RollPitchYaw, RigidTransform, RotationMatrix
)

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoname import name_count

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.transform_utils import *
from spatial_scene_grammars.drake_interop import *


class BoxNode(TerminalNode):
    ''' Terminal box type that grounds into real geometry. '''
    
    # Base rotation is specialized by RandomlyOrientedBox
    # to be a 3x3 rotation matrix.
    base_rot = None

    def _instantiate_impl(self, derived_attributes):
        # Sample own dimensions
        size_dist = dist.Uniform(torch.ones(3)*0.025,
                                 torch.ones(3)*0.2)
        self.dimensions = pyro.sample("dimensions", size_dist)

        # Sample yaw around support
        self.yaw_around_support = pyro.sample(
            "support_yaw", dist.Uniform(0., 2*np.pi))
        yaw_tf = pose_to_tf_matrix(
            torch.tensor([0., 0., 0., 0., 0., self.yaw_around_support]))

        # Resolve true pose using dimensions and yaw
        assert self.base_rot is not None, \
               "%s not specialized." % self.__class__.__name__
        # Transform vertices using base pose to calculate vertical offset.
        verts = trimesh.creation.box(extents=torch.ones(3)).vertices
        verts = torch.tensor(verts) * self.dimensions
        verts_tf = torch.matmul(self.base_rot, verts.T)
        lowest_vert = torch.min(verts_tf[2, :])
        support_tf = pose_to_tf_matrix(
            torch.tensor([0., 0., -lowest_vert, 0., 0., 0.]))

        # TODO: Is this the right order?
        base_tf = torch.eye(4)
        base_tf[:3, :3] = self.base_rot
        elevated_yawed_support = torch.matmul(yaw_tf, torch.matmul(support_tf, base_tf))
        self.tf = torch.matmul(derived_attributes["tf"], elevated_yawed_support)

        # Make own geometry using our dimensions and a randomly
        # sampled color.
        geom_tf = torch.eye(4)
        self.width, self.depth, self.height = self.dimensions

        # Color gets colormapped through self.cmap
        self.cmap = plt.cm.get_cmap("viridis")
        self.color = pyro.sample("color", dist.Uniform(0., 1.))

        self.physics_geometry_info = PhysicsGeometryInfo(
            fixed=False, spatial_inertia=default_spatial_inertia,
            is_container=False)
        geometry = Box(width=self.width, depth=self.depth, height=self.height)
        self.physics_geometry_info.register_geometry(
            geom_tf, geometry, color=self.cmap(self.color.item()))
        

class RandomlyOrientedBox(OrNode):
    ''' Intermediate single box that chooses between
    the different static configurations of a unit-scaled box.
    The base_rot supplied to generated children puts the bottom
    surface of the unit box precisely on the ground. The tf 
    of *this* node is precisely on the table surface. '''
    def __init__(self):
        # Generate candidate box orientations and specialize
        # the BoxNode for being in those orientations.
        self.valid_base_rots = self.get_valid_rotations_of_box()
        # Specialize BoxNode with base_rot set to the base poses.
        child_types = [
            type("BoxNode_Config%d" % k,
                (BoxNode,),
                {"base_rot": R})
            for k, R in enumerate(self.valid_base_rots)
        ]
        super().__init__(child_types, torch.ones(len(child_types)))

    def _instantiate_impl(self, derived_attributes):
        self.tf = derived_attributes["tf"]

    def _instantiate_children_impl(self, children):
        all_attrs = []
        for k, child in enumerate(children):
            all_attrs.append({"tf": self.tf})
        return all_attrs

    @staticmethod
    def get_valid_rotations_of_box():
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
        valid_rots = []
        
        for rpy in rpys:
            # Figure out z-offset to put the lowest verts at the
            # ground.
            valid_rots.append(torch.tensor(RotationMatrix(RollPitchYaw(rpy)).matrix()))
        return valid_rots


class Table(GeometricSetNode):
    '''
        Produces a geometric number of RandomlyOrientedBox in a bounded volume
        by randomly sampling their placement on the surface.

        Args:
            object_production_rate: Control parameter for the geometric distribution
                over object count.
            table_size: (x_len, y_len) tuple
    '''
    geometric_prob = 0.4
    table_size = torch.tensor([0.5, 0.5])
    eps = 1E-2 # Object centers won't be within this margin of edges
    def __init__(self):
        super().__init__(child_type=RandomlyOrientedBox,
                         geometric_prob=self.geometric_prob,
                         max_repeats=10)

    def _instantiate_impl(self, derived_attributes):
        self.tf = derived_attributes["tf"]
        self.physics_geometry_info = PhysicsGeometryInfo(
            fixed=True, spatial_inertia=None, is_container=True)

        # Put surface of 1-high tabletop at z=0
        geom_tf = pose_to_tf_matrix(torch.tensor([0., 0., -0.5, 0., 0., 0.]))
        geometry = Box(width=self.table_size[0],
                       depth=self.table_size[1],
                       height=1.0)
        self.physics_geometry_info.register_geometry(
            geom_tf, geometry, color=np.array([0.9, 0.9, 0.9, 1.0])
        )

    def _instantiate_children_impl(self, children):
        all_attrs = []
        for k, child in enumerate(children):
            # Put XY anywhere inside of table, with a tiny margin.
            tabletop_dist = dist.Uniform(
                -self.table_size/2. + self.eps,
                self.table_size/2. - self.eps
            )
            xy = pyro.sample("object_%d_xy_on_table" % k, tabletop_dist)
            tf = pose_to_tf_matrix(
                torch.tensor([xy[0], xy[1], 0., 0., 0., 0.]))
            all_attrs.append({"tf": torch.matmul(self.tf, tf)})
        return all_attrs