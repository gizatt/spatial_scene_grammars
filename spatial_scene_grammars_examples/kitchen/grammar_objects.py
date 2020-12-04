from functools import partial
import numpy as np
import os
import time
import glob

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


class KitchenObject():
    ''' Concrete object we might want to manipulate. '''
    pass


class MediumBoxObject(TerminalNode, PhysicsGeometryNode, KitchenObject):
    def __init__(self, name, tf):
        super().__init__(name=name, tf=tf, fixed=False)

    def _setup(self):
        # Rotate cabinet so it opens away from the wall
        geom_tf = torch.eye(4)
        # TODO(gizatt) Resource path management to be done here...
        model_path = "/home/gizatt/drake/examples/kuka_iiwa_arm/models/objects/block_for_pick_and_place_mid_size.urdf"
        self.register_model_file(tf=geom_tf, model_path=model_path)


class RandomKitchenStuff(TerminalNode, PhysicsGeometryNode, KitchenObject):
    '''
    Randomly samples sdfs of kitchen stuff. Specializably by its
    style attribute.

    TODO: Discrete attributes controlling the branching structure is dangerous...
    somehow they feel harder to guess than continuous attributes. But I think
    the logic is the same...
    '''
    foodstuffs_paths = glob.glob(
        "/home/gizatt/projects/scene_grammar/models/foodstuffs/*/model_simplified.sdf"
    )
    utensils_paths = glob.glob(
        "/home/gizatt/projects/scene_grammar/models/plates_and_things/*/model_simplified.sdf"
    )
    def __init__(self, name, tf, style_group="all"):
        super().__init__(name=name, tf=tf, style_group="all", fixed=False)

    def _setup(self, style_group):
        geom_tf = torch.eye(4)
        # TODO(gizatt) Resource path management to be done here...
        if style_group == "all":
            available_model_paths = self.foodstuffs_paths + self.utensils_paths
        elif style_group == "foodstuffs":
            available_model_paths = self.foodstuffs_paths
        elif style_group == "utensils":
            available_model_paths = self.utensils_paths
        else:
            raise ValueError("%s not a valid style_group" % style_group)

        assert len(available_model_paths) > 0
        # This is a different kind of randomness than stuff being tracked
        # within the tree -- this is a random choice affecting perceptual / 
        # geometry grounding, not tree structure. But it does impact the
        # tree, since it impacts physical feasibility... so is this the
        # right place to be making this choice? Should there be a different
        # terminal node for every geometry? (Almost certainly no to that --
        # what would I do about continuous shape variation in that case?)
        # Choose an available model at random.
        model_index = pyro.sample("model_type", dist.Categorical(
            torch.ones(len(available_model_paths)))).item()
        model_path = available_model_paths[model_index]
        self.register_model_file(tf=geom_tf, model_path=model_path)


class RandomYCBFoodstuff(TerminalNode, PhysicsGeometryNode, KitchenObject):
    '''
    Randomly samples one of the YCBs available in drake/manipulation/models/ycb.

    WARNING: At least on my computer right now, this lags out meshcat quite a lot --
    takes a few seconds to load the scene. Probably related to Drake issue #13038.
    '''
    def __init__(self, name, tf):
        super().__init__(name=name, tf=tf, fixed=False)

    def _setup(self):
        geom_tf = torch.eye(4)
        # TODO(gizatt) Resource path management to be done here...
        available_model_paths = glob.glob(
            "/home/gizatt/drake/build/install/share/drake/manipulation/models/ycb/sdf/*.sdf"
        )
        assert len(available_model_paths) > 0
        # This is a different kind of randomness than stuff being tracked
        # within the tree -- this is a random choice affecting perceptual / 
        # geometry grounding, not tree structure. But it does impact the
        # tree, since it impacts physical feasibility... so is this the
        # right place to be making this choice? Should there be a different
        # terminal node for every geometry? (Almost certainly no to that --
        # what would I do about continuous shape variation in that case?)
        # Choose an available model at random.
        model_index = pyro.sample("model_type", dist.Categorical(
            torch.ones(len(available_model_paths)))).item()
        model_path = available_model_paths[model_index]
        self.register_model_file(tf=geom_tf, model_path=model_path)


class PlanarObjectRegion(GeometricSetNode, PhysicsGeometryNode):
    '''
        Produces a geometric number of objects in a bounded volume
        by randomly sampling their placement on the surface.

        Randomly chooses between being a cluster of foodstuffs or utensils.

        Args:
            object_production_rate: Control parameter for the geometric distribution
                over object count.
            bounds: [[x_l, x_u], [y_l, y_u], [z_l, z_u]]
            show_geometry: Adds visual geometry indicating the object spawn region.
    '''
    def __init__(self, name, tf, object_production_rate, bounds, show_geometry=False):
        super().__init__(
            name=name, tf=tf, object_production_rate=object_production_rate,
            bounds=bounds, show_geometry=show_geometry, fixed=True)
        
    def _setup(self, object_production_rate, bounds, show_geometry):
        self.x_bounds = bounds[0]
        self.y_bounds = bounds[1]
        self.z_bounds = bounds[2]
        # Add some geometry for viz purposes
        geom_tf = pose_to_tf_matrix(torch.tensor([
            np.mean(self.x_bounds),
            np.mean(self.y_bounds),
            np.mean(self.z_bounds),
            0., 0., 0.]))
        geometry = Box(width=self.x_bounds[1] - self.x_bounds[0],
                       depth=self.y_bounds[1] - self.y_bounds[0],
                       height=self.z_bounds[1] - self.z_bounds[0])
        if show_geometry:
            self.register_visual_geometry(geom_tf, geometry, color=np.array([0.5, 1.0, 0.2, 0.2]))

        # Overriding style choices to just make simple boxes
        # so I can dev some collision stuff.
        style_group_options = ["utensils", "foodstuffs"]
        # Do foodstuffs more often than plates and things
        style_group_k = pyro.sample("style",
                                    dist.Categorical(torch.tensor([0.3, 0.7]))).item()
        style_group = style_group_options[style_group_k]
        # Produce a geometric number of objects within bounds.
        self.register_production_rules(
            production_rule_type=RandomRelativePoseProductionRule,
            production_rule_kwargs={
                "child_constructor": RandomKitchenStuff,
                "child_name": "object",
                "relative_tf_sampler": self._sample_object_pose,
                "style_group": style_group
            },
            geometric_prob=object_production_rate
        )

    def _sample_object_pose(self):
        # For now, hard-code cabinet size to help it not intersect the other walls...
        x_on_shelf = pyro.sample("object_x",
                                dist.Uniform(self.x_bounds[0],
                                             self.x_bounds[1]))
        y_on_shelf = pyro.sample("object_y",
                                dist.Uniform(self.y_bounds[0],
                                             self.y_bounds[1]))
        yaw = pyro.sample("object_yaw",
                          dist.Uniform(0., np.pi*2.))
        return pose_to_tf_matrix(torch.tensor([x_on_shelf, y_on_shelf, np.mean(self.z_bounds),
                                               0., 0., yaw]))