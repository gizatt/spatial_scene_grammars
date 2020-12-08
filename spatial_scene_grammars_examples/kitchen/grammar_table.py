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

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.tree import *
from spatial_scene_grammars.transform_utils import *
from spatial_scene_grammars.drake_interop import *

from .grammar_objects import *

class Table(AndNode, PhysicsGeometryNode):
    ''' Table (using the extra heavy duty table from Drake) that
    produces objects on its surface. '''
    def __init__(self, name, tf):
        # Handle geometry and physics.
        super().__init__(name=name, tf=tf, fixed=True, is_container=True)

    def _setup(self):
        geom_tf = pose_to_tf_matrix(torch.tensor([0., 0., 0., 0., 0., 0.]))
        # TODO(gizatt) Resource path management to be done here...
        model_path = "/home/gizatt/drake/examples/kuka_iiwa_arm/models/table/extra_heavy_duty_table_surface_only_collision.sdf"
        self.register_model_file(tf=geom_tf, model_path=model_path, root_body_name="link")
        geom_tf = pose_to_tf_matrix(torch.tensor([0., 0., 0.6251, 0., 0., 0.]))
        self.register_clearance_geometry(
            tf=geom_tf, 
            geometry=Box(width=1., depth=1., height=1.0))
        # Put an object spawning volume on the table surface.
        rules = []
        rules.append(DeterministicRelativePoseProductionRule(
                child_type=PlanarObjectRegion,
                relative_tf=pose_to_tf_matrix(torch.tensor(
                    [0., 0., 0.8, 0., 0., 0.])),
                object_production_rate=0.5,
                bounds=((-0.25, 0.25), (-0.25, 0.25), (0., 0.2))
        ))
        self.register_production_rules(production_rules=rules)