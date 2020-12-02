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

class Cabinet(AndNode, PhysicsGeometryNode):
    ''' Cabinets can produce objects on each internal level. '''
    def __init__(self, name, tf):
        super().__init__(name=name, tf=tf, fixed=True, is_container=True)
        
    def _setup(self):
        # Offset cabinet geometry from the wall
        geom_tf = pose_to_tf_matrix(torch.tensor([0.15, 0., 0., 0., 0., 0.]))
        # TODO(gizatt) Resource path management to be done here...
        model_path = "/home/gizatt/drake/examples/manipulation_station/models/cupboard.sdf"
        # Randomly open doors random amounts.
        # Left door is straight  open at -pi/2 and closed at 0.
        left_door_state = pyro.sample("left_door_state",
                                      dist.Uniform(-np.pi/2., 0.))
        # Right door is straight open at pi/2 and closed at 0.
        right_door_state = pyro.sample("right_door_state",
                                      dist.Uniform(0.0, np.pi/2.))
        self.register_model_file(
            tf=geom_tf, model_path=model_path, root_body_name="cupboard_body",
            q0_dict={
                "left_door_hinge": left_door_state.detach().numpy(),
                "right_door_hinge": right_door_state.detach().numpy()
            })
        # Add clearance geometry to indicate that shelves shouldn't
        # penetrate each other and should have clearance to open the doors.
        clearance_depth = 0.75
        # Offset out from wall just a bit more to give collision detection
        # a margin
        geom_tf = pose_to_tf_matrix(torch.tensor([clearance_depth/2.+0.001, 0., 0., 0., 0., np.pi/2.]))
        geometry = Box(width=0.6, depth=clearance_depth, height=1.)
        self.register_clearance_geometry(geom_tf, geometry)

        # Place shelf nodes.
        # Dimensions of a single shelf, in terms of the 
        shelf_height = 0.13115*2
        bottom_shelf_z_local = -0.3995
        num_shelves = 3
        rules = []
        for k in range(num_shelves):
            rules.append(DeterministicRelativePoseProductionRule(
                child_constructor=PlanarObjectRegion,
                child_name="cabinet_level_%02d" % k,
                relative_tf=pose_to_tf_matrix(torch.tensor(
                    [0.15, 0., bottom_shelf_z_local+shelf_height*k, 0., 0., 0.])),
                object_production_rate=0.5,
                bounds=((-0.1, 0.1), (-0.2, 0.2), (0., 0.2))
            ))
        self.register_production_rules(production_rules=rules)

