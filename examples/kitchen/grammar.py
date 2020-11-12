import numpy as np
import time

import pydrake
from pydrake.geometry import Box
from pydrake.math import (RollPitchYaw, RigidTransform)

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro import poutine

from scene_grammar.src.nodes import *
from scene_grammar.src.rules import *
from scene_grammar.src.tree import *
from scene_grammar.src.transform_utils import *
from scene_grammar.src.drake_interop import *

class Kitchen(AndNode, RootNode, SpatialNodeMixin):
    ''' Implements a square-footprint kitchen of varying length/width/height,
    with four walls and a floor. '''
    def __init__(self):
        kitchen_height = pyro.sample("kitchen_height", dist.Uniform(2.0, 3.0))
        kitchen_length = pyro.sample("kitchen_length", dist.Uniform(3.0, 6.0)) # x axis
        kitchen_width = pyro.sample("kitchen_width", dist.Uniform(3.0, 6.0)) # y axis
        # North is +y
        # East is +x
        n_wall_rule = DeterministicRelativePoseProductionRule(
            child_constructor=Wall,
            child_name="north_wall",
            relative_tf=pose_to_tf_matrix(torch.tensor([0., kitchen_width/2., 0., 0., 0., 0.])),
            height=kitchen_height,
            length=kitchen_length
        )
        e_wall_rule = DeterministicRelativePoseProductionRule(
            child_constructor=Wall,
            child_name="east_wall",
            relative_tf=pose_to_tf_matrix(torch.tensor([kitchen_length/2., 0., 0., 0., 0., -np.pi/2.])),
            height=kitchen_height,
            length=kitchen_width
        )
        w_wall_rule = DeterministicRelativePoseProductionRule(
            child_constructor=Wall,
            child_name="west_wall",
            relative_tf=pose_to_tf_matrix(torch.tensor([-kitchen_length/2., 0., 0., 0., 0., np.pi/2.])),
            height=kitchen_height,
            length=kitchen_width
        )
        s_wall_rule = DeterministicRelativePoseProductionRule(
            child_constructor=Wall,
            child_name="south_wall",
            relative_tf=pose_to_tf_matrix(torch.tensor([0., -kitchen_width/2., 0., 0., 0., np.pi])),
            height=kitchen_height,
            length=kitchen_length
        )
        floor_rule = DeterministicRelativePoseProductionRule(
            child_constructor=Floor,
            child_name="floor",
            relative_tf=torch.eye(4),
            length=kitchen_length,
            width=kitchen_width
        )
        
        AndNode.__init__(self, name="kitchen",
                         production_rules=[
                            n_wall_rule,
                            e_wall_rule,
                            w_wall_rule,
                            s_wall_rule,
                            floor_rule])
        SpatialNodeMixin.__init__(self, tf=torch.eye(4))

    @staticmethod
    def sample():
        return Kitchen()

class Wall(TerminalNode, SpatialNodeMixin, PhysicsGeometryNodeMixin):
    def __init__(self, name, tf, height, length):
        SpatialNodeMixin.__init__(self, tf)
        TerminalNode.__init__(self, name)

        # Handle geometry and physics.
        wall_thickness = 0.1
        PhysicsGeometryNodeMixin.__init__(self, fixed=True)
        # Move the collision geometry so the wall surface is at y=0 (local frame),
        # and is open in the -y direction, and the base of the wall is at z=0.
        geom_tf = pose_to_tf_matrix(torch.tensor([0., wall_thickness/2., height/2., 0., 0., 0.]))
        # "Order flip of length/width to get "length" to be the x-axis length scale, and
        # extend edge length so the corners are filled in.
        geometry = Box(width=length+wall_thickness*2., depth=wall_thickness, height=height)
        self.register_geometry(geom_tf, geometry, color=np.array([1., 0.898, 0.706, 1.0]))

class Floor(TerminalNode, SpatialNodeMixin, PhysicsGeometryNodeMixin):
    def __init__(self, name, tf, length, width):
        SpatialNodeMixin.__init__(self, tf)
        TerminalNode.__init__(self, name)

        # Handle geometry and physics.
        floor_depth = 0.1
        PhysicsGeometryNodeMixin.__init__(self, fixed=True)
        # Move the collision geometry so the surface is at z=0
        geom_tf = pose_to_tf_matrix(torch.tensor([0., 0., -floor_depth/2., 0., 0., 0.]))
        # Order flip of length/width to get "length" to be the x-axis scale.
        geometry = Box(width=length, depth=width, height=floor_depth)
        self.register_geometry(geom_tf, geometry, color=np.array([0.8, 0.8, 0.8, 1.0]))


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)

    # Draw + plot an environment and their trees
    start = time.time()
    pyro.clear_param_store()
    trace = poutine.trace(ParseTree.generate_from_root_type).get_trace(root_node_type=Kitchen)
    scene_tree = trace.nodes["_RETURN"]["value"]
    end = time.time()

    for node in scene_tree.nodes:
        if isinstance(node, Node):
            print(node.name, ": ", node.tf.tolist())
    print("Generated data in %f seconds." % (end - start))
    print("Full trace values:" )
    for node_name in trace.nodes.keys():
        if node_name in ["_INPUT", "_RETURN"]:
            continue
        print(node_name, ": ", trace.nodes[node_name]["value"].detach().numpy())

    simulate_scene_tree(scene_tree, T=1., with_meshcat=True)