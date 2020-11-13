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

class Kitchen(AndNode, RootNode, SpatialNodeMixin):
    ''' Implements a square-footprint kitchen of varying length/width/height,
    with four walls and a floor. '''
    def __init__(self):
        # TODO(gizatt) pyro @scope for local variable naming?
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
        
        AndNode.__init__(self, name="kitchen",
                         production_rules=[
                            n_wall_rule,
                            e_wall_rule,
                            #w_wall_rule,
                            #s_wall_rule,
                            floor_rule])
        SpatialNodeMixin.__init__(self, tf=torch.eye(4))

    @staticmethod
    def sample():
        return Kitchen()


class Wall(GeometricSetNode, SpatialNodeMixin, PhysicsGeometryNodeMixin):
    ''' Each wall can produce some number of cabinets on its surface and
    random (feasible) positions. '''

    def __init__(self, name, tf, height, width):
        SpatialNodeMixin.__init__(self, tf)

        # This node produces a geometric number of cabinets on its surface.
        cabinet_production_rule = RandomRelativePoseProductionRule(
            Cabinet, "%s_cabinet" % name, self._sample_cabinet_pose_on_wall
        )
        GeometricSetNode.__init__(
            self, name=name, production_rule=cabinet_production_rule,
            geometric_prob=0.5
        )

        # Handle geometry and physics.
        self.wall_thickness = 0.1
        self.width = width
        self.height = height
        PhysicsGeometryNodeMixin.__init__(self, fixed=True)
        # Move the collision geometry so the wall surface is at y=0 (local frame),
        # and is open in the -y direction, and the base of the wall is at z=0.
        geom_tf = pose_to_tf_matrix(torch.tensor([0., self.wall_thickness/2., height/2., 0., 0., 0.]))
        # Extend wall width so corners get filled in.
        geometry = Box(width=width+self.wall_thickness*2., depth=self.wall_thickness, height=height)
        self.register_geometry(geom_tf, geometry, color=np.array([1., 0.898, 0.706, 1.0]))

    def _sample_cabinet_pose_on_wall(self):
        # For now, hard-code cabinet size to help it not intersect the other walls...
        min_cab_height = 0.5
        max_cab_height = 1.5
        cabinet_width = 0.6
        x_on_wall = pyro.sample("%s_cabinet_x" % self.name,
                                dist.Uniform(-self.width/2. + cabinet_width/2.,
                                              self.width/2. - cabinet_width/2.))
        z_on_wall = pyro.sample("%s_cabinet_z" % self.name,
                                dist.Uniform(min_cab_height, max_cab_height))
        return pose_to_tf_matrix(torch.tensor([x_on_wall, 0., z_on_wall, 0., 0., 0.]))


class Floor(AndNode, SpatialNodeMixin, PhysicsGeometryNodeMixin):
    def __init__(self, name, tf, width, length):
        SpatialNodeMixin.__init__(self, tf)
        
        # Spawn a table at a determined location.
        # (Currently just for testing item placement.)
        table_spawn_rule = DeterministicRelativePoseProductionRule(
            child_constructor=Table,
            child_name="%s_table" % name,
            relative_tf=pose_to_tf_matrix(torch.tensor([1., 0., 0., 0., 0., 0.]))
        )
        AndNode.__init__(self, name=name, production_rules=[table_spawn_rule])

        # Handle geometry and physics.
        floor_depth = 0.1
        PhysicsGeometryNodeMixin.__init__(self, fixed=True)
        # Move the collision geometry so the surface is at z=0
        geom_tf = pose_to_tf_matrix(torch.tensor([0., 0., -floor_depth/2., 0., 0., 0.]))
        geometry = Box(width=width, depth=length, height=floor_depth)
        self.register_geometry(geom_tf, geometry, color=np.array([0.8, 0.8, 0.8, 1.0]))



class Cabinet(IndependentSetNode, SpatialNodeMixin, PhysicsGeometryNodeMixin):
    ''' Cabinets can potentially produce objects on each internal level. '''
    def __init__(self, name, tf):
        SpatialNodeMixin.__init__(self, tf)
        # Handle geometry and physics.
        PhysicsGeometryNodeMixin.__init__(self, fixed=True)
        # Rotate cabinet so it opens away from the wall
        geom_tf = pose_to_tf_matrix(torch.tensor([0., -0.15, 0., 0., 0., -np.pi/2.]))
        # TODO(gizatt) Resource path management to be done here...
        model_path = "/home/gizatt/drake/examples/manipulation_station/models/cupboard.sdf"
        # Randomly open doors random amounts.
        left_door_state = pyro.sample("%s_left_door_state" % name,
                                      dist.Uniform(0., np.pi/2.))
        right_door_state = pyro.sample("%s_right_door_state" % name,
                                      dist.Uniform(0., np.pi/2.))
        self.register_model_file(
            tf=geom_tf, model_path=model_path, root_body_name="cupboard_body",
            q0_dict={
                "left_door_hinge": left_door_state.detach().numpy(),
                "right_door_hinge": right_door_state.detach().numpy()
            })

        # Place shelf nodes.
        # Dimensions of a single shelf, in terms of the 
        shelf_height = 0.13115*2
        bottom_shelf_z_local = -0.3995
        num_shelves = 3
        rules = []
        for k in range(num_shelves):
            rules.append(DeterministicRelativePoseProductionRule(
                child_constructor=CabinetLevel,
                child_name="cabinet_level_%02d" % k,
                relative_tf=pose_to_tf_matrix(torch.tensor(
                    [0., -0.15, bottom_shelf_z_local+shelf_height*k, 0., 0., 0.]))
            ))
        IndependentSetNode.__init__(self, name=name, production_rules=rules,
            production_probs=torch.tensor([0.5]*num_shelves))


class CabinetLevel(GeometricSetNode, SpatialNodeMixin, PhysicsGeometryNodeMixin):
    ''' Can produce a geometric number of objects on the shelf. '''
    def __init__(self, name, tf):
        SpatialNodeMixin.__init__(self, tf)
        PhysicsGeometryNodeMixin.__init__(self, fixed=True)

        # Hardcoded local frame shelf bounds
        self.x_bounds = [-0.3, 0.3]
        self.y_bounds = [-0.15, 0.15]
        self.z_bounds = [0., 0.2]

        # Produce a geometric number of objects just above the table surface.
        object_production_rule = RandomRelativePoseProductionRule(
            Object, "%s_object" % name, self._sample_object_pose_on_shelf
        )
        GeometricSetNode.__init__(
            self, name=name, production_rule=object_production_rule,
            geometric_prob=0.5
        )

        # Add some geometry for viz purposes
        geom_tf = pose_to_tf_matrix(torch.tensor([
            np.sum(self.x_bounds)/2.,
            np.sum(self.y_bounds)/2.,
            np.sum(self.z_bounds)/2.,
            0., 0., 0.]))
        geometry = Box(width=self.x_bounds[1] - self.x_bounds[0],
                       depth=self.y_bounds[1] - self.y_bounds[0],
                       height=self.z_bounds[1] - self.z_bounds[0])
        #self.register_visual_geometry(geom_tf, geometry, color=np.array([0.5, 1.0, 0.2, 0.25]))

    def _sample_object_pose_on_shelf(self):
        # For now, hard-code cabinet size to help it not intersect the other walls...
        x_on_shelf = pyro.sample("%s_object_x" % self.name,
                                dist.Uniform(self.x_bounds[0]+0.05,
                                             self.x_bounds[1]-0.05))
        y_on_shelf = pyro.sample("%s_object_y" % self.name,
                                dist.Uniform(self.y_bounds[0]+0.05,
                                             self.y_bounds[1]-0.05))
        return pose_to_tf_matrix(torch.tensor([x_on_shelf, y_on_shelf, self.z_bounds[0]+0.1,
                                               0., 0., 0.]))


class Table(GeometricSetNode, SpatialNodeMixin, PhysicsGeometryNodeMixin):
    ''' Table (using the extra heavy duty table from Drake) that
    produces objects on its surface. '''
    def __init__(self, name, tf):
        SpatialNodeMixin.__init__(self, tf)
        
        # Produce a geometric number of objects just above the table surface.
        object_production_rule = RandomRelativePoseProductionRule(
            Object, "%s_object" % name, self._sample_object_pose_on_table
        )
        GeometricSetNode.__init__(
            self, name=name, production_rule=object_production_rule,
            geometric_prob=0.35
        )

        # Handle geometry and physics.
        PhysicsGeometryNodeMixin.__init__(self, fixed=True)
        geom_tf = pose_to_tf_matrix(torch.tensor([0., 0., 0., 0., 0., 0.]))
        # TODO(gizatt) Resource path management to be done here...
        model_path = "/home/gizatt/drake/examples/kuka_iiwa_arm/models/table/extra_heavy_duty_table_surface_only_collision.sdf"
        self.register_model_file(tf=geom_tf, model_path=model_path, root_body_name="link")

    def _sample_object_pose_on_table(self):
        # For now, hard-code cabinet size to help it not intersect the other walls...
        table_width = 0.5
        table_length = 0.5
        table_height = 1.0
        x_on_table = pyro.sample("%s_object_x" % self.name,
                                dist.Uniform(-table_width/2.,
                                              table_width/2.))
        y_on_table = pyro.sample("%s_object_y" % self.name,
                                dist.Uniform(-table_length/2.,
                                              table_length/2.))
        return pose_to_tf_matrix(torch.tensor([x_on_table, y_on_table, table_height,
                                               0., 0., 0.]))

class Object(TerminalNode, SpatialNodeMixin, PhysicsGeometryNodeMixin):
    ''' Concrete object we might want to manipulate.
        Currently just creates a green block. '''
    def __init__(self, name, tf):
        SpatialNodeMixin.__init__(self, tf)
        TerminalNode.__init__(self, name)

        # Handle geometry and physics.
        PhysicsGeometryNodeMixin.__init__(self, fixed=False)
        # Rotate cabinet so it opens away from the wall
        geom_tf = torch.eye(4)
        # TODO(gizatt) Resource path management to be done here...
        model_path = "/home/gizatt/drake/examples/kuka_iiwa_arm/models/objects/block_for_pick_and_place_mid_size.urdf"
        self.register_model_file(tf=geom_tf, model_path=model_path, root_body_name="base_link")


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)

    # Sample an environment (wrapped in some pyro messiness so I can
    # play with the resulting program trace.)
    start = time.time()
    pyro.clear_param_store()
    trace = poutine.trace(ParseTree.generate_from_root_type).get_trace(root_node_type=Kitchen)
    scene_tree = trace.nodes["_RETURN"]["value"]
    end = time.time()

    # Print out all the nodes and their transforms.
    for node in scene_tree.nodes:
        if isinstance(node, Node):
            print(node.name, ": ", node.tf.tolist())

    print("Generated data in %f seconds." % (end - start))
    print("Full trace values:" )
    for node_name in trace.nodes.keys():
        if node_name in ["_INPUT", "_RETURN"]:
            continue
        print(node_name, ": ", trace.nodes[node_name]["value"].detach().numpy())


    # Simulate the resulting scene, with a PR2 for scale.
    builder, mbp, scene_graph = compile_scene_tree_to_mbp_and_sg(
        scene_tree, timestep=0.001)
    # Add PR2 and shift it back in front of where I know the table will be.
    parser = Parser(mbp)
    pr2_model_path = "/home/gizatt/drake/build/install/share/drake/examples/pr2/models/pr2_description/urdf/pr2_simplified.urdf"
    parser.package_map().PopulateUpstreamToDrake(pr2_model_path);
    pr2_model_id = parser.AddModelFromFile(
        file_name=pr2_model_path, model_name="PR2_for_scale")
    # The PR2 is on x and y rails: find the x joint and set its default state
    # to shift back from the table.
    mbp.GetJointByName("x", model_instance=pr2_model_id).set_default_translation(-0.5)

    mbp.Finalize()
    
    visualizer = ConnectMeshcatVisualizer(builder, scene_graph,
        zmq_url="default")

    diagram = builder.Build()
    diag_context = diagram.CreateDefaultContext()

    # Fix input port for PR2 to zero.
    actuation_port = mbp.get_actuation_input_port(model_instance=pr2_model_id)
    nu = mbp.num_actuated_dofs(model_instance=pr2_model_id)
    mbp_context = diagram.GetMutableSubsystemContext(mbp, diag_context)
    actuation_port.FixValue(mbp_context, np.zeros(nu))

    sim = Simulator(diagram, diag_context)
    sim.set_target_realtime_rate(1.0)
    sim.AdvanceTo(20)