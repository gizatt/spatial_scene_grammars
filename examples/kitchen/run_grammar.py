from functools import partial
import meshcat
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
from scene_grammar.src.visualization import *
from scene_grammar.src.drake_interop import *

from grammar_room_layout import *


def rejection_sample_feasible_tree(num_attempts=999):
    satisfied = False
    for attempt_k in range(num_attempts):
        start = time.time()
        pyro.clear_param_store()
        trace = poutine.trace(ParseTree.generate_from_root_type).get_trace(root_node_type=Kitchen)
        scene_tree = trace.nodes["_RETURN"]["value"]
        end = time.time()

        print("Generated tree in %f seconds." % (end - start))

        # First check if it's got the right number of objects
        min_num_cabinets = 1
        num_cabinets = len([node for node in scene_tree.nodes if isinstance(node, Cabinet)])
        if num_cabinets < min_num_cabinets:
            continue

        min_num_objects = 1
        min_num_objects = len([node for node in scene_tree.nodes if isinstance(node, Object)])
        if min_num_objects < min_num_cabinets:
            continue

        # Draw its clearance geometry for debugging.
        # draw_clearance_geometry_meshcat(scene_tree, alpha=0.3)

        # Collision checking on the clearance geometry
        builder_clearance, mbp_clearance, sg_clearance = \
            compile_scene_tree_clearance_geometry_to_mbp_and_sg(scene_tree)
        mbp_clearance.Finalize()
        diagram_clearance = builder_clearance.Build()
        diagram_context = diagram_clearance.CreateDefaultContext()
        mbp_context = diagram_clearance.GetMutableSubsystemContext(mbp_clearance, diagram_context)
        constraint = build_clearance_nonpenetration_constraint(
            mbp_clearance, mbp_context, -0.01)
        constraint.Eval(mbp_clearance.GetPositions(mbp_context))

        q0 = mbp_clearance.GetPositions(mbp_context)
        print("CONSTRAINT EVAL: %f <= %f <= %f" % (
              constraint.lower_bound(),
              constraint.Eval(mbp_clearance.GetPositions(mbp_context)),
              constraint.upper_bound()))
        print(len(get_collisions(mbp_clearance, mbp_context)), " bodies in collision")

        if constraint.CheckSatisfied(q0):
            satisfied = True
            break
    return scene_tree, satisfied

def do_generation_and_simulation(sim_time=10):
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    scene_tree, satisfied = rejection_sample_feasible_tree(num_attempts=10000)

    # Draw generated tree in meshcat.
    #draw_scene_tree_meshcat(scene_tree, alpha=1.0, node_sphere_size=0.1)
    # Draw its clearance geometry for debugging.
    #draw_clearance_geometry_meshcat(scene_tree, alpha=0.3)

    # Simulate the resulting scene, with a PR2 for scale.
    builder, mbp, scene_graph = compile_scene_tree_to_mbp_and_sg(
        scene_tree, timestep=0.001)
    # Add PR2 and shift it back in front of where I know the table will be.
    parser = Parser(mbp)
    pr2_model_path = "/home/gizatt/drake/build/install/share/drake/examples/pr2/models/pr2_description/urdf/pr2_simplified.urdf"
    parser.package_map().PopulateUpstreamToDrake(pr2_model_path);
    pr2_model_id = parser.AddModelFromFile(
        file_name=pr2_model_path, model_name="PR2_for_scale")
    # Get the tf of the robot spawn node, and put the PR2 at that xy location.
    robot_spawn_tf = scene_tree.find_nodes_by_type(RobotSpawnLocation)[0].tf.numpy()
    mbp.GetJointByName("x", model_instance=pr2_model_id).set_default_translation(robot_spawn_tf[0, 3])
    mbp.GetJointByName("y", model_instance=pr2_model_id).set_default_translation(robot_spawn_tf[1, 3])

    mbp.Finalize()
    
    visualizer = ConnectMeshcatVisualizer(builder, scene_graph,
        zmq_url="default")
    diagram = builder.Build()
    diag_context = diagram.CreateDefaultContext()
    mbp_context = diagram.GetMutableSubsystemContext(mbp, diag_context)
    # Fix input port for PR2 to zero.
    actuation_port = mbp.get_actuation_input_port(model_instance=pr2_model_id)
    nu = mbp.num_actuated_dofs(model_instance=pr2_model_id)
    actuation_port.FixValue(mbp_context, np.zeros(nu))


    sim = Simulator(diagram, diag_context)
    sim.set_target_realtime_rate(1.0)
    if not satisfied:
        print("WARNING: SCENE TREE NOT SATISFYING CONSTRAINTS")
    try:
        sim.AdvanceTo(sim_time)
    except RuntimeError as e:
        print("Encountered error in sim: ", e)

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)

    do_generation_and_simulation(sim_time=10.0)
    