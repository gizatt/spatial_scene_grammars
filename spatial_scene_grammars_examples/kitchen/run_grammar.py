import meshcat
import numpy as np
import os
import time

import pydrake
from pydrake.all import (
    Box, RollPitchYaw, RigidTransform, Parser,
    StaticEquilibriumProblem,
    AddUnitQuaternionConstraintOnPlant,
    SnoptSolver
)

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro import poutine

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.tree import *
from spatial_scene_grammars.transform_utils import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars.drake_interop import *
from spatial_scene_grammars.serialization_model_directive import *

from spatial_scene_grammars_examples.kitchen.grammar_room_layout import *


def rejection_sample_feasible_tree(num_attempts=999):
    ''' Repeatedly samples trees from the grammar until
    one satisfies some hand-coded constraints.

    This will be simplified when constraint specification
    and sampling machinery is generalized. For now, this is
    hard-coded to work for the kitchen example. '''

    for attempt_k in range(num_attempts):
        start = time.time()
        pyro.clear_param_store()
        scene_tree = SceneTree.forward_sample_from_root_type(
            root_node_type=Kitchen,
            name="kitchen",
            tf=torch.eye(4))
        end = time.time()

        print("Generated tree in %f seconds." % (end - start))

        # Enforce  that there no cabinets
        num_cabinets = len([node for node in scene_tree.nodes if isinstance(node, Cabinet)])
        if num_cabinets != 0:
            continue
        
        # Enforce that there are at least a few objects on the table
        tables = scene_tree.find_nodes_by_type(Table)
        print("Tables: ", tables)
        table_children = sum([scene_tree.get_recursive_children_of_node(node) for node in tables], [])
        print("Table children: ", table_children)
        print("Objects on tables: ", [node for node in table_children if isinstance(node, KitchenObject)])
        num_objects_on_tables = len([node for node in table_children if isinstance(node, KitchenObject)])
        print("Num objs on table: ", num_objects_on_tables)
        if num_objects_on_tables < 3:
            continue

        # Enforce that there are at least a fews objects in cabinets
        #cabinets = scene_tree.find_nodes_by_type(Cabinet)
        #table_children = sum([scene_tree.get_recursive_children_of_node(node) for node in cabinets], [])
        #num_objects_in_cabinets = len([node for node in table_children if isinstance(node, KitchenObject)])
        #print("Num objs in cabinets: ", num_objects_in_cabinets)
        #if num_objects_in_cabinets > 1:
        #    continue

        
        # Do Collision checking on the clearance geometry, and reject
        # scenes where the collision geometry is in collision.
        # (This could be done at subtree level, and eventually I'll do that --
        # but for this scene it doesn't matter b/c clearance geometry is all
        # furniture level anyway.
        # TODO: What if I did rejection sampling for nonpenetration at the
        # container level? Is that legit as a sampling strategy?)
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

        # We can draw clearance geometry for debugging.
        # draw_clearance_geometry_meshcat(scene_tree, alpha=0.3)

        # If we failed the initial clearance check, resample.
        if not constraint.CheckSatisfied(q0):
            continue
            
        # Good solution!
        return scene_tree, True

    # Bad solution :(
    return scene_tree, False

def project_tree_to_feasibility(scene_tree, num_attempts=1):
    # Keep tree structure fixed, but try to make the objects
    # in each container satisfy a set of feasibility constraints:
    # that objects are nonpenetrating and statically stable.

    # Do the analysis for each contained, physically independent subtree.
    split_trees = split_tree_into_containers(scene_tree)
    for k, tree in enumerate(split_trees):
        print("Nodes in tree %d:" % k)
        for node in tree:
            if isinstance(node, Node):
                print("\t", node.name)
        
        builder, mbp, scene_graph = compile_scene_tree_to_mbp_and_sg(
                scene_tree, timestep=0.001)
        mbp.Finalize()
        diagram = builder.Build()
        # Currently fails at
        #   >  result = solver.Solve(opt.prog())
        # With:
        #   RuntimeError: Signed distance queries between shapes 'Box' and 'Cylinder' are not supported for scalar type drake::AutoDiffXd
        # Can not be resolved until Drake issue #10907 is resolved.
        #diagram_ad = diagram.ToAutoDiffXd()
        #mbp_ad = diagram_ad.GetSubsystemByName("plant") # Default name for MBP
        #diagram_ad_context = diagram_ad.CreateDefaultContext()
        #mbp_ad_context = diagram_ad.GetMutableSubsystemContext(mbp_ad, diagram_ad_context)
        #opt = StaticEquilibriumProblem(mbp_ad, mbp_ad_context, ignored_collision_pairs=set())
        #q_vars = opt.q_vars()
        #prog = opt.get_mutable_prog()
        #AddUnitQuaternionConstraintOnPlant(
        #    mbp_ad, q_vars, prog)
        #q_targ = mbp.GetPositions(mbp.CreateDefaultContext())
        ## Penalize deviation from target configuration
        #prog.AddQuadraticErrorCost(np.eye(q_targ.shape[0]), q_targ, q_vars)
        #prog.SetInitialGuess(q_vars, q_targ)
        #solver = SnoptSolver()
        #result = solver.Solve(opt.prog())
        #print(result, result.is_success())
    return scene_tree, False



def do_generation_and_simulation(sim_time=10):
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    scene_tree, satisfied_clearance = rejection_sample_feasible_tree(num_attempts=1000)
    scene_tree, satisfied_feasibility = project_tree_to_feasibility(scene_tree, num_attempts=3)

    serialize_scene_tree_to_package_and_model_directive(scene_tree, package_name='save', package_parent_dir=".", remove_directory=True)

    A = nx.nx_agraph.to_agraph(scene_tree)
    A.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
    A.draw('test.png')

    # Draw generated tree in meshcat.
    draw_scene_tree_meshcat(scene_tree, alpha=0.5, node_sphere_size=0.05)
    # Draw its clearance geometry for debugging.
    draw_clearance_geometry_meshcat(scene_tree, alpha=0.3)

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
    if not satisfied_clearance:
        print("WARNING: SCENE TREE NOT SATISFYING CLEARANCE")
    if not satisfied_feasibility:
        print("WARNING: SCENE TREE NOT SATISFYING FEASIBILITY, SIM MAY FAIL")
    try:
        sim.AdvanceTo(sim_time)
    except RuntimeError as e:
        print("Encountered error in sim: ", e)

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)

    do_generation_and_simulation(sim_time=10.0)
    