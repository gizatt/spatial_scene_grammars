from collections import namedtuple
import meshcat
import meshcat.geometry as meshcat_geom
import meshcat.transformations as meshcat_tf
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import yaml

import pydrake
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    ConnectMeshcatVisualizer,
    MinimumDistanceConstraint,
    MultibodyPlant,
    Parser,
    RigidTransform,
    RotationMatrix,
    Simulator
)
import pydrake.geometry as pydrake_geom

from .nodes import SpatialNode, PhysicsGeometryNode, default_friction
from .visualization import rgb_2_hex

def torch_tf_to_drake_tf(tf):
    return RigidTransform(tf.cpu().detach().numpy())

def draw_clearance_geometry_meshcat(scene_tree, zmq_url=None, alpha=0.25):
    vis = meshcat.Visualizer(zmq_url=zmq_url or "tcp://127.0.0.1:6000")
    vis["clearance_geom"].delete()

    builder, mbp, scene_graph = compile_scene_tree_clearance_geometry_to_mbp_and_sg(scene_tree)
    mbp.Finalize()

    vis = ConnectMeshcatVisualizer(builder, scene_graph,
        zmq_url="default", prefix="clearance")
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    vis.load(vis.GetMyContextFromRoot(context))
    diagram.Publish(context)


def resolve_catkin_package_path(package_map, input_str):
    if "://" in input_str:
        elements = input_str.split("://")
        assert len(elements) == 2, "Malformed path " + input_str
        package_name, path_in_package = elements
        assert package_map.Contains(package_name), "%s not in package map" % package_name
        return os.path.join(
            package_map.GetPath(package_name),
            path_in_package
        )
    else:
        return input_str


def compile_scene_tree_clearance_geometry_to_mbp_and_sg(scene_tree, timestep=0.001, alpha=0.25):
    builder = DiagramBuilder()
    mbp, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=timestep))
    parser = Parser(mbp)
    parser.package_map().PopulateFromEnvironment("ROS_PACKAGE_PATH")
    world_body = mbp.world_body()
    node_to_body_id_map = {}
    free_body_poses = []
    # For generating colors.
    node_class_to_color_dict = {}
    cmap = plt.cm.get_cmap('jet')
    cmap_counter = 0.
    for node in scene_tree.nodes:
        if isinstance(node, SpatialNode) and isinstance(node, PhysicsGeometryNode):
            # Don't have to do anything if this does not introduce geometry.
            has_clearance_geometry = len(node.clearance_geometry) > 0
            if not has_clearance_geometry:
                continue

            # Add a body for this node and register the clearance geometry.
            # TODO(gizatt) This tree body index is built in to disambiguate names.
            # But I forsee a name-to-stuff resolution crisis when inference time comes...
            # this might get resolved by the solution to that.
            body = mbp.AddRigidBody(name=node.name,
                                    M_BBo_B=node.spatial_inertia)
            node_to_body_id_map[node] = body.index()
            tf = torch_tf_to_drake_tf(node.tf)
            mbp.SetDefaultFreeBodyPose(body, tf)

            # Pick out a color for this class.
            node_type_string = node.__class__.__name__
            if node_type_string in node_class_to_color_dict.keys():
                color = node_class_to_color_dict[node_type_string]
            else:
                color = list(cmap(cmap_counter))
                color[3] = alpha
                node_class_to_color_dict[node_type_string] = color
                cmap_counter = np.fmod(cmap_counter + np.pi*2., 1.)

            # Handle adding primitive geometry by adding it all to one
            # mbp.
            if len(node.clearance_geometry) > 0:
                for k, (tf, geometry) in enumerate(node.clearance_geometry):
                    mbp.RegisterCollisionGeometry(
                        body=body,
                        X_BG=torch_tf_to_drake_tf(tf),
                        shape=geometry,
                        name=node.name + "_col_%03d" % k,
                        coulomb_friction=default_friction)
                    mbp.RegisterVisualGeometry(
                        body=body,
                        X_BG=torch_tf_to_drake_tf(tf),
                        shape=geometry,
                        name=node.name + "_vis_%03d" % k,
                        diffuse_color=color)

    return builder, mbp, scene_graph

def build_clearance_nonpenetration_constraint(mbp, mbp_context_in_diagram, signed_distance_threshold):
    ''' Given an MBP/SG pair and a signed distance threshold, returns a constraint
    function that takes a context and returns whether the MBP/SG in that configuration
    has all bodies farther than the given threshold. '''
    return MinimumDistanceConstraint(mbp, signed_distance_threshold, mbp_context_in_diagram)

def get_collisions(mbp, mbp_context_in_diagram):
    # Essentially the same logic as in ik/MinimumDistanceConstraint's distances evaluation.
    query_port = mbp.get_geometry_query_input_port()
    assert query_port.HasValue(mbp_context_in_diagram), \
        "Either the plant geometry_query_input_port() is not properly " \
        "connected to the SceneGraph's output port, or the plant_context_ is " \
        "incorrect. Please refer to AddMultibodyPlantSceneGraph on connecting " \
        "MultibodyPlant to SceneGraph."
    query_object = query_port.Eval(mbp_context_in_diagram)
    return query_object.ComputePointPairPenetration()


def expand_container_tree(full_tree, new_tree, current_node):
    # Given the original tree for reference and a new tree
    # that contains the current node, gets the current node's
    # children, adds them all (with approp connections) to the
    # new tree, and recurses on the children.
    # Does not recurse on children that are containers,
    # but will still add them to the tree. (Containers should
    # appear in the tree above *and* below them.)
    for child in full_tree.successors(current_node):
        new_tree.add_node(child)
        new_tree.add_edge(current_node, child)

        if (isinstance(child, PhysicsGeometryNode) and
             child.is_container):
            continue
        new_tree = expand_container_tree(full_tree, new_tree, child)
    return new_tree

def split_tree_into_containers(scene_tree):
    # The roots will be each container + the root
    # of the overall tree.
    roots = [node for node in scene_tree.nodes if
            (len(list(scene_tree.predecessors(node))) == 0 or
             isinstance(node, PhysicsGeometryNode) and node.is_container)]
    # Build the subtree from each root until it hits a terminal or
    # or a container.
    trees = []
    for root in roots:
        # Manually add the first 
        new_tree = nx.DiGraph()
        new_tree.add_node(root)
        trees.append(expand_container_tree(scene_tree, new_tree, root))
    return trees

def compile_scene_tree_to_mbp_and_sg(scene_tree, timestep=0.001):
    builder = DiagramBuilder()
    mbp, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=timestep))
    parser = Parser(mbp)
    parser.package_map().PopulateFromEnvironment("ROS_PACKAGE_PATH")
    world_body = mbp.world_body()
    node_to_model_id_map = {}
    free_body_poses = []
    for node in scene_tree.nodes:
        if isinstance(node, SpatialNode) and isinstance(node, PhysicsGeometryNode):
            # Don't have to do anything if this does not introduce geometry.
            has_models = len(node.model_paths) > 0
            has_prim_geometry = (len(node.visual_geometry) + len(node.collision_geometry)) > 0
            if not has_models and not has_prim_geometry:
                continue

            node_model_ids = []
            # Handle adding primitive geometry by adding it all to one
            # mbp.
            if has_prim_geometry:
                # Contain this primitive geometry in a model instance.
                model_id = mbp.AddModelInstance(
                    node.name + "::model_%d" % len(node_model_ids))
                node_model_ids.append(model_id)
                # Add a body for this node, and register any of the
                # visual and collision geometry available.
                # TODO(gizatt) This tree body index is built in to disambiguate names.
                # But I forsee a name-to-stuff resolution crisis when inference time comes...
                # this might get resolved by the solution to that.
                body = mbp.AddRigidBody(name=node.name, model_instance=model_id,
                                        M_BBo_B=node.spatial_inertia)
                tf = torch_tf_to_drake_tf(node.tf)
                if node.fixed:
                    weld = mbp.WeldFrames(world_body.body_frame(),
                                          body.body_frame(),
                                          tf)
                else:
                    mbp.SetDefaultFreeBodyPose(body, tf)
                for k, (tf, geometry, color) in enumerate(node.visual_geometry):
                    mbp.RegisterVisualGeometry(
                        body=body,
                        X_BG=torch_tf_to_drake_tf(tf),
                        shape=geometry,
                        name=node.name + "_vis_%03d" % k, 
                        diffuse_color=color)
                for k, (tf, geometry, friction) in enumerate(node.collision_geometry):
                    mbp.RegisterCollisionGeometry(
                        body=body,
                        X_BG=torch_tf_to_drake_tf(tf),
                        shape=geometry,
                        name=node.name + "_col_%03d" % k,
                        coulomb_friction=friction)

            # Handle adding each model from sdf/urdf.
            if has_models:
                for local_tf, model_path, root_body_name, q0_dict in node.model_paths:
                    model_id = parser.AddModelFromFile(
                        resolve_catkin_package_path(parser.package_map(), model_path),
                        node.name + "::" "model_%d" % len(node_model_ids))
                    node_model_ids.append(model_id)
                    if root_body_name is None:
                        root_body_ind_possibilities = mbp.GetBodyIndices(model_id)
                        assert len(root_body_ind_possibilities) == 1, \
                            "Please supply root_body_name for model with path %s" % model_path
                        root_body = mbp.get_body(root_body_ind_possibilities[0])
                    else:
                        root_body = mbp.GetBodyByName(
                            name=root_body_name,
                            model_instance=model_id)
                    node_tf = torch_tf_to_drake_tf(node.tf)
                    full_model_tf = node_tf.multiply(torch_tf_to_drake_tf(local_tf))
                    if node.fixed:
                        mbp.WeldFrames(world_body.body_frame(),
                                       root_body.body_frame(),
                                       full_model_tf)
                    else:
                        mbp.SetDefaultFreeBodyPose(root_body, full_model_tf)
                    # Handle initial joint state
                    if q0_dict is not None:
                        for joint_name in list(q0_dict.keys()):
                            q0_this = q0_dict[joint_name]
                            joint = mbp.GetMutableJointByName(
                                joint_name, model_instance=model_id)
                            # Reshape to make Drake happy.
                            q0_this = q0_this.reshape(joint.num_positions(), 1)
                            joint.set_default_positions(q0_this)

            node_to_model_id_map[node] = node_model_ids
    return builder, mbp, scene_graph

def simulate_scene_tree(scene_tree, T, timestep=0.001, with_meshcat=False):
    builder, mbp, scene_graph = compile_scene_tree_to_mbp_and_sg(
        scene_tree, timestep=timestep)
    mbp.Finalize()

    if with_meshcat:
        visualizer = ConnectMeshcatVisualizer(builder, scene_graph,
            zmq_url="default")

    diagram = builder.Build()
    diag_context = diagram.CreateDefaultContext()
    sim = Simulator(diagram)
    sim.set_target_realtime_rate(1.0)
    sim.AdvanceTo(T)
