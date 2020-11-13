import meshcat
import meshcat.geometry as meshcat_geom
import meshcat.transformations as meshcat_tf
import matplotlib.pyplot as plt
import numpy as np

import pydrake
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    ConnectMeshcatVisualizer,
    MultibodyPlant,
    Parser,
    RigidTransform,
    RotationMatrix,
    Simulator
)
import pydrake.geometry as pydrake_geom

from .nodes import SpatialNodeMixin, PhysicsGeometryNodeMixin
from .visualization import rgb_2_hex

def torch_tf_to_drake_tf(tf):
    return RigidTransform(tf.cpu().detach().numpy())

def set_meshcat_geometry_from_drake_geometry(vis, shape, color, alpha, X_FG):
    # Copied from meshcat_visualizer.py in Drake.
    material = meshcat_geom.MeshLambertMaterial(
        color=color, transparent=alpha != 1., opacity=alpha)
    if isinstance(shape, pydrake_geom.Box):
        geom = meshcat_geom.Box([shape.width(), shape.depth(),
                      shape.height()])
    elif isinstance(shape, pydrake_geom.Sphere):
        geom = meshcat_geom.Sphere(shape.radius())
    elif isinstance(shape, pydrake_geom.Cylinder):
        geom = meshcat_geom.Cylinder(shape.length(), shape.radius())
        # In Drake, cylinders are along +z
        # In meshcat, cylinders are along +y
        R_GC = RotationMatrix.MakeXRotation(np.pi/2.0).matrix()
        X_FG[0:3, 0:3] = X_FG[0:3, 0:3].dot(R_GC)
    elif isinstance(shape, pydrake_geom.Mesh):
        geom = meshcat_geom.ObjMeshGeometry.from_file(
            shape.filename()[0:-3] + "obj")
        # Attempt to find a texture for the object by looking for
        # an identically-named *.png next to the model.
        candidate_texture_path = shape.filename()[0:-3] + "png"
        if os.path.exists(candidate_texture_path):
            material = meshcat_geom.MeshLambertMaterial(
                map=meshcat_geom.ImageTexture(image=meshcat_geomPngImage.from_file(
                    candidate_texture_path)))
        X_FG = X_FG.dot(meshcat_tf.scale_matrix(shape.scale()))
    else:
        print(f"Unsupported shape {shape} ignored")
        return
    vis.set_object(geom, material)
    vis.set_transform(X_FG)

def draw_clearance_geometry_meshcat(scene_tree, zmq_url=None, alpha=0.25):
    vis = meshcat.Visualizer(zmq_url=zmq_url or "tcp://127.0.0.1:6000")
    vis["clearance_geom"].delete()

    # Assign functionally random colors to each new node
    # type we discover.
    node_class_to_color_dict = {}
    cmap = plt.cm.get_cmap('jet')
    cmap_counter = 0.
    
    k = 0
    for node in scene_tree.nodes:
        if (isinstance(node, PhysicsGeometryNodeMixin) and 
            len(node.clearance_geometry) > 0):
            # Get a color for this class and draw the clearance
            # geometry.
            node_type_string = node.__class__.__name__
            if node_type_string in node_class_to_color_dict.keys():
                color = node_class_to_color_dict[node_type_string]
            else:
                color = rgb_2_hex(cmap(cmap_counter))
                node_class_to_color_dict[node_type_string] = color
                cmap_counter = np.fmod(cmap_counter + np.pi*2., 1.)

            for l, (tf, shape) in enumerate(node.clearance_geometry):
                this_vis = vis["clearance_geom"][node.name + "%d" % k]["%d" % l]
                this_tf = torch_tf_to_drake_tf(node.tf)
                this_tf = this_tf.multiply(torch_tf_to_drake_tf(tf))
                this_tf = this_tf.GetAsMatrix4()
                set_meshcat_geometry_from_drake_geometry(
                    this_vis, shape, color, alpha, this_tf)
            k += 1

def compile_scene_tree_to_mbp_and_sg(scene_tree, timestep=0.001):
    builder = DiagramBuilder()
    mbp, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=timestep))
    parser = Parser(mbp)
    world_body = mbp.world_body()
    node_to_model_id_map = {}
    free_body_poses = []
    for node in scene_tree.nodes:
        if isinstance(node, SpatialNodeMixin) and isinstance(node, PhysicsGeometryNodeMixin):
            # Don't have to do anything if this does not introduce geometry.
            has_models = len(node.model_paths) > 0
            has_prim_geometry = (len(node.visual_geometry) + len(node.collision_geometry)) > 0
            if not has_models and not has_prim_geometry:
                continue

            # Handle adding each model from sdf/urdf.
            node_model_ids = []
            if has_models:
                for local_tf, model_path, root_body_name, q0_dict in node.model_paths:
                    model_id = parser.AddModelFromFile(
                        model_path,
                        node.name + "_model_%04d" % mbp.num_model_instances())
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


            # Handle adding primitive geometry by adding it all to one
            # mbp.
            if has_prim_geometry:
                # Contain this primitive geometry in a model instance.
                model_id = mbp.AddModelInstance(
                    node.name + "_geom_%04d" % mbp.num_model_instances())
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
