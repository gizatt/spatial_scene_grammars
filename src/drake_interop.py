import pydrake
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    ConnectMeshcatVisualizer,
    MultibodyPlant,
    RigidTransform,
    Simulator
)

from .nodes import SpatialNodeMixin, PhysicsGeometryNodeMixin

def torch_tf_to_drake_tf(tf):
    return RigidTransform(tf.cpu().detach().numpy())

def compile_scene_tree_to_mbp_and_sg(scene_tree, timestep=0.001):
    builder = DiagramBuilder()
    mbp, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=timestep))
    world_body = mbp.world_body()
    free_body_poses = []
    for node in scene_tree.nodes:
        if isinstance(node, SpatialNodeMixin) and isinstance(node, PhysicsGeometryNodeMixin):
            # Add a body for this node, and register any of the
            # visual and collision geometry available.
            body = mbp.AddRigidBody(node.name, node.spatial_inertia)
            tf = torch_tf_to_drake_tf(node.tf)
            if node.fixed:
                mbp.WeldFrames(world_body.body_frame(),
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
    # Create a context for the MBP so we can set up the state
    mbp.Finalize()
    return builder, mbp, scene_graph

def simulate_scene_tree(scene_tree, T, timestep=0.001, with_meshcat=False):
    builder, mbp, scene_graph = compile_scene_tree_to_mbp_and_sg(
        scene_tree, timestep=timestep)

    if with_meshcat:
        visualizer = ConnectMeshcatVisualizer(builder, scene_graph,
            zmq_url="default")

    diagram = builder.Build()
    diag_context = diagram.CreateDefaultContext()
    sim = Simulator(diagram)
    sim.set_target_realtime_rate(1.0)
    sim.AdvanceTo(T)
