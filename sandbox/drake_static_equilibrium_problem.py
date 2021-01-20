import trimesh
import matplotlib.pyplot as plt
import numpy as np
import pydrake

from pydrake.common.cpp_param import List as DrakeBindingList
from pydrake.all import (
    AngleAxis,
    AddMultibodyPlantSceneGraph,
    BasicVector,
    BodyIndex,
    ConnectMeshcatVisualizer,
    CoulombFriction,
    DiagramBuilder,
    LeafSystem,
    MultibodyPlant,
    RigidTransform,
    RotationMatrix,
    Simulator,
    SpatialForce,
    SpatialInertia,
    UnitInertia,
    StaticEquilibriumProblem,
    SnoptSolver,
    AddUnitQuaternionConstraintOnPlant,
    ExternallyAppliedSpatialForce,
    Value
)
import pydrake.geometry as pydrake_geom

from spatial_scene_grammars.drake_interop import DecayingForceToDesiredConfigSystem
        

def build_mbp(seed=0, verts_geom=False, convex_collision_geom=True):
    # Make some random lumpy objects
    trimeshes = []
    np.random.seed(42)
    for k in range(3):
        # Make a small random number of triangles and chull it
        # to get a lumpy object
        mesh = trimesh.creation.random_soup(5)
        mesh = trimesh.convex.convex_hull(mesh)
        trimeshes.append(mesh)

    # Create Drake geometry from those objects by adding a small
    # sphere at each vertex
    sphere_rad = 0.05
    cmap = plt.cm.get_cmap('jet')

    builder = DiagramBuilder()
    mbp, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=0.001))

    # Add ground
    friction = CoulombFriction(0.9, 0.8)
    g = pydrake_geom.Box(100., 100., 0.5)
    tf = RigidTransform(p=[0., 0., -0.25])
    mbp.RegisterVisualGeometry(
        body=mbp.world_body(),
        X_BG=tf,
        shape=g,
        name="ground",
        diffuse_color=[1.0, 1.0, 1.0, 1.0]
    )
    mbp.RegisterCollisionGeometry(
        body=mbp.world_body(),
        X_BG=tf,
        shape=g,
        name="ground",
        coulomb_friction=friction
    )

    for i, mesh in enumerate(trimeshes):
        inertia = SpatialInertia(
            mass=1.0,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(0.01, 0.01, 0.01)
        )
        body = mbp.AddRigidBody(name="body_%d" % i,
                                M_BBo_B=inertia)
        color = cmap(np.random.random())
        if verts_geom:
            for j, vert in enumerate(mesh.vertices):
                g = pydrake_geom.Sphere(radius=sphere_rad)
                tf = RigidTransform(p=vert)
                mbp.RegisterVisualGeometry(
                    body=body,
                    X_BG=tf,
                    shape=g,
                    name="body_%d_color_%d" % (i, j), 
                    diffuse_color=color)
                mbp.RegisterCollisionGeometry(
                    body=body,
                    X_BG=tf,
                    shape=g,
                    name="body_%d_collision_%d" % (i, j),
                    coulomb_friction=friction)
        # And add mesh itself for vis
        path = "/tmp/part_%d.obj" % i
        trimesh.exchange.export.export_mesh(mesh, path)
        g = pydrake_geom.Convex(path)
        mbp.RegisterVisualGeometry(
            body=body,
            X_BG=RigidTransform(),
            shape=g,
            name="body_%d_base" % i,
            diffuse_color=color
        )
        if convex_collision_geom:
            mbp.RegisterCollisionGeometry(
                body=body,
                X_BG=RigidTransform(),
                shape=g,
                name="body_%d_base_col" % i,
                coulomb_friction=friction
            )   
        mbp.SetDefaultFreeBodyPose(body, RigidTransform(p=[i % 3, i / 3., 1.]))
    mbp.Finalize()
    return builder, mbp, scene_graph

if __name__ == "__main__":
    seed = 42

    # This code would try to run a StaticEquilibriumProblem --
    # but it relies on Autodiff-converted MBP/SG, which doesn't support
    # the full matrix of geometry collisions. (Mostly does sphere/*
    # collisions.)
    #builder, mbp, scene_graph = build_mbp(seed=seed)
    #diagram = builder.Build()
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
    #q_0 = result.GetSolution(q_vars)
    #print("Q found: ", q_0)

    builder, mbp, scene_graph = build_mbp(seed=seed)
    q_des = mbp.GetPositions(mbp.CreateDefaultContext())
    forcer = builder.AddSystem(DecayingForceToDesiredConfigSystem(mbp, q_des))
    builder.Connect(mbp.get_state_output_port(),
                    forcer.get_input_port(0))
    builder.Connect(forcer.get_output_port(0),
                    mbp.get_applied_spatial_force_input_port())

    visualizer = ConnectMeshcatVisualizer(builder, scene_graph,
        zmq_url="default")
    diagram = builder.Build()
    diag_context = diagram.CreateDefaultContext()
    mbp_context = diagram.GetMutableSubsystemContext(mbp, diag_context)
    mbp.SetPositions(mbp_context, np.random.random(q_des.shape)*10.0)
    sim = Simulator(diagram, diag_context)
    #sim.set_target_realtime_rate(1.0)
    sim.AdvanceTo(10.0)

