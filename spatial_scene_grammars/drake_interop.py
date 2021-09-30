from collections import namedtuple
import contextlib
import meshcat
import meshcat.geometry as meshcat_geom
import meshcat.transformations as meshcat_tf
import matplotlib.pyplot as plt
import logging
import numpy as np
import networkx as nx
import os
import yaml
import torch

import pydrake
from pydrake.common.cpp_param import List as DrakeBindingList
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BasicVector,
    ConnectMeshcatVisualizer,
    CoulombFriction,
    DiagramBuilder,
    ExternallyAppliedSpatialForce,
    LeafSystem,
    InverseKinematics,
    MeshcatVisualizer,
    MinimumDistanceConstraint,
    ModelInstanceIndex,
    MultibodyPlant,
    SpatialInertia,
    Parser,
    RigidTransform,
    RotationMatrix,
    SpatialForce,
    Simulator,
    SnoptSolver,
    Solve,
    SolverOptions,
    UnitInertia,
    Value
)
import pydrake.geometry as pydrake_geom

def torch_tf_to_drake_tf(tf):
    return RigidTransform(tf.cpu().detach().numpy())

def drake_tf_to_torch_tf(tf):
    return torch.tensor(tf.GetAsMatrix4())


default_spatial_inertia = SpatialInertia(
    mass=1.0,
    p_PScm_E=np.zeros(3), G_SP_E=UnitInertia(0.01, 0.01, 0.01)
)
default_friction = CoulombFriction(0.9, 0.8)
class PhysicsGeometryInfo():
    '''
    Container for physics and geometry info, providing simulator and
    visualization interoperation.
    Args:
        - fixed: Whether this geometry is welded to the world (otherwise,
            it will be mobilized by a 6DOF floating base).
        - spatial_inertia: Spatial inertia of the body. If None,
            will adopt a default mass of 1.0kg and 0.01x0.01x0.01 diagonal
            rotational inertia.
        - is_container: Flag whether this object will function as a
            container for other objects for the purpose of collision
            and stability checks. If so, then objects below this one
            will be isolated from collision and clearance checks for
            objects above this one, and instead only be checked against
            this object's collision geometry and this object's
            childrens' geometry. Valid for e.g. a cabinet full of
            stuff that does not interact with anything outside of
            the cabinet.

    To construct a PhysicsGeometricInfo object, initialize the object
    with the desired arguments above, and then use registration calls
    to populate the model geometry of the following types:
        - Model files (urdf/sdf), paired with a transform from the object
          local origin, the name of the root body (which gets put at that
          transform -- required if there's more than one body in the URDF),
          and optionally, the initial joint configuration of
          the model (as a dict of joint names to joint states). These
          are added to the simulated scene with the specified link
          welded (or translated, if not fixed) to the node transform.
        - Visual and collision geometry (Drake Shape types), paired with
          transforms from the object local origin and relevant color
          and friction information.
        - Clearance geometry (Drake Shape types), paired with transforms
          from the object local origin. This represents the region around
          this object that should not intersect with any other node's
          clearance geometry: e.g., the space in front of a cabinet should
          be clear so the doors can open.
    '''
    def __init__(self, fixed=True, spatial_inertia=None, is_container=False):
        self.fixed = fixed
        self.is_container = is_container
        self.model_paths = []
        self.spatial_inertia = spatial_inertia or default_spatial_inertia
        self.visual_geometry = []
        self.collision_geometry = []
        self.clearance_geometry = []
    def register_model_file(self, tf, model_path, root_body_name=None,
                            q0_dict={}):
        self.model_paths.append((tf, model_path, root_body_name, q0_dict))
    def register_geometry(self, tf, geometry, color=np.ones(4), friction=default_friction):
        # Shorthand for registering the same geometry as collision + visual.
        self.register_visual_geometry(tf, geometry, color)
        self.register_collision_geometry(tf, geometry, friction)
    def register_visual_geometry(self, tf, geometry, color=np.ones(4)):
        assert isinstance(tf, torch.Tensor) and tf.shape == (4, 4)
        assert isinstance(geometry, pydrake.geometry.Shape)
        self.visual_geometry.append((tf, geometry, color))
    def register_collision_geometry(self, tf, geometry, friction=default_friction):
        assert isinstance(tf, torch.Tensor) and tf.shape == (4, 4)
        assert isinstance(geometry, pydrake.geometry.Shape)
        assert isinstance(friction, CoulombFriction)
        self.collision_geometry.append((tf, geometry, friction))
    def register_clearance_geometry(self, tf, geometry):
        assert isinstance(tf, torch.Tensor) and tf.shape == (4, 4)
        assert isinstance(geometry, pydrake.geometry.Shape)
        self.clearance_geometry.append((tf, geometry))


def sanity_check_node_tf_and_physics_geom_info(node):
    assert isinstance(node.tf, torch.Tensor), type(node.tf)
    assert node.tf.shape == (4, 4), node.tf.shape
    assert isinstance(node.physics_geometry_info, PhysicsGeometryInfo), type(node.physics_geometry_info)


class StochasticLangevinForceSource(LeafSystem):
    ''' Connect to a MBP to apply ghost forces. The forces are:
    1) Random noise whose magnitude decays with sim time.
    2) A force proportional to the gradient of the log-prob of the
    object poses w.r.t the log-prob of the scene tree.

    This probably doesn't work for systems with tough inter-node
    constraints like planar-ity. Need to do some reparameterization
    of the system under sim for that to work?
    
    MBP + scene_tree should be corresponded to each other through
    the mbp construction method in this file.
    '''
    def __init__(self, mbp, scene_tree, node_to_free_body_ids_map, body_id_to_node_map):
        LeafSystem.__init__(self)
        self.set_name('StochasticLangevinForceSystem')

        self.robot_state_input_port = self.DeclareVectorInputPort(
            "robot_state", BasicVector(mbp.num_positions() + mbp.num_velocities()))
        forces_cls = Value[DrakeBindingList[ExternallyAppliedSpatialForce]]
        self.spatial_forces_output_port = self.DeclareAbstractOutputPort(
            "spatial_forces_vector",
            lambda: forces_cls(),
            self.DoCalcAbstractOutput)

        self.scene_tree = scene_tree
        self.node_to_free_body_ids_map = node_to_free_body_ids_map
        self.body_id_to_node_map = body_id_to_node_map
        self.mbp = mbp
        self.mbp_current_context = mbp.CreateDefaultContext()

        for node, body_ids in self.node_to_free_body_ids_map.items():
            for body_id in body_ids:
                self.mbp.SetFreeBodyPose(self.mbp_current_context, self.mbp.get_body(body_id), torch_tf_to_drake_tf(node.tf))

    def DoCalcAbstractOutput(self, context, y_data):
        t = context.get_time()
        # TODO: Hook up random input on the right kind of random port.
        print(t)
        noise_scale = 0.25 * 0.25**t
        ll_scale = 2.0 * 0.25**t
        
        x_in = self.EvalVectorInput(context, 0).get_value()
        self.mbp.SetPositionsAndVelocities(self.mbp_current_context, x_in)

        # Copy state over to scene tree.
        free_bodies = self.mbp.GetFloatingBaseBodies()
        body_tf_vars = {}
        for body_id, node in self.body_id_to_node_map.items():
            if body_id not in free_bodies:
                continue
            tf_dec_var = drake_tf_to_torch_tf(self.mbp.GetFreeBodyPose(self.mbp_current_context, self.mbp.get_body(body_id)))
            tf_dec_var.requires_grad = True
            body_tf_vars[body_id] = tf_dec_var
            node.tf = tf_dec_var
            
        # Compute log prob and backprop.
        score = self.scene_tree.score(include_discrete=False, include_continuous=True)
        score.backward() 
        
        forces = []
        for body_id in free_bodies:
            body = self.mbp.get_body(body_id)
            
            # Get pose of body in world frame
            body_tf = self.mbp.GetFreeBodyPose(self.mbp_current_context, body)
            body_tfd = self.mbp.EvalBodySpatialVelocityInWorld(self.mbp_current_context, body)

            # Get mass info so we can calc correct force scaling
            si = body.CalcSpatialInertiaInBodyFrame(self.mbp_current_context)
            m = si.get_mass()
            I = si.CalcRotationalInertia().CopyToFullMatrix3()
            I_w = body_tf.rotation().matrix().dot(I)
            
            # Calculate total wrench
            # Noise term
            f_noise = np.random.normal(0., noise_scale, size=3)
            tau_noise = np.random.normal(0., noise_scale, size=3)
            # Force maximizing log prob
            t_grad = body_tf_vars[body_id].grad[:3, 3].numpy()
            f_ll = t_grad*m

            force = SpatialForce(
                tau=tau_noise - 0.01*body_tfd.rotational(),
                f=f_noise + f_ll*ll_scale - body_tfd.translational()*0.5
            )
            out = ExternallyAppliedSpatialForce()
            out.F_Bq_W = force
            out.body_index = body.index()
            forces.append(out)
        y_data.set_value(forces)


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
    free_body_poses = []
    # For generating colors.
    node_class_to_color_dict = {}
    cmap = plt.cm.get_cmap('jet')
    cmap_counter = 0.
    for node in scene_tree.nodes:
        if node.tf is not None and node.physics_geometry_info is not None:
            # Don't have to do anything if this does not introduce geometry.
            sanity_check_node_tf_and_physics_geom_info(node)
            phys_geom_info = node.physics_geometry_info
            has_clearance_geometry = len(phys_geom_info.clearance_geometry) > 0
            if not has_clearance_geometry:
                continue

            # Add a body for this node and register the clearance geometry.
            # TODO(gizatt) This tree body index is built in to disambiguate names.
            # But I forsee a name-to-stuff resolution crisis when inference time comes...
            # this might get resolved by the solution to that.
            body = mbp.AddRigidBody(name=node.name,
                                    M_BBo_B=phys_geom_info.spatial_inertia)
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
            if len(phys_geom_info.clearance_geometry) > 0:
                for k, (tf, geometry) in enumerate(phys_geom_info.clearance_geometry):
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

def build_nonpenetration_constraint(mbp, mbp_context_in_diagram, signed_distance_threshold):
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

def resolve_sg_proximity_id_to_mbp_id(sg, mbp, geometry_id):
    for model_k in range(mbp.num_model_instances()):
        model_k = ModelInstanceIndex(model_k)
        for body_k in mbp.GetBodyIndices(model_k):
            if geometry_id in mbp.GetCollisionGeometriesForBody(mbp.get_body(body_k)):
                return model_k, body_k
    raise ValueError("Geometry ID not registered by this MBP.")

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

        if (child.physics_geometry_info is not None and
                child.physics_geometry_info.is_container):
            continue
        new_tree = expand_container_tree(full_tree, new_tree, child)
    return new_tree

def split_tree_into_containers(scene_tree):
    # The roots will be each container + the root
    # of the overall tree.
    roots = [node for node in scene_tree.nodes if
            (len(list(scene_tree.predecessors(node))) == 0 or
             (node.physics_geometry_info is not None and
              node.physics_geometry_info.is_container))]
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

    node_to_free_body_ids_map = {}
    body_id_to_node_map = {}

    free_body_poses = []
    for node in scene_tree.nodes:
        node_to_free_body_ids_map[node] = []
        if node.tf is not None and node.physics_geometry_info is not None:
            # Don't have to do anything if this does not introduce geometry.
            sanity_check_node_tf_and_physics_geom_info(node)
            phys_geom_info = node.physics_geometry_info

            # Don't have to do anything if this does not introduce geometry.
            has_models = len(phys_geom_info.model_paths) > 0
            has_prim_geometry = (len(phys_geom_info.visual_geometry)
                                 + len(phys_geom_info.collision_geometry)) > 0
            if not has_models and not has_prim_geometry:
                continue

            node_model_ids = []
            # Handle adding primitive geometry by adding it all to one
            # mbp.
            if has_prim_geometry:
                # Contain this primitive geometry in a model instance.
                model_id = mbp.AddModelInstance(
                    node.name + "::model_%d" % len(node_model_ids))
                # Add a body for this node, and register any of the
                # visual and collision geometry available.
                # TODO(gizatt) This tree body index is built in to disambiguate names.
                # But I forsee a name-to-stuff resolution crisis when inference time comes...
                # this might get resolved by the solution to that.
                body = mbp.AddRigidBody(name=node.name, model_instance=model_id,
                                        M_BBo_B=phys_geom_info.spatial_inertia)
                body_id_to_node_map[body.index()] = node
                tf = torch_tf_to_drake_tf(node.tf)
                if phys_geom_info.fixed:
                    weld = mbp.WeldFrames(world_body.body_frame(),
                                          body.body_frame(),
                                          tf)
                else:
                    node_to_free_body_ids_map[node].append(body.index())
                    mbp.SetDefaultFreeBodyPose(body, tf)
                for k, (tf, geometry, color) in enumerate(phys_geom_info.visual_geometry):
                    mbp.RegisterVisualGeometry(
                        body=body,
                        X_BG=torch_tf_to_drake_tf(tf),
                        shape=geometry,
                        name=node.name + "_vis_%03d" % k, 
                        diffuse_color=color)
                for k, (tf, geometry, friction) in enumerate(phys_geom_info.collision_geometry):
                    mbp.RegisterCollisionGeometry(
                        body=body,
                        X_BG=torch_tf_to_drake_tf(tf),
                        shape=geometry,
                        name=node.name + "_col_%03d" % k,
                        coulomb_friction=friction)

            # Handle adding each model from sdf/urdf.
            if has_models:
                for local_tf, model_path, root_body_name, q0_dict in phys_geom_info.model_paths:
                    model_id = parser.AddModelFromFile(
                        resolve_catkin_package_path(parser.package_map(), model_path),
                        node.name + "::" "model_%d" % len(node_model_ids))
                    if root_body_name is None:
                        root_body_ind_possibilities = mbp.GetBodyIndices(model_id)
                        assert len(root_body_ind_possibilities) == 1, \
                            "Please supply root_body_name for model with path %s" % model_path
                        root_body = mbp.get_body(root_body_ind_possibilities[0])
                    else:
                        root_body = mbp.GetBodyByName(
                            name=root_body_name,
                            model_instance=model_id)
                    body_id_to_node_map[root_body.index()] = node
                    node_tf = torch_tf_to_drake_tf(node.tf)
                    full_model_tf = node_tf.multiply(torch_tf_to_drake_tf(local_tf))
                    if phys_geom_info.fixed:
                        mbp.WeldFrames(world_body.body_frame(),
                                       root_body.body_frame(),
                                       full_model_tf)
                    else:
                        node_to_free_body_ids_map[node].append(root_body.index())
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

    return builder, mbp, scene_graph, node_to_free_body_ids_map, body_id_to_node_map

def project_tree_to_feasibility(tree, constraints=[], jitter_q=None, do_forward_sim=False, zmq_url=None, prefix="projection", timestep=0.001, T=1.):
    print("Starting")
    # Mutates tree into tree with bodies in closest
    # nonpenetrating configuration.
    builder, mbp, sg, node_to_free_body_ids_map, body_id_to_node_map = \
        compile_scene_tree_to_mbp_and_sg(tree, timestep=timestep)
    mbp.Finalize()
    # Connect visualizer if requested. Wrap carefully to keep it
    # from spamming the console.
    if zmq_url is not None:
         with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                visualizer = ConnectMeshcatVisualizer(builder, sg, zmq_url=zmq_url, prefix=prefix)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    mbp_context = diagram.GetMutableSubsystemContext(mbp, diagram_context)
    q0 = mbp.GetPositions(mbp_context)
    nq = len(q0)
    
    print("setting up ik")
    
    # Set up projection NLP.
    ik = InverseKinematics(mbp, mbp_context)
    q_dec = ik.q()
    prog = ik.prog()
    # It's always a projection, so we always have this
    # Euclidean norm error between the optimized q and
    # q0.
    prog.AddQuadraticErrorCost(np.eye(nq), q0, q_dec)
    # Nonpenetration constraint.
    ik.AddMinimumDistanceConstraint(0.01)
    # Other requested constraints.
    for constraint in constraints:
        constraint.add_to_ik_prog(tree, ik, mbp, mbp_context, node_to_free_body_ids_map)
    # Initial guess, which can be slightly randomized by request.
    q_guess = q0
    if jitter_q:
        q_guess = q_guess + np.random.normal(0., jitter_q, size=q_guess.size)
    prog.SetInitialGuess(q_dec, q_guess)
    
    # Solve.
    solver = SnoptSolver()
    options = SolverOptions()
    logfile = "/tmp/snopt.log"
    os.system("rm %s" % logfile)
    options.SetOption(solver.id(), "Print file", logfile)
    options.SetOption(solver.id(), "Major feasibility tolerance", 1E-6)
    print("Solving")
    result = solver.Solve(prog, None, options)
    print("Solved")
    if not result.is_success():
        logging.warn("Projection failed.")
        print("Logfile: ")
        with open(logfile) as f:
            print(f.read())
    qf = result.GetSolution(q_dec)
    mbp.SetPositions(mbp_context, qf)
    
    # If forward sim is requested, do a quick forward sim to get to
    # a statically stable config.
    if do_forward_sim:
        sim = Simulator(diagram, diagram_context)
        sim.set_target_realtime_rate(1000.)
        sim.AdvanceTo(T)
        
    # Reload poses back into tree
    free_bodies = mbp.GetFloatingBaseBodies()
    for body_id, node in body_id_to_node_map.items():
        if body_id not in free_bodies:
            continue
        node.tf = drake_tf_to_torch_tf(mbp.GetFreeBodyPose(mbp_context, mbp.get_body(body_id)))
    return tree

def simulate_scene_tree(scene_tree, T, timestep=0.001, target_realtime_rate=1.0, meshcat=None):
    builder, mbp, scene_graph, _, _ = compile_scene_tree_to_mbp_and_sg(
        scene_tree, timestep=timestep)
    mbp.Finalize()

    if meshcat:
        visualizer = ConnectMeshcatVisualizer(builder, scene_graph,
            zmq_url=meshcat)

    diagram = builder.Build()
    diag_context = diagram.CreateDefaultContext()
    sim = Simulator(diagram)
    sim.set_target_realtime_rate(target_realtime_rate)
    sim.AdvanceTo(T)

