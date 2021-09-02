'''
# MIP Scene Parsing for this grammar

We're given the grammar description (implicit in the node definitions); can we recover a MAP parse tree?

Parameterize the space of parse trees via the "super tree" for the grammar (with a recursion
limit). Create decision variables for each possible node pose and activation.

Impose feasibility constraints on each parent/child pair:
- Child activation implies parent activation
- If the parent is an AND node and is active, all children are active.
- If the parent is an OR node and is active, exactly one child is active.
- Child pose in feasible region w.r.t parent node.

Impose additional symmetry breaking constraints:
- Prefer children activate in left-to-right order, and are assigned ascending poses in the x coordinate (symmetry breaking).

Add costs summing up activation of each child node.
'''

import time
import networkx as nx
import numpy as np
import sys
from copy import deepcopy
from collections import namedtuple
from functools import partial

import pydrake
from pydrake.all import (
    AngleAxis,
    ClpSolver,
    CommonSolverOption,
    MathematicalProgram,
    MakeSolver,
    MixedIntegerBranchAndBound,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    GurobiSolver,
    SnoptSolver,
    OsqpSolver,
    Solve,
    SolverOptions,
    SolutionResult,
    VPolytope,
    MixedIntegerRotationConstraintGenerator,
    IntervalBinning
)

from .nodes import *
from .rules import *
from .scene_grammar import *
from .drake_interop import *

# These accept an XyzProductionRule subclass, the MathematicalProgram,
# and the parent and child nodes, and add appropriate constraints to
# constrain the parent and child to follow the rule.
def encode_WorldBBoxRule(rule, prog, parent, child):
    # Child translation should be within the translation bounds in
    # world frame.
    lb_world = rule.lb.detach().cpu().numpy()
    ub_world = rule.ub.detach().cpu().numpy()
    # X should be within a half-bound-width of the centerl.
    for k in range(3):
        prog.AddLinearConstraint(child.t_optim[k] >= lb_world[k])
        prog.AddLinearConstraint(child.t_optim[k] <= ub_world[k])

def encode_AxisAlignedBBoxRule(rule, prog, parent, child):
    # Child translation should be within relative translation bounds
    # parent, but added in world frame (with no rotations).
    lb_world = rule.lb.detach().cpu().numpy() + parent.t_optim
    ub_world = rule.ub.detach().cpu().numpy() + parent.t_optim
    # X should be within a half-bound-width of the centerl.
    for k in range(3):
        prog.AddLinearConstraint(child.t_optim[k] >= lb_world[k])
        prog.AddLinearConstraint(child.t_optim[k] <= ub_world[k])

xyz_rule_to_encode_map = {
    WorldBBoxRule: encode_WorldBBoxRule,
    AxisAlignedBBoxRule: encode_AxisAlignedBBoxRule
}

# These accept a RotationProductionRule subclass, the MathematicalProgram,
# and the parent and child nodes, and add appropriate constraints to
# constrain the parent and child to follow the rule. These should return
# True if the rotation was *completely constrained*, or False otherwise,
# to enable some logic in the optimization setup.
def encode_UnconstrainedRotationRule(rule, prog, parent, child):
    # No constraints to add! Just report that the child rotation
    # was unconstrained.
    return False

def encode_UniformBoundedRevoluteJointRule(rule, prog, parent, child):
    axis = rule.axis.detach().cpu().numpy()
    min_angle = rule.lb
    max_angle = rule.ub
    assert min_angle <= max_angle

    if max_angle - min_angle <= 1E-6:
        # In this case, the child rotation is exactly equal to the
        # parent rotation, so we can short-circuit.
        relative_rotation = RotationMatrix(AngleAxis(max_angle, axis)).matrix()
        target_rotation = parent.R_optim.dot(relative_rotation)
        for i in range(3):
            for j in range(3):
                prog.AddLinearEqualityConstraint(child.R_optim[i, j] == target_rotation[i, j])
        return True

    # Child rotation should be within a relative rotation of the parent around
    # the specified axis, and the axis should *not* be rotated between the
    # parent and child frames. This is similar to the revolute joint constraints
    # used by Hongkai Dai in his global IK formulation.
    # (1): The direction of the rotation axis doesn't change between
    # parent and child frames.
    # The axis is the same in both the parent and child frame
    # (see https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_revolute_joint.html).
    # Though there may be an additional offset according to the axis offset
    # in the parent and child frames.
    #axis_offset_in_parent = RigidTransform()
    #axis_offset_in_child = RigidTransform()
    parent_view_of_axis_in_world = parent.R_optim.dot(axis)
    child_view_of_axis_in_world = child.R_optim.dot(axis)
    for k in range(3):
        prog.AddLinearEqualityConstraint(
            parent_view_of_axis_in_world[k] == child_view_of_axis_in_world[k]
        )
    
    # Short-circuit if there is no rotational constraint other than axis alignment.
    if max_angle - min_angle >= 2.*np.pi:
        return False

    # If we're only allowed a limited rotation around this axis, apply a constraint
    # to enforce that.
    # (2): Eq(10) in the global IK paper. Following implementation in
    # https://github.com/RobotLocomotion/drake/blob/master/multibody/inverse_kinematics/global_inverse_kinematics.cc
    # First generate a vector normal to the rotation axis via cross products.

    v_c = np.cross(axis, np.array([0., 0., 1.]))
    if np.linalg.norm(v_c) <= np.sqrt(2)/2:
        # Axis is too close to +z; try a different axis.
        v_c = np.cross(axis, np.array([0., 1., 0.]))
    v_c = v_c / np.linalg.norm(v_c)
    # TODO: Hongkai uses multiple perpendicular vectors for tighter
    # bound. Why does that make it tighter? Maybe worth a try?

    # Translate into a symmetric bound by finding a rotation to
    # "center" us in the bound region, and the symmetric bound size alpha.
    # -alpha <= theta - (a+b)/2 <= alpha
    # where alpha = (b-a) / 2
    alpha = (max_angle - min_angle) / 2.
    offset_angle = (max_angle + min_angle) / 2.
    R_offset = RotationMatrix(AngleAxis(offset_angle, axis)).matrix()
    # |R_WC*R_CJc*v - R_WP * R_PJp * R(k,(a+b)/2)*v | <= 2*sin (α / 2) in
    # global ik code; for us, I'm assuming the joint frames are aligned with
    # the body frames, so R_CJc and R_PJp are identitiy.
    lorentz_bound = 2 * np.sin(alpha / 2.)
    vector_diff = (
        child.R_optim.dot(v_c) - 
        parent.R_optim.dot(R_offset).dot(v_c)
    )
    # TODO: Linear approx?
    prog.AddLorentzConeConstraint(np.r_[lorentz_bound, vector_diff])
    return False

rotation_rule_to_encode_map = {
    UnconstrainedRotationRule: encode_UnconstrainedRotationRule,
    UniformBoundedRevoluteJointRule: encode_UniformBoundedRevoluteJointRule
}


# Return type of infer_mle_tree_with_mip
TreeInferenceResults = namedtuple("TreeInferenceResults", ["solver", "optim_result", "super_tree", "observed_nodes"])
# TODO(gizatt) Remote max_scene_extent_in_any_dir and calculate from grammar.
def infer_mle_tree_with_mip(grammar, observed_nodes, max_recursion_depth=10, solver="gurobi", verbose=False,
                            num_intervals_per_half_axis=2, max_scene_extent_in_any_dir=10.):
    ''' Given a grammar and an observed node set, find the MLE tree induced
    by that grammar (up to a maximum recursion depth) that reproduces the
    observed node set, or report that it's infeasible. '''

    start_time = time.time()
    if verbose:
        print("Starting setup.")

    super_tree = grammar.make_super_tree(max_recursion_depth=max_recursion_depth)
    # Copy observed node set -- we'll be annotating the nodes with decision variables.
    observed_nodes = deepcopy(observed_nodes)

    prog = MathematicalProgram()

    # Every node gets a binary variable to indicate
    # whether it's active or node. This is equivalent to a
    # binary variable on the incoming edge for the node.
    active_vars = prog.NewBinaryVariables(len(super_tree.nodes), "nodes_active")
    for k, node in enumerate(super_tree.nodes):
        node.active = active_vars[k]

    if verbose:
        print("Activation vars allocated.")

    # Every node gets an optimized pose.
    mip_rot_gen = MixedIntegerRotationConstraintGenerator(
        approach = MixedIntegerRotationConstraintGenerator.Approach.kBilinearMcCormick,
        num_intervals_per_half_axis=num_intervals_per_half_axis,
        interval_binning = IntervalBinning.kLogarithmic
    )
    for node_k, node in enumerate(super_tree.nodes):
        node.R_optim = prog.NewContinuousVariables(3, 3, "%s_%03d_R" % (node.__class__.__name__, node_k))
        node.t_optim = prog.NewContinuousVariables(3, "%s_%03d_t" % (node.__class__.__name__, node_k))
        # Trivial constraint: elements of R bounded in [-1, 1].
        prog.AddBoundingBoxConstraint(-np.ones(9), np.ones(9), node.R_optim.flatten())
        # We'll need to constrain some R's to be in SO(3), but
        # many R's are directly constrained and don't actually need
        # this (very expensive to create) constraint. So we delay
        # setup until we know the rotation is unconstrained.
        
    if verbose:
        print("Continuous variables allocated.")

    # Constraint root node to have the grammar's root position.
    root_node = super_tree.get_root()
    root_tf = torch_tf_to_drake_tf(grammar.root_node_tf)
    prog.AddBoundingBoxConstraint(
        root_tf.translation(), root_tf.translation(),
        root_node.t_optim
    )
    R_target = root_tf.rotation().matrix()
    for i in range(3):
        for j in range(3):
            prog.AddBoundingBoxConstraint(
                R_target[i, j], R_target[i, j],
                root_node.R_optim[i, j]
            )

    ## For each node in the super tree, add relationships between the parent
    ## and that node.
    for parent_node in super_tree:
        children = super_tree.get_children(parent_node)
        ## Get child rule list.
        if isinstance(parent_node, GeometricSetNode):
            rules = [parent_node.rule for k in range(len(children))]
        elif isinstance(parent_node, (AndNode, OrNode, IndependentSetNode)):
            rules = parent_node.rules
        elif isinstance(parent_node, TerminalNode):
            rules = []
        else:
            raise ValueError("Unexpected node type: ", type(parent_node))

        child_actives = [c.active for c in children]
        parent_node.child_actives = child_actives

        # Children active requires parent active.
        for child_active in child_actives:
            prog.AddLinearConstraint(parent_node.active >= child_active)

        ## Parent/child activation relationships and symmetry breaking.
        if isinstance(parent_node, GeometricSetNode):
            # Geometric node child ordering to reduce duplicate parse: a child can
            # only be active if the previous child is also active.
            for k in range(len(child_actives) - 1):
                # child_{k+1} implies child_k
                # i.e. child_k >= child_{k+1}
                prog.AddLinearConstraint(child_actives[k] >= child_actives[k+1])

            # Child supports should be ordered, to break symmetries.
            # TODO(gizatt) Does this apply to AND nodes? I don't think so, since
            # each node under an AND node has its own distribution. But I could conceivably
            # detect "equivalent" children and break symmetries there.
            for child, next_child in zip(children[:-1], children[1:]):
                prog.AddLinearConstraint(next_child.t_optim[0] >= child.t_optim[0])

            # The geometric process we used (np.random.geometric) is only supported
            # on k=1, ..., so constrain that the # of active children must be
            # nonzero if this node is on.
            if len(child_actives) > 0:
                prog.AddLinearConstraint(sum(child_actives) >= parent_node.active)

        elif isinstance(parent_node, AndNode):
            # All children should be on if the parent node is on.
            for k in range(len(child_actives)):
                prog.AddLinearConstraint(child_actives[k] >= parent_node.active)

        elif isinstance(parent_node, OrNode):
            # Exactly one child can be on if the parent is on.
            prog.AddLinearConstraint(sum(child_actives) == parent_node.active)

        elif isinstance(parent_node, (TerminalNode, IndependentSetNode)):
            # No additional constraint on which set of children should be active.
            pass
        
        else:
            raise ValueError("Unexpected node type: ", type(parent_node))
        

        ## Child location constraints relative to parent.
        for rule, child_node in zip(rules, children):
            xyz_rule, rotation_rule = rule.xyz_rule, rule.rotation_rule

            # Look up how to encode these rule types, and dispatch.
            assert type(xyz_rule) in xyz_rule_to_encode_map.keys(), type(xyz_rule)
            xyz_rule_to_encode_map[type(xyz_rule)](xyz_rule, prog, parent_node, child_node)
            assert type(rotation_rule) in rotation_rule_to_encode_map.keys(), type(rotation_rule)
            rotation_was_fully_constrained = \
                rotation_rule_to_encode_map[type(rotation_rule)](rotation_rule, prog, parent_node, child_node)

            if not child_node.observed and not rotation_was_fully_constrained:
                # In this case, and only this case, we need to make sure R_optim
                # is in SO(3).
                mip_rot_gen.AddToProgram(child_node.R_optim, prog)


    # For each observed node, add a binary variable for each possible
    # correspondence to a node in the observed set, where an active correspondence
    # forces the corresponded node to be the same position as the observed node.
    for n in super_tree:
        # (first prep some bookkeeping)
        n.outgoings = []
    for observed_node in observed_nodes:
        obs_tf = torch_tf_to_drake_tf(observed_node.tf)
        possible_sources = [n for n in super_tree if type(n) == type(observed_node)]
        source_actives = prog.NewBinaryVariables(len(possible_sources), observed_node.__class__.__name__ + "_sources")

        # Store these variables
        observed_node.source_actives = source_actives
        for k, n in enumerate(possible_sources):
            n.outgoings.append(source_actives[k])

        # Each observed node needs exactly one explaining input.
        prog.AddLinearEqualityConstraint(sum(source_actives) == 1)

        for k, node in enumerate(possible_sources):
            M = max_scene_extent_in_any_dir # Should upper bound positional error in any single dimension/

            # When correspondence is active, force the node to match the observed node.
            # Otherwise, it can vary within a big M of the observed node.
            obs_t = obs_tf.translation()
            obs_R = obs_tf.rotation().matrix()
            for i in range(3):
                prog.AddLinearConstraint(node.t_optim[i] <= obs_t[i] + 1E-6 + (1. - source_actives[k]) * M)
                prog.AddLinearConstraint(node.t_optim[i] >= obs_t[i] - 1E-6 - (1. - source_actives[k]) * M)
            M = 2. # Max error in a rotation matrix entry
            for i in range(3):
                for j in range(3):
                    prog.AddLinearConstraint(node.R_optim[i, j] <= obs_R[i, j] + 1E-6 + (1. - source_actives[k]) * M)
                    prog.AddLinearConstraint(node.R_optim[i, j] >= obs_R[i, j] - 1E-6 - (1. - source_actives[k]) * M)

    # Go back and make sure no node in the super tree is being used
    # to explain more than one observed node, and that the "observed"
    # nodes are only active if they are explaining something.
    # TODO(gizatt) These constraints are a little frustrating: without
    # them, the parsing likes to hallucinate unnecessary hidden nodes
    # since adding them increases the tree likelihood (since the net log-prob
    # of adding a new node is positive due to concentrated densities).
    # It makes me feel like I'm doing something fundamentally wrong, like using
    # total model prob to do model comparisons between models of different size.
    for node in super_tree:
        if node.observed:
            if len(node.outgoings) > 0:
                prog.AddLinearConstraint(sum(node.outgoings) <= 1)
                prog.AddLinearConstraint(node.active == sum(node.outgoings))
            else:
                # Never observed this type in the scene, so this node can't be active.
                prog.AddLinearConstraint(node.active == 0)
            
    # Finally, build the objective.
    for parent_node in super_tree:
        # For the discrete states, do maximum likelihood.
        children = super_tree.get_children(parent_node)
        if isinstance(parent_node, (AndNode, TerminalNode)):
            pass
        elif isinstance(parent_node, (OrNode, IndependentSetNode)):
            # Binary variables * log of probabilities.
            for p, child in zip(parent_node.rule_probs, children):
                prog.AddLinearCost(-np.log(p) * child.active)
        elif isinstance(parent_node, GeometricSetNode):
            # TODO(gizatt) Is this accurate given max_children causes a truncation
            # of the geometric dist? That might need special handling.
            p = parent_node.p
            for child in children:
                prog.AddLinearCost(-np.log(1.-p) * child.active)
        else:
            raise ValueError("Unexpected node in cost assembly: ", type(parent_node))

        # TODO(gizatt) Continuous costs, if I add rules that are expressed as
        # Gaussian costs (e.g. Gaussian relative placement).

    setup_time = time.time()
    if verbose:
        print("Setup time: ", setup_time - start_time)
        print("Num vars: ", prog.num_vars())
        print("Num constraints: ", sum([c.evaluator().num_constraints() for c in prog.GetAllConstraints()]))
        sys.stdout.flush()

    if solver == "gurobi":
        solver = GurobiSolver()
        options = SolverOptions()
        logfile = "/tmp/gurobi.log"
        os.system("rm %s" % logfile)
        options.SetOption(solver.id(), "LogFile", logfile)
        result = solver.Solve(prog, None, options)
        if verbose:
            print("Optimization success?: ", result.is_success())
            print("Logfile: ")
            with open(logfile) as f:
                print(f.read())

    elif solver == "branch_and_bound":
        # Mostly for testing; supplies a different result type.
        solver = MixedIntegerBranchAndBound(prog, ClpSolver().solver_id())
        result = solver.Solve()
        if verbose:
            print("Optimization success?: ", result == SolutionResult.kSolutionFound)
        # Return solver, since result is just a SolutionResult object.
        # (Why does the MIBNB solver not use SolverInterface?)
        result = [solver, result == SolutionResult.kSolutionFound]
    else:
        raise ValueError("Unknown solver: ", solver)

    solve_time = time.time() 
    if verbose:
            print("Solve time: ", solve_time-setup_time)
            print("Total time: ", solve_time - start_time)

    return TreeInferenceResults(solver, result, super_tree, observed_nodes)

def get_optimized_tree_from_mip_results(inference_results, assert_on_failure=False):
    ''' From the specified inference results, extract the scene tree. '''
    # Grab that supertree from optimization
    
    #TODO packing of TreeInferenceResults should handle this complexity
    if isinstance(inference_results.solver,  MixedIntegerBranchAndBound):
        optim_result, success = (
            inference_results.solver,
            inference_results.optim_result
        )
    elif isinstance(inference_results.solver, GurobiSolver):
        optim_result, success = (
            inference_results.optim_result,
            inference_results.optim_result.is_success()
        )
    else:
        raise ValueError(inference_results.solver)
    super_tree = inference_results.super_tree
    observed_nodes = inference_results.observed_nodes

    # Sanity-check observed nodes are explained properly.
    for observed_node in observed_nodes:
        if not np.isclose(np.sum(optim_result.GetSolution(observed_node.source_actives)), 1.):
            if assert_on_failure:
                raise ValueError("observed node %s not explained by MLE sol." % str(observed_node))
            else:
                print("WARN: observed node %s not explained by MLE sol." % str(observed_node))

    optimized_tree = SceneTree()
    for node in super_tree:
        if optim_result.GetSolution(node.active) > 0.5:
            optimized_tree.add_node(node)
            # May have to post-process R to closest good R?
            node.tf = drake_tf_to_torch_tf(RigidTransform(
                p=optim_result.GetSolution(node.t_optim),
                R=RotationMatrix(optim_result.GetSolution(node.R_optim))
            ))
            parents = list(super_tree.predecessors(node))
            assert len(parents) <= 1
            if len(parents) == 1:
                parent = parents[0]
                assert parent.active
                optimized_tree.add_edge(parent, node)
    return optimized_tree


TreeRefinementResults = namedtuple("TreeRefinementResults", ["optim_result", "refined_tree", "unrefined_tree"])
def optimize_scene_tree_with_nlp(scene_tree, objective="mle", verbose=False, max_scene_extent_in_any_dir=10.):
    ''' Given a scene tree, set up a nonlinear optimization:
    1) Keeps the tree structure the same, but tweaks non-observed,
        non-root node poses.
    2) Optimizes for feasibility (of relative node poses).
    3) Allows choice of objective: maximum tree score ("mle"), or closeness to
        current tree configuration ("projection")
    4) Uses the scene tree's current (possibly infeasible) configuration
        as the initial guess.
    '''
    eps = 1E-6
    start_time = time.time()
    prog = MathematicalProgram()

    # Add pose decision variables with constraints.
    root_node = scene_tree.get_root()
    for k, node in enumerate(scene_tree.nodes):
        # Declare decision variables for pose and seed them from
        # the input tree's poses.
        node.R_optim = prog.NewContinuousVariables(3, 3, "R_%d" % k)
        prog.SetInitialGuess(node.R_optim, node.rotation.detach().cpu().numpy())
        node.t_optim = prog.NewContinuousVariables(3, "t_%d" % k)
        prog.SetInitialGuess(node.t_optim, node.translation.detach().cpu().numpy())

        # If it's an observed node or the root node, constrain the pose
        # to not change, and that's it.
        if node.observed or node is root_node:
            # TODO(gizatt) Rip this apart and figure out why I need such loose feasibility
            # tolerances.
            R_target = node.rotation.cpu().detach().numpy()
            prog.AddBoundingBoxConstraint(R_target.flatten()-eps, R_target.flatten()+eps, node.R_optim.flatten())
            t_target = node.translation.cpu().detach().numpy()
            prog.AddBoundingBoxConstraint(t_target-eps, t_target+eps, node.t_optim)
        else:
            # Otherwise, constraint the pose to be a legal and good pose.
            # R.' R = I
            RtR = node.R_optim.T.dot(node.R_optim)
            I = np.eye(3)
            for i in range(3):
                for j in range(3):
                    prog.AddConstraint(RtR[i, j] >= I[i, j] - eps)
                    prog.AddConstraint(RtR[i, j] <= I[i, j] + eps)
            # det(R) = +1; using cross product form and expressing it loosely, just to
            # keep SNOPT from falsely returning a flip-and-rotation as a legitimate
            # solution. Idea here is that the first two columns of the rotation
            # are the X and Y axes of the new coordinate; the Z axis of that right-handed
            # coordinate system should be the same direction as the last column.
            z_dir = np.cross(node.R_optim[:, 0], node.R_optim[:, 1])
            prog.AddConstraint(np.dot(z_dir, node.R_optim[:, 2]) >= 0)
            # Strong bounding box on rotation matrix elements.
            #prog.AddBoundingBoxConstraint(-np.ones(9), np.ones(9), node.R_optim.flatten())
            # Translation
            # Add really loose bbox constraint, to keep SNOPT from running away.
            # TODO(gizatt) Update these to use scene tree production bounds, if I
            # ever get around to adding that.
            prog.AddBoundingBoxConstraint(
                -np.ones(3)*max_scene_extent_in_any_dir,
                np.ones(3)*max_scene_extent_in_any_dir,
            node.t_optim)

        
    # Constraint parent/child relationships.
    for parent_node in scene_tree.nodes:
        children, rules = scene_tree.get_children_and_rules(parent_node)

        ## Child location constraints relative to parent.
        for rule, child_node in zip(rules, children):
            xyz_rule, rotation_rule = rule.xyz_rule, rule.rotation_rule

            # Look up how to encode these rule types, and dispatch.
            assert type(xyz_rule) in xyz_rule_to_encode_map.keys(), type(xyz_rule)
            xyz_rule_to_encode_map[type(xyz_rule)](xyz_rule, prog, parent_node, child_node)
            assert type(rotation_rule) in rotation_rule_to_encode_map.keys(), type(rotation_rule)
            rotation_was_fully_constrained = \
                rotation_rule_to_encode_map[type(rotation_rule)](rotation_rule, prog, parent_node, child_node)

    def penalize_rotation_error(R_goal, R_optim):
        # We don't always have perfect rotations, so instead of using
        # arccos((tr(R_diff) - 1) / 2) (the angular distance),
        # I'll use the distance chord distance of a few vectors rotated
        # by the rotation difference, which goes to zero if the
        # rotations are the same. I use multiple vectors so the rotation
        # axis is the same as a vector, we still get signal.
        # TODO(gizatt) Would just quadratic error between desired + actual R
        # work just as well but be computationally easier?
        R_diff = R_goal.T.dot(R_optim)
        for k in range(3):
            vec = np.zeros(3)
            vec[k] = 1.
            vec_rot = R_diff.dot(vec)
            prog.AddCost((vec_rot - vec).T.dot(vec_rot - vec))

    def penalize_pose_error(pose_goal, t_optim, R_optim):
        prog.AddQuadraticErrorCost(np.eye(3), pose_goal.translation(), t_optim)
        penalize_rotation_error(pose_goal.rotation().matrix(), R_optim)

    if objective == "mle":
        # Add costs for MLE tree estimate
        pass
    elif objective == "projection":
        # Try to get optimized tree as close as possible to current config.
        for node in scene_tree.nodes:
            if not node.observed and node is not root_node:
                pose_goal = RigidTransform(
                    p=node.translation.cpu().detach().numpy(),
                    R=RotationMatrix(node.rotation.cpu().detach().numpy())
                )
                penalize_pose_error(pose_goal, node.t_optim, node.R_optim)


    else:
        raise ValueError("Unknown objective spec \"%s\"" % objective)

    ## Solve
    setup_time = time.time()
    solver = SnoptSolver()
    options = SolverOptions()
    logfile = "/tmp/snopt.log"
    os.system("rm %s" % logfile)
    options.SetOption(solver.id(), "Print file", logfile)
    options.SetOption(solver.id(), "Major feasibility tolerance", eps)
    result = solver.Solve(prog, None, options)
    solve_time = time.time()

    if verbose:
        print("Success?: ", result.is_success())
        print("Logfile: ")
        with open(logfile) as f:
            print(f.read())
        print("Solve time: ", solve_time-setup_time)
        print("Total time: ", solve_time - start_time)

    out_tree = deepcopy(scene_tree)
    if result.is_success():
        # Copy results into scene tree node positions
        for out_node, orig_node in zip(out_tree, scene_tree):
            if orig_node is root_node or orig_node.observed:
                # Don't mutate the fixed nodes
                continue
            out_node.translation = torch.tensor(result.GetSolution(orig_node.t_optim))
            out_node.rotation = torch.tensor(result.GetSolution(orig_node.R_optim))
    return TreeRefinementResults(result, out_tree, scene_tree)
