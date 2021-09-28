'''
# Scene Parsing utilities for this grammar

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
import logging

import pydrake
from pydrake.all import (
    AngleAxis,
    AngleAxis_,
    ClpSolver,
    CommonSolverOption,
    Expression,
    MathematicalProgram,
    MakeSolver,
    MixedIntegerBranchAndBound,
    RandomGenerator,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    RotationMatrix_,
    GurobiSolver,
    SnoptSolver,
    OsqpSolver,
    Solve,
    SolverOptions,
    SolutionResult,
    VPolytope,
    MixedIntegerRotationConstraintGenerator,
    IntervalBinning,
    UniformlyRandomRotationMatrix,
    Variable
)

from .nodes import *
from .rules import *
from .scene_grammar import *
from .drake_interop import *


def prepare_grammar_for_parsing(prog, grammar, optimize_parameters=False, inequality_eps=1E-6):
    # Populates grammar.rule_params_by_node_type_optim, which has parallel
    # structure to grammar.rule_params_by_node_type, which with appropriately
    # constrained decision variables if optimize_parameters is True, or
    # placeholder numpy values of the current parameter setting otherwise.

    def _make_constrained_param(param, name):
        assert isinstance(param, ConstrainedParameter)
        val = param().detach().cpu().numpy()
        if optimize_parameters:
            new_vars = prog.NewContinuousVariables(*val.shape, name)
            if isinstance(param.constraint, constraints._Real):
                # No constraint necessary
                pass
            elif isinstance(param.constraint, constraints._Interval):
                prog.AddBoundingBoxConstraint(
                    param.constraint.lower_bound.detach().cpu().numpy(),
                    param.constraint.upper_bound.detach().cpu().numpy(),
                    new_vars
                )
            elif isinstance(param.constraint, constraints._Positive):
                for var in new_vars:
                    prog.AddLinearConstraint(prog >= inequality_eps)
            else:
                raise NotImplementedError("Constraint type %s" % param.constraint)
            return new_vars
        else:
            return val

    grammar.rule_params_by_node_type_optim = {}
    for node_type in grammar.all_types:
        param_dicts_by_rule = grammar.rule_params_by_node_type[node_type.__name__]
        grammar.rule_params_by_node_type_optim[node_type.__name__] = []
        rules = node_type.generate_rules()
        for k, (rule, (xyz_params, rot_params)) in enumerate(zip(rules, param_dicts_by_rule)):
            prefix = "%s:%s(%d):" % (node_type.__name__, type(rule).__name__, k)
            xyz_var_dict = {}
            for key, param in xyz_params.items():
                xyz_var_dict[key] = _make_constrained_param(param, name=prefix+key)
            rot_var_dict = {}
            for key, param in rot_params.items():
                rot_var_dict[key] = _make_constrained_param(param, name=prefix+key)
            grammar.rule_params_by_node_type_optim[node_type.__name__].append(
                [xyz_var_dict, rot_var_dict]
            )
    return grammar


def add_mle_tree_parsing_to_prog(
        prog, grammar, observed_nodes, max_recursion_depth=10,
        num_intervals_per_half_axis=2, max_scene_extent_in_any_dir=10.,
        verbose=False, use_random_rotation_offset=True):
    # The grammar should be pre-processed by `prepare_grammar_for_parsing`,
    # which creates decision variables or placeholders for the grammar parameters,
    # which we access from here.
    super_tree = grammar.make_super_tree(max_recursion_depth=max_recursion_depth, detach=True)
    root_node = super_tree.get_root()
    # Copy observed node set -- we'll be annotating the nodes with decision variables.
    observed_nodes = deepcopy(observed_nodes)

    # Support using a random rotation offset. For a random offset R_o, when a rotation matrix
    # encoded in decision variables R_dec is used, it's first (left) multiplied with R_o. Since many
    # optimal node poses may be axis-aligned (i.e. have zeros in the rotation matrix),
    # this means R_dec usually *won't* have zeros. Not having zeros in R_dec is important
    # to minimize the number of equivalent binary variable settings that represent the
    # same rotation; zeros in the rotation matrix are at the overlap of the piecewise
    # McCormick envelope setup used to approximate the rotation constraint.
    # (see sandbox/mip_rotation_constraint_solution_uniqueness).
    if use_random_rotation_offset:
        R_random_offset = UniformlyRandomRotationMatrix(RandomGenerator(42))
    else:
        R_random_offset = RotationMatrix(np.eye(3))

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
        node.R_optim_pre_offset = prog.NewContinuousVariables(3, 3, "%s_%03d_R" % (node.__class__.__name__, node_k))
        node.R_optim = R_random_offset.matrix().dot(node.R_optim_pre_offset)
        node.t_optim = prog.NewContinuousVariables(3, "%s_%03d_t" % (node.__class__.__name__, node_k))
        # Trivial constraint: elements of R bounded in [-1, 1].
        prog.AddBoundingBoxConstraint(-np.ones(9), np.ones(9), node.R_optim_pre_offset.flatten())
        # We'll need to constrain some R's to be in SO(3), but
        # many R's are directly constrained and don't actually need
        # this (very expensive to create) constraint. So we delay
        # setup until we know the rotation is unconstrained. These
        # vars track if
        #  1) The node rotation will constrained by correspondence to
        #     an observation or root-node tf.
        #  2) The parent/child rotation relationship is fully constrained
        #     (i.e. equality or fixed offset).
        if node.observed or node is root_node:
            node.rotation_is_fully_constrained = True
        else:
            node.rotation_is_fully_constrained = False
        # Default to not constrained; encoded rules may update this value.
        node.rotation_fully_constrained_relative_to_parent = False

        # For rules, annotate them with their parameter set for
        # convenience later. (This saves some effort, since the
        # node-type-to-rule mapping is a big trickier to work out in
        # a supertree context since we have to hand-construct child-to-rule
        # mappings for different node types.)
        param_vars_by_rule = grammar.rule_params_by_node_type_optim[node.__class__.__name__]
        for rule, (xyz_optim_params, rot_optim_params) in zip(node.rules, param_vars_by_rule):
            rule.xyz_optim_params = xyz_optim_params
            rule.rot_optim_params = rot_optim_params
        
    if verbose:
        print("Continuous variables allocated.")

    # Constraint root node to have the grammar's root position.
    root_tf = torch_tf_to_drake_tf(grammar.root_node_tf)
    prog.AddBoundingBoxConstraint(
        root_tf.translation(), root_tf.translation(),
        root_node.t_optim
    )
    # R_optim = R_offset * R_optim_pre_offset = R_target
    # R_optim_pre_offset = R_offset.T * R_target
    R_target_pre_offset = R_random_offset.matrix().T.dot(root_tf.rotation().matrix())
    prog.AddBoundingBoxConstraint(R_target_pre_offset.flatten(), R_target_pre_offset.flatten(), root_node.R_optim_pre_offset.flatten())

    ## For each node in the super tree, add relationships between the parent
    ## and that node.
    for parent_node in super_tree:
        children = super_tree.get_children(parent_node)
        ## Get child rule list. Can't use get_children_and_rules
        # here since we're operating on a supertree, so the standard
        # scene tree logic for getting rules isn't correct.
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
            child_node.rotation_fully_constrained_relative_to_parent = rule.encode_constraint(
                prog, rule.xyz_optim_params, rule.rot_optim_params, parent_node, child_node
            )

    # Propagate rotations being fully constrained up and down the tree to be sure
    # we really only add rotation constraints where they're necessary. This is done
    # in a two-step, inside-out process: first pass constraints up the tree as far
    # as possible, then pass back down to any children that aren't already constrained.
    # Step 1: For every node that is already fully constrained, if its parent/child
    # relationship is fully constrained, fully constrain the parent.
    fully_constrained_nodes = [node for node in super_tree if node.rotation_is_fully_constrained]
    while len(fully_constrained_nodes) > 0:
        node = fully_constrained_nodes.pop()
        parent = super_tree.get_parent(node)
        if parent and not parent.rotation_is_fully_constrained and node.rotation_fully_constrained_relative_to_parent:
            parent.rotation_is_fully_constrained = True
            fully_constrained_nodes.append(parent)
    # Step 2: Same thing, but go downwards instead of upwards.
    fully_constrained_nodes = [node for node in super_tree if node.rotation_is_fully_constrained]
    while len(fully_constrained_nodes) > 0:
        node = fully_constrained_nodes.pop()
        children = super_tree.get_children(node)
        for child in children:
            if not child.rotation_is_fully_constrained and child.rotation_fully_constrained_relative_to_parent:
                child.rotation_is_fully_constrained = True
                fully_constrained_nodes.append(child)

    for node in super_tree:
        if not node.rotation_is_fully_constrained:
            # In this case, and only this case, we need to make sure R_optim
            # is in SO(3).
            mip_rot_gen.AddToProgram(node.R_optim_pre_offset, prog)


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
    # to explain more than one observed node, that the "observed"
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
        ## Get child rule list.
        if isinstance(parent_node, GeometricSetNode):
            rules = [parent_node.rule for k in range(len(children))]
        elif isinstance(parent_node, (AndNode, OrNode, IndependentSetNode)):
            rules = parent_node.rules
        elif isinstance(parent_node, TerminalNode):
            rules = []
        else:
            raise ValueError("Unexpected node type: ", type(parent_node))

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

        ## Child location costs relative to parent.
        for rule, child_node in zip(rules, children):
            rule.encode_cost(
                prog, rule.xyz_optim_params, rule.rot_optim_params, parent_node, child_node
            )
    return super_tree, observed_nodes, R_random_offset

# Return type of infer_mle_tree_with_mip
TreeInferenceResults = namedtuple("TreeInferenceResults", ["solver", "optim_result", "super_tree", "observed_nodes", "R_random_offset"])
# TODO(gizatt) Remote max_scene_extent_in_any_dir and calculate from grammar.
def infer_mle_tree_with_mip(grammar, observed_nodes, max_recursion_depth=10, solver="gurobi", verbose=False,
                            num_intervals_per_half_axis=2, max_scene_extent_in_any_dir=10.,
                            N_solutions=1, use_random_rotation_offset=True):
    ''' Given a grammar and an observed node set, find the MLE tree induced
    by that grammar (up to a maximum recursion depth) that reproduces the
    observed node set, or report that it's infeasible. '''
    start_time = time.time()
    if verbose:
        print("Starting setup.")

    prog = MathematicalProgram()

    grammar = prepare_grammar_for_parsing(prog, grammar, optimize_parameters=False)
    super_tree, observed_nodes, R_random_offset = add_mle_tree_parsing_to_prog(
        prog, grammar, observed_nodes, max_recursion_depth=max_recursion_depth,
        verbose=verbose, num_intervals_per_half_axis=num_intervals_per_half_axis,
        max_scene_extent_in_any_dir=max_scene_extent_in_any_dir,
        use_random_rotation_offset=use_random_rotation_offset
    )
    
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
        os.system("rm -f %s" % logfile)
        options.SetOption(solver.id(), "LogFile", logfile)
        options.SetOption(solver.id(), "MIPGap", 1E-3)
        if N_solutions > 1:
            options.SetOption(solver.id(), "PoolSolutions", N_solutions)
            options.SetOption(solver.id(), "PoolSearchMode", 2)

        result = solver.Solve(prog, None, options)
        # Hacky method getter because `num_suboptimal_solution()` was bound with () in its
        # method name. Should fix this upstream!
        actual_N_solutions = getattr(result, "num_suboptimal_solution()")()
        if actual_N_solutions != N_solutions:
            logging.warning("MIP got %d solutions, but requested %d. ", actual_N_solutions, N_solutions)
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

    return TreeInferenceResults(solver, result, super_tree, observed_nodes, R_random_offset)


def get_optimized_tree_from_mip_results(inference_results, assert_on_failure=False, solution_k=0):
    ''' From the specified inference results, extract the k^th suboptimal scene tree. For k=0,
    returns the optimal solution. '''
    
    #TODO packing of TreeInferenceResults should handle this complexity
    if isinstance(inference_results.solver,  MixedIntegerBranchAndBound):
        optim_result, success = (
            inference_results.solver,
            inference_results.optim_result
        )
        assert solution_k == 0, "MIBnB doesn't support suboptimal sols."
        def get_sol(var):
            return optim_result.GetSolution(var)

    elif isinstance(inference_results.solver, GurobiSolver):
        optim_result, success = (
            inference_results.optim_result,
            inference_results.optim_result.is_success()
        )
        def get_sol(var):
            return optim_result.GetSuboptimalSolution(var, solution_k)

    else:
        raise ValueError(inference_results.solver)
    super_tree = inference_results.super_tree
    observed_nodes = inference_results.observed_nodes

    # Sanity-check observed nodes are explained properly.
    for observed_node in observed_nodes:
        if not np.isclose(np.sum(get_sol(observed_node.source_actives)), 1.):
            if assert_on_failure:
                raise ValueError("observed node %s not explained by MLE sol." % str(observed_node))
            else:
                logging.warning("Observed node %s not explained by MLE sol." % str(observed_node))

    if not success:
        logging.warning("MIP structure finding unsuccessful.")
    optimized_tree = SceneTree()
    # Build tree top-down so we know parent is already in new tree.
    potential_node_queue = [(super_tree.get_root(), None)]
    while len(potential_node_queue) > 0:
        node, parent = potential_node_queue.pop(0)
        if get_sol(node.active) > 0.5:
            t_sol = get_sol(node.t_optim)
            R_sol = RotationMatrix(
                inference_results.R_random_offset.matrix().dot(
                    get_sol(node.R_optim_pre_offset))
            )
            new_tf = drake_tf_to_torch_tf(RigidTransform(
                p=t_sol, R=R_sol
            ))
            new_node = deepcopy(node)
            new_node.tf = new_tf
            optimized_tree.add_node(new_node)
            if parent is not None:
                optimized_tree.add_edge(parent, new_node)
            children = list(super_tree.successors(node))
            for child in children:
                potential_node_queue.append((child, new_node))
    return optimized_tree

def get_optimized_trees_from_mip_results(inference_results, assert_on_failure=False):
    if isinstance(inference_results.solver, MixedIntegerBranchAndBound):
        return [get_optimized_tree_from_mip_results(inference_results, assert_on_failure)]
    # Hacky method getter because `num_suboptimal_solution()` was bound with () in its
    # method name. Should fix this upstream!
    N_solutions = getattr(inference_results.optim_result, "num_suboptimal_solution()")()
    assert N_solutions >= 1
    return [get_optimized_tree_from_mip_results(inference_results, assert_on_failure, k) for k in range(N_solutions)]

TreeRefinementResults = namedtuple("TreeRefinementResults", ["optim_result", "refined_tree", "unrefined_tree"])
def optimize_scene_tree_with_nlp(grammar, scene_tree, initial_guess_tree=None, objective="mle", verbose=False, max_scene_extent_in_any_dir=10.):
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
    grammar = prepare_grammar_for_parsing(prog, grammar, optimize_parameters=False)

    if initial_guess_tree is None:
        initial_guess_tree = scene_tree

    # Add pose decision variables with constraints.
    root_node = scene_tree.get_root()
    for k, (node, hint_node) in enumerate(zip(scene_tree.nodes, initial_guess_tree)):
        # Declare decision variables for pose and seed them from
        # the input tree's poses.
        node.R_optim = prog.NewContinuousVariables(3, 3, "R_%d" % k)
        prog.SetInitialGuess(node.R_optim, hint_node.rotation.detach().cpu().numpy())
        node.t_optim = prog.NewContinuousVariables(3, "t_%d" % k)
        prog.SetInitialGuess(node.t_optim, hint_node.translation.detach().cpu().numpy())

        # If it's an observed node or the root node, constrain the pose
        # to not change, and that's it.
        if node.observed or node is root_node:
            # TODO(gizatt) Rip this apart and figure out why I need such loose feasibility
            # tolerances.
            R_target = node.rotation.cpu().detach().numpy()
            prog.AddBoundingBoxConstraint(R_target.flatten()-eps, R_target.flatten()+eps, node.R_optim.flatten())
            t_target = node.translation.cpu().detach().numpy()
            prog.AddBoundingBoxConstraint(t_target, t_target, node.t_optim)
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
        optim_params_by_rule = grammar.rule_params_by_node_type_optim[type(parent_node).__name__]
        for rule, child_node, (xyz_optim_params, rot_optim_params) in zip(rules, children, optim_params_by_rule):
            rule.encode_constraint(
                prog, xyz_optim_params, rot_optim_params, parent_node, child_node
            )

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
        for prent_node in scene_tree.nodes:
            children, rules = scene_tree.get_children_and_rules(parent_node)
            ## Child location costs relative to parent.
            optim_params_by_rule = grammar.rule_params_by_node_type_optim[type(parent_node).__name__]
            for rule, child_node, (xyz_optim_params, rot_optim_params) in zip(rules, children, optim_params_by_rule):
                rule.encode_cost(
                    prog, xyz_optim_params, rot_optim_params, parent_node, child_node
                )

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
    else:
        logging.warning("Nonlinear refinement unsuccessful.")
    return TreeRefinementResults(result, out_tree, scene_tree)
