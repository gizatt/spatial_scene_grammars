'''
Utilities for fitting grammar parameters given sets of
observed trees.
'''

import time
import networkx as nx
import numpy as np
import sys
from copy import deepcopy
from collections import namedtuple
from collections.abc import Iterable   # import directly from collections for Python < 3.3
from functools import partial
import logging
import multiprocessing as mp

try:
    mp.set_start_method('spawn')
except RuntimeError as e:
    logging.warn(e)

import torch
torch.set_default_tensor_type(torch.DoubleTensor)

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
    IntervalBinning,
    Variable
)

from .nodes import *
from .rules import *
from .scene_grammar import *
from .drake_interop import *
from .sampling import *
from .parsing import *
from .visualization import *


def fit_grammar_params_to_sample_sets_with_uninformative_prior(grammar, posterior_sample_sets):
    ## Fit node and rule parameters using closed-form solutions, assuming
    # an uninformative prior.

    # Collect parent/child sets for each node type.
    observed_child_sets_per_node_type = {}
    for node_type in grammar.all_types:
        observed_child_sets_per_node_type[node_type.__name__] = []
    for sample_set in posterior_sample_sets:
        for tree in sample_set:
            for node in tree:
                if isinstance(node, TerminalNode):
                    pass
                observed_child_sets_per_node_type[type(node).__name__].append( (node, tree.get_children(node)) )

    for node_type in grammar.all_types:
        # Fit the child weights for the node type.
        observed_child_sets = observed_child_sets_per_node_type[node_type.__name__]
        if len(observed_child_sets) == 0:
            # Don't try to update if we didn't get any observations of
            # this node type.
            continue
        if issubclass(node_type, AndNode):
            continue
        elif issubclass(node_type, OrNode):
            # Sum up the rule_k weights, and set
            # the new weighting to the average.
            count = torch.zeros(len(node_type.generate_rules()))
            for (_, children) in observed_child_sets:
                for child in children:
                    count[child.rule_k] += 1
            avg_count = count / torch.sum(count)
            # Bound the avg count so we don't make nodes absolutely
            # impossible to see.
            avg_count = torch.clip(avg_count, 1E-4, 1.-1E-4)
            grammar.params_by_node_type[node_type.__name__].set(avg_count)
        elif issubclass(node_type, GeometricSetNode):
            # Record average count of children, whose inverse
            # is a maximum likelihood estimate of p.
            # https://en.wikipedia.org/wiki/Geometric_distribution#Statistical_inference
            n_children = [len(children) for (_, children) in observed_child_sets]
            # TODO: Pretty sure this isn't right because the underlying geometric
            # set node has capped # of outputs. Maybe I should change its parameter
            # space to be the # of outputs probabilities, and call it a "repeating item"
            # node?
            p = 1./torch.mean(torch.tensor(n_children, dtype=torch.double))
            p = torch.clip(p, 1E-4, 1.-1E-4)
            grammar.params_by_node_type[node_type.__name__].set(p)
        elif issubclass(node_type, IndependentSetNode):
            # For each child, record how often it's active.
            count = torch.zeros(len(node_type.generate_rules()))
            for (_, children) in observed_child_sets:
                for child in children:
                    count[child.rule_k] += 1.
            count /= len(observed_child_sets)
            count = torch.clip(count, 1E-4, 1.-1E-4)
            grammar.params_by_node_type[node_type.__name__].set(count)
        elif issubclass(node_type, TerminalNode):
            continue
        else:
            raise NotImplementedError(node_type)

        ## For each rule type expected under this node, find optimal parameters.
        rules = node_type.generate_rules()

        # Go collect all parent/child pairs for the rules.
        parent_child_pairs_for_rules = [[] for k in range(len(rules))]
        if issubclass(node_type, GeometricSetNode):
            # Special case: only one rule that all children
            # correspond to.
            assert len(rules) == 1
            for parent, children in observed_child_sets:
                for child in children:
                    parent_child_pairs_for_rules[0].append((parent, child))
        else:
            for parent, children in observed_child_sets:
                for child in children:
                    parent_child_pairs_for_rules[child.rule_k].append((parent, child))

        for rule_k, (xyz_param_dict, rot_param_dict) in enumerate(
                grammar.rule_params_by_node_type[node_type.__name__]):
            xyz_rule = rules[rule_k].xyz_rule
            rot_rule = rules[rule_k].rotation_rule
            parent_child_pairs = parent_child_pairs_for_rules[rule_k]
            if len(parent_child_pairs) == 0:
                continue
            ## XYZ Rules
            if type(xyz_rule) == WorldBBoxRule:
                # The inferred lb/ub (from which we'll derive center/width)
                # will be the biggest deviation between parent and child.
                offsets = torch.stack([child.translation for (_, child) in parent_child_pairs])
                lb = torch.min(offsets, axis=0)[0]
                ub = torch.max(offsets, axis=0)[0]
                xyz_param_dict["center"].set((lb + ub) / 2.)
                xyz_param_dict["width"].set(ub - lb)
            elif type(xyz_rule) == AxisAlignedBBoxRule:
                # The inferred lb/ub (from which we'll derive center/width)
                # will be the biggest deviation between parent and child.
                offsets = torch.stack([child.translation - parent.translation for (parent, child) in parent_child_pairs])
                lb = torch.min(offsets, axis=0)[0]
                ub = torch.max(offsets, axis=0)[0]
                xyz_param_dict["center"].set((lb + ub) / 2.)
                xyz_param_dict["width"].set(ub - lb)
            elif type(xyz_rule) == AxisAlignedGaussianOffsetRule:
                # The inferred mean and variance are fit from deviations
                # between parent and child.
                offsets = torch.stack([child.translation - parent.translation for (parent, child) in parent_child_pairs])
                mean = torch.mean(offsets, axis=0)
                variance = torch.var(offsets, dim=0, unbiased=True)
                xyz_param_dict["mean"].set(mean)
                xyz_param_dict["variance"].set(variance)
            elif type(xyz_rule) == SamePositionRule:
                # No parameters
                pass
            else:
                raise NotImplementedError("type %s under node %s" % (type(xyz_rule), node_type))
            ## Rotation rules
            if type(rot_rule) == (SameRotationRule, UnconstrainedRotationRule):
                # No parameters
                pass
            elif type(rot_rule) == UniformBoundedRevoluteJointRule:
                # The inferred lb/ub (from which we'll derive center/width)
                # will be the biggest deviation between parent and child in terms
                # of axis/angle rotation.
                offsets = []
                for parent, child in parent_child_pairs:
                    angle, _ = recover_relative_angle_axis(parent, child, rot_rule.axis)
                    offsets.append(angle)
                offsets = torch.stack(offsets)
                lb = torch.min(offsets)
                ub = torch.max(offsets)
                rot_param_dict["center"].set((lb + ub) / 2.)
                rot_param_dict["width"].set(ub - lb)
            elif type(rot_rule) == GaussianChordOffsetRule:
                # Infer concentration and variance by gathering parent/child
                # angles around axis, and apply parameter estimates from
                # https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
                offsets = []
                for parent, child in parent_child_pairs:
                    angle, _ = recover_relative_angle_axis(parent, child, rot_rule.axis)
                    offsets.append(angle)
                angle_offsets = torch.stack(offsets)
                # Turn the set of angle offsets into vectors
                xs = torch.stack([torch.cos(angle_offsets), torch.sin(angle_offsets)], axis=1)
                # The loc is the angle of the average vector
                x_avg = torch.mean(xs, dim=0)
                x_norm = x_avg.square().sum().sqrt()
                x_avg = x_avg / x_norm
                loc = torch.atan2(x_avg[1], x_avg[0])
                # The concentration is approximately 
                concentration = (x_norm * (2 - x_norm**2)) / (1 - x_norm**2)
                rot_param_dict["loc"].set(loc)

    return grammar

def get_posterior_tree_samples_from_observation(
            grammar, observed_nodes, verbose=0,
            num_samples=20, subsample_step=1, hmc_strategy="NUTS",
            vis=None):
    if vis is not None:
        draw_scene_tree_contents_meshcat(
            SceneTree.make_from_observed_nodes(observed_nodes), zmq_url=vis.window.zmq_url, prefix="observed"
        )
    
    # Use a MIP to get MAP structure.
    mip_results = infer_mle_tree_with_mip(
        grammar, observed_nodes, verbose=verbose>1, max_scene_extent_in_any_dir=10.
    )
    mip_optimized_tree = get_optimized_tree_from_mip_results(mip_results)
    if not mip_optimized_tree:
        return None

    if vis is not None:
        draw_scene_tree_structure_meshcat(mip_optimized_tree, zmq_url=vis.window.zmq_url, prefix="mip_optimized")
                                      
    # Use NLP to refine that to a MAP estimate.
    refinement_results = optimize_scene_tree_with_nlp(mip_optimized_tree, verbose=verbose>1)
    refined_tree = refinement_results.refined_tree

    if vis is not None:
        draw_scene_tree_structure_meshcat(mip_optimized_tree, zmq_url=vis.window.zmq_url, prefix="nlp_optimized")
    
    # And sample trees around that MAP estimate with the
    # same structure.
    # Langevin-esque:
    if hmc_strategy=="langevin":
        sampled_trees = do_fixed_structure_hmc_with_constraint_penalties(
            grammar, refined_tree, num_samples=num_samples, subsample_step=subsample_step, verbose=verbose,
            kernel_type="HMC", num_steps=1, step_size=1E-3, adapt_step_size=True
        )
    # NUTS, defaults except limiting tree depth, to save on calls to
    # the slow model. Much slower, but should theoretically get much more
    # diversity as long as it doesn't diverge.
    elif hmc_strategy=="NUTS":
        sampled_trees = do_fixed_structure_hmc_with_constraint_penalties(
            grammar, refined_tree, num_samples=num_samples, subsample_step=subsample_step, verbose=verbose,
            kernel_type="NUTS", max_tree_depth=4
        )
    else:
        raise ValueError(hmc_strategy)
    
    
    # Finally, subsample the sampled trees as requested and return
    # the sampled set.
    return sampled_trees

def _contains(obj, type_of_interest):
    if isinstance(obj, type_of_interest):
        return True
    try:
        return any([_contains(subobj, type_of_interest) for subobj in obj if subobj is not obj])
    except TypeError:
        return False

def _cleanup_object(obj):
    to_remove = []
    for key, value in vars(obj).items():
        if _contains(value, Variable):
            to_remove.append(key)
        if isinstance(value, torch.Tensor):
            setattr(obj, key, value.detach())
    for key in to_remove:
        delattr(obj, key)
    
def _cleanup_tree_for_pickling(tree):
    # Remove all variables, detach all tensors.
    _cleanup_object(tree)
    for node in tree.nodes:
        _cleanup_object(node)
        for rule in node.rules:
            _cleanup_object(rule)
    return tree

def _get_samples_from_observation(arg_tuple):
    grammar, observed_nodes, vis = arg_tuple
    try:
        posterior_samples = get_posterior_tree_samples_from_observation(
            grammar, observed_nodes, verbose=1, hmc_strategy="NUTS",
            num_samples=100, subsample_step=5, vis=vis
        )
    except Exception as e:
        logging.warning("Unexpected error: ", e)
        return None
    return [_cleanup_tree_for_pickling(tree) for tree in posterior_samples]

def collect_posterior_sample_sets(grammar, observed_node_sets, num_workers=1, tqdm=None, vis=None):
    # Given a grammar and a set of observed node sets, retrieves a list of
    # lists of scene trees sampled (approximately) from the posterior of
    # trees given that observed set.
    # If num_workers > 1, will work on different observed node sets
    # in different threads.
    posterior_sample_sets = []

    if vis is not None:
        assert num_workers == 1, "Multiprocessing + meshcat not implemented"

    if num_workers > 1:
        logging.warning("Multiprocessing takes tons of FDs: make sure ulimit is large (~100k).")

    if num_workers == 1:
        # Single-threaded case.
        if tqdm is None:
            iterator = observed_node_sets
        else:
            iterator = tqdm(observed_node_sets, desc='Collecting posterior samples')
        for observed_nodes in iterator:
            posterior_samples = _get_samples_from_observation((grammar, observed_nodes, vis))
            if posterior_samples is not None:
                posterior_sample_sets.append(posterior_samples)
    else:
        # Multi-processing case.
        pool = mp.Pool(min(num_workers, mp.cpu_count()))
        args = [ (grammar, observed_nodes, None) for observed_nodes in observed_node_sets ]
        imap = pool.imap(func=_get_samples_from_observation, iterable=args)
        if tqdm is not None:
            imap = tqdm(imap, total=len(args))
        for posterior_samples in imap:
            if posterior_samples is not None:
                posterior_sample_sets.append(posterior_samples)
        pool.close()

    return posterior_sample_sets