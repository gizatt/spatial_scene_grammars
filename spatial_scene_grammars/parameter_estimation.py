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
from torch.distributions import transform_to

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
    PiecewisePolynomial,
    VPolytope,
    MixedIntegerRotationConstraintGenerator,
    IntervalBinning,
    Variable, Expression, Formula
)

from .nodes import *
from .rules import *
from .scene_grammar import *
from .drake_interop import *
from .sampling import *
from .parsing import *
from .visualization import *


def fit_grammar_params_to_sample_sets_with_uninformative_prior(grammar, posterior_sample_sets, weight_by_sample_prob=False, min_weight=1e-4):
    ## Fit node and rule parameters using closed-form solutions, assuming
    # an uninformative prior.

    # Annotate trees with weights.
    for posterior_set in posterior_sample_sets:
        log_weights = []
        for posterior_tree in posterior_set:
            if weight_by_sample_prob:
                log_weights.append(posterior_tree.score(include_discrete=True, include_continuous=True).detach())
            else:
                log_weights.append(torch.tensor(0.))
        # Normalize log weights across this sample batch
        log_weights = torch.tensor(log_weights)
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
        for log_weight, tree in zip(log_weights, posterior_set):
            tree.__log_weight = log_weight

    # Collect parent/child sets for each node type.
    observed_child_sets_per_node_type = {}
    for node_type in grammar.all_types:
        observed_child_sets_per_node_type[node_type.__name__] = []
    for sample_set in posterior_sample_sets:
        for tree in sample_set:
            for node in tree:
                if isinstance(node, TerminalNode):
                    pass
                observed_child_sets_per_node_type[type(node).__name__].append( (node, tree.get_children(node), tree.__log_weight) )

    for node_type in grammar.all_types:
        # Fit the child weights for the node type.
        observed_child_sets = observed_child_sets_per_node_type[node_type.__name__]
        if len(observed_child_sets) == 0:
            logging.warning("%s had zero observed child sets. Skipping..." % node_type.__name__)
            continue
        # Pre-normalize child set weights to head off any numeric issues
        observed_child_sets_weights = torch.stack([weight for (_, _, weight) in observed_child_sets])
        observed_child_sets_weights = torch.exp(observed_child_sets_weights - torch.logsumexp(observed_child_sets_weights, dim=0)).flatten()
        if len(observed_child_sets) == 0:
            # Don't try to update if we didn't get any observations of
            # this node type.
            continue
        if issubclass(node_type, AndNode):
            # We might have child rules to update, so don't continue;
            # but we don't have any discrete-choice params to update. 
            pass
        elif issubclass(node_type, OrNode):
            # Sum up the rule_k weights, and set
            # the new weighting to the average.
            count = torch.zeros(len(node_type.generate_rules()))
            for (_, children, _), weight in zip(observed_child_sets, observed_child_sets_weights):
                for child in children:
                    # Maybe precision problems here? Need more bookkeeping
                    # to do proper logspace a
                    count[child.rule_k] += weight
            avg_count = count / torch.sum(count)
            # Bound the avg count so we don't make nodes absolutely
            # impossible to see.
            avg_count = torch.clip(avg_count, min_weight, 1.-min_weight)
            grammar.params_by_node_type[node_type.__name__].set(avg_count)
        elif issubclass(node_type, GeometricSetNode):
            # Record weighted-average count of children, whose inverse
            # is a maximum likelihood estimate of p.
            # https://en.wikipedia.org/wiki/Geometric_distribution#Statistical_inference

            n_children = [len(children) for (_, children, _) in observed_child_sets]
            # TODO: Pretty sure this isn't right because the underlying geometric
            # set node has capped # of outputs. Maybe I should change its parameter
            # space to be the # of outputs probabilities, and call it a "repeating item"
            # node?
            p = 1./torch.sum(torch.tensor(n_children, dtype=torch.double) * observed_child_sets_weights)
            p = torch.clip(p, min_weight, 1.-min_weight)
            grammar.params_by_node_type[node_type.__name__].set(p)
        elif issubclass(node_type, IndependentSetNode):
            # For each child, record weighted average of how often it's active.
            count = torch.zeros(len(node_type.generate_rules()))
            for (_, children, _), weight in zip(observed_child_sets, observed_child_sets_weights):
                for child in children:
                    count[child.rule_k] += weight
            count = torch.clip(count, min_weight, 1.-min_weight)
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
            for (parent, children, _), weight in zip(observed_child_sets, observed_child_sets_weights):
                for child in children:
                    parent_child_pairs_for_rules[0].append((parent, child, weight))
        else:
            for (parent, children, _), weight in zip(observed_child_sets, observed_child_sets_weights):
                for child in children:
                    parent_child_pairs_for_rules[child.rule_k].append((parent, child, weight))

        for rule_k, (xyz_param_dict, rot_param_dict) in enumerate(
                grammar.rule_params_by_node_type[node_type.__name__]):
            xyz_rule = rules[rule_k].xyz_rule
            rot_rule = rules[rule_k].rotation_rule
            parent_child_pairs = parent_child_pairs_for_rules[rule_k]
            if len(parent_child_pairs) == 0:
                continue
            ## XYZ Rules
            if xyz_rule.fix_parameters is True:
                pass
            elif type(xyz_rule) == WorldFrameBBoxRule:
                # The inferred lb/ub (from which we'll derive center/width)
                # will be the biggest deviation between parent and child, irrespective of weights.
                offsets = torch.stack([child.translation for (_, child, _) in parent_child_pairs])
                lb = torch.min(offsets, axis=0)[0]
                ub = torch.max(offsets, axis=0)[0]
                xyz_param_dict["center"].set((lb + ub) / 2.)
                xyz_param_dict["width"].set(ub - lb)
            elif type(xyz_rule) == WorldFrameBBoxOffsetRule:
                # The inferred lb/ub (from which we'll derive center/width)
                # will be the biggest deviation between parent and child, irrespective of weights.
                offsets = torch.stack([child.translation - parent.translation for (parent, child, _) in parent_child_pairs])
                lb = torch.min(offsets, axis=0)[0]
                ub = torch.max(offsets, axis=0)[0]
                xyz_param_dict["center"].set((lb + ub) / 2.)
                xyz_param_dict["width"].set(ub - lb)
            elif type(xyz_rule) == WorldFrameGaussianOffsetRule:
                # The inferred mean and variance are fit from deviations
                # between parent and child.
                offsets = torch.stack([child.translation - parent.translation for (parent, child, _) in parent_child_pairs]) # Nx3
                weights = torch.stack([weight for (_, _, weight) in parent_child_pairs]) # N
                weights = weights / torch.sum(weights) # Renormalize, as we may not have one-to-one parent-child pair to rules.
                mean = torch.sum(offsets.T * weights, axis=1) # 3
                # Weighted average of squared deviations as variance estimate
                variance = torch.sum( torch.square((offsets - mean)).T * weights, axis=1 )
                xyz_param_dict["mean"].set(mean)
                xyz_param_dict["variance"].set(variance)
            elif type(xyz_rule) == SamePositionRule:
                # No parameters
                pass
            else:
                raise NotImplementedError("type %s under node %s" % (type(xyz_rule), node_type))
            ## Rotation rules
            if rot_rule.fix_parameters is True:
                pass
            elif type(rot_rule) == SameRotationRule or type(rot_rule) == UnconstrainedRotationRule:
                # No parameters
                pass
            elif type(rot_rule) == UniformBoundedRevoluteJointRule:
                # The inferred lb/ub (from which we'll derive center/width)
                # will be the biggest deviation between parent and child in terms
                # of axis/angle rotation, irrespective of weights.
                offsets = []
                for parent, child, _ in parent_child_pairs:
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
                raise NotImplementedError("Need to update this to work with weights.")
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
            elif type(rot_rule) == WorldFrameBinghamRotationRule:
                child_quats = [matrix_to_quaternion(child.rotation) for _, child, _ in parent_child_pairs]
                child_quats = torch.stack(child_quats)
                weights = torch.stack([weight for (_, _, weight) in parent_child_pairs]) 
                weights = weights / torch.sum(weights) # Renormalize, as we may not have one-to-one parent-child pair to rules.
                new_m, new_z = BinghamDistribution.fit(child_quats, weights)
                rot_param_dict["M"].set(new_m)
                rot_param_dict["Z"].set(new_z)

            else:
                raise NotImplementedError("type %s under node %s" % (type(rot_rule), node_type))

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
    refinement_results = optimize_scene_tree_with_nlp(grammar, mip_optimized_tree, verbose=verbose>1)
    refined_tree = refinement_results.refined_tree

    if vis is not None:
        draw_scene_tree_structure_meshcat(mip_optimized_tree, zmq_url=vis.window.zmq_url, prefix="nlp_optimized")
    
    # And sample trees around that MAP estimate with the
    # same structure.
    # Langevin-esque:
    if hmc_strategy=="langevin":
        sampled_trees = do_fixed_structure_hmc_with_constraint_penalties(
            grammar, refined_tree, num_samples=num_samples, subsample_step=subsample_step, verbose=verbose,
            kernel_type="HMC", num_steps=1, step_size=1.0, adapt_step_size=False, target_accept_prob=0.75
        )
    # NUTS, defaults except limiting tree depth, to save on calls to
    # the slow model. Much slower, but should theoretically get much more
    # diversity as long as it doesn't diverge.
    elif hmc_strategy=="NUTS":
        sampled_trees = do_fixed_structure_hmc_with_constraint_penalties(
            grammar, refined_tree, num_samples=num_samples, subsample_step=subsample_step, verbose=verbose,
            kernel_type="NUTS", max_tree_depth=5, target_accept_prob=0.75
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
        return any([_contains(subobj, type_of_interest) for subobj in iter(obj) if subobj is not obj])
    except TypeError:
        return False

def _cleanup_object(obj):
    to_remove = []
    for key, value in vars(obj).items():
        if _contains(value, (Variable, Expression, Formula)):
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

def _get_samples_from_observation(arg_dict):
    try:
        posterior_samples = get_posterior_tree_samples_from_observation(
            **arg_dict
        )
    except Exception as e:
        logging.warning("Unexpected error: ", e)
        return None
    return [_cleanup_tree_for_pickling(tree) for tree in posterior_samples]

def collect_posterior_sample_sets(grammar, observed_node_sets, num_workers=1, tqdm=None, vis=None,
                                  verbose=1, hmc_strategy="NUTS", num_samples=100, subsample_step=5, **kwargs):
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

    def make_arg_dict(k):
        return {
            "grammar": grammar,
            "observed_nodes": observed_node_sets[k],
            "verbose": verbose,
            "hmc_strategy": hmc_strategy,
            "num_samples": num_samples,
            "subsample_step": subsample_step,
            "vis": vis,
            **kwargs
        }
    
    if num_workers == 1:
        # Single-threaded case.
        if tqdm is None:
            iterator = range(len(observed_node_sets))
        else:
            iterator = tqdm(range(len(observed_node_sets)), desc='Collecting posterior samples')
        for k in iterator:
            posterior_samples = _get_samples_from_observation(make_arg_dict(k))
            if posterior_samples is not None:
                posterior_sample_sets.append(posterior_samples)
    else:
        # Multi-processing case.
        pool = mp.Pool(min(num_workers, mp.cpu_count()))
        args = [ make_arg_dict(k) for k in range(len(observed_node_sets)) ]
        imap = pool.imap(func=_get_samples_from_observation, iterable=args)
        if tqdm is not None:
            imap = tqdm(imap, total=len(args))
        for posterior_samples in imap:
            if posterior_samples is not None:
                posterior_sample_sets.append(posterior_samples)
        pool.close()

    return posterior_sample_sets

def get_supertree_rules_from_parent(parent_node, children):
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
    return rules

class VariationalPosteriorSuperTree(torch.nn.Module):
    ''' Torch Module whose parameters encode a variational
        posterior distribution over the poses of all nodes in a
        super tree. Enables a data update that consumes a MAP-optimized
        tree and resets the posterior of nodes in that tree to
        be tightly matching the poses of nodes in that MAP tree.

        Distribution is a mean-field over the sample sites in the forward
        sampling process for the continuous sites in this tree.'''
    def __init__(self, super_tree, grammar):
        super().__init__()
        self.super_tree = deepcopy(super_tree)
        self.grammar_in_list = [grammar] # Hide in list so params don't slip into the posterior super tree
        # Keyed by sample site name
        self.supertree_mean_params = torch.nn.ModuleDict()
        self.supertree_var_params = torch.nn.ModuleDict()
        self.supertree_param_has_been_set = {}
        self.supertree_site_constraints = {}
        for parent in self.super_tree:
            children = self.super_tree.get_children(parent)
            rules = get_supertree_rules_from_parent(parent, children)
            for child, rule in zip(children, rules):
                for key, site_value in rule.get_site_values(parent, child).items():
                    site_name = "%s/%s/%s" % (parent.name, child.name, key)
                    constrained_value = site_value.value.detach()
                    site_constraint = site_value.fn.support
                    unconstrained_value = transform_to(site_constraint).inv(constrained_value)
                    self.supertree_site_constraints[site_name] = site_constraint
                    self.supertree_mean_params[site_name] = ConstrainedParameter(
                        unconstrained_value, constraints.real
                    )
                    #self.supertree_var_params[site_name] = ConstrainedParameter(
                    #    torch.ones(unconstrained_value.shape) * 0.01,
                    #    constraints.positive
                    #)
                    self.supertree_param_has_been_set[site_name] = False

    def make_shallow_copy(self):
        new_copy = VariationalPosteriorSuperTree(self.super_tree, self.grammar_in_list[0])
        new_copy.supertree_mean_params = self.supertree_mean_params
        new_copy.supertree_var_params = self.supertree_var_params
        return new_copy

    def update_map_tree(self, map_tree, update_posterior):
        # Traverse map tree and supertree in lockstep, updating
        # parameter values for nodes in the map tree. This prepares
        # SVI for a minor iteration (after parsing has found a new
        # good MAP tree with good discrete structure).
        self.map_tree = deepcopy(map_tree)

        self.map_tree_mean_params = {}
        self.map_tree_var_params = {}
        self.map_tree_site_constraints = {}

        node_queue = [(map_tree.get_root(), self.super_tree.get_root())]
        while len(node_queue) > 0:
            map_parent, super_parent = node_queue.pop() 
            map_children, map_rules = map_tree.get_children_and_rules(map_parent)
            super_children = self.super_tree.get_children(super_parent)
            super_rules = get_supertree_rules_from_parent(super_parent, super_children)

            for map_child, map_rule in zip(map_children, map_rules):
                super_child = super_children[map_child.rule_k]
                super_rule = super_rules[map_child.rule_k]
                super_site_values = super_rule.get_site_values(super_parent, super_child)
                map_site_values = map_rule.get_site_values(map_parent, map_child)
                for ((_, site_value), (key, super_site_value)) in zip(map_site_values.items(), super_site_values.items()):
                    # Params will be addressed by supertree node names
                    supertree_site_name = "%s/%s/%s" % (super_parent.name, super_child.name, key)
                    # Remap to map tree node names
                    map_site_name = "%s/%s/%s" % (map_parent.name, map_child.name, key)
                    
                    # Constraint should not have changed
                    site_constraint = site_value.fn.support
                    assert type(site_constraint) is type(super_site_value.fn.support)

                    # Update parameter to have the map value.
                    constrained_value = site_value.value.detach()
                    unconstrained_value = transform_to(site_constraint).inv(constrained_value)
                    supertree_mean_param = self.supertree_mean_params[supertree_site_name]
                    #supertree_var_param = self.supertree_var_params[supertree_site_name]
                    if update_posterior: # or not self.supertree_param_has_been_set[supertree_site_name]:
                        supertree_mean_param.set(unconstrained_value)
                        #supertree_var_param.set(torch.ones(unconstrained_value.shape) * 0.001)
                        self.supertree_param_has_been_set[supertree_site_name] = True

                    # Register these parameters to correspond to this node+rule, so we can
                    # condition rerolls of the map  tree.
                    self.map_tree_mean_params[map_site_name] = supertree_mean_param
                    #self.map_tree_var_params[map_site_name] = supertree_var_param
                    self.map_tree_site_constraints[map_site_name] = site_constraint

                node_queue.append((map_child, super_child))

    def forward_model(self, grammar):
        # Resample the continuous structure of this tree into
        # a book-keeping copy.
        scene_tree = deepcopy(self.map_tree)
        # Hook grammar back into tree
        grammar.update_tree_grammar_parameters(scene_tree)
        node_queue = [scene_tree.get_root()]
        while len(node_queue) > 0:
            parent = node_queue.pop(0)
            children, rules = scene_tree.get_children_and_rules(parent)
            for child, rule in zip(children, rules):
                with scope(prefix=parent.name):
                    rule.sample_child(parent, child)
                node_queue.append(child)

        # Implement observation constraints
        xyz_observed_variance = 1E-3
        rot_observed_variance = 1E-3
        for node, original_node in zip(scene_tree.nodes, self.map_tree.nodes):
            if node.observed:
                xyz_observed_dist = dist.Normal(original_node.translation, xyz_observed_variance)
                rot_observed_dist = dist.Normal(original_node.rotation, rot_observed_variance)
                pyro.sample("%s_xyz_observed" % node.name, xyz_observed_dist, obs=node.translation)
                pyro.sample("%s_rotation_observed" % node.name, rot_observed_dist, obs=node.rotation)
        # Implement joint axis constraints
        # Should not be necessary since we're in configuration space (random sample
        # sites of forward model) instead of maximal coords (poses).
        #axis_alignment_variance = 1E-2
        #for node in scene_tree.nodes:
        #    children, rules = scene_tree.get_children_and_rules(parent)
        #    for child, rule in zip(children, rules):
        #        if type(rule) == GaussianChordOffsetRule or type(rule) == UniformBoundedRevoluteJointRule:
        #            # Both of these rule types require that parent/child rotation is
        #            # about an axis.
        #            axis_from_parent = torch.matmul(node.rotation, node.axis)
        #            axis_from_child = torch.matmul(child.rotation, child.axis)
        #            inner_product = (axis_from_parent*axis_from_child).sum()
        #            pyro.sample("%s_axis_error_observed" % node.name,
        #                dist.Normal(1., axis_alignment_variance),
        #                obs=inner_product
        #            )
        return scene_tree

    def evaluate_log_density(self, grammar, verbose=0):
        # Sample from variational posterior, and use that to condition
        # the sample sites in the forward model.
        
        conditioning = {}
        for key in self.map_tree_mean_params.keys():
            q_density = dist.Delta(self.map_tree_mean_params[key]()) #, self.map_tree_var_params[key]())
            unconstrained_sample = q_density.rsample()
            # Map the unconstrained sample into the contrained parameter space
            conditioning[key] = transform_to(self.map_tree_site_constraints[key])(unconstrained_sample)
        
        with pyro.condition(data=conditioning):
            trace = pyro.poutine.trace(self.forward_model).get_trace(grammar)
        log_p_continuous = trace.log_prob_sum()
        assert torch.isfinite(log_p_continuous)
        scoring_tree = trace.nodes["_RETURN"]["value"]
        # Add score for the discrete decisions in the tree, which isn't captured in
        # our continuous-only forward model.
        log_p_discrete = torch.zeros(1)
        for node in scoring_tree.nodes:
            children = scoring_tree.get_children(node)
            log_p_discrete = log_p_discrete + node.score_child_set(children)
        assert torch.isfinite(log_p_discrete)

        # Sanity-check conditioning matched out tree
        for key in conditioning.keys():
            assert key in trace.nodes.keys()

        return log_p_continuous, log_p_discrete


def _get_map_trees(grammar, observed_nodes, verbose, max_scene_extent_in_any_dir, N_solutions, throw_on_failure):
    mip_results = infer_mle_tree_with_mip(
        grammar, observed_nodes, verbose=verbose>1, max_scene_extent_in_any_dir=max_scene_extent_in_any_dir,
        N_solutions=N_solutions
    )
    mip_optimized_trees = get_optimized_trees_from_mip_results(mip_results)
    refined_trees = []
    for mip_optimized_tree in mip_optimized_trees:
        if mip_optimized_tree is None:
            error_msg = "MIP optimization failed for observed set %d" % k
            if throw_on_failure:
                raise RuntimeError(error_msg)
            else:
                logging.error(error_msg)
                refined_trees = None
                break
        
        refinement_results = optimize_scene_tree_with_nlp(grammar, mip_optimized_tree, verbose=verbose>1,
                                                          max_scene_extent_in_any_dir=max_scene_extent_in_any_dir)
        refined_tree = refinement_results.refined_tree
        if refined_tree is None:
            error_msg = "Nonlinear refinement failed for observed set %d" % k
            if throw_on_failure:
                raise RuntimeError(error_msg)
            else:
                logging.warning(error_msg)
                refined_trees = None
                break
        
        refined_trees.append(refined_tree)
    return refined_trees

def _get_map_trees_thread_wrapper(arg_dict):
    try:
        refined_trees = _get_map_trees(**arg_dict)
        return [_cleanup_tree_for_pickling(tree) for tree in refined_trees]
    except Exception as e:
        logging.error("Error in thread: %s" % e)
        return []
    
def get_map_trees_for_observed_node_sets(grammar, observed_node_sets,
        throw_on_failure=False, verbose=0, tqdm=None, N_solutions=1,
        num_workers=1, max_scene_extent_in_any_dir=10.):
    ''' Get MAP parse list for each tree in dataset using our current grammar params.
        If parsing fails for a tree, will return None for that tree and throw
        if requsted. '''
    refined_tree_sets = []

    def make_arg_dict(k):
        return {
            "grammar": grammar,
            "observed_nodes": observed_node_sets[k],
            "verbose": verbose,
            "max_scene_extent_in_any_dir": max_scene_extent_in_any_dir,
            "N_solutions": N_solutions,
            "throw_on_failure": throw_on_failure
        }
    
    if num_workers == 1:
        # Single-threaded case.
        iterator = enumerate(observed_node_sets)
        if tqdm:
            iterator = tqdm(iterator, desc="Getting MAP parses", total=len(observed_node_sets))

        for k, observed_nodes in iterator:
            refined_tree_sets.append(
                _get_map_trees(**make_arg_dict(k))
            )
    else:
        # Multi-processing case.
        pool = mp.Pool(min(num_workers, mp.cpu_count()))

        args = [ make_arg_dict(k) for k in range(len(observed_node_sets)) ]

        imap = pool.imap(func=_get_map_trees_thread_wrapper, iterable=args)
        if tqdm is not None:
            imap = tqdm(imap, total=len(args))
        refined_tree_sets = []
        for refined_trees in imap:
            refined_tree_sets.append(refined_trees)
        pool.close()
    return refined_tree_sets


class EMWrapper():
    '''
        Given a grammar with params \theta, repeatedly:
            1) For each observed environment, produce the `N_solutions` best parses
            (i.e. trees sampled from the posterior) using the MIPMAP parser.
            Compute a weight for each posterior tree based on its tree
            joint probability (discrete + continuous), normalized across the N_solutions
            for that observed environment.

            2) Update the parameters in closed form given that set of solutions for
            each observed environment.
                - For each node type in the grammar, collect all parent/child set pairs with
                the parent of the right type. Count how often each child type appears, weighting
                by the tree weight of the tree that the parent/child pair came from. Update
                parameters accordingly.
                - For each rule type in the grammar, collect all parent/child pairs related
                by that rule. Fit the parameters of the rule to the parent/child observations,
                weighting by the tree weight of the tree that the parent/child pair came from.
    '''
    def __init__(self, grammar, observed_node_sets, max_scene_extent_in_any_dir=10.):
        self.grammar = grammar
        self.observed_node_sets = observed_node_sets
        self.max_scene_extent_in_any_dir = max_scene_extent_in_any_dir

    def do_iterated_em_fitting(self, em_iterations=5, throw_on_map_failure=False,
                               verbose=0, tqdm=None, max_recursion_depth=10, N_solutions=1,
                               num_workers=1, weight_by_sample_prob=True, min_weight=1E-4):
        self.grammar_iters = [deepcopy(self.grammar.state_dict())]
        if tqdm is None:
            iterator = range(em_iterations)
        else:
            iterator = tqdm(range(em_iterations), desc="EM Iteration", total=em_iterations)
        for iter_k in iterator:
            refined_tree_sets = get_map_trees_for_observed_node_sets(
                self.grammar, self.observed_node_sets, N_solutions=N_solutions, tqdm=tqdm,
                num_workers=num_workers, max_scene_extent_in_any_dir=self.max_scene_extent_in_any_dir
            )
            self.grammar = fit_grammar_params_to_sample_sets_with_uninformative_prior(
                self.grammar, refined_tree_sets, weight_by_sample_prob=weight_by_sample_prob,
                min_weight=min_weight
            )
            self.grammar_iters.append(deepcopy(self.grammar.state_dict()))
        return self.grammar

    def plot_grammar_parameter_history(self, node_type):
         # Plot param history for a node type in the grammar.
        assert hasattr(self, "grammar_iters") and len(self.grammar_iters) > 0
        assert node_type in self.grammar.all_types
        
        node_param_history = []
        rule_param_history = None
        for state_dict in self.grammar_iters:
            self.grammar.load_state_dict(state_dict)
            possible_params = self.grammar.params_by_node_type[node_type.__name__]
            if possible_params:
                node_param_history.append(possible_params().detach())
            rule_params = self.grammar.rule_params_by_node_type[node_type.__name__]
            if rule_param_history is None:
                rule_param_history = [[{}, {}] for k in range(len(rule_params))]
            
            for k, (xyz_rule_params, rot_rule_params) in enumerate(rule_params):
                for key, value in xyz_rule_params.items():
                    value = deepcopy(value().detach())
                    hist_dict = rule_param_history[k][0]
                    if key not in hist_dict:
                        hist_dict[key] = [value]
                    else:
                        hist_dict[key].append(value)
                for key, value in rot_rule_params.items():
                    value = deepcopy(value().detach())
                    hist_dict = rule_param_history[k][1]
                    if key not in hist_dict:
                        hist_dict[key] = [value]
                    else:
                        hist_dict[key].append(value)

        if len(node_param_history) > 0:
            node_param_history = torch.stack(node_param_history)
        for entry in rule_param_history:
            for k in range(2): # xyz / rot rule
                for key, value in entry[k].items():
                    entry[k][key] = torch.stack(value)

        if len(node_param_history) > 0:
            plt.figure().set_size_inches(6, 6)
            for k in range(node_param_history.shape[1]):
                plt.plot(node_param_history[:, k], label="%d" % k)
            plt.title("%s params" % node_type.__name__)
            plt.xlabel("EM iter")
            plt.legend()
            print("Final params: ", node_param_history[-1, :])
        
        # Rules
        N_rules = len(rule_param_history)
        plt.figure().set_size_inches(12, 6)
        for k, entry in enumerate(rule_param_history):
            plt.suptitle("%s rule %d params" % (node_type.__name__, k))
            # XYZ
            plt.subplot(2, N_rules, k+1)
            plt.title("XYZ Rule")
            for key, value in entry[0].items():
                print(key, value)
                for col in range(value.shape[1]):
                    plt.plot(value[:, col], label=key + "%d" % col)
                print("%d:xyz:%s final: %s" % (k, key, value[-1, :]))
            plt.xlabel("EM iter")
            plt.legend()
            plt.subplot(2, N_rules, k + N_rules + 1)
            plt.title("Rot rule")
            for key, value in entry[1].items():
                for col in range(value.shape[1]):
                    plt.plot(value[:, col], label=key + "%d" % col)
                print("%d:rot:%s final: %s" % (k, key, value[-1, :]))
            plt.xlabel("EM iter")
            plt.legend()
        plt.tight_layout()


class SVIWrapper():
    '''
    Initialize a variational posterior $q_\phi(z_k)$ for each observation $k$,
    where $z_k$ describes a distribution over the latent poses for each node
    in the supertree corresponding to observation $k$.
    Also initialize model parameters $\theta$ for any parameters in the grammar.

    Repeatedly:
    1. Find the MAP tree structure $t_k$ for all observations with the
        MIP using the current $\theta$.
    2. For the nodes that show up in $t_k$, set the posterior parameters
        $\phi$ to be tightly peaked around the MAP-optimal setting.
    3. Run a few gradient step updates on the ELBO evaluated using discrete tree
        structure $t_k$ and continuous poses sampled from the variational posterior.

    TODO: This *was* an SVI implementation, but the constraints are so stiff that
    the posteriors tended to be very tight. So I've swapped it out to Delta-distribution
    posteriors, i.e. this is just simultaneous MLE estimation of params + latent variables.

    This seems to conceptually work pretty well; however, it is really numerically unstable, especially
    when it's seeded with a MAP solution out of the MIP process. So it tends to do better when run
    with very small learning rates (~0.01) and lots of minor iterations (~200-500?). But that is *very*
    slow right now, because it evaluates the score for each tree sequentially. Vectorizing this would require
    vectoring Rule.score_child and Node.score_child_set; if I can do that, I can vectorize this cleanly
    and hopefully get a huge speedup.
    '''
    def __init__(self, grammar, observed_node_sets):
        self.grammar = grammar
        self.observed_node_sets = observed_node_sets

    def do_iterated_vi_fitting(self, major_iterations=1, minor_iterations=50, subsample=None, base_lr=0.1,
                               throw_on_map_failure=False, verbose=0, tqdm=None, max_recursion_depth=10,
                               clip=None, N_solutions=1):
        '''
        
            throw_on_map_failure: If there's a failure in the MAP tree optimization routine,
                should we throw a RuntimeError? If not, that observation will be ignored
                in the iterations where the optimization failed.
        '''
        
        # Logging: each major iteration produces a ModuleList of variational_posterior 
        # modules, along with a list of grammar state_dicts and variational_posterior_list state_dicts
        # from the minor iterations.
        self.elbo_history = []
        self.grammar_major_iters = []
        self.posterior_major_iters = []
        
        # Initialize a list of VariationalPosteriorSuperTrees for each observation -- one for each distinct
        # integer solution we expect.
        super_tree = self.grammar.make_super_tree(max_recursion_depth=max_recursion_depth, detach=True)

        variational_posterior_sets = []
        for k in range(len(self.observed_node_sets)):
            variational_posterior_set = []
            # Make one variational posterior, and then shallow-copy it so that they
            # all reference the same underlying supertree params. (I'm thinking of these
            # as being different "views" into the params, with the view changed by calling
            # update_map_tree.)
            orig_posterior = VariationalPosteriorSuperTree(super_tree, self.grammar)
            variational_posterior_set = [orig_posterior.make_shallow_copy() for k in range(N_solutions)]
            variational_posterior_sets.append(torch.nn.ModuleList(variational_posterior_set))
        variational_posterior_sets = torch.nn.ModuleList(variational_posterior_sets)
        
        # Set up common optimizer for the whole process.
        param_groups = [
         {"params": variational_posterior_sets.parameters(), "lr": base_lr},
         {"params": self.grammar.parameters(), "lr": base_lr}
        ]
        params = [*variational_posterior_sets.parameters(), *self.grammar.parameters()]
        optimizer = torch.optim.Adam(param_groups, lr=base_lr)#, betas = (0.95, 0.999))
        lr_schedule = PiecewisePolynomial.FirstOrderHold(
            breaks=[0., 10.],
            samples=[[0., 1.]]
        )
        def get_lr(epoch):
            return lr_schedule.value(epoch)[0, 0] * 0.999**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

        major_iterator = range(major_iterations)
        if tqdm:
            major_iterator = tqdm(major_iterator, desc="Major iteration", total=major_iterations)
        for major_iteration in major_iterator:
            if tqdm:
                major_iterator.set_description("Major %03d: calculating MAP trees" % (major_iteration))
            refined_tree_sets = get_map_trees_for_observed_node_sets(
                self.grammar, self.observed_node_sets, throw_on_failure=throw_on_map_failure,
                verbose=verbose, tqdm=tqdm, N_solutions=N_solutions
            )
            # Initialize variational posterior at the MAP tree.
            for variational_posterior_set, trees in zip(variational_posterior_sets, refined_tree_sets):
                for k, variational_posterior in enumerate(variational_posterior_set):
                    if k < len(trees):
                        variational_posterior.update_map_tree(trees[k], update_posterior=True)
                    else:
                        logging.warn("# of trees less than expected: ", len(trees))
                        variational_posterior.map_tree = None # No solution corresponds to this posterior, so don't update it.

            if tqdm:
                major_iterator.set_description("Major %03d: doing SVI iters" % (major_iteration))

            grammar_history = []
            variational_posterior_history = []
            
            minor_iterator = range(minor_iterations)
            if tqdm:
                minor_iterator = tqdm(minor_iterator, desc="Minor iteration", total=minor_iterations)
            for minor_iteration in minor_iterator:
                optimizer.zero_grad()
                
                grammar_history.append(deepcopy(self.grammar.state_dict()))
                variational_posterior_history.append(deepcopy(variational_posterior_sets.state_dict()))
                
                # Evaluate ELBO and do gradient updates.
                if subsample:
                    posterior_set_inds = np.random.choice(range(len(variational_posterior_sets)), size=subsample, replace=False)
                    posterior_sets = [variational_posterior_sets[k] for k in posterior_set_inds]
                else:
                    posterior_sets = variational_posterior_sets

                total_log_prob = torch.zeros(1)
                for posterior_set in posterior_sets:
                    all_continuous = []
                    all_discrete = []
                    for posterior in posterior_set:
                        if posterior.map_tree is not None:
                            log_p_continuous, log_p_discrete = posterior.evaluate_log_density(
                                self.grammar, verbose=verbose
                            )
                            all_continuous.append(log_p_continuous)
                            all_discrete.append(log_p_discrete)
                    if len(all_continuous) == 0:
                        continue
                    # p = sum_{discrete}[p(discrete) * p(continuous | discrete)] / sum_{discrete}[p(discrete)]
                    all_continuous = torch.stack(all_continuous).flatten()
                    all_discrete = torch.stack(all_discrete).flatten()
                    assert all_continuous.shape == all_discrete.shape
                    mode_probs = all_discrete + all_continuous
                    normalizer = torch.logsumexp(all_discrete, dim=0)
                    total_log_prob = total_log_prob + torch.logsumexp(mode_probs, dim=0) - normalizer
                mean_log_prob = total_log_prob / len(posterior_sets)
                self.elbo_history.append(mean_log_prob.detach())
                if tqdm:
                    minor_iterator.set_description("Minor %05d: ELBO %f" % (minor_iteration, mean_log_prob.item()))
                else:
                    logging.info("%03d/%05d: ELBO %f" % (major_iteration, minor_iteration, mean_log_prob.item()))
                
                if minor_iteration < minor_iterations - 1:
                    (-mean_log_prob).backward(retain_graph=False)
                    if clip is not None:
                        torch.nn.utils.clip_grad_norm_(params, clip)

                    optimizer.step()
                    scheduler.step()
                
            self.grammar_major_iters.append(grammar_history)
            self.posterior_major_iters.append((variational_posterior_sets, variational_posterior_history))
       
    def plot_elbo_history(self):
        assert hasattr(self, "elbo_history") and len(self.elbo_history) > 0
        plt.figure()
        loss_history = -torch.stack(self.elbo_history).detach()
        offset = torch.min(loss_history)
        loss_history += -offset + 1.
        plt.plot(loss_history)
        plt.yscale('log')
        plt.title("(Vertically shifted) ELBO history, reached min %f" % offset)
        plt.xlabel("Iter")
        plt.ylabel("ELBO")

    def plot_grammar_parameter_history(self, node_type):
         # Plot param history for a node type in the grammar.
        assert hasattr(self, "grammar_major_iters") and len(self.grammar_major_iters) > 0
        assert node_type in self.grammar.all_types
        
        all_state_dicts = [x for l in self.grammar_major_iters for x in l]
        node_param_history = []
        rule_param_history = None
        for state_dict in all_state_dicts:
            self.grammar.load_state_dict(state_dict)
            possible_params = self.grammar.params_by_node_type[node_type.__name__]
            if possible_params:
                node_param_history.append(possible_params().detach())
            rule_params = self.grammar.rule_params_by_node_type[node_type.__name__]
            if rule_param_history is None:
                rule_param_history = [[{}, {}] for k in range(len(rule_params))]
            
            for k, (xyz_rule_params, rot_rule_params) in enumerate(rule_params):
                for key, value in xyz_rule_params.items():
                    hist_dict = rule_param_history[k][0]
                    if key not in hist_dict:
                        hist_dict[key] = [value().detach()]
                    else:
                        hist_dict[key].append(value().detach())
                for key, value in rot_rule_params.items():
                    hist_dict = rule_param_history[k][1]
                    if key not in hist_dict:
                        hist_dict[key] = [value().detach()]
                    else:
                        hist_dict[key].append(value().detach())

        if len(node_param_history) > 0:
            node_param_history = torch.stack(node_param_history)
        for entry in rule_param_history:
            for k in range(2): # xyz / rot rule
                for key, value in entry[k].items():
                    entry[k][key] = torch.stack(value)

        if len(node_param_history) > 0:
            plt.figure()
            plt.plot(node_param_history)
            plt.title("%s params" % node_type.__name__)
            print("Final params: ", node_param_history[-1, :])
        
        # Rules
        plt.figure()
        N_rules = len(rule_param_history)
        for k, entry in enumerate(rule_param_history):
            plt.suptitle("%s rule %d params" % (node_type.__name__, k))
            # XYZ
            plt.subplot(2, N_rules, k+1)
            plt.title("XYZ Rule")
            for key, value in entry[0].items():
                plt.plot(value, label=key)
                print("%d:xyz:%s final: %s" % (k, key, value[-1, :]))
            plt.legend()
            plt.subplot(2, N_rules, k + N_rules + 1)
            plt.title("Rot rule")
            for key, value in entry[1].items():
                plt.plot(value, label=key)
                print("%d:rot:%s final: %s" % (k, key, value[-1, :]))
            plt.legend()
        plt.tight_layout()

