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
# https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999
torch.multiprocessing.set_sharing_strategy('file_system')

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
from .torch_utils import calculate_mmd


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
                    # to do proper logspace calculations.
                    count[child.rule_k] += weight
            avg_count = count / torch.sum(count)
            # Bound the avg count so we don't make nodes absolutely
            # impossible to see.
            avg_count = torch.clip(avg_count, min_weight, 1.-min_weight)
            grammar.params_by_node_type[node_type.__name__].set(avg_count)
        elif issubclass(node_type, RepeatingSetNode):
            # Record frequency of each child count, adjusting
            # for the child set occurance weight.
            child_counts = torch.zeros(grammar.params_by_node_type[node_type.__name__]().shape[0])
            for (_, children, _), weight in zip(observed_child_sets, observed_child_sets_weights):
                # Maybe precision problems here? Need more bookkeeping
                # to do proper logspace calculations.
                child_counts[len(children)] += weight
            avg_child_counts = child_counts / torch.sum(child_counts)    
            grammar.params_by_node_type[node_type.__name__].set(avg_child_counts)
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
        if issubclass(node_type, RepeatingSetNode):
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
            elif type(xyz_rule) == ParentFrameGaussianOffsetRule:
                # Same as WorldFrame case, but calculate offsets using rotations.
                offsets = torch.stack([
                    torch.matmul(parent.rotation.T, (child.translation - parent.translation))
                    for (parent, child, _) in parent_child_pairs
                ]) # Nx3
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
                    angle, _, _ = recover_relative_angle_axis(parent, child, rot_rule.axis)
                    offsets.append(angle)
                offsets = torch.stack(offsets)
                lb = torch.min(offsets)
                ub = torch.max(offsets)
                rot_param_dict["center"].set((lb + ub) / 2.)
                rot_param_dict["width"].set(ub - lb)
            elif type(rot_rule) == WorldFrameBinghamRotationRule:
                child_quats = [matrix_to_quaternion(child.rotation) for _, child, _ in parent_child_pairs]
                child_quats = torch.stack(child_quats)
                weights = torch.stack([weight for (_, _, weight) in parent_child_pairs]) 
                weights = weights / torch.sum(weights) # Renormalize, as we may not have one-to-one parent-child pair to rules.
                new_m, new_z = BinghamDistribution.fit(child_quats, weights)
                # Sanity-check  before reassigning
                if torch.allclose(torch.matmul(new_m.T, new_m), torch.eye(4), atol=1E-4) and all(torch.isfinite(new_z)):
                    rot_param_dict["M"].set(new_m)
                    rot_param_dict["Z"].set(torch.clip(new_z, -1E4, 0.))
                else:
                    logging.error("Got bad M and Z from fit: ", new_m, new_z, torch.matmul(new_m.T, new_m))
            elif type(rot_rule) == ParentFrameBinghamRotationRule:
                # Same as  above case, but use parent/child rotation delta.
                child_quats = [
                    matrix_to_quaternion(torch.matmul(parent.rotation.T, child.rotation))
                    for _, child, _ in parent_child_pairs
                ]
                child_quats = torch.stack(child_quats)
                weights = torch.stack([weight for (_, _, weight) in parent_child_pairs]) 
                weights = weights / torch.sum(weights) # Renormalize, as we may not have one-to-one parent-child pair to rules.
                new_m, new_z = BinghamDistribution.fit(child_quats, weights)
                # Sanity-check  before reassigning
                if torch.allclose(torch.matmul(new_m.T, new_m), torch.eye(4), atol=1E-4) and all(torch.isfinite(new_z)):
                    rot_param_dict["M"].set(new_m)
                    rot_param_dict["Z"].set(torch.clip(new_z, -1E4, 0.))
                else:
                    logging.error("Got bad M and Z from fit: ", new_m, new_z, torch.matmul(new_m.T, new_m))
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


def _get_samples_from_observation(arg_dict):
    try:
        posterior_samples = get_posterior_tree_samples_from_observation(
            **arg_dict
        )
    except Exception as e:
        logging.warning("Unexpected error: ", e)
        return None
    return [cleanup_tree_for_pickling(tree) for tree in posterior_samples]

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
    if isinstance(parent_node, RepeatingSetNode):
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


def _get_map_trees(grammar, observed_nodes, verbose, max_scene_extent_in_any_dir, N_solutions, throw_on_failure, parsing_strategy, do_nlp_refinement):

    if parsing_strategy == "micp":
        mip_results = infer_mle_tree_with_mip(
            grammar, observed_nodes, verbose=verbose>1, max_scene_extent_in_any_dir=max_scene_extent_in_any_dir,
            N_solutions=N_solutions
        )
        mip_optimized_trees = get_optimized_trees_from_mip_results(mip_results)
    elif parsing_strategy == "ip" or parsing_strategy == "ip_noproposals":
        if parsing_strategy == "ip":
            proposed_poses_by_type = generate_candidate_node_pose_sets(
                grammar, observed_nodes, max_recursion_depth=10, verbose=verbose>1
            )
        else:
            proposed_poses_by_type = {}
        mip_optimized_trees = infer_mle_tree_with_mip_from_proposals(
            grammar, observed_nodes, proposed_poses_by_type, verbose=verbose>1,
            N_solutions=N_solutions
        )
    else:
        logging.error("Bad parsing strategy %s" % parsing_strategy)
        return []

    # Skip refinement if not requested.
    if do_nlp_refinement is False:
        return mip_optimized_trees

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
        start_time = time.time()
        refined_trees = _get_map_trees(**arg_dict)
        elapsed = time.time() - start_time
        return [cleanup_tree_for_pickling(tree) for tree in refined_trees], elapsed
    except Exception as e:
        logging.error("Error in thread: %s" % e)
        return []
    
def get_map_trees_for_observed_node_sets(grammar, observed_node_sets,
        throw_on_failure=False, verbose=0, tqdm=None, N_solutions=1,
        num_workers=1, max_scene_extent_in_any_dir=10., report_timing=False,
        parsing_strategy="ip", do_nlp_refinement=True):
    ''' Get MAP parse list for each tree in dataset using our current grammar params.
        If parsing fails for a tree, will return None for that tree and throw
        if requsted. '''

    assert parsing_strategy in ["ip", "ip_noproposals", "micp"]

    refined_tree_sets = []

    def make_arg_dict(k):
        return {
            "grammar": grammar,
            "observed_nodes": observed_node_sets[k],
            "verbose": verbose,
            "max_scene_extent_in_any_dir": max_scene_extent_in_any_dir,
            "N_solutions": N_solutions,
            "throw_on_failure": throw_on_failure,
            "parsing_strategy": parsing_strategy,
            "do_nlp_refinement": do_nlp_refinement
        }
    
    if num_workers == 1:
        # Single-threaded case.
        iterator = enumerate(observed_node_sets)
        if tqdm:
            iterator = tqdm(iterator, desc="Getting MAP parses", total=len(observed_node_sets))

        times = []
        for k, observed_nodes in iterator:
            start_time = time.time()
            refined_tree_sets.append(
                _get_map_trees(**make_arg_dict(k))
            )
            times.append(start_time - time.time())
        if report_timing:
            print("Elapsed %fs +/ %f (min %f, median %f, max %f)" % (np.mean(times), np.std(times), np.median(times), min(times), max(times)))

    else:
        # Multi-processing case.

        args = [ make_arg_dict(k) for k in range(len(observed_node_sets)) ]

        refined_tree_sets = []
        times = []
        
        def chunks(l, n):
            n = max(1, n)
            return (l[i:i+n] for i in range(0, len(l), n))

        chunk_size = 20
        for chunk_k, args_subset in enumerate(chunks(args, chunk_size)):

            pool = mp.Pool(min(num_workers, mp.cpu_count()))
            imap = pool.imap(func=_get_map_trees_thread_wrapper, iterable=args_subset)

            if tqdm is not None:
                imap = tqdm(imap, total=len(args_subset), desc="Chunk %d/%d" % (chunk_k + 1, len(args) // chunk_size + 1))
            for refined_trees, elapsed in imap:
                refined_tree_sets.append(refined_trees)
                times.append(elapsed)
            pool.close()

        if report_timing:
            print("Elapsed %fs +/ %f (min %f, median %f, max %f)" % (np.mean(times), np.std(times), np.median(times), min(times), max(times)))

    if report_timing:
        return refined_tree_sets, times
    else:
        return refined_tree_sets


class EMWrapper():
    '''
        Given a grammar with params \theta, repeatedly:
            1) For each observed environment, produce the `N_solutions` best parses
            (i.e. trees sampled from the posterior) using the chosen parsing strategy
            (strategy="ip", "ip_noproposals" (optimal when the grammar has no unobserved
            equivalent sets), or "micp"). 
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
    def __init__(self, grammar, observed_node_sets, max_scene_extent_in_any_dir=10.,
                 parsing_strategy="ip", do_nlp_refinement=True):
        self.grammar = grammar
        assert parsing_strategy in ["ip", "ip_noproposals", "micp"]
        self.parsing_strategy = parsing_strategy
        self.do_nlp_refinement = do_nlp_refinement
        self.observed_node_sets = observed_node_sets
        self.max_scene_extent_in_any_dir = max_scene_extent_in_any_dir

        # For storing fitting process. If empty, fitting has not been
        # done yet.
        self.grammar_iters = []
        self.log_evidence_iters = []

    def do_iterated_em_fitting(self, em_iterations=5, throw_on_map_failure=False,
                               verbose=0, tqdm=None, max_recursion_depth=10, N_solutions=1,
                               num_workers=1, weight_by_sample_prob=True, min_weight=1E-4):
        self.grammar_iters = [deepcopy(self.grammar.state_dict())]
        self.log_evidence_iters = []
        if tqdm is None:
            iterator = range(em_iterations)
        else:
            iterator = tqdm(range(em_iterations), desc="EM Iteration", total=em_iterations)
        for iter_k in iterator:
            refined_tree_sets = get_map_trees_for_observed_node_sets(
                self.grammar, self.observed_node_sets, N_solutions=N_solutions, tqdm=tqdm,
                num_workers=num_workers, max_scene_extent_in_any_dir=self.max_scene_extent_in_any_dir,
                parsing_strategy=self.parsing_strategy, do_nlp_refinement=self.do_nlp_refinement
            )
            # Compute tree score for each tree, and sum them to get a (very approximate)
            # estimate of the log-evidence.
            log_evidence_population = []
            for tree_set in refined_tree_sets:
                approx_log_evidence = torch.logsumexp(torch.tensor([tree.score().detach() for tree in tree_set]), dim=0).item()
                log_evidence_population.append(approx_log_evidence)
            self.log_evidence_iters.append(log_evidence_population)
            # Do grammar update.
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

    
class SampleBasedFittingWrapper():
    '''
        Given a grammar and a population of observed samples compatible with
        the observed types from that grammar, repeatedly sample populations
        from the grammar, calculate a differentiable distribution distance
        between that sample population and the target population, and
        optimize the grammar parameters to improve that distance.

        We provide a differentiable MMD-based population distance metric to
        help with this. We use Pyro's tracing functionality to reparameterize
        which sample sites we can, and for all others, use REINFORCE to optimize
        this objective.
    '''
    def __init__(self, grammar, observed_node_sets, distance_metric="kde_poses"):
        self.grammar = grammar
        self.observed_node_sets = observed_node_sets
        self.distance_metric = distance_metric
        if distance_metric == "mean_mmd_poses":
            self._prep_for_mean_mmd_poses()
        elif distance_metric == "kde_poses":
            self.observed_kde_kernels_by_type = self._prep_kde_of_poses(self.observed_node_sets)
            self.moving_average_of_divergences = None
        else:
            raise ValueError(distance_metric)

        # For storing fitting process. If empty, fitting has not been
        # done yet.
        self.grammar_iters = []
        self.loss_iters = []

    def do_sample_based_fitting(
            self, num_iterations=100, num_samples=100, verbose=0, tqdm=None,
            lr=0.01, clip=10.):
        self.grammar_iters = [deepcopy(self.grammar.state_dict())]
        self.loss_iters = []

        # Set up parameter optimizer by attaching to grammar parameters.
        named_params = {key: value for key, value in self.grammar.named_parameters()}
        params = list(named_params.values())
        optimizer = torch.optim.Adam(params, lr=lr)
        lr_warmup_schedule = PiecewisePolynomial.FirstOrderHold(
            breaks=[0., 10.],
            samples=[[0., 1.]]
        )
        def get_lr(epoch):
            return lr_warmup_schedule.value(epoch)[0, 0] * 1.00**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

        if tqdm is None:
            iterator = range(num_iterations)
        else:
            iterator = tqdm(range(num_iterations), desc="Epoch", total=num_iterations)
        moving_average_loss = None
        for iter_k in iterator:
            # Sample a population of trees.
            sampled_trees = []
            for sample_k in range(num_samples):
                tree = self.grammar.sample_tree(detach=False)
                sampled_trees.append(tree)

            # Pass that population through to get a loss function.
            if self.distance_metric == "mean_mmd_poses":
                proxy_loss, log_loss = self._calc_loss_mean_mmd_poses(sampled_trees)
            elif self.distance_metric == "kde_poses":
                proxy_loss, log_loss = self._calc_loss_kde_poses(sampled_trees)
            else:
                raise ValueError(self.distance_metric)

            proxy_loss.backward(retain_graph=False)

            grad_norms = torch.tensor(
                [torch.norm(param.grad) for param in params if param.grad is not None]
            )

            
            self.loss_iters.append(log_loss)
            if iter_k % 10 == 0:
                print({"%s: %s, %s" % (name, param.detach().numpy(), param.grad) for (name, param) in named_params.items()})
            print("%d: Loss %f, Proxy Loss %f, Gradient Norm (%f +/- %f)" % (
                iter_k, log_loss.item(), proxy_loss.item(),
                torch.mean(grad_norms), torch.std(grad_norms))
            )
            
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(params, clip)
            optimizer.step()
            scheduler.step()

            self.grammar_iters.append(deepcopy(self.grammar.state_dict()))
        return self.grammar

    '''
        KDE: option "kde_poses"
        Related to tactic of MetaSim2 paper (https://arxiv.org/pdf/2008.09092.pdf),
        but minus the latent space calculation.
        We pre-compute a KDE for the set of observed poses p(x) = KDE_obs. Given
        a population of samples and compute
        the loss as KL(q(x) - p(x)), where q(x) is the model our samples come from.
        This translates to
            min_theta x ~ q_theta(x) [log q_theta(x) - log p(x)]
        which tries to cover p(x) as concisely as possible with q(x).
    '''
    def _prep_kde_of_poses(self, node_sets, bw=1):
        observed_population_by_type = {}
        for observed_nodes in node_sets:
            for observed_node in observed_nodes:
                key = type(observed_node).__name__
                if key not in observed_population_by_type.keys():
                    observed_population_by_type[key] = []
                observed_population_by_type[key].append(observed_node.tf)
        for key, value in observed_population_by_type.items():
            observed_population_by_type[key] = torch.stack(value)

        # Set up KDE kernels
        # TODO(gizatt) Better handling of rotation!
        return {
            key: dist.Normal(loc=value, scale=bw)
            for key, value in observed_population_by_type.items()
        }
    def _calc_kde_densities_per_sample(self, sampled_node_sets, kernel_dict):
        lls_by_scene = []
        for sampled_nodes in sampled_node_sets:
            total_ll = torch.tensor(0)
            for node in sampled_nodes:
                key = type(node).__name__
                if key not in kernel_dict.keys():
                    logging.warning("Sampled object not in kernel dict, applying large ll penalty instead: %s" % key)
                    total_ll = total_ll - 100
                    continue
                kernels = kernel_dict[key]
                # TODO(gizatt) Is this normalization correct?
                ll = torch.logsumexp(
                    kernels.log_prob(node.tf.unsqueeze(0).expand(kernels.batch_shape)),
                    dim=(0, 1, 2)
                ) -  torch.log(torch.tensor(kernels.batch_shape[0]))
                total_ll = total_ll + ll
            lls_by_scene.append(total_ll)
        return lls_by_scene
    def _calc_loss_kde_poses(self, sampled_trees):
        sampled_node_sets = [tree.get_observed_nodes() for tree in sampled_trees]
        obs_lls_by_scene = self._calc_kde_densities_per_sample(sampled_node_sets, self.observed_kde_kernels_by_type)
        
        # Prep symmetric KDE for the samples
        sampled_kde_kernels_by_type = self._prep_kde_of_poses(sampled_node_sets)
        sampled_lls_by_scene = self._calc_kde_densities_per_sample(sampled_node_sets, sampled_kde_kernels_by_type)
        
        # Perform partially-reparameterized REINFORCE update on KL divergence.
        proxy_loss = torch.zeros(1)
        divergences = []
        for k, (tree, ll_against_obs_kde, ll_against_sampled_kde) in enumerate(
                zip(sampled_trees, obs_lls_by_scene, sampled_lls_by_scene)):
            tree.trace.compute_score_parts()
            
            nonreparam_ll = torch.zeros(1)
            reparam_ll = torch.zeros(1)
            for name, site in tree.trace.nodes.items():
                if site["type"] == "sample":
                    # This will be zero if it's reparameterizable:
                    f = site["score_parts"].score_function
                    if isinstance(f, torch.Tensor):
                        f = f.sum()
                    nonreparam_ll = nonreparam_ll + f
                    
                    f = site["score_parts"].entropy_term
                    if isinstance(f, torch.Tensor):
                        f = f.sum()
                    reparam_ll = reparam_ll + f

            divergence = ll_against_sampled_kde - ll_against_obs_kde
            divergences.append(divergence.detach())
            # TODO(gizatt) Variance reduction
            for node in tree:
                node.tf = node.tf.detach()
            #proxy_loss = proxy_loss + (divergence.detach() * nonreparam_ll * 0 + divergence)
            if self.moving_average_of_divergences is None:
                proxy_loss = proxy_loss + (divergence.detach() * tree.score())
            else:
                proxy_loss = proxy_loss + ((divergence.detach() - self.moving_average_of_divergences[k]) * tree.score())
        if self.moving_average_of_divergences is None:
            self.moving_average_of_divergences = torch.stack(divergences)
        else:
            self.moving_average_of_divergences = self.moving_average_of_divergences*0.8 + torch.stack(divergences)*0.2

        proxy_loss = proxy_loss / len(sampled_trees)
        return proxy_loss, torch.tensor(ll_against_obs_kde).mean().detach()

    '''
        MMD: option "mean_mmd_poses"
        As noted in the MetaSim2 paper, this generates one score for the entire sampled
        dataset, so it can't easily be used to decide things like the relative weight
        of different samples. I've also seen that it struggles mightily to converge
        on the 3-mode GMM, even for the means, which I would hope would converge well.
    '''
    def _prep_for_mean_mmd_poses(self):
        # Precalculate pose set dictionaries for observed nodes
        self.observed_population_by_type = {}
        self.mmds_by_type_history = []
        for observed_nodes in self.observed_node_sets:
            for observed_node in observed_nodes:
                key = type(observed_node).__name__
                if key not in self.observed_population_by_type.keys():
                    self.observed_population_by_type[key] = []
                self.observed_population_by_type[key].append(observed_node.tf)
        for key, value in self.observed_population_by_type.items():
            self.observed_population_by_type[key] = torch.stack(value)
        # Set up variance reduction baseline
        self.moving_average_of_loss = None
    def _calc_average_mmd_for_observed_poses(self, sampled_node_sets):
        ''' Returns a dictionary of {node_type_name: mmd}. If an observed type is never
        sampled, or a sampled type is never observed, then an arbitrary penalty distance of
        +10 will be returned in its place. '''
        # Collect pose sets by type
        sampled_population_by_type = {key: [] for key in self.observed_population_by_type.keys()}
        for sampled_nodes in sampled_node_sets:
            for sampled_node in sampled_nodes:
                key = type(sampled_node).__name__
                if key not in sampled_population_by_type:
                    logging.warning("Sampled object not in observations: %s" % key)
                    sampled_population_by_type[key] = []
                sampled_population_by_type[key].append(sampled_node.tf)
        for key, value in sampled_population_by_type.items():
            sampled_population_by_type[key] = torch.stack(value)

        # Compute the actual MMDs
        mmds_by_type = {}
        mmd_alphas = [0.01, 0.05, 0.1, 0.5, 1.]
        no_match_mmd = 10.
        for key in self.observed_population_by_type.keys():
            observed_pop = self.observed_population_by_type[key]
            if key not in sampled_population_by_type.keys():
                mmds_by_type[key] = no_match_mmd
                logging.warning("Applying no-match MMD for unsampled type %s" % key)
                continue

            sampled_pop = sampled_population_by_type[key]
            if observed_pop.shape[0] > 0 and sampled_pop.shape[0] > 0:
                mmds_by_type[key] = calculate_mmd(observed_pop, sampled_pop, alphas=mmd_alphas, use_se3_metric=True)
            else:
                logging.warning("%d observed, %d sampled, can't compare." % (observed_pop.shape[0], sampled_pop.shape[0]))
                mmds_by_type[key] = no_match_mmd
        return mmds_by_type
    def _calc_loss_mean_mmd_poses(self, sampled_trees):
        sampled_node_sets = [tree.get_observed_nodes() for tree in sampled_trees]
        mmds_by_type = self._calc_average_mmd_for_observed_poses(sampled_node_sets)
        self.mmds_by_type_history.append({key: value.detach() for key, value in mmds_by_type.items()})
        loss = torch.mean(torch.cat([x.view(1) for x in mmds_by_type.values()]))
        # Perform partially-reparameterized REINFORCE update.
        nonreparam_ll = torch.zeros(1)
        for tree in sampled_trees:
            tree.trace.compute_score_parts()
            for name, site in tree.trace.nodes.items():
                if site["type"] == "sample":
                    # This will be zero if it's reparameterizable:
                    nonreparam_ll = nonreparam_ll + site["score_parts"].score_function
        mean_nonreparam_ll = nonreparam_ll / len(sampled_trees)

        # TODO: Variance reduction baseline?
        if self.moving_average_of_loss is None:
            proxy_loss = mean_nonreparam_ll * loss.detach() + loss
            self.moving_average_of_loss = loss.detach()
        else:
            proxy_loss = mean_nonreparam_ll * (loss.detach() - self.moving_average_of_loss) + loss
            self.moving_average_of_loss = self.moving_average_of_loss * 0.9 + loss.detach() * 0.1
        return proxy_loss, loss