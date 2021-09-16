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


class VariationalPosteriorTree(torch.nn.Module):
    ''' Torch Module whose parameters encode a variational
        posterior distribution over the poses of a scene tree with
        structure matching a supplied base scene tree, whose parameters
        are initialized to put a tight peak at the base tree.
        
        Distribution is a mean-field over the sample sites in the forward
        sampling process for the continuous sites in this tree.'''
    def __init__(self, base_tree, grammar):
        super().__init__()
        self.base_tree = base_tree
        # Keyed by sample site name
        self.mean_params = torch.nn.ModuleDict()
        self.var_params = torch.nn.ModuleDict()
        self.site_constraints = {}
        for parent in self.base_tree.nodes:
            children, rules = base_tree.get_children_and_rules(parent)
            for child, rule in zip(children, rules):
                for key, site_value in rule.get_site_values(parent, child).items():
                    site_name = "%s/%s/%s" % (parent.name, child.name, key)
                    self.site_constraints[site_name] = site_value.fn.support
                    self.mean_params[site_name] = ConstrainedParameter(
                        site_value.value, constraints.real
                    )
                    self.var_params[site_name] = ConstrainedParameter(
                        torch.ones(site_value.value.shape) * 0.01,
                        constraints.positive
                    )
                    
    def forward_model(self, grammar):
        # Resample the continuous structure of the tree into
        # a book-keeping copy.
        scene_tree = deepcopy(self.base_tree)
        # Hook grammar back into tree
        grammar.update_tree_grammar_parameters(scene_tree)
        node_queue = [scene_tree.get_root()]
        while len(node_queue) > 0:
            parent = node_queue.pop(0)
            children, rules = scene_tree.get_children_and_rules(parent)
            for child, rule in zip(children, rules):
                # TODO: node score children too!
                with scope(prefix=parent.name):
                    rule.sample_child(parent, child)
                node_queue.append(child)

        # Implement observation constraints
        xyz_observed_variance = 1E-3
        rot_observed_variance = 1E-3
        for node, original_node in zip(scene_tree.nodes, self.base_tree.nodes):
            if node.observed:
                xyz_observed_dist = dist.Normal(original_node.translation, xyz_observed_variance)
                rot_observed_dist = dist.Normal(original_node.rotation, rot_observed_variance)
                pyro.sample("%s_xyz_observed" % node.name, xyz_observed_dist, obs=node.translation)
                pyro.sample("%s_rotation_observed" % node.name, rot_observed_dist, obs=node.rotation)
        # Implement joint axis constraints
        axis_alignment_variance = 1E-2
        for node in scene_tree.nodes:
            children, rules = scene_tree.get_children_and_rules(parent)
            for child, rule in zip(children, rules):
                if type(rule) == GaussianChordOffsetRule or type(rule) == UniformBoundedRevoluteJointRule:
                    # Both of these rule types require that parent/child rotation is
                    # about an axis.
                    axis_from_parent = torch.matmul(node.rotation, node.axis)
                    axis_from_child = torch.matmul(child.rotation, child.axis)
                    inner_product = (axis_from_parent*axis_from_child).sum()
                    pyro.sample("%s_axis_error_observed" % node.name,
                        dist.Normal(1., axis_alignment_variance),
                        obs=inner_product
                    )

    def evaluate_elbo(self, grammar, num_samples=5, verbose=0):
        total_elbo = torch.tensor([0.])
        for sample_k in range(num_samples):
            # Sample from variational posterior, and use that to condition
            # the sample sites in the forward model.
            total_q_ll = torch.tensor([0.])
            conditioning = {}
            for key in self.mean_params.keys():
                q_density = dist.Normal(self.mean_params[key](), self.var_params[key]())
                unconstrained_sample = q_density.rsample()
                # Map the unconstrained sample into the contrained parameter space
                conditioning[key] = transform_to(self.site_constraints[key])(unconstrained_sample)
                total_q_ll = total_q_ll + q_density.log_prob(unconstrained_sample).sum()
            
            with pyro.condition(data=conditioning):
                trace = pyro.poutine.trace(self.forward_model).get_trace(grammar)
                log_p = trace.log_prob_sum()
            
            for key in conditioning.keys():
                assert key in trace.nodes.keys()

            # Update ELBO
            # For now, no non-reparameterized sites in the variational
            # posterior, so we can calculate expectation as mean of
            # samples.
            total_elbo = total_elbo + (log_p - total_q_ll)
        total_elbo = total_elbo / num_samples
        return total_elbo


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
    '''
    def __init__(self, grammar, observed_node_sets):
        self.grammar = grammar
        self.observed_node_sets = observed_node_sets
                
    def get_map_trees(self, throw_on_failure=False, verbose=0, tqdm=None):
        ''' Get MAP parse for each tree in dataset using our current grammar params.
            If parsing fails for a tree, will return None for that tree and throw
            if requsted. '''
        refined_trees = []

        iterator = enumerate(self.observed_node_sets)
        if tqdm:
            iterator = tqdm(iterator, desc="Getting MAP parses", total=len(self.observed_node_sets))
            
        for k, observed_nodes in iterator:
            mip_results = infer_mle_tree_with_mip(
                self.grammar, observed_nodes, verbose=verbose>1, max_scene_extent_in_any_dir=10.
            )
            mip_optimized_tree = get_optimized_tree_from_mip_results(mip_results)
            if mip_optimized_tree is None:
                error_msg = "MIP optimization failed for observed set %d" % k
                if throw_on_failure:
                    raise RuntimeError(error_msg)
                else:
                    logging.warning(error_msg)
                    refined_trees.append(None)
                    continue
            
            refinement_results = optimize_scene_tree_with_nlp(self.grammar, mip_optimized_tree, verbose=verbose>1)
            refined_tree = refinement_results.refined_tree
            if refined_tree is None:
                error_msg = "Nonlinear refinement failed for observed set %d" % k
                if throw_on_failure:
                    raise RuntimeError(error_msg)
                else:
                    logging.warning(error_msg)
                    refined_trees.append(None)
                    continue
            
            refined_trees.append(refined_tree)

        return refined_trees
            

    def do_iterated_vi_fitting(self, major_iterations=1, minor_iterations=50, throw_on_map_failure=False,
                               num_elbo_samples=3, verbose=0, tqdm=None):
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
        
        major_iterator = range(major_iterations)
        if tqdm:
            major_iterator = tqdm(major_iterator, desc="Major iteration", total=major_iterations)
        for major_iteration in major_iterator:
            if tqdm:
                major_iterator.set_description("Major %03d: calculating MAP trees" % (major_iteration))
            refined_trees = self.get_map_trees(throw_on_failure=throw_on_map_failure, verbose=verbose, tqdm=tqdm)
            # Initialize variational posterior at the MAP tree.
            variational_posteriors = torch.nn.ModuleList(
                [VariationalPosteriorTree(tree, self.grammar) for tree in refined_trees]
            )
            
            if tqdm:
                major_iterator.set_description("Major %03d: doing SVI iters" % (major_iteration))

            params = [*variational_posteriors.parameters(), *self.grammar.parameters()]
            optimizer = torch.optim.Adam(params, lr=0.1)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

            grammar_history = []
            variational_posterior_history = []
            
            minor_iterator = range(minor_iterations)
            if tqdm:
                minor_iterator = tqdm(minor_iterator, desc="Minor iteration", total=minor_iterations)
            for minor_iteration in minor_iterator:
                optimizer.zero_grad()
                
                grammar_history.append(deepcopy(self.grammar.state_dict()))
                variational_posterior_history.append(deepcopy(variational_posteriors.state_dict()))
                
                # Evaluate ELBO and do gradient updates.
                total_elbo = torch.mean(torch.stack([
                    posterior.evaluate_elbo(self.grammar, num_elbo_samples, verbose=verbose)
                    for posterior in variational_posteriors
                ]))
                
                self.elbo_history.append(total_elbo.detach())
                if tqdm:
                    minor_iterator.set_description("Minor %05d: ELBO %f" % (minor_iteration, total_elbo.item()))
                else:
                    logging.info("%03d/%05d: ELBO %f" % (major_iteration, minor_iteration, total_elbo.item()))
                
                if minor_iteration < minor_iterations - 1:
                    (-total_elbo).backward()
                    optimizer.step()
                    scheduler.step()
                
            self.grammar_major_iters.append(grammar_history)
            self.posterior_major_iters.append((variational_posteriors, variational_posterior_history))
       
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

