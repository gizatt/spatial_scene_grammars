from copy import deepcopy
import numpy as np
import networkx as nx
import torch
import pyro
import pyro.distributions as dist
from .nodes import Node
from .rules import ProductionRule
from .factors import TopologyConstraint, ContinuousVariableConstraint
from .tree import SceneTree, get_tree_root


def eval_total_constraint_set_violation(scene_tree, constraints):
    # Returns the total violation across all constraints, summed
    # into one number.
    total_violation = 0.
    for constraint in constraints:
        max_violation, lower_violation, upper_violation = constraint.eval_violation(scene_tree)
        total_violation += torch.sum(torch.clamp(lower_violation, 0., np.inf))
        total_violation += torch.sum(torch.clamp(upper_violation, 0., np.inf))
    return total_violation


def eval_tree_log_prob(scene_tree, constraints, clamped_error_distribution):
    # Calculates the log prob of the tree under the constraint set.
    # Topology constraints get -infinity for being violated;
    # continuous constraint violations get penalized under the
    # clamped error distribution.
    # TODO: This should get formalized and rolled in to a constraint-factor
    # sort of framework?
    topology_constraints, continuous_constraints = split_constraints(constraints)
    for c in topology_constraints:
        max_violation, _, _ = c.eval_violation(scene_tree)
        if max_violation > 0:
            # Violated a discrete constraint, so we can't accept this tree,
            # so return an accept prob of zero.
            print("Rejecting jump due to discrete constraint violation.")
            return torch.tensor(-np.inf)
    total_constraint_log_prob = torch.tensor(0.)
    for c in continuous_constraints:
        # TODO: How do I actually score these factors properly? This isn't
        # clean or right...
        max_violation, _, _ = c.eval_violation(scene_tree)
        # Score the violation numbers themselves with an exponential that falls
        # off rapidly as the violation goes positive. Same code as in HMC.
        positive_violations = torch.clamp(max_violation, 0., np.inf)
        total_constraint_log_prob += clamped_error_distribution.log_prob(positive_violations).sum()
    return total_constraint_log_prob


def rejection_sample_under_constraints_with_trace(
        root_node_type, root_node_instantiation_dict,
        constraints, max_num_attempts):
    # Try to rejection sample a reasonable configuration.
    # (Keep track of the "current best" in case we never accept a
    # configuration so we have *something* to return.)
    current_best_scene_tree = None
    current_best_trace = None
    current_best_violation = torch.tensor(np.inf)
    for k in range(max_num_attempts):
        trace = pyro.poutine.trace(
            SceneTree.forward_sample_from_root_type).get_trace(
                root_node_type, root_node_instantiation_dict)
        scene_tree = trace.nodes["_RETURN"]["value"]
        total_violation = eval_total_constraint_set_violation(scene_tree, constraints)
        if total_violation <= torch.tensor(0.0):
            return scene_tree, trace, True
        if torch.isinf(current_best_violation) or total_violation <= current_best_violation:
            current_best_violation = total_violation
            current_best_scene_tree = scene_tree
            current_best_trace = trace
    return current_best_scene_tree, current_best_trace, False


def _sample_backend_rejection(
        root_node_type, root_node_instantiation_dict,
        constraints, max_num_attempts=100):
    scene_tree, _, success = rejection_sample_under_constraints_with_trace(
        root_node_type, root_node_instantiation_dict, constraints, max_num_attempts)
    return [scene_tree], success


def split_constraints(constraints):
    topology_constraints = []
    continuous_constraints = []
    for c in constraints:
        if isinstance(c, TopologyConstraint):
            topology_constraints.append(c)
        elif isinstance(c, ContinuousVariableConstraint):
            continuous_constraints.append(c)
        else:
            raise ValueError("Bad constraint: not a topology or continuous variable constraint.")
    return topology_constraints, continuous_constraints


def do_fixed_structure_mcmc(root_node_type, root_node_instantiation_dict, feasible_trace,
                           constraints, callback, num_samples, verbose=False,
                           kernel_constructor=pyro.infer.mcmc.RandomWalkKernel, **kwargs):
    # Kernel type should be one of ["RandomWalkKernel", "HMC", "NUTS"].
    # kwargs is passed to the mcmc kernel constructor.

    scene_tree = deepcopy(feasible_trace.nodes["_RETURN"]["value"])
    root = get_tree_root(scene_tree)
    topology_constraints, continuous_constraints = split_constraints(constraints)

    # 2) Create a sub-model for HMC by conditioning the original forward model on the
    # remaining variables, and adding additional factors to the sub-model implementing
    # the continuous variable constraints as Gibbs factors.
    def mcmc_model():
        # Run the *instantiations only* forward, top-down.
        scene_tree.resample_instantiations(root, root_node_instantiation_dict)
        
        clamped_error_distribution = dist.Normal(0., 0.01)
        for k, c in enumerate(continuous_constraints):
            # TODO: How do I actually score these factors properly? This isn't
            # clean or right...
            max_violation, _, _ = c.eval_violation(scene_tree)
            # Score the violation numbers themselves with an exponential that falls
            # off rapidly as the violation goes positive.
            positive_violations = torch.clamp(max_violation, 0., np.inf)
            #print("positive_violations: ", positive_violations)
            pyro.sample("dist_%d_err" % k, clamped_error_distribution, obs=positive_violations)
            
        if callback:
            callback(scene_tree)
        return scene_tree

    # 3) Run HMC.
    initial_values = {key: site["value"] for key, site in feasible_trace.nodes.items()
                      if site["type"] is "sample"}
    print("initial values: ", initial_values)
    init_params, potential_fn, transforms, _ = pyro.infer.mcmc.util.initialize_model(
        mcmc_model, model_args=(),
        init_strategy=pyro.infer.autoguide.init_to_value(values=initial_values))
    
    mcmc_kernel = kernel_constructor(potential_fn=potential_fn, transforms=transforms, **kwargs)
    mcmc = pyro.infer.mcmc.MCMC(mcmc_kernel, num_samples=num_samples, initial_params=init_params, disable_progbar=~verbose)
    mcmc.run()
    return mcmc, scene_tree

def _sample_backend_rejection_and_hmc(
        root_node_type, root_node_instantiation_dict,
        constraints, max_num_attempts=100, callback=None, num_samples=25):
    # Rejection sample to get a feasible configuration w.r.t. topology
    # constraints, and then perform HMC for the continuous constraints.
    # CRITICAL TODO: I don't think this is actually sampling from the
    # constrained distribution properly if it doesn't circle back after
    # the HMC and resample the tree structure in a blocked Gibbs sampler
    # sort of scheme. Though I'm not 100% sure of that... this strategy is
    # what prior work uses occasionally.

    # Start out sampling a tree with rejection sampling, but only
    # check the topology-related constraints.
    topology_constraints, continuous_constraints = split_constraints(constraints)
    scene_tree, orig_trace, topology_feasible = rejection_sample_under_constraints_with_trace(
        root_node_type, root_node_instantiation_dict,
        topology_constraints, max_num_attempts=max_num_attempts)
    if not topology_feasible:
        print("Couldn't achieve even just the topology constraints on their own via rejection sampling.")
        return scene_tree, False

    # Now do HMC on the continuous variables that are involved
    # in the constraints.
    mcmc, discrete_site_data = do_fixed_structure_mcmc(root_node_type, root_node_instantiation_dict, orig_trace, constraints,
        callback=callback, num_samples=num_samples,
        kernel_constructor=pyro.infer.mcmc.HMC,
        num_steps=1, step_size=0.1, target_accept_prob=0.5, adapt_step_size=True,
        adapt_mass_matrix=True
    )
    

    print("MCMC Summary: ",)
    mcmc.summary()


    samples = mcmc.get_samples()
    out_trees = []
    for k in range(num_samples):
        sample_data = {}
        for key in sample_data.keys():
            sample_data[key] = sample_data[key][k, ...]
        new_tree = deepcopy(scene_tree)
        pyro.poutine.condition(
            new_tree.resample_instantiations,
            sample_data)(get_tree_root(new_tree), root_node_instantiation_dict)
        out_trees.append(new_tree)
    return out_trees, True

def _sample_backend_metroplis_procedural_modeling(
        root_node_type, root_node_instantiation_dict,
        constraints, callback=None, max_num_attempts=100,
        num_samples=100, diffusion_rate=0.5, estimated_branching_factor=2):
    ''' Implements RJMCMC as described in Metroplis Procedural Modeling:
    for a number of steps, randomly interleaves diffusion moves
    (i.e. fixed-tree-structure MCMC steps) and jumps. Jumps randomly
    select a node and re-derive it and its subtree.

    Hopefully eventually supports parallel tempering and delayed
    rejection to get as-reasonable-as-possible mixing.'''

    # Generate an initial feasible tree that satisfies the topology
    # constraints via rejection sampling.

    topology_constraints, continuous_constraints = split_constraints(constraints)
    current_scene_tree, current_trace, topology_feasible = rejection_sample_under_constraints_with_trace(
        root_node_type, root_node_instantiation_dict,
        topology_constraints, max_num_attempts=max_num_attempts)
    if not topology_feasible:
        print("Couldn't achieve even just the topology constraints on their own via rejection sampling.")
        return current_scene_tree, False

    samples = [current_scene_tree]

    # Now do the primary MCMC steps as requested.
    for step_k in range(num_samples):
        if pyro.sample("step_%d_type" % step_k, dist.Bernoulli(diffusion_rate)):
            # Do diffusion step with constraints in mind
            mcmc, discrete_site_data = do_fixed_structure_mcmc(root_node_type, root_node_instantiation_dict, current_trace,
                constraints, callback=callback, num_samples=1,
                kernel_constructor=pyro.infer.mcmc.RandomWalkKernel,
                variance=0.1)
            if not mcmc._samples:
                print("MCMC didn't get any samples. Rejected this fixed-structure pass.")
            else:
                sample = mcmc.get_samples(1)
                sample_data = {}
                for key in sample.keys():
                    sample_data[key] = sample[key][0, ...]
                # Use resulting site data to regenerate our current tree and trace.
                current_trace = pyro.poutine.trace(
                    pyro.poutine.condition(
                        SceneTree.forward_sample_from_root_type,
                        {**sample_data, **discrete_site_data})
                    ).get_trace(root_node_type, root_node_instantiation_dict)
                current_scene_tree = current_trace.nodes["_RETURN"]["value"]
        else:
            # Do jump step.
            # First randomly select a node.
            def get_node_choices_and_probs(scene_tree):
                # Apply nonterminal selection bias towards the tree, following
                # sec 8.1 of MPM. For each node, figure out its depth relative to the
                # max depth of the supplied tree.
                node_depth = []
                for node in scene_tree:
                    depth = 0
                    parent = node
                    while True:
                        depth += 1
                        parent = scene_tree.get_node_parent_or_none(parent)
                        if not parent:
                            break
                        assert depth < 10000, "Excessive depth, probably malformed [cyclic] scene tree."
                    node_depth.append(depth)
                node_depth = torch.tensor(node_depth)
                node_selection_probs = torch.ones(len(scene_tree))*estimated_branching_factor
                node_selection_probs = torch.pow(node_selection_probs, torch.max(node_depth) - node_depth)
                node_selection_probs /= torch.sum(node_selection_probs)
                return node_choices, node_selection_probs
            node_choices, node_selection_probs = get_node_choices_and_probs(current_scene_tree)
            reseed_node_index = pyro.sample(
                "step_%d_reseed_node" % step_k,
                dist.Categorical(node_selection_probs)
            )
            reseed_node = node_choices[reseed_node_index.item()]

            # Regenerate tree, keeping everything outside of that subtree fixed
            # (via conditioning). Every random choice site name under in this subtree
            # will have the unique node name in it, so select out sites that
            # don't have that name to preserve. (Assumes the unique node name
            # doesn't appear in any variables outside of the subtree. This is
            # relatively safe; you'd have to try very hard to have it happen.
            # The deeper in the tree the node is, the longer its mostly-auto-generated
            # unique name is.)
            # TODO: I really ought to be able to just ask all of the
            # nodes the report all the variables that are associated with
            # themselves and not make this dangerous assumption. Part of my
            # continuing node-variable and naming woes...
            keep_data = {}
            for key, site in current_trace.nodes.items():
                if site["type"] == "sample" and reseed_node.name not in key:
                    keep_data[key] = site["value"]

            rerun_tree = pyro.poutine.condition(
                SceneTree.forward_sample_from_root_type,
                data=keep_data)
            new_trace = pyro.poutine.trace(rerun_tree).get_trace(root_node_type, root_node_instantiation_dict)
            new_tree = new_trace.nodes["_RETURN"]["value"]

            # Decide whether to accept the jump by calculating an MH accept
            # probability. This is the equation immediately before [Diamond] in
            # 7.2 in Talton et al 2011; I'm computing full tree probabilities because
            # we don't partition into P(s | parent(s)) * P(params(S)) like they do,
            # so their simplifications don't line up for us. Instead, we jump from
            # subtree T_v to subtree T_v' (both having nodes + attributes fully
            # populated out) splitting at split node v:
            #    q(T_v' | T_v) = q_T(v) * p(T_v')
            # where q_T(v) is the prob of selecting node v from tree T, and p(T_v')
            # is the total prob of subtree T_v'.
            def calc_accept_prob():
                # Find the forward and reverse selection probs. To find the node in the
                # new tree, we search by name, since it should be identical.
                forward_node_selection_prob = node_selection_probs[reseed_node_index]
                reverse_node_choices, reverse_node_selection_probs = get_node_choices_and_probs(new_tree)
                reverse_node_selection_prob = None
                for k, node in enumerate(reverse_node_choices):
                    if node.name == reseed_node.name:
                        reverse_node_selection_prob = reverse_node_selection_probs[k]
                assert reverse_node_selection_prob is not None

                # Score both trees under the constraint set (which we treat as the scoring function.)
                clamped_error_distribution = dist.Normal(0., 0.01)
                L_pre_tree = eval_tree_log_prob(current_scene_tree, constraints, clamped_error_distribution)
                assert torch.isfinite(L_pre_tree), current_scene_tree.nodes
                L_post_tree = eval_tree_log_prob(new_tree, constraints, clamped_error_distribution)
                if not torch.isfinite(L_post_tree):
                    # Short-circuit: proposed tree is completely infeasible.
                    print("Short-circuit rejecting jump because of infeasible proposed tree.")
                    return torch.tensor(0.)
    
                # And score the pre/post subtree forward sampling probabilities to compute
                # the proposal density correction term.
                def score_non_keep_sites(trace):
                    score = torch.tensor(0.0)
                    # Ensure log probs are available at all sites.
                    trace.compute_log_prob()
                    for key, site in trace.nodes.items():
                        if site["type"] == "sample" and key not in keep_data.keys():
                            score += site["log_prob_sum"]
                    return score

                pre_subtree_log_prob = score_non_keep_sites(current_trace)
                post_subtree_log_prob = score_non_keep_sites(new_trace)

                pre_full_tree_log_prob = current_trace.log_prob_sum()
                post_full_tree_log_prob = new_trace.log_prob_sum()

                # Finally assemble the accept prob.
                # TODO: I'm really not sure if this is right. Any part pertaining to tree probability winds
                # up canceling out precisely here. Which is good for me -- seems to lead to empirically reasonable
                # mixing and could make computing this faster. But it feels wrong / unjustified.
                num = L_post_tree + post_full_tree_log_prob + torch.log(reverse_node_selection_prob) + pre_subtree_log_prob
                denom = L_pre_tree + pre_full_tree_log_prob + torch.log(forward_node_selection_prob) + post_subtree_log_prob
                print("All terms:\n\tL post %f, full tree post %f, Reverse select %f, pre subtree %f" % (L_post_tree.item(), post_full_tree_log_prob.item(), torch.log(reverse_node_selection_prob).item(), pre_subtree_log_prob.item()))
                print("\t L pre: %f, full tree pre %f, Forward select %f, post subtree %f" % (L_pre_tree.item(), pre_full_tree_log_prob.item(), torch.log(forward_node_selection_prob).item(), post_subtree_log_prob.item()))
                print("\t %f / %f -> %f" % (torch.exp(num).item(), torch.exp(denom).item(), torch.exp(num - denom)))
                return torch.clamp(torch.exp(num - denom), max=1.0)

            a = calc_accept_prob()
            print("Accept prob for jump is %f..." % a.item())
            if pyro.sample("accept_step_%d" % step_k, dist.Uniform(0., 1.)) <= a:
                print("\tAccepting jump.")
                current_scene_tree = new_tree
                current_trace = new_trace
            else:
                print("\tRejecting jump.")
        samples.append(current_scene_tree)
        if callback is not None:
            callback(current_scene_tree)
    return samples, True
            

def sample_tree_from_root_type_with_constraints(
        root_node_type,
        root_node_instantiation_dict,
        constraints=[],
        backend="rejection",
        **backend_kwargs):
    '''
    Args:
        - root_node_type: A Node type.
        - root_args: A dictionary of arguments to initialize
        the root type.
        - constraints: A list of functions that map from
        a scene tree to constraint satisfaction info, which
        should all evaluate true for the tree to be accepted.
            (TODO: Generalize to factors. Need to decide if I can
            handle Gibbs distributions and that they're going to
            be unnormalize-able.)
            (TODO: maybe instead of generators, the constriant or
            factor can just return multiple values for the
            different sub-constraints? I need a way of tracking which
            constraints are influenced by which nodes, so I can track
            sparsity and independence.)
        - <extra args to control the sampling method>
    Returns: (scene_tree, constraints_satisfied)
        - A scene tree that is (approximately) drawn from
        the joint distribution induced by the scene grammar
        and the supplied factor set.
    '''
    assert issubclass(root_node_type, Node)
    assert isinstance(root_node_instantiation_dict, dict)
    assert isinstance(constraints, list)

    # Short-circuit if there are no factors -- forward sampling
    # is enough.
    if len(constraints) == 0:
        return SceneTree.forward_sample_from_root_type(
            root_node_type, root_node_instantiation_dict), True

    if backend == "rejection":
        backend_handler = _sample_backend_rejection
    elif backend == "rejection_then_hmc":
        backend_handler = _sample_backend_rejection_and_hmc
    elif backend == "metropolis_procedural_modeling":
        backend_handler = _sample_backend_metroplis_procedural_modeling
    else:
        raise ValueError("Backend type %s" % backend)

    return backend_handler(root_node_type, root_node_instantiation_dict,
                           constraints, **backend_kwargs)
