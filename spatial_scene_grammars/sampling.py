import numpy as np
import networkx as nx
import torch
import pyro
import pyro.distributions as dist
from .nodes import Node
from .rules import ProductionRule
from .factors import TopologyConstraint, ContinuousVariableConstraint
from .tree import SceneTree


def eval_total_constraint_set_violation(scene_tree, constraints):
    # Returns the total violation across all constraints, summed
    # into one number.
    total_violation = 0.
    for constraint in constraints:
        max_violation, lower_violation, upper_violation = constraint.eval_violation(scene_tree)
        total_violation += torch.sum(torch.clamp(lower_violation, 0., np.inf))
        total_violation += torch.sum(torch.clamp(upper_violation, 0., np.inf))
    return total_violation


def _sample_backend_rejection(
        root_node_type, root_node_type_kwargs,
        constraints, max_num_attempts=100):
    # Try to rejection sample a reasonable configuration.
    # (Keep track of the "current best" in case we never accept a
    # configuration so we have *something* to return.)
    current_best_scene_tree = None
    current_best_violation = torch.tensor(np.inf)
    for k in range(max_num_attempts):
        scene_tree = SceneTree.forward_sample_from_root_type(
            root_node_type, **root_node_type_kwargs)
        total_violation = eval_total_constraint_set_violation(scene_tree, constraints)
        if total_violation <= torch.tensor(0.0):
            return scene_tree, True
        if torch.isinf(current_best_violation) or total_violation <= current_best_violation:
            current_best_violation = total_violation
            current_best_scene_tree = scene_tree

    print("Current best tree has violation ", current_best_violation)
    return current_best_scene_tree, False


def _sample_backend_rejection_and_hmc(
        root_node_type, root_node_type_kwargs,
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
    topology_constraints = []
    continuous_constraints = []
    for c in constraints:
        if isinstance(c, TopologyConstraint):
            topology_constraints.append(c)
        elif isinstance(c, ContinuousVariableConstraint):
            continuous_constraints.append(c)
        else:
            raise ValueError("Bad constraint: not a topology or continuous variable constraint.")

    scene_tree = None
    current_best_violation = torch.tensor(np.inf)
    topology_feasible = False
    for k in range(max_num_attempts):
        orig_trace = pyro.poutine.trace(
            SceneTree.forward_sample_from_root_type).get_trace(
                root_node_type, **root_node_type_kwargs)
        scene_tree = orig_trace.nodes["_RETURN"]["value"]
        total_violation = eval_total_constraint_set_violation(scene_tree, topology_constraints)
        if total_violation <= torch.tensor(0.0):
            topology_feasible = True
            break
        if torch.isinf(current_best_violation) or total_violation <= current_best_violation:
            current_best_violation = total_violation

    if not topology_feasible:
        print("Couldn't achieve even just the topology constraints on their own via rejection sampling.")
        return scene_tree, False

    # Now do HMC on the continuous variables that are involved
    # in the constraints.

    # 1) Separate out a list of all of the site names of continuous variables that can affect
    # the continuous constraints.
    continuous_var_names = []
    for node in scene_tree:
        if isinstance(node, ProductionRule):
            continuous_var_names += node.get_local_variable_names()

    # 2) Create a sub-model for HMC by conditioning the original forward model on the
    # remaining variables, and adding additional factors to the sub-model implementing
    # the continuous variable constraints as Gibbs factors.
    discrete_var_names_and_vals = {}
    for key, value in orig_trace.nodes.items():
        if key not in ["_RETURN", "_INPUT"] and key not in continuous_var_names:
            discrete_var_names_and_vals[key] = value["value"]

    def hmc_model():
        # This handle runs the continuous model forward, conditioning all the same
        # discrete choices to be made.
        only_continuous_model = pyro.poutine.condition(
                SceneTree.forward_sample_from_root_type,
                data=discrete_var_names_and_vals)
        scene_tree = only_continuous_model(root_node_type, **root_node_type_kwargs)

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
    #init_params, potential_fn, transforms, _ = pyro.infer.mcmc.util.initialize_model(hmc_model, model_args=())
    hmc_kernel = pyro.infer.mcmc.NUTS(hmc_model, target_accept_prob=0.3, adapt_step_size=True)
    #hmc_kernel = pyro.infer.mcmc.HMC(hmc_model, num_steps=1, step_size=0.1, target_accept_prob=0.5, adapt_step_size=True, adapt_mass_matrix=True,
    #    init_strategy=pyro.infer.autoguide.initialization.init_to_sample)
    mcmc = pyro.infer.mcmc.MCMC(hmc_kernel, num_samples=num_samples)
    mcmc.run()

    print("MCMC Summary: ",)
    mcmc.summary()

    sample_data = mcmc.get_samples(1)
    for key in sample_data.keys():
        sample_data[key] = sample_data[key][0, ...]
    scene_tree = pyro.poutine.condition(hmc_model, sample_data)()
    return scene_tree, True

def sample_tree_from_root_type_with_constraints(
        root_node_type,
        root_node_type_kwargs,
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
    assert isinstance(root_node_type_kwargs, dict)
    assert isinstance(constraints, list)

    # Short-circuit if there are no factors -- forward sampling
    # is enough.
    if len(constraints) == 0:
        return SceneTree.forward_sample_from_root_type(
            root_node_type, **root_node_type_kwargs), True

    if backend == "rejection":
        backend_handler = _sample_backend_rejection
    elif backend == "rejection_then_hmc":
        backend_handler = _sample_backend_rejection_and_hmc
    else:
        raise ValueError("Backend type %s" % backend)

    return backend_handler(root_node_type, root_node_type_kwargs,
                           constraints, **backend_kwargs)
