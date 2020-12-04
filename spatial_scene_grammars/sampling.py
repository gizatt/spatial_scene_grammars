import numpy as np
import networkx as nx
import torch
from .nodes import Node
from .factors import ExtraSceneFactor
from .tree import SceneTree

def _sample_backend_rejection(
        root_node_type, root_node_type_kwargs,
        constraints, max_num_attempts=100):
    # Try to rejection sample a reasonable configuration.
    # (Keep track of the "current best" in case we never accept a
    # configuration so we have *something* to return.)
    current_best_scene_tree = None
    current_best_violation = torch.tensor(np.inf)
    for k in range(max_num_attempts):
        print("Attempt %d" % k)
        scene_tree = SceneTree.forward_sample_from_root_type(
            root_node_type, **root_node_type_kwargs)
        total_violation = 0.
        for constraint in constraints:
            val = constraint.eval(scene_tree)
            # If the lower bound is -inf, go ahead and say no violation here.
            # TODO: This'll break for vector-valued constraints...
            if torch.isinf(constraint.lower_bound) and constraint.lower_bound < 0:
                lower_violation = torch.tensor(0.)
            else:
                lower_violation = torch.max(constraint.lower_bound - val)
            if torch.isinf(constraint.upper_bound) and constraint.upper_bound > 0:
                upper_violation = torch.tensor(0.)
            else:
                upper_violation = torch.max(val - constraint.upper_bound)
            max_violation = torch.max(lower_violation, upper_violation)
            total_violation += torch.max(max_violation, torch.tensor(0.0))
        if total_violation <= torch.tensor(0.0):
            return scene_tree, True
        if torch.isinf(current_best_violation) or total_violation <= current_best_violation:
            current_best_violation = total_violation
            current_best_scene_tree = scene_tree

    print("Current best tree has violation ", current_best_violation)
    return current_best_scene_tree, False


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
    else:
        raise ValueError("Backend type %s" % backend)

    return backend_handler(root_node_type, root_node_type_kwargs,
                           constraints, **backend_kwargs)
