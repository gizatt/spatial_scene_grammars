from copy import deepcopy
import logging

import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, HMC
from pyro.contrib.autoname import scope

from .random_walk_kernel import RandomWalkKernel
from .rules import *
from .parsing import optimize_scene_tree_with_nlp
from .torch_utils import interp_translation, interp_rotation

from pytorch3d.transforms.rotation_conversions import (
    euler_angles_to_matrix, axis_angle_to_matrix
)

def do_fixed_structure_mcmc(grammar, scene_tree, num_samples=500,
                            perturb_in_config_space=False, verbose=0,
                            vis_callback=None,
                            translation_variance=0.1,
                            rotation_variance=0.1,
                            do_hit_and_run_postprocess=False):
    ''' Given a scene tree, resample its continuous variables
    (i.e. the node poses) while keeping the root and observed
    node poses fixed. Returns a population of trees sampled
    from the joint distribution over node poses given the
    fixed structure.

    Verbose = 0: Print nothing
    Verbose = 1: Print updates about accept rate and steps
    Verbose = 2: Print NLP output and scoring info
    '''

    # Strategy sketch:
    # - Initialize a random walk at the current scene tree.
    # - Repeatedly:
    #     1) Apply a random perturbation to the non-observed, non-root
    #        node poses. If perturb_in_config_space, then this perturbation
    #        is applied to the root node first, and then that change propagated
    #        down to its children. Otherwise, every node pose is perturbed
    #        randomly simultaneously.
    #     2) Use an NLP to project the perturbed tree configuration to
    #        the nearest feasible tree configuration (i.e. satisfyingq
    #        uniform and joint angle bounds).
    #     2b) If do_hit_and_run_postprocess, randomly sample a point uniformly
    #         between the projected config and the initial config, and project
    #         that to the nearest good config to get a proposed config.
    #     3) Use the Accept/reject with a Metroplis acceptance ratio
    #        (without proposal probabilities, right now).
    #          x_proj = projected pose
    #          x_init = initial pose
    #        Accept if u ~ [0, 1] <= min(1, p(x_proj)/p(x_init))
    #
    #           TODO: This isn't gonna have a perfect underlying distribution
    #            unless I account for the Hastings ratio. But given my use of
    #            nasty nonlinear projection, that seems rough.


    current_tree = deepcopy(scene_tree) 
    
    # Do steps of random-walk MCMC on those variables.
    n_accept = 0
    old_score = current_tree.score()
    assert torch.isfinite(old_score), "Bad initialization for MCMC."

    sample_trees = []
    for step_k in range(num_samples):
        new_tree = deepcopy(current_tree)
        # Update tree from root down.
        node_queue = [(current_tree.get_root(), new_tree.get_root())]
        while len(node_queue) > 0:
            current_parent, new_parent = node_queue.pop()
            current_children, current_rules = current_tree.get_children_and_rules(current_parent)
            new_children = new_tree.get_children(new_parent)
            ## Apply pertubrations to children according to their generating rules.
            for current_child, rule, new_child in zip(current_children, current_rules, new_children):
                if current_child.observed:
                    # No perturbation to observed nodes necessary.
                    continue
                ## The exact perturbation we'll apply
                # depends on the relationship to the parent.
                xyz_rule, rotation_rule = rule.xyz_rule, rule.rotation_rule
                if type(xyz_rule) is WorldBBoxRule:
                    # Perturb in axes that are not equality constrained
                    perturb = dist.Normal(torch.zeros(3), torch.ones(3)*translation_variance).sample()
                    perturb[xyz_rule.xyz_dist.delta_mask] = 0.
                    new_child.translation = current_child.translation + perturb
                elif type(xyz_rule) is AxisAlignedBBoxRule:
                    # Perturb in axes that are not equality constrained
                    perturb = dist.Normal(torch.zeros(3), torch.ones(3)*translation_variance).sample()
                    perturb[xyz_rule.xyz_dist.delta_mask] = 0.
                    if perturb_in_config_space:
                        current_offset = current_child.translation - current_parent.translation
                        new_child.translation = new_parent.translation + current_offset + perturb
                    else:
                        new_child.translation = current_child.translation + perturb
                else:
                    raise NotImplementedError("%s" % xyz_rule)

                if type(rotation_rule) is UnconstrainedRotationRule:
                    # Apply small random rotation 
                    random_small_rotation = euler_angles_to_matrix(
                        dist.Normal(torch.zeros(3), torch.ones(3)*rotation_variance).sample().unsqueeze(0),
                        convention="ZYX"
                    )[0, ...]
                    new_child.rotation = torch.matmul(current_child.rotation, random_small_rotation)
                elif type(rotation_rule) is UniformBoundedRevoluteJointRule:
                    # Apply small rotation around axis, unless the rotation is fully constrained
                    if not np.isclose(rotation_rule.lb, rotation_rule.ub):
                        random_angle = dist.Normal(torch.zeros(1), torch.ones(1)*translation_variance).sample()
                        orig_angle, orig_axis = rotation_rule._recover_relative_angle_axis(current_parent, current_child)
                        # Add angle to orig angle, and rotate around the joint's actual axis to get
                        # the new rotation offset.
                        new_angle_axis = rotation_rule.axis * (orig_angle + random_angle)
                        new_R_offset = axis_angle_to_matrix(new_angle_axis.unsqueeze(0))[0, ...]
                        if perturb_in_config_space:
                            new_child.rotation = torch.matmul(new_parent.rotation, new_R_offset)
                        else:
                            new_child.rotation = torch.matmul(current_child.rotation, new_R_offset)

            # Add children to the node queue.
            for current_child, new_child in zip(current_children, new_children):
                node_queue.append((current_child, new_child))

        # Now project the tree to the closest feasible tree to that config.
        projection_results = optimize_scene_tree_with_nlp(new_tree, initial_guess_tree=current_tree, objective="projection", verbose=verbose>1)
        if projection_results.optim_result.is_success():
            new_tree = projection_results.refined_tree
            try:
                new_score = new_tree.score(verbose=verbose>1)
            except ValueError as e:
                logging.warn("Unexpected ValueError: %s", e)
                new_score = -torch.tensor(np.inf)
        else:
            logging.warning("Post-random-step projection failed")
            new_score = -torch.tensor(np.inf)

        if torch.isfinite(new_score) and do_hit_and_run_postprocess:
            # Randomly sample point between this new tree and the
            # initial tree.
            interp_factor = dist.Uniform(torch.zeros(1), torch.ones(1)).sample()
            current_root = current_tree.get_root()
            for original_node, new_node in zip(current_tree.nodes, new_tree.nodes):
                new_node.translation = interp_translation(original_node.translation, new_node.translation, interp_factor)
                new_node.rotation = interp_rotation(original_node.rotation, new_node.rotation, interp_factor)

            projection_results = optimize_scene_tree_with_nlp(new_tree, initial_guess_tree=current_tree, objective="projection", verbose=verbose>1)
            if projection_results.optim_result.is_success():
                new_tree = projection_results.refined_tree
                try:
                    new_score = new_tree.score(verbose=verbose>1)
                except ValueError as e:
                    logging.warning("Unexpected ValueError after hit-and-run projection: %s", e)
                    new_score = -torch.tensor(np.inf)
            else:
                logging.warning("Post-hit-and-run projection failed")
                new_score = -torch.tensor(np.inf)

        reject = True
        if torch.isfinite(new_score):
            # MH acceptance ratio: just score_new / score_old, since proposal is symmetric
            alpha = min(1, torch.exp(new_score - old_score))
            if verbose:
                print("New score %f, old score %f, alpha %f" % (new_score, old_score, alpha))
            if dist.Uniform(0., 1.).sample() <= alpha:
                # Accepted. Update our temp pose variables to reflect new "good" state.
                n_accept += 1
                current_tree = new_tree
        sample_trees.append(deepcopy(current_tree))
        if vis_callback is not None:
            vis_callback(current_tree)
        if verbose:
            print("%d: Accept rate %f" % (step_k, n_accept / (step_k + 1)))
    return sample_trees

