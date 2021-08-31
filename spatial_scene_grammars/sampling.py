from copy import deepcopy

import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, HMC
from pyro.contrib.autoname import scope

from .random_walk_kernel import RandomWalkKernel
from .rules import *
from .parsing import optimize_scene_tree_with_nlp

from pytorch3d.transforms.rotation_conversions import (
    euler_angles_to_matrix, axis_angle_to_matrix
)

def do_fixed_structure_mcmc(grammar, scene_tree, num_samples=500, verbose=False):
    ''' Given a scene tree, resample its continuous variables
    (i.e. the node poses) while keeping the root and observed
    node poses fixed. Returns a population of trees sampled
    from the joint distribution over node poses given the
    fixed structure. '''

    # Strategy sketch:
    # - Initialize a random walk at the current scene tree.
    # - Repeatedly:
    #     1) Apply a random perturbation to the non-observed, non-root
    #        node poses.
    #     2) Use an NLP to project the perturbed tree configuration to
    #        the nearest feasible tree configuration (i.e. satisfying
    #        uniform and joint angle bounds).
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
        current_root = current_tree.get_root()
        for current_node, new_node in zip(current_tree.nodes, new_tree.nodes):
            if current_node.observed or current_node is current_root:
                continue
            ## The exact perturbation we'll apply depends on the relationship to the parent.
            current_parent = current_tree.get_parent(current_node)
            rule = current_tree.get_rule_for_child(current_parent, current_node)
            xyz_rule, rotation_rule = rule.xyz_rule, rule.rotation_rule
            if isinstance(xyz_rule, WorldBBoxRule):
                # Perturb in axes that are not equality constrained
                perturb = dist.Normal(torch.zeros(3), torch.ones(3)*0.1).sample()
                perturb[xyz_rule.xyz_dist.delta_mask] = 0.
                new_node.translation = current_node.translation + perturb
            elif isinstance(xyz_rule, AxisAlignedBBoxRule):
                perturb = dist.Normal(torch.zeros(3), torch.ones(3)*0.1).sample()
                perturb[xyz_rule.xyz_offset_dist.delta_mask] = 0.
                new_node.translation = current_node.translation + perturb
            else:
                raise NotImplementedError("%s" % xyz_rule)

            if isinstance(rotation_rule, UnconstrainedRotationRule):
                # Apply small random rotation 
                random_small_rotation = euler_angles_to_matrix(
                    dist.Normal(torch.zeros(3), torch.ones(3)*0.1).sample().unsqueeze(0),
                    convention="ZYX"
                )[0, ...]
                new_node.rotation = torch.matmul(current_node.rotation, random_small_rotation)
            elif isinstance(rotation_rule, UniformBoundedRevoluteJointRule):
                # Apply small rotation around axis, unless the rotation is fully constrained
                if not np.isclose(rotation_rule.lb, rotation_rule.ub):
                    random_angle = dist.Normal(torch.zeros(0), torch.ones(1)*0.1).sample()
                    orig_angle, orig_axis = rotation_rule._recover_relative_angle_axis(current_parent, current_node)
                    # Add angle to orig angle, and rotate around the joint's actual axis to get
                    # the new rotation offset.
                    new_angle_axis = rotation_rule.axis * (orig_angle + random_angle)
                    new_R_offset = axis_angle_to_matrix(new_angle_axis.unsqueeze(0))[0, ...]
                    new_node.rotation = torch.matmul(current_parent.rotation, new_R_offset)

        # Now project the tree to the closest feasible tree to that config.
        projection_results = optimize_scene_tree_with_nlp(new_tree, objective="projection", verbose=False)
        if projection_results.optim_result.is_success():
            new_tree = projection_results.refined_tree
            try:
                new_score = new_tree.score()
            except ValueError as e:
                print("Unexpected ValueError: ", e)
        else:
            new_score = -torch.tensor(np.inf)

        reject = True
        if torch.isfinite(new_score):
            # MH acceptance ratio: just score_new / score_old, since proposal is symmetric
            alpha = min(1, torch.exp(new_score - old_score))
            print("New score %f, old score %f, alpha %f" % (new_score, old_score, alpha))
            if dist.Uniform(0., 1.).sample() <= alpha:
                # Accepted. Update our temp pose variables to reflect new "good" state.
                n_accept += 1
                current_tree = new_tree
        if reject:
            pass
        sample_trees.append(deepcopy(current_tree))

        print("%d: Accept rate %f" % (step_k, n_accept / (step_k + 1)))
    return sample_trees

