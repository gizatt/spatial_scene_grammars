from copy import deepcopy

import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, HMC
from pyro.contrib.autoname import scope

from .random_walk_kernel import RandomWalkKernel
from .rules import *

from pytorch3d.transforms.rotation_conversions import (
    euler_angles_to_matrix, axis_angle_to_matrix
)

def do_fixed_structure_mcmc(grammar, scene_tree, num_samples=500, verbose=False):
    ''' Given a scene tree, resample its continuous variables
    (i.e. the node poses) while keeping the root and observed
    node poses fixed. Returns a population of trees sampled
    from the joint distribution over node poses given the
    fixed structure. '''

    # We can use the scene tree itself to generate a conditioning
    # map that fixes the discrete choices, but allows the continuous
    # ones to vary. Then we can use this model in Pyro to perform
    # MCMC, initialized at the continuous site values.

    # Better alt method:
    #  1) Build the list of actual choices we need to do sampling over (i.e. poses of
    #     intermediate nodes).
    #  2) Propose a jump on those choices.
    #  3) Accept/reject based on resulting tree prob.
    # Should be able to do that hella fast, and without torch.

    # TODO this is scrappy as hell

    # Make local tree copy that we'll be decorating.
    scene_tree = deepcopy(scene_tree) 
    node_to_pose_variables = {}

    root_node = scene_tree.get_root()
    for node in scene_tree.nodes:
        if node is not root_node and not node.observed:
            # This is a node whose pose we'll want to explore.
            node_to_pose_variables[node] = deepcopy((node.translation, node.rotation))

    # Do steps of random-walk MCMC on those variables.
    n_accept = 0
    old_score = scene_tree.score()
    assert torch.isfinite(old_score), "Bad initialization for MCMC."

    sample_trees = []
    for step_k in range(num_samples):
        for node, value in node_to_pose_variables.items():
            translation, rotation = value
            ## The exact perturbation we'll apply depends on the relationship to the parent.

            parent = scene_tree.get_parent(node)
            rule = scene_tree.get_rule_for_child(parent, node)
            xyz_rule, rotation_rule = rule.xyz_rule, rule.rotation_rule
            if isinstance(xyz_rule, WorldBBoxRule):
                # Perturb in axes that are not equality constrained
                perturb = dist.Normal(torch.zeros(3), torch.ones(3)*0.01).sample()
                perturb[xyz_rule.xyz_dist.delta_mask] = 0.
                node.translation = translation + perturb
            elif isinstance(xyz_rule, AxisAlignedBBoxRule):
                perturb = dist.Normal(torch.zeros(3), torch.ones(3)*0.01).sample()
                perturb[xyz_rule.xyz_offset_dist.delta_mask] = 0.
                node.translation = translation + perturb
            else:
                raise NotImplementedError("%s" % xyz_rule)

            if isinstance(rotation_rule, UnconstrainedRotationRule):
                # Apply small random rotation 
                random_small_rotation = euler_angles_to_matrix(
                    dist.Normal(torch.zeros(3), torch.ones(3)*0.01).sample().unsqueeze(0),
                    convention="ZYX"
                )[0, ...]
                node.rotation = torch.matmul(rotation, random_small_rotation)
            elif isinstance(rotation_rule, UniformBoundedRevoluteJointRule):
                # Apply small rotation around axis, unless the rotation is fully constrained
                if not np.isclose(rotation_rule.lb, rotation_rule.ub):
                    random_angle = dist.Normal(torch.zeros(0), torch.ones(1)*0.01).sample()
                    orig_angle, orig_axis = rotation_rule._recover_relative_angle_axis(parent, node)
                    # Add angle to orig angle, and rotate around the joint's actual axis to get
                    # the new rotation offset.
                    new_angle_axis = rotation_rule.axis * (orig_angle + random_angle)
                    new_R_offset = axis_angle_to_matrix(new_angle_axis.unsqueeze(0))[0, ...]
                    node.rotation = torch.matmul(parent.rotation, new_R_offset)
                    print("New rotation")
                    

        new_score = scene_tree.score()
        reject = True
        if torch.isfinite(new_score):
            # MH acceptance ratio: just score_new / score_old, since proposal is symmetric
            alpha = torch.min(1, new_score - old_score)
            if dist.Uniform(0., 1.).sample() <= alpha:
                # Accepted. Update our temp pose variables to reflect new "good" state.
                n_accept += 1
                for node in node_to_pose_variables.keys():
                    node_to_pose_variables[node] = deepcopy((node.translation, node.rotation))
        if reject:
            for node, value in node_to_pose_variables.items():
                node.translation, node.rotation = value
        
        sample_trees.append(deepcopy(scene_tree))

        print("%d: Accept rate %f" % (step_k, n_accept / num_samples))
    return sample_trees

