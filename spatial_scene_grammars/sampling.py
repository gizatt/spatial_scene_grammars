from copy import deepcopy
import logging
import time

import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, HMC
from pyro.contrib.autoname import scope

from pydrake.all import (
    ConnectMeshcatVisualizer
)
from .random_walk_kernel import RandomWalkKernel
from .rules import *
from .constraints import PoseConstraint, StructureConstraint
from .parsing import optimize_scene_tree_with_nlp
from .torch_utils import interp_translation, interp_rotation
from .drake_interop import (
    compile_scene_tree_to_mbp_and_sg,
    build_nonpenetration_constraint,
    resolve_sg_proximity_id_to_mbp_id
)
from .visualization import draw_scene_tree_structure_meshcat
from pytorch3d.transforms.rotation_conversions import (
    euler_angles_to_matrix, axis_angle_to_matrix
)

def split_constraints(constraints):
    topology_constraints = []
    continuous_constraints = []
    for c in constraints:
        if isinstance(c, StructureConstraint):
            topology_constraints.append(c)
        elif isinstance(c, PoseConstraint):
            continuous_constraints.append(c)
        else:
            raise ValueError("Bad constraint: not a topology or continuous variable constraint.")
    return topology_constraints, continuous_constraints

def eval_total_constraint_set_violation(scene_tree, constraints):
    # Returns the total violation across all constraints, summed
    # into one number.
    total_violation = 0.
    for constraint in constraints:
        max_violation, lower_violation, upper_violation = constraint.eval_violation(scene_tree)
        total_violation += torch.sum(torch.clamp(lower_violation, 0., np.inf))
        total_violation += torch.sum(torch.clamp(upper_violation, 0., np.inf))
    return total_violation

def rejection_sample_under_constraints(
        grammar, constraints, max_num_attempts, detach=False, verbose=0):
    # Try to rejection sample a reasonable configuration.
    # (Keep track of the "current best" in case we never accept a
    # configuration so we have a least-violating-tree to return.)
    best_tree = None
    best_violation = torch.tensor(np.inf)
    for k in range(max_num_attempts):
        tree = grammar.sample_tree(detach=detach)
        total_violation = eval_total_constraint_set_violation(tree, constraints)
        if total_violation <= torch.tensor(0.0):
            return tree, True
        if torch.isinf(best_violation) or total_violation <= best_violation:
            best_violation = total_violation
            best_tree = tree
    if verbose > 0:
        print("Sampled in %d iters" % k)
    return best_tree, False

class NonpenetrationConstraint(PoseConstraint):
    def __init__(self, allowed_penetration_margin=0.0):
        ''' penetration_margin > 0, specifies penetration amounts we'll allow. '''
        self.allowed_penetration_margin = allowed_penetration_margin
        self.cached_scene_tree = None
        super().__init__(lower_bound=torch.tensor(-np.inf),
                         upper_bound=torch.tensor(0.0))

    def eval(self, scene_tree):
        # Convert into MBP/SceneGraph and find penetrating point pairs
        # using that machinery. For each penetrating pair, get the
        # penetrating points and normal in body frame, and then recompute
        # the penalty score using (pb - pa)^n directly, applying transformation
        # into world frame using torch tensors from the scene tree to get
        # the continuous parameters (partially) autodiffable from torch.

        # This will NOT work for dynamically sized objects, since
        # only the transform gradients make it through.

        if self.cached_scene_tree is None or scene_tree is not self.cached_scene_tree:
            self.cached_scene_tree = scene_tree
            builder, mbp, sg, node_to_free_body_ids_map, body_id_to_node_map = \
                compile_scene_tree_to_mbp_and_sg(scene_tree)
            mbp.Finalize()
            diagram = builder.Build()
            diagram_context = diagram.CreateDefaultContext()
            mbp_context = diagram.GetMutableSubsystemContext(mbp, diagram_context)
            sg_context = diagram.GetMutableSubsystemContext(sg, diagram_context)
            self.mbp = mbp
            self.sg = sg
            self.mbp_context = mbp_context
            self.sg_context = sg_context
            self.node_to_free_body_ids_map = node_to_free_body_ids_map
            self.body_id_to_node_map = body_id_to_node_map
        else:
            # Update node tf for every node in the tree
            for node in scene_tree:
                for body_id in self.node_to_free_body_ids_map[node]:
                    self.mbp.SetFreeBodyPose(
                        self.mbp_context, self.mbp.get_body(body_id),
                        torch_tf_to_drake_tf(node.tf)
                    )

        # Get colliding point pairs and their normal info
        # using the SceneGraph.
        query_object = self.sg.get_query_output_port().Eval(self.sg_context)
        point_pairs = query_object.ComputePointPairPenetration()
        total_score = torch.tensor([0.])
        for point_pair in point_pairs:
            assert point_pair.depth > 0
            if point_pair.depth <= self.allowed_penetration_margin:
                continue
            model_id_A, body_id_A = resolve_sg_proximity_id_to_mbp_id(
                self.sg, self.mbp, point_pair.id_A)
            model_id_B, body_id_B = resolve_sg_proximity_id_to_mbp_id(
                self.sg, self.mbp, point_pair.id_B)

            # Transform the points on the objects and the normal vector
            # into the object frames
            tf_WA = self.mbp.CalcRelativeTransform(
                self.mbp_context, frame_A=self.mbp.world_frame(),
                frame_B=self.mbp.get_body(body_id_A).body_frame())
            tf_WB = self.mbp.CalcRelativeTransform(
                self.mbp_context, frame_A=self.mbp.world_frame(),
                frame_B=self.mbp.get_body(body_id_B).body_frame())
            p_ACa = torch.tensor(tf_WA.inverse().multiply(point_pair.p_WCa))
            p_BCb = torch.tensor(tf_WB.inverse().multiply(point_pair.p_WCb))
            nhat_BA_B = torch.tensor(tf_WB.inverse().rotation().multiply(point_pair.nhat_BA_W))
            
            # Retrieve their corresponding node TFs and compute error term
            tf_WA_torch = self.body_id_to_node_map[body_id_A].tf
            p_WCa_torch = torch.matmul(tf_WA_torch[:3, :3], p_ACa) + tf_WA_torch[:3, 3]
            tf_WB_torch = self.body_id_to_node_map[body_id_B].tf
            p_WCb_torch = torch.matmul(tf_WB_torch[:3, :3], p_BCb) + tf_WB_torch[:3, 3]
            nhat_BA_W_torch = torch.matmul(tf_WB_torch[:3, :3], nhat_BA_B)
            depth = torch.sum((p_WCb_torch - p_WCa_torch) * nhat_BA_W_torch)

            # Sanity-check that we did things right.
            assert np.abs(point_pair.depth - depth.item()) < 1E-6, "%s vs %s" % (point_pair.depth, depth.item())
            total_score += depth*10.0 # TODO Why the scaling? It's tied to my factor definition
                                      # for evaluation constraints with HMC -- currently errors
                                      # on the ~0.01 scale get reasonable probability. This
                                      # makes sure contact is actually really tightly handled.

        return total_score

def is_discrete_distribution(fn):
    # Returns whether the distribution type ("fn" from
    # a Pyro trace) is a known discrete distribution type.
    # This is used to distinguish discrete and continuous choices
    # made in a trace.
    return isinstance(fn, (dist.Categorical, dist.Geometric, dist.Bernoulli))

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
                if type(xyz_rule) is WorldFrameBBoxRule:
                    # Perturb in axes that are not equality constrained
                    scale = xyz_rule.ub - xyz_rule.lb
                    perturb = dist.Normal(torch.zeros(3), scale*translation_variance).sample()
                    perturb[xyz_rule.xyz_dist.delta_mask] = 0.
                    new_child.translation = current_child.translation + perturb
                elif type(xyz_rule) is WorldFrameBBoxOffsetRule:
                    # Perturb in axes that are not equality constrained
                    scale = xyz_rule.ub - xyz_rule.lb
                    perturb = dist.Normal(torch.zeros(3), scale*translation_variance).sample()
                    perturb[xyz_rule.xyz_dist.delta_mask] = 0.
                    if perturb_in_config_space:
                        current_offset = current_child.translation - current_parent.translation
                        new_child.translation = new_parent.translation + current_offset + perturb
                    else:
                        new_child.translation = current_child.translation + perturb
                elif type(xyz_rule) is WorldFrameGaussianOffsetRule:
                    # Perturb all axes
                    scale = xyz_rule.variance
                    perturb = dist.Normal(torch.zeros(3), scale*translation_variance).sample()
                    if perturb_in_config_space:
                        current_offset = current_child.translation - current_parent.translation
                        new_child.translation = new_parent.translation + current_offset + perturb
                    else:
                        new_child.translation = current_child.translation + perturb
                elif type(xyz_rule) is SamePositionRule:
                    pass
                else:
                    raise NotImplementedError("%s" % xyz_rule)

                if type(rotation_rule) is UnconstrainedRotationRule:
                    # Apply small random rotation 
                    random_small_rotation = euler_angles_to_matrix(
                        dist.Normal(torch.zeros(3), torch.ones(3)*rotation_variance).sample().unsqueeze(0),
                        convention="ZYX"
                    )[0, ...]
                    new_child.rotation = torch.matmul(current_child.rotation, random_small_rotation)
                elif type(rotation_rule) is SameRotationRule:
                    pass
                elif type(rotation_rule) is UniformBoundedRevoluteJointRule:
                    # Apply small rotation around axis, unless the rotation is fully constrained
                    if not np.isclose(rotation_rule.lb, rotation_rule.ub):
                        scale = rotation_rule.ub - rotation_rule.lb
                        random_angle = dist.Normal(torch.zeros(1), scale*translation_variance).sample()
                        orig_angle, orig_axis, in_range = recover_relative_angle_axis(current_parent, current_child, target_axis=rotation_rule.axis)
                        assert in_range
                        # Add angle to orig angle, and rotate around the joint's actual axis to get
                        # the new rotation offset.
                        new_angle_axis = rotation_rule.axis * (orig_angle + random_angle)
                        new_R_offset = axis_angle_to_matrix(new_angle_axis.unsqueeze(0))[0, ...]
                        if perturb_in_config_space:
                            new_child.rotation = torch.matmul(new_parent.rotation, new_R_offset)
                        else:
                            new_child.rotation = torch.matmul(current_child.rotation, new_R_offset)
                elif type(rotation_rule) is GaussianChordOffsetRule:
                    # Apply small rotation around axis
                    scale = rotation_rule.concentration
                    random_angle = dist.Normal(torch.zeros(1), scale*translation_variance).sample()
                    orig_angle, orig_axis, in_range = recover_relative_angle_axis(current_parent, current_child, target_axis=rotation_rule.axis)
                    assert in_range
                    # Add angle to orig angle, and rotate around the joint's actual axis to get
                    # the new rotation offset.
                    new_angle_axis = rotation_rule.axis * (orig_angle + random_angle)
                    new_R_offset = axis_angle_to_matrix(new_angle_axis.unsqueeze(0))[0, ...]
                    if perturb_in_config_space:
                        new_child.rotation = torch.matmul(new_parent.rotation, new_R_offset)
                    else:
                        new_child.rotation = torch.matmul(current_child.rotation, new_R_offset)
                else:
                    raise NotImplementedError("%s" % xyz_rule)

            # Add children to the node queue.
            for current_child, new_child in zip(current_children, new_children):
                node_queue.append((current_child, new_child))

        # Now project the tree to the closest feasible tree to that config.
        projection_results = optimize_scene_tree_with_nlp(grammar, new_tree, initial_guess_tree=current_tree, objective="projection", verbose=verbose>1)
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

            projection_results = optimize_scene_tree_with_nlp(grammar, new_tree, initial_guess_tree=current_tree, objective="projection", verbose=verbose>1)
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


def do_fixed_structure_hmc_with_constraint_penalties(
        grammar, original_tree, num_samples=100, subsample_step=5, verbose=0, kernel_type="NUTS",
        fix_observeds=False, with_nonpenetration=False, with_static_stability=False, 
        zmq_url=None, prefix="hmc_sample", constraints=[], structure_vis_kwargs={}, **kwargs):
    ''' Given a scene tree, resample its continuous variables
    (i.e. the node poses) while keeping the root and observed
    node poses fixed, and trying to keep the constraints implied
    by the tree and grammar satisfied.. Returns a population of trees sampled
    from the joint distribution over node poses given the fixed structure.

    Verbose = 0: Print nothing
    Verbose = 1: Print updates about accept rate and steps
    Verbose = 2: Print NLP output and scoring info
    '''

    # Strategy sketch:
    # - Initialize at the current scene tree, asserting that it's a feasible configuration.
    # - Form a probabilistic model that samples all of the node poses,
    #   and uses observe() statements to implement constraint factors as tightly peaked
    #   energy terms.
    # - Use Pyro HMC to sample from the model.

    # Make a bookkeeping copy of the tree
    scene_tree = deepcopy(original_tree) 
    
    # Do steps of random-walk MCMC on those variables.
    initial_score = scene_tree.score()
    assert torch.isfinite(initial_score), "Bad initialization for MCMC."

    if with_nonpenetration:
        constraints.append(NonpenetrationConstraint(0.00))

    # Form probabilistic model
    root = scene_tree.get_root()
    def model():
        # Resample the continuous structure of the tree.
        node_queue = [root]
        while len(node_queue) > 0:
            parent = node_queue.pop(0)
            children, rules = scene_tree.get_children_and_rules(parent)
            for child, rule in zip(children, rules):
                with scope(prefix=parent.name):
                    rule.sample_child(parent, child)
                node_queue.append(child)

        # Implement observation constraints
        if fix_observeds:
            xyz_observed_variance = 1E-2
            rot_observed_variance = 1E-2
            for node, original_node in zip(scene_tree.nodes, original_tree.nodes):
                if node.observed:
                    xyz_observed_dist = dist.Normal(original_node.translation, xyz_observed_variance)
                    rot_observed_dist = dist.Normal(original_node.rotation, rot_observed_variance)
                    pyro.sample("%s_xyz_observed" % node.name, xyz_observed_dist, obs=node.translation)
                    pyro.sample("%s_rotation_observed" % node.name, rot_observed_dist, obs=node.rotation)

        for k, constraint in enumerate(constraints):
            clamped_error_distribution = dist.Normal(0., 0.001)
            violation, _, _ = constraint.eval_violation(scene_tree)
            positive_violations = torch.clamp(violation, 0., np.inf)
            pyro.sample("%s_%d_err" % (type(constraint).__name__, k), clamped_error_distribution, obs=positive_violations)


    # Ugh, I shouldn't need to manually reproduce the site names here.
    # Can I rearrange how traces get assembled to extract these?
    initial_values = {}
    for parent in original_tree.nodes:
        children, rules = original_tree.get_children_and_rules(parent)
        for child, rule in zip(children, rules):
            for key, site_value in rule.get_site_values(parent, child).items():
                initial_values["%s/%s/%s" % (parent.name, child.name, key)] = site_value.value
    trace = pyro.poutine.trace(model).get_trace()
    for key in initial_values.keys():
        if key not in trace.nodes.keys():
            print("Trace keys: ", trace.nodes.keys())
            print("Initial values keys: ", initial_values.keys())
            raise ValueError("%s not in trace keys" % key)
    
    print("Initial trace log prob: ", trace.log_prob_sum())
    # If I let MCMC auto-tune its step size, it seems to do well,
    # but sometimes seems to get lost, and then gets stuck with big step size and
    # zero acceptances.
    if kernel_type == "NUTS":
        kernel = NUTS(model,
            init_strategy=pyro.infer.autoguide.init_to_value(values=initial_values),
            **kwargs
        )
    elif kernel_type is "HMC":
        kernel = HMC(model,
            init_strategy=pyro.infer.autoguide.init_to_value(values=initial_values),
            **kwargs
        )
    else:
        raise NotImplementedError(kernel_type)
        
    # Get MBP for viz
    if zmq_url is not None:
        builder, mbp, sg, node_to_free_body_ids_map, body_id_to_node_map = compile_scene_tree_to_mbp_and_sg(scene_tree)
        mbp.Finalize()
        visualizer = ConnectMeshcatVisualizer(builder, sg,
            zmq_url=zmq_url, prefix=prefix)
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()
        mbp_context = diagram.GetMutableSubsystemContext(mbp, diagram_context)
        vis_context = diagram.GetMutableSubsystemContext(visualizer, diagram_context)
        visualizer.load()
        def hook_fn(kernel, samples, stage, i):
            # Set MBP context to 
            for node, body_ids in node_to_free_body_ids_map.items():
                for body_id in body_ids:
                    mbp.SetFreeBodyPose(mbp_context, mbp.get_body(body_id), torch_tf_to_drake_tf(node.tf))
            diagram.Publish(diagram_context)
            draw_scene_tree_structure_meshcat(scene_tree, zmq_url=zmq_url,  prefix=prefix + "/structure", **structure_vis_kwargs)
            time.sleep(0.1)
    else:
        hook_fn = None

    mcmc = MCMC(
        kernel,
        num_samples=num_samples,
        warmup_steps=min(int(num_samples/2), 50),
        num_chains=1,
        disable_progbar=(verbose==-1),
        hook_fn=hook_fn
    )
    mcmc.run()
    if verbose > 1:
        mcmc.summary(prob=0.5)

    samples = mcmc.get_samples()
    sampled_trees = []
    for k in range(0, num_samples, subsample_step):
        condition = {key: value[k, ...] for key, value in samples.items()}
        with pyro.condition(data=condition):
            model()
        sampled_trees.append(deepcopy(scene_tree))

    return sampled_trees

