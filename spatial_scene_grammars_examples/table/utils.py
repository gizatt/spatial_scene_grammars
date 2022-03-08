

def sample_realistic_scene(seed=None):
    if seed is not None:
        torch.random.manual_seed(seed)
    topology_constraints, continuous_constraints = split_constraints(constraints)
    if len(topology_constraints) > 0:
        tree, success = rejection_sample_under_constraints(grammar, topology_constraints, 1000)
        if not success:
            logging.error("Couldn't rejection sample a feasible tree config.")
            return None
    else:
        tree = grammar.sample_tree(detach=True)
    samples = do_fixed_structure_hmc_with_constraint_penalties(
        grammar, tree, num_samples=25, subsample_step=5,
        with_nonpenetration=False, zmq_url=vis.window.zmq_url,
        constraints=continuous_constraints,
        kernel_type="NUTS", max_tree_depth=6, target_accept_prob=0.8, adapt_step_size=True
    )
    # Step through samples backwards in HMC process and pick out a tree that satisfies
    # the constraints.
    good_tree = None
    for candidate_tree in samples[::-1]:
        if eval_total_constraint_set_violation(candidate_tree, constraints) <= 0.:
            good_tree = candidate_tree
            break
    if good_tree == None:
        logging.error("No tree in samples satisfied constraints.")
        return None
    
    return project_tree_to_feasibility(good_tree, do_forward_sim=True, timestep=0.001, T=1.)