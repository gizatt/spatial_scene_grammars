import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, HMC
from pyro.contrib.autoname import scope
from .random_walk_kernel import RandomWalkKernel

def do_fixed_structure_mcmc(grammar, scene_tree, observation_variance=0.01, num_samples=500, verbose=False):
    ''' Given a scene tree, resample its continuous variables
    (i.e. the node poses) while keeping the root and observed
    node poses fixed. Returns a population of trees sampled
    from the joint distribution over node poses given the
    fixed structure. '''

    # We can use the scene tree itself to generate a conditioning
    # map that fixes the discrete choices, but allows the continuous
    # ones to vary. Then we can use this model in Pyro to perform
    # MCMC, initialized at the continuous site values.

    # TODO(gizatt) NOT RIGHT, for a few reasons:
    #  1) Biggest: Need to constrain root + observed nodes.
    #     Root is easy; but observed nodes aren't. I need some form of surrogate
    #     model where the choice before the observed node is a function of
    #     the parent node and the observed node location. Or an energy-based thing.
    #  2) Smaller: Since this doesn't work as written, having pyro woven through the
    #     whole system is a little excessive, right?

    # Better alt method:
    #  1) Build the list of actual choices we need to do sampling over (i.e. poses of
    #     intermediate nodes).
    #  2) Propose a jump on those choices.
    #  3) Accept/reject based on resulting tree prob.
    # Should be able to do that hella fast, and without torch.

    trace = scene_tree.trace
    choice_map = {}
    initial_values = {}
    known_continuous_dists = (dist.Uniform, dist.Normal)
    known_discrete_dists = (dist.Categorical)
    for key, value in trace.nodes.items():
        if value["type"] == "sample":
            dist_fn = value["fn"]
            if isinstance(dist_fn, known_discrete_dists):
                # Variable to fix!
                choice_map[key] = value["value"]
            else:
                assert isinstance(dist_fn, known_continuous_dists), \
                    "Unknown distribution %s for site %f" % (dist_fn, key)
                initial_values[key] = value["value"]

    def mcmc_model():
        new_tree = pyro.condition(grammar.sample_tree, choice_map)()

        # Add additional observed sampling of each observed node's pose.
        obs_dist = dist.Normal(torch.zeros(1), torch.ones(1) * observation_variance)
        for node_k, node in enumerate(new_tree.nodes):
            if node.observed:
                with scope(prefix="%d" % node_k):
                    # Ugly; ought to use a dist over SO(3) instead.
                    pyro.sample("obs_rotation", obs_dist.expand(torch.Size((3, 3))), obs=node.rotation)
                    # This one's OK
                    pyro.sample("obs_translation", obs_dist.expand(torch.Size((3,))), obs=node.translation)

    # 3) Run HMC.
    mcmc_kernel = NUTS(
        mcmc_model,
        init_strategy=pyro.infer.autoguide.init_to_value(values=initial_values)
    )
    mcmc = MCMC(mcmc_kernel, num_samples=num_samples, warmup_steps=10, disable_progbar=False)
    mcmc.run()
    if verbose:
        mcmc.summary()
    return mcmc, choice_map