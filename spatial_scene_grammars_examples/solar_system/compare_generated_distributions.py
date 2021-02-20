from spatial_scene_grammars_examples.solar_system.solar_system_grammar import *
from spatial_scene_grammars.tree import get_tree_root
import numpy as np
import matplotlib.pyplot as plt


def collect_statistics(solar_system_trees):
    def num_planets(tree):
        sun = get_tree_root(tree)
        return len(list(tree.successors(sun)))

    def num_moons(tree):
        sun = get_tree_root(tree)
        planets = tree.successors(sun)
        total_moons = 0
        for planet in planets:
            total_moons += len(list(tree.successors(planet)))
        return total_moons

    def num_bodies(tree):
        return len(list(tree.nodes))

    def all_planet_positions(tree):
        return [node.x for node in tree if isinstance(node, OrbitalBody)]

    all_positions = []
    all_n_bodies = []
    all_n_moons = []
    all_n_planets = []
    for tree in solar_system_trees:
        all_positions += all_planet_positions(tree)
        all_n_bodies.append(num_bodies(tree))
        all_n_moons.append(num_moons(tree))
        all_n_planets.append(num_planets(tree))

    return np.array(all_positions), np.array(all_n_bodies), np.array(all_n_moons), np.array(all_n_planets)


def compare_generated_distributions(N=100):
    # Runs rejection sampling and MPM a bunch of times
    # and compares sample data between the two.

    constraint_set = [
                ClearNeighborhoodConstraint(),
                PlanetCountConstraint(),
                #MoonCountConstraint()
    ]
    root_kwargs = {
        "radius": torch.tensor(100.),
        "x": torch.tensor(0.),
        "x_local": torch.tensor(0.)
    }

    # Rejection sampling
    rejection_sampled_scene_trees = []
    while len(rejection_sampled_scene_trees) < N:
        new_trees, success = sample_tree_from_root_type_with_constraints(
                root_node_type=Sun,
                root_node_instantiation_dict=root_kwargs,
                constraints=constraint_set,
                max_num_attempts=1000,
                backend="rejection",
                #callback=partial(draw_solar_system, fig=fig, ax=ax)
        )
        if success:
            rejection_sampled_scene_trees.append(new_trees[0])
        print("%d/%d rejection sampled" % (len(rejection_sampled_scene_trees), N))


    fig = plt.figure()
    ax = plt.gca()
    sampled_trees, success = sample_tree_from_root_type_with_constraints(
            root_node_type=Sun,
            root_node_instantiation_dict=root_kwargs,
            constraints=constraint_set,
            max_num_attempts=1000,
            backend="metropolis_procedural_modeling",
            num_samples=N,
            #callback=partial(draw_solar_system, fig=fig, ax=ax)
    )

    # Now I have pile of traces, so collect statistics from them both.
    gt_positions, gt_n_bodies, gt_n_moons, gt_n_planets = collect_statistics(rejection_sampled_scene_trees)
    positions, n_bodies, n_moons, n_planets = collect_statistics(sampled_trees)

    # Plot out the planets and moons over time to try to indicate mixing
    plt.figure()
    plt.title("Mixing")
    plt.ylabel("#Bodies")
    plt.xlabel("Sample #")
    plt.plot(gt_n_bodies, label="RS")
    plt.plot(n_bodies, label="MPM")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.subplot(4, 1, 1)
    _, bins, _ = plt.hist(gt_positions, label="RS", alpha=0.5, density=True)
    plt.hist(positions, label="MPM", alpha=0.5, bins=bins, density=True)
    plt.ylabel("Body X")

    plt.subplot(4, 1, 2)
    bin_centers = np.arange(0, max(max(gt_n_bodies), max(n_bodies)) + 2)
    bins = bin_centers - 0.5
    plt.hist(gt_n_bodies, label="RS", alpha=0.5, bins=bins, density=True)
    plt.hist(n_bodies, label="MPM", alpha=0.5, bins=bins, density=True)
    plt.xticks(bin_centers)
    plt.ylabel("#Bodies")

    plt.subplot(4, 1, 3)
    bin_centers = np.arange(0, max(max(gt_n_planets), max(n_planets)) + 2)
    bins = bin_centers - 0.5
    plt.hist(gt_n_planets, label="RS", alpha=0.5, bins=bins, density=True)
    plt.hist(n_planets, label="MPM", alpha=0.5, bins=bins, density=True)
    plt.xticks(bin_centers)
    plt.ylabel("#Planets")

    plt.subplot(4, 1, 4)
    plt.ylabel("#Moons")
    bin_centers = np.arange(0, max(max(gt_n_moons), max(n_moons)) + 2)
    bins = bin_centers - 0.5
    plt.hist(gt_n_moons, label="RS", alpha=0.5, bins=bins, density=True)
    plt.hist(n_moons, label="MPM", alpha=0.5, bins=bins, density=True)
    plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.xticks(bin_centers)
    plt.show()    

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)
    torch.manual_seed(43)

    compare_generated_distributions(N=10000)