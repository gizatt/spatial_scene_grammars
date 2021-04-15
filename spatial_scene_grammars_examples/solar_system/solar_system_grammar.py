from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import sys

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.sampling import *

class OrbitalBody(GeometricSetNode):
    '''
    Metaclass for orbital body that can produce some number of
    children in orbits around itself.

    (Modeled in 1d, in radial coordinates.)

    The body will produce children in a radius range based
    on its radius, and will produce children of significantly smaller
    radius than itself. Smaller radii bodies produce less children.
    '''

    # Geometric prob will be varied via metaclass instantiation
    # for sun vs planet vs moon.
    geometric_prob = None
    child_type = None

    def __init__(self):
        super().__init__(child_type=self.child_type, geometric_prob=self.geometric_prob, max_repeats=10)

    def _instantiate_children_impl(self, children):
        all_attrs = []
        for k, child in enumerate(children):
            child_orbital_radius = pyro.sample(
                "child_%d_orbital_radius" % k,
                dist.Uniform(self.min_child_orbital_radius,
                             self.max_child_orbital_radius))
                
            child_radius = pyro.sample(
                "child_%d_radius" % k,
                dist.Uniform(self.min_child_radius,
                             self.max_child_radius))
            child_x = self.x + child_orbital_radius
            all_attrs.append({
                "x": child_x,
                "radius": child_radius,
                "x_local": child_orbital_radius
            })
        return all_attrs

    def _instantiate_impl(self, derived_attributes):
        self.radius = derived_attributes["radius"]
        self.x = derived_attributes["x"]
        self.x_local = derived_attributes["x_local"]
        
        self.color = pyro.sample("color", dist.Uniform(0.0, 1.0))

        self.min_child_orbital_radius = self.radius*2.
        self.max_child_orbital_radius = self.radius*10.
        self.min_child_radius = self.radius * 0.001
        self.max_child_radius = self.radius * 0.1

# TODO(gizatt): Again, are metaclasses the answer here?
Moon = type("Moon", (OrbitalBody,), {"geometric_prob": torch.tensor(1.0), "child_type": None})
Planet = type("Planet", (OrbitalBody,), {"geometric_prob": torch.tensor(0.5), "child_type": Moon})
Sun = type("Sun", (OrbitalBody,), {"geometric_prob": torch.tensor(0.5), "child_type": Planet})


class ClearNeighborhoodConstraint(ContinuousVariableConstraint):
    def __init__(self):
        # Hard-coded "neighborhood" size
        self.neighborhood_size_ratio = 15.0
        super().__init__(lower_bound=torch.tensor(0.0),
                         upper_bound=torch.tensor(np.inf))

    def _eval_for_single_body(self, scene_tree, body):
        # Collect child local x and exclusion radii
        # by looking at children of this body.
        children = list(scene_tree.successors(body))
        child_local_x = [child.x_local for child in children]
        child_exclusion_radii = [child.radius*self.neighborhood_size_ratio
                                 for child in children]

        min_sdf = torch.tensor(np.inf)
        # Do N^2 comparison of all bodies
        for child_i in range(len(children)):
            for child_j in range(child_i+1, len(children)):
                if child_i == child_j:
                    continue
                dist = torch.abs(child_local_x[child_i] - child_local_x[child_j])
                sdf = dist - (child_exclusion_radii[child_j] + child_exclusion_radii[child_i])
                if sdf < min_sdf:
                    min_sdf = sdf
        return min_sdf
        
    def eval(self, scene_tree):
        # Returns signed distance between all exclusion zones
        # for any pairs of body.
        constraints = []
        all_bodies = scene_tree.find_nodes_by_type(OrbitalBody)
        signed_dist = [self._eval_for_single_body(scene_tree, body) for body in all_bodies]
        return min(signed_dist)

class PlanetCountConstraint(TopologyConstraint):
    def __init__(self):
        super().__init__(lower_bound=torch.tensor(2.0), upper_bound=torch.tensor(np.inf))
    def eval(self, scene_tree):
        # Counts how many planets there are
        num_planets = len(list(scene_tree.find_nodes_by_type(Planet)))
        return torch.tensor(num_planets)

class MoonCountConstraint(TopologyConstraint):
    def __init__(self):
        super().__init__(lower_bound=torch.tensor(1.0), upper_bound=torch.tensor(np.inf))
    def eval(self, scene_tree):
        # Counts how many moons each planet has
        planets = scene_tree.find_nodes_by_type(Planet)
        if len(planets) == 0:
            return torch.tensor(np.inf)
        num_children_per_child = torch.tensor([
            len(list(scene_tree.successors(planet)))
            for planet in planets
        ])
        return torch.min(num_children_per_child)


def draw_solar_system(scene_tree, fig=None, ax=None, block=False):
    sun = get_tree_root(scene_tree)
    # Override sun color to yellow
    sun.color = torch.tensor(1.0)
    all_bodies = scene_tree.find_nodes_by_type(OrbitalBody)
    
    planet_locations = np.vstack([planet.x.item() for planet in all_bodies])
    planet_radii = [planet.radius.item() for planet in all_bodies]
    planet_colors = [planet.color.item() for planet in all_bodies]
    print("Radii: ", planet_radii)
    print("Locations:", planet_locations)

    if not fig:
        print("New fig")
        fig = plt.figure(dpi=300, facecolor='black')
        fig.set_size_inches(13, 2)
    if not ax:
        ax = plt.gca()
    else:
        ax.clear()
    cm = plt.get_cmap("viridis")

    ax.axhline(0., linestyle="--", color="white", linewidth=1, zorder=-1)
    # For each planet, plot the orbits of the children and the planet istelf
    for k, planet in enumerate(all_bodies):
        children = list(scene_tree.successors(planet))
        for child in children:
            ax.add_artist(
                plt.Circle([planet_locations[k], 0.], child.x_local.item(), edgecolor=cm(planet_colors[k]),
                           fill=False, linestyle="--", linewidth=0.2)
            )
        # Planet core
        ax.add_artist(
            plt.Circle([planet_locations[k, :], 0.], planet_radii[k], color=cm(planet_colors[k]))
        )
    ax.set_xlim(-100, 1100)
    ax.set_ylim(-100, 100)
    ax.axis("off")
    ax.set_aspect('equal')
    plt.pause(0.001)
    #print("Planetl ocations: ", planet_locations)
    #print("Planet radii: ", planet_radii)
    if block:
        plt.waitforbuttonpress()

def sample_and_plot_solar_system(block=False):
    # Create clear-your-neighborhood constraint

    fig = plt.figure(dpi=300, facecolor='black')
    fig.set_size_inches(13, 2)
    ax = plt.gca()

    scene_trees, success = sample_tree_from_root_type_with_constraints(
            root_node_type=Sun,
            root_node_instantiation_dict={
                "radius": torch.tensor(100.),
                "x": torch.tensor(0.),
                "x_local": torch.tensor(0.)
            },
            constraints=[
                ClearNeighborhoodConstraint(),
                PlanetCountConstraint(),
                MoonCountConstraint()
            ],
            max_num_attempts=1000,
            backend="rejection",#"metropolis_procedural_modeling",
            #num_samples=10,
            #callback=partial(draw_solar_system, fig=fig, ax=ax)
    )
    if not success:
        print("WARNING: SAMPLING UNSUCCESSFUL")

    '''
    sun = Sun()
    sun.instantiate({"x": torch.tensor(0.),
                     "radius": torch.tensor(100.),
                     "x_local": torch.tensor(0.)})
    scene_tree = SceneTree.forward_sample_from_root(sun)
    '''
    draw_solar_system(scene_trees[0], fig=fig, block=block)
    
    return scene_trees[0]


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)
    torch.manual_seed(42)
    # Print a trace of a solar system generation
    #trace = pyro.poutine.trace(
    #    SceneTree.forward_sample_from_root_type).get_trace(
    #    root_node_type=OrbitalBody,
    #    radius=torch.tensor(100.),
    #    x=torch.tensor(0.))
    #print(trace.format_shapes())

    for k in range(1):
        sample_and_plot_solar_system()
        plt.savefig("solar_system_%03d.png" % k,
                    facecolor=plt.gcf().get_facecolor(), edgecolor='none',
                    pad_inches=0., dpi=300, bbox_inches='tight')
        plt.show()
        #plt.waitforbuttonpress()
        #plt.close(plt.gcf())
    