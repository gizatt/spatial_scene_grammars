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
from spatial_scene_grammars.tree import *
from spatial_scene_grammars.sampling import *
from spatial_scene_grammars.factors import *


class OrbitalBody(GeometricSetNode):
    '''
    Orbital body that can produce some number of
    children in orbits around itself.

    (Modeled in 1d, in radial coordinates.)

    The body will produce children in a radius range based
    on its radius, and will produce children of significantly smaller
    radius than itself. Smaller radii bodies produce less children.
    '''

    class ChildProductionRule(ProductionRule):
        ''' Randomly produces a child planet from a parent planet. '''
        def _sample_products(self, parent, child_names):
            # Child planet location and size is a function of the parent.
            # Both are saved associated with the rule to make writing
            # the neighborhood constraint more convenient.

            self.child_orbital_radius = pyro.sample(
                "child_orbital_radius",
                dist.Uniform(parent.min_child_orbital_radius,
                             parent.max_child_orbital_radius))
                
            self.child_radius = pyro.sample(
                "child_radius",
                dist.Uniform(parent.min_child_radius,
                             parent.max_child_radius))
            child_x = parent.x + self.child_orbital_radius
            return [OrbitalBody(
                name=child_names[0],
                x=child_x,
                radius=self.child_radius)]

    def __init__(self, name, x, radius):
        self.radius = radius
        self.x = x
        super().__init__(name=name)

    def _setup(self):
        self.color = pyro.sample("color", dist.Uniform(0.0, 1.0))

        self.min_child_orbital_radius = self.radius*2.
        self.max_child_orbital_radius = self.radius*10.
        self.min_child_radius = self.radius * 0.001
        self.max_child_radius = self.radius * 0.01

        # Geometric rate increases linearly with
        # radius until saturating at 0.7 (i.e. 30%
        # chance of stopping at each new planet)
        # when the radius hits 0.7.
        assert self.radius > 0.
        reproduction_prob = torch.min(self.radius, torch.tensor(0.5))
        
        self.register_production_rules(
            production_rule_type=OrbitalBody.ChildProductionRule,
            production_rule_kwargs={"child_types":[OrbitalBody]},
            geometric_prob= 1.-reproduction_prob
        )

class ClearNeighborhoodConstraint(ContinuousVariableConstraint):
    def __init__(self):
        # Hard-coded "neighborhood" size
        self.neighborhood_size_ratio = 15.0
        super().__init__(lower_bound=torch.tensor(0.0),
                         upper_bound=torch.tensor(np.inf))

    def _eval_for_single_body(self, scene_tree, body):
        # Collect child local x and exclusion radii
        # by looking at the production rules under this body.
        child_rules = list(scene_tree.successors(body))
        child_local_x = [rule.child_orbital_radius for rule in child_rules]
        child_exclusion_radii = [rule.child_radius*self.neighborhood_size_ratio
                                 for rule in child_rules]

        min_sdf = torch.tensor(np.inf)
        # Do N^2 comparison of all bodies
        for child_i in range(len(child_rules)):
            for child_j in range(child_i+1, len(child_rules)):
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
        #print("Min SDF: ", min(signed_dist))
        return min(signed_dist)

class PlanetCountConstraint(TopologyConstraint):
    def __init__(self):
        super().__init__(lower_bound=torch.tensor(2.0), upper_bound=torch.tensor(np.inf))
    def eval(self, scene_tree):
        # Counts how many planets the sun has
        sun = get_tree_root(scene_tree)
        #print("Num planets: ", len(list(scene_tree.successors(sun))))
        return torch.tensor(len(list(scene_tree.successors(sun))))

class MoonCountConstraint(TopologyConstraint):
    def __init__(self):
        super().__init__(lower_bound=torch.tensor(2.0), upper_bound=torch.tensor(2.0))
    def eval(self, scene_tree):
        # Counts how many moons each planet has
        simplified_tree = scene_tree.get_tree_without_production_rules()
        sun = get_tree_root(simplified_tree)
        planets = list(simplified_tree.successors(sun))
        if len(planets) == 0:
            return torch.tensor(np.inf)
        num_children_per_child = torch.tensor([
            len(list(simplified_tree.successors(planet)))
            for planet in planets
        ])
        #print("Num children per child: ", num_children_per_child)
        return torch.min(num_children_per_child)

def draw_solar_system(scene_tree, fig=None, ax=None):
    sun = get_tree_root(scene_tree)
    # Override sun color to yellow
    sun.color = torch.tensor(1.0)
    all_bodies = scene_tree.find_nodes_by_type(OrbitalBody)
    
    planet_locations = np.vstack([planet.x.item() for planet in all_bodies])
    planet_radii = [planet.radius.item() for planet in all_bodies]
    planet_colors = [planet.color.item() for planet in all_bodies]

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
        child_rules = list(scene_tree.successors(planet))
        for rule in child_rules:
            ax.add_artist(
                plt.Circle([planet_locations[k], 0.], rule.child_orbital_radius.item(), edgecolor=cm(planet_colors[k]),
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
    plt.pause(0.01)
    #print("Planetl ocations: ", planet_locations)
    #print("Planet radii: ", planet_radii)
    #plt.waitforbuttonpress()

def sample_and_plot_solar_system():
    # Create clear-your-neighborhood constraint

    fig = plt.figure(dpi=300, facecolor='black')
    fig.set_size_inches(13, 2)
    ax = plt.gca()

    scene_tree, success = sample_tree_from_root_type_with_constraints(
            root_node_type=OrbitalBody,
            root_node_type_kwargs={
                "name":"sun",
                "radius": torch.tensor(100.),
                "x": torch.tensor(0.)
            },
            constraints=[
                ClearNeighborhoodConstraint(),
                PlanetCountConstraint(),
                MoonCountConstraint()
            ],
            max_num_attempts=1000,
            backend="rejection_then_hmc",
            callback=partial(draw_solar_system, fig=fig, ax=ax)
    )
    if not success:
        print("WARNING: SAMPLING UNSUCCESSFUL")
    draw_solar_system(scene_tree)
    

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)
    torch.manual_seed(43)
    # Print a trace of a solar system generation
    #trace = pyro.poutine.trace(
    #    SceneTree.forward_sample_from_root_type).get_trace(
    #    root_node_type=OrbitalBody,
    #    radius=torch.tensor(100.),
    #    x=torch.tensor(0.))
    #print(trace.format_shapes())
#
    #sys.exit(0)
    for k in range(1):
        sample_and_plot_solar_system()
        plt.savefig("solar_system_%03d.png" % k,
                    facecolor=plt.gcf().get_facecolor(), edgecolor='none',
                    pad_inches=0., dpi=300, bbox_inches='tight')
        plt.show()
        #plt.waitforbuttonpress()
        #plt.close(plt.gcf())
    