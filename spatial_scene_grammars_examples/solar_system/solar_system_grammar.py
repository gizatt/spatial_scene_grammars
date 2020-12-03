import matplotlib.pyplot as plt
import numpy as np
import os
import time

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.tree import *
from spatial_scene_grammars.transform_utils import *


class OrbitalBody(GeometricSetNode, SpatialNode):
    '''Orbital body that can produce some number of
    children in orbits around itself.

    The body will produce children in a radius range based
    on its radius, and will produce children of significantly smaller
    radius than itself. Smaller radii bodies produce less children.
    '''

    class ChildProductionRule(ProductionRule):
        ''' Randomly produces a child planet from a parent planet. '''
        def sample_products(self, parent):
            # Child planet is at random phase.
            self.child_angle = pyro.sample("child_angle", dist.Uniform(0., np.pi*2.))
            # Child planet is at random radius in the parent's region of
            # influence.
            self.child_orbital_radius = pyro.sample("child_orbital_radius", 
                dist.Uniform(parent.min_child_orbital_radius, parent.max_child_orbital_radius))
            # Child mass is going to be a fraction of parent mass.
            child_radius = pyro.sample("child_radius",
                dist.Uniform(parent.min_child_radius, parent.max_child_radius))
            # Figure out relative transform for the planet in world frame.
            relative_tf = torch.eye(4)
            relative_tf[:2, 3] =  self.child_orbital_radius*torch.tensor([
                torch.cos(self.child_orbital_radius), torch.sin(self.child_orbital_radius)])
            new_tf = torch.mm(parent.tf, relative_tf)
            return [OrbitalBody(
                name="orbital_body",
                tf=new_tf,
                radius=child_radius)]

    def __init__(self, name, tf, radius):
        self.radius = radius
        super().__init__(name=name, tf=tf)

    def _setup(self):
        self.color = pyro.sample("color", dist.Uniform(0.0, 1.0))

        self.min_child_orbital_radius = self.radius*2.
        self.max_child_orbital_radius = self.radius*10.
        self.min_child_radius = self.radius * 0.01
        self.max_child_radius = self.radius * 0.05

        # Geometric rate increases linearly with
        # radius until saturating at 0.5 (i.e. 20%
        # chance of stopping at each new planet)
        # when the radius hits 10.
        assert self.radius > 0.
        reproduction_prob = torch.min(self.radius, torch.tensor(0.8))
        
        self.register_production_rules(
            production_rule_type=OrbitalBody.ChildProductionRule,
            production_rule_kwargs={},
            geometric_prob= 1.-reproduction_prob
        )

if __name__ == "__main__":

    scene_tree = SceneTree.forward_sample_from_root_type(
            root_node_type=OrbitalBody,
            name="sun",
            radius=torch.tensor(100.),
            tf=torch.eye(4))
    all_bodies = scene_tree.find_nodes_by_type(OrbitalBody)
    
    planet_locations = np.vstack([planet.tf[:2, 3].numpy() for planet in all_bodies])
    print(planet_locations.shape)
    planet_radii = [planet.radius.item() for planet in all_bodies]
    planet_colors = [planet.color.item() for planet in all_bodies]

    plt.figure(dpi=300).set_size_inches(8, 8)
    ax = plt.gca()
    cm = plt.get_cmap("brg")

    print("Radii: ", planet_radii)
    # For each planet, plot the orbits of the children and the planet istelf
    for k, planet in enumerate(all_bodies):
        child_rules = list(scene_tree.successors(planet))
        for rule in child_rules:
            ax.add_artist(
                plt.Circle(planet_locations[k, :], rule.child_orbital_radius.item(), edgecolor=cm(planet_colors[k]),
                           fill=False, linestyle="--", linewidth=0.2)
            )
        # Planet core
        ax.add_artist(
            plt.Circle(planet_locations[k, :], planet_radii[k], color=cm(planet_colors[k]))
        )
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)
    plt.show()
