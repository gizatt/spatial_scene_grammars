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


# Ground -> Stack or Non-Stack
# Stack -> Stack of 2, or 3
# Stacks of N -> N objects with approximate vertical alignment
# Non stack -> non-stack of 1, 2, or 3
# Non-stacks of N -> N objects at ground level at random positions.

# Explicitly implemented without SET nodes for prototyping some
# parsing stuff a little easier.


class Box(TerminalNode):
    def _instantiate_impl(self, derived_attributes):
        self.xy = derived_attributes["xy"]

class StackOfN(AndNode):
    N = None
    def __init__(self):
        super().__init__(child_types=[Box]*self.__class__.N)
    def _instantiate_children_impl(self, children):
        all_attrs = []
        for k, child in enumerate(children):
            child_x = pyro.sample("child_%d_x" % k,
                dist.Normal(torch.tensor(0.), torch.tensor(0.1))).item()
            child_y = self.xy[1] + torch.tensor(k)
            all_attrs.append({
                "xy": torch.tensor([child_x, child_y]),
            })
        return all_attrs
    def _instantiate_impl(self, derived_attributes):
        self.xy = derived_attributes["xy"]
StackOf2 = type("StackOf2", (StackOfN,), {"N": 2})
StackOf3 = type("StackOf3", (StackOfN,), {"N": 3})


class GroupOfN(AndNode):
    N = None
    def __init__(self):
        super().__init__(child_types=[Box]*self.__class__.N)
    def _instantiate_children_impl(self, children):
        all_attrs = []
        for k, child in enumerate(children):
            child_x = pyro.sample("child_%d_x" % k,
                dist.Normal(torch.tensor(0.), torch.tensor(1.0))).item()
            child_y = self.xy[1]
            all_attrs.append({
                "xy": torch.tensor([child_x, child_y]),
            })
        return all_attrs
    def _instantiate_impl(self, derived_attributes):
        self.xy = derived_attributes["xy"]
GroupOf1 = type("GroupOf1", (GroupOfN,), {"N": 2})
GroupOf2 = type("GroupOf2", (GroupOfN,), {"N": 2})
GroupOf3 = type("GroupOf3", (GroupOfN,), {"N": 2})


class Ground(OrNode):
    def __init__(self):
        child_types = [StackOf2, StackOf3, GroupOf1, GroupOf2, GroupOf3]
        super().__init__(child_types=child_types,
                         production_weights=torch.ones(len(child_types)))
    def _instantiate_children_impl(self, children):
        all_attrs = []
        for k, child in enumerate(children):
            child_x = pyro.sample("child_%d_x" % k,
                dist.Normal(torch.tensor(0.), torch.tensor(1.0))).item()
            child_y = self.xy[1]
            all_attrs.append({
                "xy": torch.tensor([child_x, child_y]),
            })
        return all_attrs
    def _instantiate_impl(self, derived_attributes):
        self.xy = derived_attributes["xy"]



def draw_boxes(scene_tree, fig=None, ax=None, block=False, xlim=[-10., 10.]):
    if fig is None:
        plt.figure()
    if ax is None:
        ax = plt.gca()
    ax.clear()

    ground = scene_tree.find_nodes_by_type(Ground)[0]
    ground_level = ground.xy[1].item()
    ax.fill_between(xlim,[ground_level, ground_level], y2=ground_level-1000, color='red', alpha=0.8)
    ax.axhline(ground_level)

    boxes = scene_tree.find_nodes_by_type(Box)
    cm = plt.get_cmap("viridis")
    for k, box in enumerate(boxes):
        print("Box at %f, %f" % (box.xy[0], box.xy[1]))
        color = cm(float(k) / (len(boxes) - 1))
        ax.add_artist(
                plt.Rectangle([item.item() for item in box.xy],
                              width=1., height=1., angle=0., fill=True, alpha=0.8,
                              color=color)
            )

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ground_level-1, ground_level + 5)
    ax.axis("off")
    ax.set_aspect('equal')
    plt.pause(0.001)
    #print("Planetl ocations: ", planet_locations)
    #print("Planet radii: ", planet_radii)
    if block:
        plt.waitforbuttonpress()


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)
    torch.manual_seed(42)

    fig = plt.figure()
    for k in range(100):
        tree = SceneTree.forward_sample_from_root_type(
            Ground,
            {"xy": torch.tensor([0., 0.])}
        )
        draw_boxes(tree, fig=fig, block=False)
        plt.pause(1.0)
