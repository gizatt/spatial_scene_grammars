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
from pyro.nn import PyroParam

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.sampling import *


# Ground -> Stack or Non-Stack
# Stack -> Stack of 2, or 3
# Stacks of N -> N objects with approximate vertical alignment
# Non stack -> non-stack of 1, 2, or 3
# Non-stacks of N -> N objects at ground level at random positions.

# Explicitly implemented without SET nodes (for legacy reasons).


# When this mixin is used, it must come before
# the base node type in the parent class list so
# it overloads the default variable info methods.
class HasDerivedXy():
    @classmethod
    def get_derived_variable_info(cls):
        return {"xy": (2,)}


class Box(HasDerivedXy, TerminalNode):
    pass
    

class StackOfN(HasDerivedXy, AndNode):
    N = None
    def __init__(self, **kwargs):
        super().__init__(child_types=[Box]*self.__class__.N, **kwargs)
    @classmethod
    def get_default_parameters(cls):
        return {
            "x_mean": NodeParameter(torch.tensor([0.])),
            "x_variance": NodeParameter(torch.tensor([0.1]), constraint=constraints.positive)
        }
    def get_derived_variable_dists_for_children(self, child_types):
        all_child_dist_dicts = []
        for k, child in enumerate(child_types):
            offset = torch.empty(2)
            offset[0] = self.x_mean()[0]
            offset[1] = float(k)
            variance = torch.empty(2)
            variance[0] = self.x_variance()[0]
            variance[1] = 0.01
            child_xy_dist = dist.Normal(self.xy + offset, variance)
            all_child_dist_dicts.append({
                "xy": child_xy_dist
            })
        return all_child_dist_dicts
StackOf2 = type("StackOf2", (StackOfN,), {"N": 2})
StackOf3 = type("StackOf3", (StackOfN,), {"N": 3})


class GroupOfN(HasDerivedXy, AndNode):
    N = None
    def __init__(self, **kwargs):
        super().__init__(child_types=[Box]*self.__class__.N, **kwargs)
    @classmethod
    def get_default_parameters(cls):
        return {
            "x_mean": NodeParameter(torch.tensor([0.])),
            "x_variance": NodeParameter(torch.tensor([1.]), constraint=constraints.positive)
        }
    def get_derived_variable_dists_for_children(self, child_types):
        all_child_dist_dicts = []
        for k, child in enumerate(child_types):
            offset = torch.zeros(2)
            offset[0] = self.x_mean()[0]
            variance = torch.empty(2)
            variance[0] = self.x_variance()[0]
            variance[1] = 0.01
            child_xy_dist = dist.Normal(self.xy + offset, variance)
            all_child_dist_dicts.append({
                "xy": child_xy_dist
            })
        return all_child_dist_dicts
GroupOf1 = type("GroupOf1", (GroupOfN,), {"N": 1})
GroupOf2 = type("GroupOf2", (GroupOfN,), {"N": 2})
GroupOf3 = type("GroupOf3", (GroupOfN,), {"N": 3})


class Ground(HasDerivedXy, OrNode):
    def __init__(self, parameters, **kwargs):
        child_types = [StackOf2, StackOf3, GroupOf1, GroupOf2, GroupOf3]
        super().__init__(child_types=child_types,
                         production_weights=parameters["child_weights"](),
                         parameters=parameters,
                         **kwargs)
    @classmethod
    def get_default_parameters(cls):
        return {
            "x_mean": NodeParameter(torch.tensor([0.])),
            "x_variance": NodeParameter(torch.tensor([1.]), constraint=constraints.positive),
            "child_weights": NodeParameter(torch.ones(5)/5., constraint=constraints.simplex)
        }
    def get_derived_variable_dists_for_children(self, child_types):
        all_child_dist_dicts = []
        for k, child in enumerate(child_types):
            # Spawn the child group at the center of a single box's
            # height, randomly somewhere along the x axis.
            offset = torch.empty(2)
            offset[0] = self.x_mean()[0]
            offset[1] = self.xy[1] + 0.5
            variance = torch.empty(2)
            variance[0] = self.x_variance()[0]
            variance[1] = 0.01
            child_xy_dist = dist.Normal(self.xy + offset, variance)
            all_child_dist_dicts.append({
                "xy": child_xy_dist,
            })
        return all_child_dist_dicts


class NonpenetrationConstraint(ContinuousVariableConstraint):
    def __init__(self, allowed_penetration_margin=0.0):
        ''' penetration_margin > 0, specifies penetration amounts we'll allow. '''
        self.allowed_penetration_margin = allowed_penetration_margin
        super().__init__(lower_bound=torch.tensor(-np.inf),
                         upper_bound=torch.tensor(0.0))

    def eval(self, scene_tree):
        # For all pairs of boxes, compute the overlap region
        # and add the area to the total penetration.
        boxes = scene_tree.find_nodes_by_type(Box)
        N = len(boxes)
        total_penetration_area = torch.tensor([0.0])
        for i in range(N):
            for j in range(i+1, N):
                # determine the coordinates of the intersection rectangle
                box_i_l = boxes[i].xy - torch.tensor([0.5, 0.5])
                box_i_u = boxes[i].xy + torch.tensor([0.5, 0.5])
                box_j_l = boxes[j].xy - torch.tensor([0.5, 0.5])
                box_j_u = boxes[j].xy + torch.tensor([0.5, 0.5])

                bb_l = torch.maximum(box_i_l, box_j_l)
                bb_u = torch.minimum(box_i_u, box_j_u)
                edge_lengths = bb_u - bb_l - self.allowed_penetration_margin
                if torch.all(edge_lengths > 0):
                    total_penetration_area += torch.prod(edge_lengths)
        return total_penetration_area


def draw_boxes(scene_tree, fig=None, ax=None, block=False, xlim=[-5., 5.]):
    if fig is None:
        plt.figure()
    if ax is None:
        ax = plt.gca()
    ax.clear()

    # Hacky support for sometimes-there ground node;
    # it might not be there in half-parsed scenes.
    grounds = scene_tree.find_nodes_by_type(Ground)
    if len(grounds) > 0:
        ground = grounds[0]
        ground_level = ground.xy[1].item()
        ax.fill_between(xlim,[ground_level, ground_level], y2=ground_level-1000, color='red', alpha=0.8)
    else:
        ground_level = 0. # Guess

    boxes = scene_tree.find_nodes_by_type(Box)
    cm = plt.get_cmap("viridis")
    for k, box in enumerate(boxes):
        color = cm(float(k) / (len(boxes)))
        ax.add_artist(
                plt.Rectangle([item.item() - 0.5 for item in box.xy],
                              width=1., height=1., angle=0., fill=True, alpha=0.8,
                              color=color)
            )

    # Draw underlying scene tree
    colors = []
    colors_by_type = {
        "Box": [0., 0., 0., 0.],
        "StackOf1": [0.5, 0., 0.5, 0.5],
        "StackOf2": [0.25, 0., 0.75, 0.5],
        "StackOf3": [0., 0., 1.0, 0.5],
        "GroupOf1": [0.5, 1., 0.5, 0.5],
        "GroupOf2": [0.25, 1., 0.75, 0.5],
        "GroupOf3": [0., 1., 0.1, 0.5],
        "Ground": [0., 0., 0., 0.5]
    }
    pos = {}
    for node in scene_tree:
        colors.append(colors_by_type[node.__class__.__name__])
        pos[node] = node.xy.detach().numpy()

    nx.draw_networkx(scene_tree, pos=pos, node_color=colors, with_labels=False, node_size=3.)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ground_level-1, ground_level + 5)
    ax.axis("off")
    ax.set_aspect('equal')
    if block:
        plt.waitforbuttonpress()


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)
    torch.manual_seed(42)

    fig = plt.figure()
    for k in range(100):
        scene_trees, success = sample_tree_from_root_type_with_constraints(
            root_node_type=Ground,
            root_node_instantiation_dict={
                "xy": torch.tensor([0., 0.])
            },
            constraints=[
                NonpenetrationConstraint(0.001),
            ],
            max_num_attempts=1000,
            backend="rejection",#"metropolis_procedural_modeling",
        )
        if not success:
            print("WARNING: SAMPLING UNSUCCESSFUL")
        draw_boxes(scene_trees[0], fig=fig, block=False)
        plt.pause(1.0)
