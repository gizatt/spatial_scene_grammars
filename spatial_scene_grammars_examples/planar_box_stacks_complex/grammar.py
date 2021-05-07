from dataclasses import dataclass
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


# Ground -> geometric number of groups
# Group -> Stack or Individual
# Stack -> Geometric number vertically aligned
# Non-stacks of N -> N objects at ground level at random positions.

# When this mixin is used, it must come before
# the base node type in the parent class list so
# it overloads the default variable info methods.
class HasDerivedXy():
    @classmethod
    def get_derived_variable_info(cls):
        return {"xy": NodeVariableInfo(shape=(2,))}


class Box(HasDerivedXy, TerminalNode):
    pass

class ObjectStack(HasDerivedXy, RepeatingObjectSetNode):
    def __init__(self, **kwargs):
        super().__init__(
            child_type=StackedObject,
            repeat_probs=torch.tensor([0., 1., 1., 1., 1.]),
            **kwargs
        )
    def get_derived_variable_dists_for_children(self, child_types):
        dists = []
        for k, _ in enumerate(child_types):
            offset = torch.tensor([0., 1.])
            var = torch.tensor([0.01, 0.0001])
            new_xy = dist.Normal(self.xy + offset, var)
            dists.append({"xy": new_xy})
        return dists

class Stack(HasDerivedXy, RepeatingObjectSetNode):
    def __init__(self, parameters, **kwargs):
        super().__init__(
            child_type=Box,
            repeat_probs=torch.tensor([0., 1., 1., 1., 1.]),
            parameters=parameters,
            **kwargs
        )
    @classmethod
    def get_default_parameters(cls):
        return {
            "x_mean": ConstrainedParameter(torch.tensor([0.])),
            "x_variance": ConstrainedParameter(torch.tensor([0.1]), constraint=constraints.positive)
        }
    def get_derived_variable_dists_for_children(self, child_types):
        all_child_dist_dicts = []
        for k, child in enumerate(child_types):
            offset = torch.empty(2)
            offset[0] = self.x_mean()[0]
            offset[1] = float(k)
            variance = torch.empty(2)
            variance[0] = self.x_variance()[0]
            variance[1] = 0.001
            child_xy_dist = dist.Normal(self.xy + offset, variance)
            all_child_dist_dicts.append({
                "xy": child_xy_dist
            })
        return all_child_dist_dicts

class Group(HasDerivedXy, OrNode):
    def __init__(self, parameters, **kwargs):
        super().__init__(
            child_types=[Stack, Box],
            production_weights=parameters["child_weights"](),
            parameters=parameters,
            **kwargs
        )
    @classmethod
    def get_default_parameters(cls):
        return {
            "child_weights": ConstrainedParameter(torch.tensor([0.8, 0.2]), constraint=constraints.simplex)
        }
    def get_derived_variable_dists_for_children(self, child_types):
        # Pass down XY directly to whichever child we resolve to.
        assert len(child_types) == 1
        return [{
            "xy": dist.Normal(self.xy, torch.ones(2)*0.001)
        }]

class Ground(HasDerivedXy, RepeatingObjectSetNode):
    def __init__(self, parameters, **kwargs):
        super().__init__(
            child_type=Group,
            repeat_probs=torch.tensor([0., 1., 1., 1., 1.]),
            parameters=parameters,
            **kwargs
        )

    @classmethod
    def get_default_parameters(cls):
        return {
            "x_mean": ConstrainedParameter(torch.tensor([0.])),
            "x_variance": ConstrainedParameter(torch.tensor([2.]), constraint=constraints.positive),
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


class BoundsConstraint(ContinuousVariableConstraint):
    def __init__(self, xbounds=[-4., 4]):
        super().__init__(lower_bound=torch.tensor([xbounds[0], -np.inf]),
                         upper_bound=torch.tensor([np.inf, xbounds[1]]))

    def eval(self, scene_tree):
        # Get min / max x of any box
        boxes = scene_tree.find_nodes_by_type(Box)
        box_xs = torch.stack([box.xy[0] for box in boxes])
        return torch.tensor([torch.min(box_xs), torch.max(box_xs)])


class BoxCountConstraint(TopologyConstraint):
    def __init__(self):
        super().__init__(lower_bound=torch.tensor(1.0), upper_bound=torch.tensor(np.inf))
    def eval(self, scene_tree):
        # Counts how many boxes there are
        num_boxes = len(list(scene_tree.find_nodes_by_type(Box)))
        return torch.tensor(num_boxes)


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
        "Group": [0.5, 0., 0.5, 0.5],
        "Stack": [0.25, 0., 0.75, 0.5],
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
                "xy": dist.Normal(torch.tensor([0., 0.]), torch.tensor([0.001, 0.001]))
            },
            constraints=[
                NonpenetrationConstraint(0.01),
                BoundsConstraint(),
                BoxCountConstraint()
            ],
            max_num_attempts=1000,
            backend="rejection",#"metropolis_procedural_modeling",
        )
        if not success:
            print("WARNING: SAMPLING UNSUCCESSFUL")
        draw_boxes(scene_trees[0], fig=fig, block=False)
        plt.pause(1.0)
