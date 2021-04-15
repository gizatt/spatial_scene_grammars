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
from spatial_scene_grammars_examples.solar_system.solar_system_grammar import *

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

    scene_tree = sample_and_plot_solar_system(block=False)
    trace = scene_tree.get_trace()
    #print(trace.format_shapes())
    #plt.show()

    # Build the "observed" tree, which in this case is the
    # set of planets (but without connections).
    observed_tree = SceneTree()
    for node in scene_tree:
        observed_tree.add_node(node)
        # No edges

    print("Observed tree has nodes: ", [n.name for n in observed_tree.nodes])
    reconstructed_tree = SceneTree.parse_greedily_from_partial_tree(
        root_node_type=Sun, partial_tree=observed_tree
    )


