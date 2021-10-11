import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import time
from tqdm.notebook import tqdm

import torch
torch.set_default_tensor_type(torch.DoubleTensor)

from spatial_scene_grammars.constraints import *
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars_examples.dish_bin.grammar import *
from spatial_scene_grammars.parsing import *
from spatial_scene_grammars.sampling import *
from spatial_scene_grammars.parameter_estimation import *
from spatial_scene_grammars.dataset import *

import meshcat
import meshcat.geometry as meshcat_geom

# Set up grammar
grammar = SpatialSceneGrammar(
    root_node_type = DishBin,
    root_node_tf = drake_tf_to_torch_tf(RigidTransform(p=[0.5, 0., 0.]))
)
grammar.load_state_dict(torch.load("fit_grammar.torch"))

vis = meshcat.Visualizer()
print("Meshcat port: ", vis.url())

class InBinConstraint(Constraint):
    # XY coord of each object inside .56 x .83 dish bin
    def __init__(self):
        lb = torch.tensor([-0.56/2+0.1, -0.83/2+0.1, 0.])
        ub = torch.tensor([0.56/2-0.1, 0.83/2-0.1, 1.])
        super().__init__(
            lower_bound=lb,
            upper_bound=ub
        )
    def eval(self, scene_tree):
        xys = []
        bin_pos = scene_tree.find_nodes_by_type(DishBin)[0].translation
        for node in scene_tree.find_nodes_by_type(ObjectModel):
            xys.append(node.translation - bin_pos)
        return torch.stack(xys, axis=0)
    def add_to_ik_prog(self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map):
        bin_pos = scene_tree.find_nodes_by_type(DishBin)[0].translation.detach().numpy()
        for node in scene_tree.find_nodes_by_type(ObjectModel):
            for body_id in node_to_free_body_ids_map[node]:
                body = mbp.get_body(body_id)
                print(bin_pos + self.lower_bound.detach().numpy(), bin_pos + self.upper_bound.detach().numpy())
                ik.AddPositionConstraint(
                    body.body_frame(), np.zeros(3),
                    mbp.world_frame(),
                    bin_pos + self.lower_bound.detach().numpy(),
                    bin_pos + self.upper_bound.detach().numpy()
                )
constraints = [
    InBinConstraint()
]

# Draw a lot of projected environments
torch.random.manual_seed(0)
for k in range(100):
    tree = grammar.sample_tree(detach=True)
    tree, success = rejection_sample_structure_to_feasibility(tree, constraints=constraints)
    draw_scene_tree_contents_meshcat(tree, zmq_url=vis.window.zmq_url, prefix="pre_projected_sample/contents")
    draw_scene_tree_structure_meshcat(tree, zmq_url=vis.window.zmq_url, prefix="pre_projected_sample/structure", alpha=0.25, node_sphere_size=0.01)
    tree = project_tree_to_feasibility_via_sim(tree, constraints=constraints, zmq_url=vis.window.zmq_url)
    draw_scene_tree_contents_meshcat(tree, zmq_url=vis.window.zmq_url, prefix="projected_samples/content")
    #draw_scene_tree_structure_meshcat(tree, zmq_url=vis.window.zmq_url, prefix="projected_samples/structure")
    input()