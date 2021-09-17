import networkx as nx
import numpy as np
import os
import pickle
import time
from tqdm import tqdm
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars_examples.planar_clusters_gaussians.grammar import *
from spatial_scene_grammars.parsing import *
from spatial_scene_grammars.sampling import *
from spatial_scene_grammars.parameter_estimation import *

# Sample a dataset of scenes from the default grammar params.
# Draw a random sample from the grammar and visualize it.
# (Cache output.)
torch.random.manual_seed(2)
N_samples = 5

ground_truth_grammar = SpatialSceneGrammar(
    root_node_type = Desk,
    root_node_tf = torch.eye(4)
)

samples = []
for k in tqdm(range(N_samples)):
    tree = ground_truth_grammar.sample_tree(detach=True)
    observed_nodes = tree.get_observed_nodes()
    samples.append((tree, observed_nodes))

observed_node_sets = [x[1] for x in samples]

# Randomly reset parameters and try to recover them.
torch.random.manual_seed(42)
random_grammar = SpatialSceneGrammar(
    root_node_type = Desk,
    root_node_tf = torch.eye(4),
    sample_params_from_prior=True
)
svi = SVIWrapper(random_grammar, [sample[1] for sample in samples])
svi.do_iterated_vi_fitting(major_iterations=2, minor_iterations=5, tqdm=tqdm, num_elbo_samples=5)