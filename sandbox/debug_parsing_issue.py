import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import time

import torch
torch.set_default_tensor_type(torch.DoubleTensor)

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars.parsing import *
from spatial_scene_grammars.sampling import *
from spatial_scene_grammars.parameter_estimation import *
from spatial_scene_grammars.dataset import *

import meshcat
import meshcat.geometry as meshcat_geom

import glob
import os
from functools import lru_cache

import torch
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.drake_interop import *

import pydrake
import pydrake.geometry as pydrake_geom
from pydrake.all import (
    RollPitchYaw,
    RigidTransform
)

'''

 Simple grammar that forces parsing to make a choice between two explanations for a scene:

Root -> Mode 1 -> Observed
Root -> Mode 2 -> Observed

'''

class Observed(TerminalNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=True
        )

class Mode1(AndNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Observed,
                xyz_rule=WorldFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.1, 0.2]),
                    variance=torch.tensor([1., 0.005, 3.]),
                ),
                # Assume world-frame vertically-oriented plate stacks
                rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), [1000., 10., 0.1]
                )
            )
        ]

class Mode2(AndNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Observed,
                xyz_rule=WorldFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.1, 0.2]),
                    variance=torch.tensor([1., 0.005, 0.1]),
                ),
                rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), [1000., 10., 0.1]
                )
            )
        ]

class Root(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.9, 0.1]),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Mode1,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=Mode2,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
        ]


# Set up grammar
grammar = SpatialSceneGrammar(
    root_node_type = Root,
    root_node_tf = drake_tf_to_torch_tf(RigidTransform())
)

observed_nodes = [Observed(tf=torch.eye(4))]
results = infer_mle_tree_with_mip(grammar, observed_nodes, N_solutions=5, verbose=False, use_random_rotation_offset=False)
trees = get_optimized_trees_from_mip_results(results)
for k, tree in enumerate(trees):
    print("Computed score %f, optimization score %f" % (tree.score(verbose=0), results.optim_result.get_suboptimal_objective(k)))