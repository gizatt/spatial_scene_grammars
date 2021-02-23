import trimesh
import matplotlib.pyplot as plt
import numpy as np
import pydrake
import torch

from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    MultibodyPlant,
    Parser
)
import pydrake.geometry as pydrake_geom

from spatial_scene_grammars_examples.tabletop_objects.nodes import *
from spatial_scene_grammars_examples.tabletop_objects.constraints import *

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.tree import *
from spatial_scene_grammars.sampling import *
from spatial_scene_grammars.factors import *


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)
    torch.manual_seed(1)

    scene_trees, success = sample_tree_from_root_type_with_constraints(
            root_node_type=Table,
            root_node_instantiation_dict={
                "tf": pose_to_tf_matrix(torch.tensor([0., 0., 0.0, 0., 0., 0.]))
            },
            constraints=[
                NonpenetrationConstraint(),
                ObjectCountConstraint(min=2, max=None),
            ],
            max_num_attempts=1000,
            #backend="rejection",
            backend="rejection_then_hmc",
            #backend="metropolis_procedural_modeling",
            callback=draw_scene_tree_meshcat,
            num_samples=100,
    )
    if not success:
        print("WARNING: SAMPLING UNSUCCESSFUL")
    #simulate_scene_tree(scene_trees[-1], T=1.0, target_realtime_rate=1.0, meshcat="default")

    input("Press enter to start showing scenes...")
    for scene_tree in scene_trees:
        draw_scene_tree_meshcat(scene_tree, alpha=1.0)
        time.sleep(0.1)