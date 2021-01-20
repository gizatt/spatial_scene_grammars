import trimesh
import matplotlib.pyplot as plt
import numpy as np
import pydrake

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
    scene_tree, success = sample_tree_from_root_type_with_constraints(
            root_node_type=Table,
            root_node_type_kwargs={
                "name":"table",
                "tf": torch.eye(4),
                "object_production_rate": 0.4,
                "table_size": torch.tensor([0.5, 0.5])
            },
            constraints=[
                NonpenetrationConstraint()
            ],
            max_num_attempts=1000,
            backend="rejection_then_hmc",
            callback=draw_scene_tree_meshcat,
            num_samples=30
    )
    if not success:
        print("WARNING: SAMPLING UNSUCCESSFUL")
    simulate_scene_tree(scene_tree, T=1.0, target_realtime_rate=1.0, meshcat="default")