from spatial_scene_grammars_examples.kitchen.run_grammar import rejection_sample_feasible_tree
from spatial_scene_grammars.serialization_sdf import serialize_scene_tree_to_package_and_single_sdf
import torch
import pyro

import pydrake
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Parser,
    ConnectMeshcatVisualizer,
    Simulator,
    DiagramBuilder
)

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)

    scene_tree, satisfied_clearance = rejection_sample_feasible_tree(num_attempts=1000)
    #scene_tree, satisfied_feasibility = project_tree_to_feasibility(scene_tree, num_attempts=3)

    serialize_scene_tree_to_package_and_single_sdf(scene_tree, "test.sdf")

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    parser = Parser(plant)
    parser.AddAllModelsFromFile("test.sdf")
    meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url="default")
    plant.Finalize()
    diagram = builder.Build()
    diag_context = diagram.CreateDefaultContext()
    sim = Simulator(diagram)
    sim.set_target_realtime_rate(1.0)
    sim.AdvanceTo(0.001)
