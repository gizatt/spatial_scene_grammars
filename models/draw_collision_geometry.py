import argparse

import numpy as np

import pydrake
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Meshcat,
    MeshcatVisualizerCpp,
    MeshcatVisualizerParams,
    MultibodyPlant,
    Parser,
    Role,
    RigidTransform,
    Simulator
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("model_path", help="File to vis")
    parser.add_argument("--port", help="Meshcat port", type=int, default=None)

    args = parser.parse_args()
    
    builder = DiagramBuilder()
    mbp, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=0.001))
    parser = Parser(mbp)
    model_id = parser.AddModelFromFile(args.model_path)
    mbp.Finalize()
    
    print(args.port)
    meshcat = Meshcat(port=args.port)
    params = MeshcatVisualizerParams()
    params.role = Role.kProximity
    params.prefix = "geometry_draw"
    params.delete_on_initialization_event = False
    vis = MeshcatVisualizerCpp.AddToBuilder(
        builder, scene_graph, meshcat, params
    )
    
    diagram = builder.Build()
    diagram.Publish(diagram.CreateDefaultContext())
    
    input()