import argparse

import pydrake
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Meshcat,
    MeshcatVisualizerParams,
    MeshcatVisualizerCpp,
    Parser,
    Role,
    Simulator,
    MultibodyPlant
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Do interactive placement of objects.')
    parser.add_argument('model_path',
                        help='Path to model SDF/URDF.')
    args = parser.parse_args()
    
    # Build MBP
    builder = DiagramBuilder()
    mbp, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=1E-3)
    )
    # Parse requested file
    parser = Parser(mbp, scene_graph)
    model_id = parser.AddModelFromFile(args.model_path)

    mbp.Finalize()

    # Visualizer
    meshcat = Meshcat()
    vis = MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat=meshcat)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    mbp_context = diagram.GetSubsystemContext(mbp, diagram_context)
    simulator = Simulator(diagram, diagram_context)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(1000.)
    
