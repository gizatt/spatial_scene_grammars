from functools import partial
import numpy as np
import os
import time

import pydrake
from pydrake.all import (
    Box, RollPitchYaw, RigidTransform, Parser
)

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoname import name_count

from scene_grammar.src.nodes import *
from scene_grammar.src.rules import *
from scene_grammar.src.tree import *
from scene_grammar.src.transform_utils import *
from scene_grammar.src.drake_interop import *

from grammar_room_layout import *


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)

    # Sample an environment (wrapped in some pyro messiness so I can
    # play with the resulting program trace.)
    start = time.time()
    pyro.clear_param_store()
    trace = poutine.trace(ParseTree.generate_from_root_type).get_trace(root_node_type=Kitchen)
    scene_tree = trace.nodes["_RETURN"]["value"]
    end = time.time()

    # Print out all the nodes and their transforms.
    for node in scene_tree.nodes:
        if isinstance(node, Node):
            print(node.name, ": ", node.tf.tolist())

    print("Generated data in %f seconds." % (end - start))
    print("Full trace values:" )
    for node_name in trace.nodes.keys():
        if node_name in ["_INPUT", "_RETURN"]:
            continue
        print(node_name, ": ", trace.nodes[node_name]["value"].detach().numpy())


    # Simulate the resulting scene, with a PR2 for scale.
    builder, mbp, scene_graph = compile_scene_tree_to_mbp_and_sg(
        scene_tree, timestep=0.001)
    # Add PR2 and shift it back in front of where I know the table will be.
    parser = Parser(mbp)
    pr2_model_path = "/home/gizatt/drake/build/install/share/drake/examples/pr2/models/pr2_description/urdf/pr2_simplified.urdf"
    parser.package_map().PopulateUpstreamToDrake(pr2_model_path);
    pr2_model_id = parser.AddModelFromFile(
        file_name=pr2_model_path, model_name="PR2_for_scale")
    # The PR2 is on x and y rails: find the x joint and set its default state
    # to shift back from the table.
    mbp.GetJointByName("x", model_instance=pr2_model_id).set_default_translation(-0.5)

    mbp.Finalize()
    
    visualizer = ConnectMeshcatVisualizer(builder, scene_graph,
        zmq_url="default")

    diagram = builder.Build()
    diag_context = diagram.CreateDefaultContext()

    # Fix input port for PR2 to zero.
    actuation_port = mbp.get_actuation_input_port(model_instance=pr2_model_id)
    nu = mbp.num_actuated_dofs(model_instance=pr2_model_id)
    mbp_context = diagram.GetMutableSubsystemContext(mbp, diag_context)
    actuation_port.FixValue(mbp_context, np.zeros(nu))

    sim = Simulator(diagram, diag_context)
    sim.set_target_realtime_rate(1.0)
    sim.AdvanceTo(20)