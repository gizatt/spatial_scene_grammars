from spatial_scene_grammars_examples.kitchen.run_grammar import rejection_sample_feasible_tree
from spatial_scene_grammars.serialization_sdf import serialize_scene_tree_to_package_and_single_sdf
import pybullet
import torch
import pyro

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)

    scene_tree, satisfied_clearance = rejection_sample_feasible_tree(num_attempts=1000)

    serialize_scene_tree_to_package_and_single_sdf(
        scene_tree, "test.sdf",
        include_static_tag=True, 
        include_model_files=True,
        pybullet_compat=True
    )


    physicsClient = pybullet.connect(pybullet.GUI)
    pybullet.setGravity(0,0,-9.81)
    pybullet.loadSDF("test.sdf")
    pybullet.setRealTimeSimulation(1)
    while 1:
        pass
    pybullet.disconnect()