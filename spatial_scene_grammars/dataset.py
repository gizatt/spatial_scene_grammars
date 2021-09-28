from copy import deepcopy
from collections import namedtuple
import numpy as np
import logging

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import pydrake
from pydrake.all import RigidTransform, RollPitchYaw

from .drake_interop import drake_tf_to_torch_tf


def convert_scenes_yaml_to_observed_nodes(dataset_file, type_map={}, model_map={}):
    '''
    Converts a scene list YAML file of the format described at
    https://github.com/gizatt/drake_hydra_interact/blob/master/environments/README.md
    into a list of nodes, using the dictionary type_map to map from metadata "type"
    string to Node types.
    '''

    with open(dataset_file, "r") as f:
        scenes_dict = yaml.load(f, Loader=Loader)
    observed_node_sets = []
    for scene_name, scene_info in scenes_dict.items():
        observed_nodes = []
        for object_info in scene_info["objects"]:
                class_name = object_info["metadata"]["class"]
                model_name = object_info["model_file"]
                if model_name in model_map.keys():
                    this_type = model_map[model_name]
                elif class_name in type_map.keys():
                    this_type = type_map[class_name]
                else:
                    logging.warn("Environment had unknown model name / class name: %s, %s", model_name, class_name)
                    continue
                tf = RigidTransform(
                    p = object_info["pose"]["xyz"],
                    rpy = RollPitchYaw(object_info["pose"]["rpy"])
                )
                observed_nodes.append(
                    this_type(drake_tf_to_torch_tf(tf))
                )
        for object_info in scene_info["world_description"]["models"]:
            if "metadata" in object_info.keys():
                class_name = object_info["metadata"]["class"]
                model_name = object_info["model_file"]
                if model_name in model_map.keys():
                    this_type = model_map[model_name]
                elif class_name in type_map.keys():
                    this_type = type_map[class_name]
                else:
                    logging.warn("Environment had unknown model name / class name: %s, %s", model_name, class_name)
                    continue
                tf = RigidTransform(
                    p = object_info["pose"]["xyz"],
                    rpy = RollPitchYaw(object_info["pose"]["rpy"])
                )
                observed_nodes.append(
                    this_type(drake_tf_to_torch_tf(tf))
                )
        observed_node_sets.append(observed_nodes)
    return observed_node_sets
