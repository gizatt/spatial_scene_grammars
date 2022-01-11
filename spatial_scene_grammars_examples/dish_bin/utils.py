import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import time
import logging

import torch
torch.set_default_tensor_type(torch.DoubleTensor)

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars_examples.dish_bin.grammar import *
from spatial_scene_grammars.parsing import *
from spatial_scene_grammars.sampling import *
from spatial_scene_grammars.parameter_estimation import *
from spatial_scene_grammars.dataset import *

DATASET_YAML_FILE = "sink/saved_scenes.yaml"
DATASET_SAVE_FILE = "observed_node_sets.dat"
OUTLIER_DATASET_YAML_FILE = "sink/saved_outlier_scenes.yaml"
OUTLIER_DATASET_SAVE_FILE = "observed_outlier_node_sets.dat"

def _load_dataset(yaml_file, save_file, reconvert=False):
    if reconvert or not os.path.exists(save_file):
        type_map = {
            "bin": DishBin
        }
        model_map = {
        }
        for model_type_set in [PlateModels, CupModels, BowlModels]:
            for model_type in model_type_set:
                # Have to cut off the "sink" folder to match model names;
                # dataset management is ugly and should get reorganized...
                model_map[os.path.join(*model_type.sdf.split("/")[1:])] = model_type
        observed_node_sets = convert_scenes_yaml_to_observed_nodes(yaml_file, type_map, model_map)
        logging.debug("Saving %s" % save_file)
        with open(save_file, "wb") as f:
            pickle.dump(observed_node_sets, f)

    logging.debug("Loading %s" % save_file)
    with open(save_file, "rb") as f:
        observed_node_sets = pickle.load(f)
    return observed_node_sets

def get_observed_node_sets(reconvert=False):
    return _load_dataset(DATASET_YAML_FILE, DATASET_SAVE_FILE, reconvert=reconvert), \
           _load_dataset(OUTLIER_DATASET_YAML_FILE, OUTLIER_DATASET_SAVE_FILE, reconvert=reconvert)
