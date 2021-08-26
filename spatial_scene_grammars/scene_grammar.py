from copy import deepcopy
from collections import namedtuple, OrderedDict
from dataclasses import dataclass
from functools import partial
import networkx as nx

import pyro
from pyro.contrib.autoname import scope, name_count
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints

from .torch_utils import ConstrainedParameter
from .nodes import (
    NonTerminalNode, TerminalNode, Node,
    AndNode, OrNode, GeometricSetNode, IndependentSetNode,
    RepeatingObjectSetNode
)
from .rules import ProductionRule
from .scene_generative_program import SceneGenerativeProgram
from .distributions import LeftSidedRepeatingOnesDist


def get_tree_root(tree):
    # Warning: will infinite loop if this isn't a tree.
    # I don't check...
    root_node = list(tree.nodes)[0]
    while len(list(tree.predecessors(root_node))) > 0:
        root_node = tree.predecessors(root_node)[0]
    return root_node


class SpatialSceneGrammar():
    '''
    Manages a scene grammar that produces scene trees by composition
    of subclasses of the node types in this repo.
    '''

    def __init__(self, root_node_type, root_node_tf, do_sanity_checks=True):
        ''' Given a root node type and its tf, prepares this grammar for use. '''
        self.root_node_type = root_node_type
        self.root_node_tf = root_node_tf
        self.do_sanity_checks = do_sanity_checks

    def forward(self, params=None):
        # Samples a tree, ensuring our stored parameters get substituted
        # into every node that is generated.
        if params is None:
            params = self.get_default_param_dict()

        scene_tree = SceneTree()
        root_node = self._spawn_node_with_our_params(self.root_node_type, params)
        root_node.instantiate(self.root_node_instantiation_dict)
        scene_tree.add_node(root_node)
        return self._generate_from_node_recursive(scene_tree, root_node, params)
