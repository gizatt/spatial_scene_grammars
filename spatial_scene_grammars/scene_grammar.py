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
from .nodes import (Node, TerminalNode,
    AndNode, OrNode, GeometricSetNode
)
from .rules import ProductionRule


def get_tree_root(tree):
    # Warning: will infinite loop if this isn't a tree.
    # I don't check...
    root_node = list(tree.nodes)[0]
    while len(list(tree.predecessors(root_node))) > 0:
        root_node = tree.predecessors(root_node)[0]
    return root_node


class SceneTree(nx.DiGraph):

    def get_observed_nodes(self):
        # Pulls out only nodes in the tree that are
        # "observable."
        return [n for n in self if n.observed is True]

    def get_root(self):
        # Assumes tree structure; might infinite loop otherwise.
        k = 0
        root = list(self.nodes)[0]
        while len(list(self.predecessors(root))) > 0:
            root = list(self.predecessors(root))[0]
            k += 1
            if (k > 10000):
                raise ValueError(">10k iters when finding root. Not a tree?")
        return root

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

    def sample_tree(self):
        tree = SceneTree()

        root = self.root_node_type(tf=self.root_node_tf)
        tree.add_node(root)
        node_queue = [root]
        while len(node_queue) > 0:
            parent = node_queue.pop(0)
            # Ask node to sample its children.
            children = parent.sample_children()
            for child in children:
                tree.add_node(child)
                tree.add_edge(parent, child)
                node_queue.append(child)
        return tree
