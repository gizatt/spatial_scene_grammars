from copy import deepcopy
import os
import time
import pydrake
from pydrake.all import (
    CommonSolverOption,
    MathematicalProgram,
    MakeSolver,
    GurobiSolver,
    Solve,
    SolverOptions,
    VPolytope
)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

'''
Minimal simple geometric-production-only grammar definition.

Minimal grammar definition: each node has:
- a pose x
- a type name

A node class defines:
- its child type (by name or None)
- max # of children
- the geometric stop prob p (1. = always 1 child, 0. = infinite children)
- the region in which children will be produced (uniformly at random), in the frame of the node, in the form of an axis-aligned bounding box

E.g. object groups in plane:
- Root node produces object clusters and uniform random locations inside [0, 1]^2.
- Each cluster produces up points uniformly in a 0.1-length box centered at the cluster center.
'''

from collections import namedtuple
NodeDefinition = namedtuple("NodeDefinition",
    ["child_type",   # string name of child type (or None if no productions),
     "max_children", # max # of children
     "p",            # geometric keep-going probability
     "bounds"        # Generation bounds for children relative to node x
    ])
class Node():
    '''
        x: np.array position of node. 2D in this notebook.
        type: string, type name of node.
    '''
    def __init__(self, type, x):
        self.x = x
        self.type = type

def sample_tree(grammar):
    # Given a grammar description as a dict of {node name: NodeDefinition} pairs,
    # sample a scene tree as a networkx DiGraph.
    tree = nx.DiGraph()
    root = Node("root", x=np.array([0., 0.]))
    tree.add_node(root)
    node_queue = [root]
    
    while len(node_queue) > 0:
        parent = node_queue.pop(0)
        assert parent.type in grammar.keys()
        parent_def = grammar[parent.type]
        if parent_def.child_type is None:
            continue
        n_children = min(np.random.geometric(parent_def.p), parent_def.max_children)

        for k in range(n_children):
            child_x = parent.x + np.random.uniform(*parent_def.bounds)
            child = Node(parent_def.child_type, child_x)
            tree.add_node(child)
            tree.add_edge(parent, child)
            node_queue.append(child)
    return tree

def get_observed_nodes(tree, observed_types):
    # Given a scene tree (nx.DiGraph) and a list of observed
    # node type names (list of strings), pulls out only nodes
    # in the  tree with matching type into a list of Nodes.
    return [n for n in tree if n.type in observed_types]

# Drawing utilities for trees.
def draw_tree(tree, draw_pos=True, with_labels=False, node_color_dict=None, alpha=0.5, node_size=200, **kwargs):
    # Decide a coloring for the node tpyes.
    unique_types = sorted(list(set([n.type for n in tree])))
    n_types = len(unique_types)
    cm = plt.get_cmap("viridis")
    if node_color_dict is not None:
        node_color = [node_color_dict[node] for node in tree]
    else:
        color_mapping = {unique_type: cm(float(k)/n_types) for k, unique_type in enumerate(unique_types)}
        node_color = [color_mapping[node.type] for node in tree]
    if draw_pos:
        pos={node: node.x for node in tree}
    else:
        pos=None
    nx.draw_networkx(
        tree,
        labels={node: node.type for node in tree},
        with_labels=with_labels,
        pos=pos,
        node_size=node_size,
        node_color=node_color,
        alpha=alpha,
        **kwargs
    )    
    plt.gca().set_xlim([-0.1, 1.1])
    plt.gca().set_ylim([-0.1, 1.1])

def draw_observed_nodes(nodes):
    tree = nx.Graph()
    tree.add_nodes_from(nodes)
    draw_tree(tree)