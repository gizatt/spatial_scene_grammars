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
    # Minor additional bookkeeping on top of a digraph:
    # - Keeps track of node - parent rule correspondence
    # - Accessors for observed nodes, nodes by type, + tree root
    # - Score calculation
    # - Optionally can store the Pyro trace of generation of this
    #   tree, for internal use in sampling routines.

    def __init__(self, **kwargs):
        self._trace = None
        super().__init__(self, **kwargs)

    @property
    def trace(self):
        assert self._trace is not None, "Trace not recorded for this tree."
        return self._trace
    @trace.setter
    def trace(self, trace):
        self._trace = trace

    def get_parent(self, child):
        parents = list(self.predecessors(child))
        if len(parents) == 0:
            return None
        else:
            assert len(parents) == 1
            return parents[0]

    def get_children(self, parent):
        assert parent in self.nodes
        return sorted(list(self.successors(parent)), key=lambda x: x.rule_k)

    def get_rule_for_child(self, parent, child):
        if isinstance(parent, GeometricSetNode):
            return parent.rule
        elif isinstance(parent, (AndNode, OrNode)):
            return parent.rules[child.rule_k]
        else:
            raise ValueError("Parent %s of child %s is of bad type for getting rules." % (parent, child))

    def get_children_and_rules(self, parent):
        ''' Return the child nodes and their corresponding
        rules (selected from the parent node rule list according
        to the parent type and child rule_k index) as matching lists. '''
        children = self.get_children(parent)
        # Get child rule list.
        if isinstance(parent, GeometricSetNode):
            rules = [parent.rule for k in range(len(children))]
        elif isinstance(parent, AndNode):
            rules = parent.rules
        elif isinstance(parent, OrNode):
            assert len(children) == 1
            rules = [parent.rules[children[0].rule_k]]
        elif isinstance(parent, TerminalNode):
            rules = []
        else:
            raise ValueError("Unexpected node type: ", type(parent))
        return children, rules

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

    def find_nodes_by_type(self, node_type):
        return [n for n in self.nodes if isinstance(n, node_type)]

    def score(self, include_discrete=True, include_continuous=True, verbose=False):
        # Compute total score of parents and children.
        total_score = torch.tensor(0.)
        for node in self.nodes:
            children = list(self.successors(node))
            if include_discrete:
                contrib = node.score_child_set(children)
                total_score = total_score + contrib
                if verbose:
                    print(node, ": ", contrib.item())
            if include_continuous:
                for child in children:
                    assert child.rule_k is not None
                    assert child.rule_k >= 0
                    if isinstance(node, (AndNode, OrNode)):
                        rule = node.rules[child.rule_k]
                    elif isinstance(node, GeometricSetNode):
                        rule = node.rule
                    else:
                        raise ValueError("Unknown node type has children.")
                    contrib = rule.score_child(node, child, verbose=verbose)
                    total_score = total_score + contrib
                    if verbose:
                        print(node, " -> ", child, ": ", contrib.item())
        return total_score


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

        def do_sampling():
            root = self.root_node_type(tf=self.root_node_tf)
            tree.add_node(root)
            node_queue = [root]
            k = 0
            while len(node_queue) > 0:
                parent = node_queue.pop(0)
                # Ask node to sample its children.
                with scope(prefix="%d" % k):
                    children = parent.sample_children()
                k += 1
                for child in children:
                    tree.add_node(child)
                    tree.add_edge(parent, child)
                    node_queue.append(child)

        tree.trace = pyro.poutine.trace(do_sampling).get_trace()
        return tree

    def make_super_tree(self, max_recursion_depth=15):
        # Forms a graph of nodes for which any actual sampled tree would be a subgraph.
        # (TF's are all set to 0.)
        tree = SceneTree()

        root = self.root_node_type(tf = torch.eye(4))
        # Label recursion depth in on nodes of super tree.
        root._recursion_depth = 0
        tree.add_node(root)
        node_queue = [root]
        while len(node_queue) > 0:
            parent = node_queue.pop(0)
            if isinstance(parent, (AndNode, OrNode)):
                maximal_children = [r.child_type for r in parent.rules]
            elif isinstance(parent, GeometricSetNode):
                maximal_children = [parent.rule.child_type for k in range(parent.max_children)]
            elif isinstance(parent, TerminalNode):
                maximal_children = []
            else:
                raise ValueError(type(parent))

            for k, child_type in enumerate(maximal_children):
                child = child_type(tf = torch.eye(4))
                child.rule_k = k
                child._recursion_depth = parent._recursion_depth + 1
                if child._recursion_depth <= max_recursion_depth:
                    tree.add_node(child)
                    tree.add_edge(parent, child)
                    node_queue.append(child)
        return tree