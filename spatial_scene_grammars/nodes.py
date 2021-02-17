from copy import deepcopy
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.contrib.autoname import scope
import torch

import pydrake
from pydrake.all import (
    CoulombFriction,
    SpatialInertia,
    UnitInertia
)

# Global name store for avoiding name collisions.
class NodeNameManager():
    def __init__(self):
        self.num_existing_by_class_name = {}
    def get_name(self, requesting_class):
        class_name = requesting_class.__class__.__name__
        if class_name not in self.num_existing_by_class_name.keys():
            self.num_existing_by_class_name[class_name] = 1
            n = 0
        else:
            n = self.num_existing_by_class_name[class_name]
            self.num_existing_by_class_name[class_name] += 1
        return class_name + "_%d" % n
node_name_store = NodeNameManager()


class Node():
    ''' Every node (symbol) in the grammar derives from this base.
    At construction time, a node will be supplied a name and a dictionary
    of attribute assignments, which are stored in the node instance.
     '''
    def __init__(self):
        self.name = node_name_store.get_name(self)
        self.instantiated = False
        super().__init__()

    def instantiate(self, derived_attributes):
        ''' Given a list of derived attributes, sets self up.
        This can include local random choices and deterministic
        setup (of things like graphics assets). '''
        with scope(prefix=self.name + "_instantiate"):
            self._instantiate_impl(derived_attributes)
        self.instantiated = True

    def _instantiate_impl(self, derived_attributes):
        raise NotImplementedError()


class TerminalNode(Node):
    ''' The leafs of a generated scene tree will be terminal nodes. '''
    pass

class NonTerminalNode(Node):
    ''' Abstract interface for nonterminal nodes, which are responsible
    for sampling a set of production rules to produce new nodes.'''

    def sample_children(self):
        ''' Samples a list of ProductionRules to enact to create children
        for this node. '''
        raise NotImplementedError()

    def get_maximal_child_list(self):
        ''' Returns a list of nodes, such that the child
        set of any sampled instance of this node will be a subset
        of that list. '''
        raise NotImplementedError()

    def instantiate_children(self, children):
        ''' Instantiates the supplied children of this node, creating
        derived attribute dictionaries for each of them and passing them
        into their instantiate_self methods. '''
        assert self.instantiated, "Node should be instantiated before instantiating children."
        with scope(prefix=self.name + "_instantiate_children"):
            child_attributes = self._instantiate_children_impl(children)
        assert len(child_attributes) == len(children)
        for child, attr in zip(children, child_attributes):
            child.instantiate(attr)
        self.instantiated = True

    def _instantiate_children_impl(self, children):
        raise NotImplementedError()


class OrNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: will choose one
    of the available children with probabilities given by the production weights.'''
    def __init__(self, child_types, production_weights):
        if len(production_weights) != len(child_types):
            raise ValueError("# of production rules and weights must match.")
        if len(production_weights) == 0:
            raise ValueError("Must have nonzero number of production rules.")
        self.child_types = child_types
        self.production_weights = production_weights
        self.production_dist = dist.Categorical(production_weights)
        super().__init__()

    def sample_children(self):
        active_rule = pyro.sample("or_sample", self.production_dist)
        return [self.child_types[active_rule]()]

    def get_maximal_child_list(self):
        return [child() for child in self.child_types]


class AndNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: will choose all
    of the available children all the time.'''
    def __init__(self, child_types):
        self.child_types = child_types
        super().__init__()

    def sample_children(self):
        return [child() for child in self.child_types]

    def get_maximal_child_list(self):
        return [child() for child in self.child_types]


class IndependentSetNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: has a list of
    children that can occur, and chooses each one as an independent
    Bernoulli choice..'''
    def __init__(self, child_types, production_probs):
        self.child_types = child_types
        self.production_probs = production_probs
        self.production_dist = dist.Bernoulli(production_probs)
        super().__init__()

    def sample_children(self):
        active_rules = pyro.sample("independent_set_sample", self.production_dist)
        return [child() for k, child in enumerate(self.child_types)
                if active_rules[k]]

    def get_maximal_child_list(self):
        return [child() for child in self.child_types]
