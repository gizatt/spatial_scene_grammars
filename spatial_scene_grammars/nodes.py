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


# TODO: Expand to exhaustive list of
# all continuous distribution constraint / support
# types. I didn't see any obvious inheritence hierarchy
# I can take advantage of to make this easier...
continuous_support_constraints = (
    torch.distributions.constraints._Real,
    torch.distributions.constraints._Interval
)
def assert_trace_sites_are_all_continuous(trace):
    for key, value in trace.nodes.items():
        if value["type"] == "sample":
            support = value["fn"].support
            assert isinstance(support, continuous_support_constraints), \
               "Sample sites in instantiate methods should only sample continuous " \
               "values: %s has support %s" % (key, str(support))

def assert_trace_sites_are_all_discrete(trace):
    for key, value in trace.nodes.items():
        if value["type"] == "sample":
            support = value["fn"].support
            assert not isinstance(support, continuous_support_constraints), \
              "Sample sites in sample_products methods should only sample discrete " \
              "values: %s has support %s" % (key, str(support))


class Node():
    ''' Every node (symbol) in the grammar derives from this base.
    At construction time, a node will be supplied a name and a dictionary
    of attribute assignments, which are stored in the node instance.
     '''
    def __init__(self):
        self.name = node_name_store.get_name(self)
        self.instantiated = False
        # Place to register physics/geometry info and
        # node tf.
        self.physics_geometry_info = None
        self.tf = None
        super().__init__()

    def instantiate(self, derived_attributes):
        ''' Given a list of derived attributes, sets self up.
        This can include local random choices and deterministic
        setup (of things like graphics assets). Traces the actual
        instantiate implementation and records identity of random
        variables sampled. '''
        self.derived_attributes = derived_attributes
        with scope(prefix=self.name + "_instantiate"):
            self.instantiate_trace = pyro.poutine.trace(
                self._instantiate_impl
            ).get_trace(derived_attributes)
            assert_trace_sites_are_all_continuous(self.instantiate_trace)
        self.instantiated = True

    def _instantiate_impl(self, derived_attributes):
        raise NotImplementedError()

    def get_instantiate_ll(self):
        assert self.instantiated
        return self.instantiate_trace.log_prob_sum()


class TerminalNode(Node):
    ''' The leafs of a generated scene tree will be terminal nodes. '''
    pass

class NonTerminalNode(Node):
    ''' Abstract interface for nonterminal nodes, which are responsible
    for sampling a set of production rules to produce new nodes.'''
    def __init__(self):
        self.children_sampled = False
        self.children_instantiated = False
        super().__init__()

    def sample_children(self):
        ''' Samples a list of ProductionRules to enact to create children
        for this node. '''
        with scope(prefix=self.name + "_choose_children"):
            self.sample_children_trace = pyro.poutine.trace(
                self._sample_children_impl
            ).get_trace()
        assert_trace_sites_are_all_discrete(self.sample_children_trace)
        self.children_sampled = True
        return self.sample_children_trace.nodes["_RETURN"]["value"]

    def _sample_children_impl(self):
        raise NotImplementedError()

    def get_children_ll(self):
        assert self.children_sampled
        return self.sample_children_trace.log_prob_sum()

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
            self.instantiate_children_trace = pyro.poutine.trace(
                self._instantiate_children_impl
            ).get_trace(children)
        assert_trace_sites_are_all_continuous(self.instantiate_children_trace)
        child_attributes = self.instantiate_children_trace.nodes["_RETURN"]["value"]
        assert len(child_attributes) == len(children)
        for child, attr in zip(children, child_attributes):
            child.instantiate(attr)
        self.children_instantiated = True

    def _instantiate_children_impl(self, children):
        raise NotImplementedError()

    def get_instantiate_children_ll(self):
        assert self.children_instantiated
        return self.instantiate_children_trace.log_prob_sum()


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

    def _sample_children_impl(self):
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

    def _sample_children_impl(self):
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

    def _sample_children_impl(self):
        active_rules = pyro.sample("independent_set_sample", self.production_dist)
        return [child() for k, child in enumerate(self.child_types)
                if active_rules[k]]

    def get_maximal_child_list(self):
        return [child() for child in self.child_types]


class GeometricSetNode(NonTerminalNode):
    ''' Convenience specialization: has a single child type that can occur,
    and chooses to repeat it according to a geometric distribution, capped at
    a total number of instantiations.'''
    def __init__(self, child_type, geometric_prob, max_repeats):
        if child_type is None or geometric_prob == 0.:
            self.child_types = []
            self.production_dist = None
            self.max_repeats = 0
        else:
            self.child_types = [child_type] * max_repeats
            self.production_dist = dist.Geometric(geometric_prob)
            self.max_repeats = torch.tensor(max_repeats, dtype=torch.int)
        super().__init__()

    def _sample_children_impl(self):
        if len(self.child_types) == 0:
            # Short circuit trivial case.
            return []
        num_active_base = pyro.sample("geometric_set_sample", self.production_dist)
        num_active = min(int(num_active_base.item()), self.max_repeats)
        return [self.child_types[0]() for k in range(num_active)]

    def get_maximal_child_list(self):
        return [child() for child in self.child_types]
