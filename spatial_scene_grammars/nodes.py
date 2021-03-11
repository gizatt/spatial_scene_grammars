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

def trace_to_observe_dict(trace):
    return {key: site["value"] for key, site in trace.nodes.items()
            if site["type"] is "sample"}


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

def assert_traces_have_same_sites(trace1, trace2):
    for key, value in trace1.nodes.items():
        if value["type"] == "sample":
            assert key in trace2.nodes.keys(), "%s not in trace2 sites" % key
    for key, value in trace2.nodes.items():
        if value["type"] == "sample":
            assert key in trace1.nodes.keys(), "%s not in trace1 sites" % key

class Node():
    ''' Every node (symbol) in the grammar derives from this base.
    At construction time, a node will be supplied a name and a dictionary
    of attribute assignments, which are stored in the node instance.
     '''
    def __init__(self, do_sanity_checks=True):
        self.name = node_name_store.get_name(self)
        self.instantiated = False
        self.do_sanity_checks = True
        # Place to register physics/geometry info and
        # node tf.
        self.physics_geometry_info = None
        self.tf = None
        super().__init__()

    @classmethod
    def get_derived_attribute_info(cls):
        # Need (unfortunately) for parsing experiments; I can't
        # conclude this information statically for all node types
        # in a grammar easily.
        # Should return a dict keyed by attribute names
        # with the expected tensor shape as the value.
        return {}

    @classmethod
    def get_local_attribute_info(cls):
        # Need (unfortunately) for parsing experiments; I can't
        # conclude this information statically for all node types
        # in a grammar easily.
        # Should return a dict keyed by attribute names
        # with the expected tensor shape as the value.
        return {}

    def get_num_local_variables(self):
        all_attr_shapes = {**self.get_derived_attribute_info(), **self.get_local_attribute_info()}
        return sum([sum(shape) for shape in all_attr_shapes.values()])

    def _sanity_check_attribute_dict(self, input_dict, expected_dict_of_shapes):
        for key, shape in expected_dict_of_shapes.items():
            assert key in input_dict.keys(), "Attribute %s not in input in class %s." % (key, self.__class__.__name__)
            got_shape = input_dict[key].shape
            assert got_shape == shape, "Attribute %s of wrong shape: %s vs expected %s in class %s." % (key, got_shape, shape, self.__class__.__name__)
        for key, value in input_dict.items():
            assert key in expected_dict_of_shapes.keys(), "Attribute %s not expected in class %s." % (key, self.__class__.__name__)
            got_shape = value.shape
            shape = expected_dict_of_shapes[key]
            assert got_shape == shape, "Attribute %s of wrong shape: %s vs expected %s in class %s." % (key, got_shape, shape, self.__class__.__name__)

    def instantiate(self, derived_attributes):
        ''' Given a list of derived attributes, sets self up.
        This can include local random choices and deterministic
        setup (of things like graphics assets). Traces the actual
        instantiate implementation and records identity of random
        variables sampled. '''

        # Sanity-checking input.
        if self.do_sanity_checks:
            self._sanity_check_attribute_dict(
                derived_attributes,
                self.get_derived_attribute_info()
            )
        self.derived_attributes = derived_attributes
        # TODO(gizatt) Is this too risky?
        for key, value in self.derived_attributes.items():
            # Try to protect from overwriting stuff we shouldn't, while
            # still allowing attributes to be updated.
            if hasattr(self, key):
                assert type(getattr(self, key)) == type(value)
            setattr(self, key, value)

        # Call implementation and record what variables were sampled.
        with scope(prefix=self.name + "_instantiate"):
            self.instantiate_trace = pyro.poutine.trace(
                self._instantiate_impl
            ).get_trace(derived_attributes)

        self.local_attributes = self.instantiate_trace.nodes["_RETURN"]["value"]
        # TODO(gizatt) Is this too risky?
        for key, value in self.local_attributes.items():
            # Try to protect from overwriting stuff we shouldn't, while
            # still allowing attributes to be updated.
            if hasattr(self, key):
                assert type(getattr(self, key)) == type(value)
            setattr(self, key, value)
        if self.do_sanity_checks:
            # Sanity-check the local variables are continuous
            # and match the expected set of attributes.
            assert_trace_sites_are_all_continuous(self.instantiate_trace)
            self._sanity_check_attribute_dict(self.local_attributes, self.get_local_attribute_info())
        self.instantiated = True

    def conditioned_instantiate(self, derived_attributes, local_attributes):
        '''
        Given both derived and local attributes,
        sets up this node (including trace) as if that is what was sampled.
        '''
        if self.do_sanity_checks:
            self._sanity_check_attribute_dict(
                derived_attributes,
                self.get_derived_attribute_info())
            self._sanity_check_attribute_dict(
                local_attributes,
                self.get_local_attribute_info())
        self.derived_attributes = derived_attributes
        self.local_attributes = local_attributes
        # Block outside scope so scope gets applied.
        with pyro.poutine.block():
            with scope(prefix=self.name + "_instantiate"):
                self.conditioned_instantiate_trace = pyro.poutine.trace(
                    self._conditioned_instantiate_impl
                ).get_trace(derived_attributes, local_attributes)
        # Also run a forward pass using those sites to generate a forward
        # trace, so we can evaluate score if requested, and have this one
        # be the "externally visible" version.
        # Condition needs to be outside the scope, for some reason, for the
        # conditioning to stick...?
        with pyro.poutine.condition(data=trace_to_observe_dict(self.conditioned_instantiate_trace)):
            with scope(prefix=self.name + "_instantiate"):
                self.instantiate_trace = pyro.poutine.trace(
                    self._instantiate_impl
                ).get_trace(self.derived_attributes)

        if self.do_sanity_checks:
            # Those two traces should be identical in terms of sample sites.
            assert_traces_have_same_sites(
                self.instantiate_trace,
                self.conditioned_instantiate_trace
            )
            assert_trace_sites_are_all_continuous(self.instantiate_trace)
            assert_trace_sites_are_all_continuous(self.conditioned_instantiate_trace)
        self.instantiated = True

    def _conditioned_instantiate_impl(self, derived_attributes, local_attributes):
        if len(self.get_local_attribute_info().items()) > 0:
            raise NotImplementedError("Override _conditioned_instantiate_impl.")

    def _instantiate_impl(self, derived_attributes):
        return {}

    def get_instantiate_ll(self):
        assert self.instantiated
        return self.instantiate_trace.log_prob_sum()

    def get_all_attributes(self):
        assert self.instantiated
        return {**self.derived_attributes, **self.local_attributes}

    def get_all_attributes_as_vector(self):
        assert self.instantiated
        attr_dict = self.get_all_attributes()
        elements = [v.flatten() for v in attr_dict.values()]
        if len(elements) > 0:
            return torch.cat(elements)
        else:
            return torch.empty((0,))


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
        ''' Samples a list of children for this node. '''
        with scope(prefix=self.name + "_choose_children"):
            self.sample_children_trace = pyro.poutine.trace(
                self._sample_children_impl
            ).get_trace()
        assert_trace_sites_are_all_discrete(self.sample_children_trace)
        self.children_sampled = True
        return self.sample_children_trace.nodes["_RETURN"]["value"]

    def conditioned_sample_children(self, children):
        ''' Given a list of children, recreate my own sample trace as if
        those children were chosen. '''

        # Block outside scope so scope gets applied.
        with pyro.poutine.block():
            with scope(prefix=self.name + "_choose_children"):
                self.conditioned_sample_children_trace = pyro.poutine.trace(
                    self._conditioned_sample_children_impl
                ).get_trace(children)
        # Also run a forward pass using those sites to generate a forward
        # trace, so we can evaluate score if requested, and have this one
        # be the "externally visible" version.
        # Condition needs to be outside the scope, for some reason, for the
        # conditioning to stick...?
        with pyro.poutine.condition(data=trace_to_observe_dict(self.conditioned_sample_children_trace)):
            with scope(prefix=self.name + "_choose_children"):
                self.sample_children_trace = pyro.poutine.trace(
                    self._sample_children_impl
                ).get_trace()

        if self.do_sanity_checks:
            assert_trace_sites_are_all_discrete(self.conditioned_sample_children_trace)
            assert_trace_sites_are_all_discrete(self.sample_children_trace)
            assert_traces_have_same_sites(
                self.conditioned_sample_children_trace,
                self.sample_children_trace
            )
        self.children_sampled = True

    def _conditioned_sample_children_impl(self, children):
        if len(children) > 0:
            raise NotImplementedError("Override _conditioned_sample_children_impl.")

    def _sample_children_impl(self):
        raise NotImplementedError("Override _sample_children_impl.")

    def get_children_ll(self):
        assert self.children_sampled
        return self.sample_children_trace.log_prob_sum()

    def get_maximal_child_list(self):
        ''' Returns a list of nodes, such that the child
        set of any sampled instance of this node will be a subset
        of that list. '''
        raise NotImplementedError("Override get_maximal_child_list.")

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

    def conditioned_instantiate_children(self, children):
        for child in children:
            assert child.instantiated
        assert self.instantiated

        # Block outside scope so scope gets applied.
        with pyro.poutine.block():
            with scope(prefix=self.name + "_instantiate_children"):
                self.conditioned_instantiate_children_trace = pyro.poutine.trace(
                    self._conditioned_instantiate_children_impl
                ).get_trace(children)
        # Also run a forward pass using those sites to generate a forward
        # trace, so we can evaluate score if requested, and have this one
        # be the "externally visible" version.
        # Condition needs to be outside the scope, for some reason, for the
        # conditioning to stick...?
        with pyro.poutine.condition(data=trace_to_observe_dict(self.conditioned_instantiate_children_trace)):
            with scope(prefix=self.name + "_instantiate_children"):
                self.instantiate_children_trace = pyro.poutine.trace(
                    self._instantiate_children_impl
                ).get_trace(children)
        
        if self.do_sanity_checks:
            assert_trace_sites_are_all_continuous(self.conditioned_instantiate_children_trace)
            assert_trace_sites_are_all_continuous(self.instantiate_children_trace)
            # Those two traces should be identical in terms of sample sites.
            assert_traces_have_same_sites(
                self.instantiate_children_trace,
                self.conditioned_instantiate_children_trace
            )
            
        self.children_instantiated = True

    def _conditioned_instantiate_children_impl(self, children):
        # Given instantiated child set, provide proposals
        # for my sample sites.
        for child in children:
            if len(child.get_derived_attribute_info().items()) > 0:
                raise NotImplementedError("Override _conditioned_instantiate_children_impl.")

    def _instantiate_children_impl(self, children):
        if len(children) > 0:
            for child in children:
                if len(child.get_derived_attribute_info().items()) > 0:
                    raise NotImplementedError("Override _instantiate_children_impl.")

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

    def _conditioned_sample_children_impl(self, children):
        assert len(children) == 1
        production_weights = torch.zeros(len(self.child_types))
        child = children[0]
        inds = [k for k in range(len(self.child_types)) if self.child_types[k] == type(child)]
        if len(inds) > 1:
            # In this case, the child matches multiple of the production types.
            # This is legal under the grammar, but is strange in the sense that
            # the two OR rules lead to functionally identically trees, making the
            # grammar ambiguous.
            # In this case, we could sample "or_sample" using the OR production weights
            # for the feasible children? I need to understand downstream users of
            # this method better to know if nondeterminism here is OK.
            raise NotImplementedError("Multiple potential children.")
        for k in inds:
            production_weights[k] = self.production_weights[k]
        pyro.sample("or_sample", dist.Categorical(production_weights))

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

    def _conditioned_sample_children_impl(self, children):
        # No sample sites, so no work necessary.
        pass

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

    def _conditioned_sample_children_impl(self, children):
        # Recover if each entry is active.
        activations = torch.zeros(len(self.child_types))
        for child_type in self.child_types:
            matching_children_inds = [
                k for (k, child) in enumerate(children)
                if type(child) == child_type
            ]
            if len(matching_children_inds) > 1:
                # Same case as in the OR rule -- this is legal under the grammar but would
                # only happen under a weird / ambiguous grammar, I think.
                raise NotImplementedError("Child could come from multiple productions.")
            elif len(matching_children_inds) == 1:
                activations[matching_children_inds[0]] = 1
        pyro.sample("independent_set_sample", dist.Bernoulli(activations))

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

    def _conditioned_sample_children_impl(self, children):
        for child in children:
            # All our child types should be the same.
            assert type(child) == self.child_type[0]
        num_active = len(children)
        assert num_active <= self.max_repeats
        choices = torch.zeros(self.max_repeats + 1)
        choices[num_active] = 1.
        # This can't be exact, as _sample_children_impl loses information.
        # It should be reimplemented as a Categorical with geometric densities
        # at everything but the last bin, and the sum of the truncated tail
        # at the last bin.
        pyro.sample("geometric_set_sample", dist.Categorical(choices))

    def get_maximal_child_list(self):
        return [child() for child in self.child_types]
