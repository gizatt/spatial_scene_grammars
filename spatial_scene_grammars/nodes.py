from copy import deepcopy
from collections import namedtuple
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.distributions.torch_distribution import (
    TorchDistribution, TorchDistributionMixin
)
from pyro.contrib.autoname import scope
import torch
from torch.distributions import constraints

from .distributions import VectorCappedGeometricDist, LeftSidedRepeatingOnesDist
from .torch_utils import ConstrainedParameter

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


class NodeVariableInfo:
    def __init__(self, shape, support=None):
        self.shape = shape
        self.support = support


# TODO: Expand to exhaustive list of
# all continuous distribution constraint / support
# types. I didn't see any obvious inheritence hierarchy
# I can take advantage of to make this easier...
continuous_support_constraints = (
    torch.distributions.constraints._Real,
    torch.distributions.constraints._Interval
)
def assert_dists_are_all_continuous(dict_of_dists):
    for key, value in dict_of_dists.items():
        assert isinstance(value, (TorchDistribution, TorchDistributionMixin)), \
            "Variable %s distribution is type %s, not a pyro distribution type." % (key, type(value))
        support = value.support
        assert isinstance(support, continuous_support_constraints), \
            "Distribution of %s (type %s) does not have continuous support." % (key, type(value))


class Node():
    '''
    Every node (symbol) in the grammar derives from this base.
    '''
    def __init__(self, parameters={}, do_sanity_checks=True):
        self.name = node_name_store.get_name(self)
        self.instantiated = False
        self.do_sanity_checks = True
        # Place to register physics/geometry info and
        # node tf.
        self.physics_geometry_info = None
        self.tf = None
        if do_sanity_checks:
            # Check that passed parameters are the right size with
            # the right entries compared to our registered defaults.
            expected_param_dict = self.get_default_parameters()
            assert set(expected_param_dict.keys()) == set(parameters.keys())
            for key, value in expected_param_dict.items():
                assert value.get_value().shape == parameters[key].get_value().shape
        self.parameters = parameters
        for name, value in self.parameters.items():
            assert not hasattr(self, name), "Parameter name %s conflicts with existing class value." % name
            setattr(self, name, value)
        super().__init__()

    @classmethod
    def init_with_default_parameters(cls):
        return cls(parameters=cls.get_default_parameters())

    @classmethod
    def get_default_parameters(cls):
        ''' Provide a dictionary of default values of parameters, keyed by
        their names, as ConstrainedParameter objects. They will be wrapped
        tracked by the containing grammar, and stored both in self.parameters
        and as members of the class (i.e. parameter named "foo" will be at self.foo)
        when the class is constructed. '''
        return {}

    def get_parameters(self):
        return self.parameters

    @classmethod
    def get_derived_variable_info(cls):
        ''' 
        Provide a dictionary of derived variables that instances of this
        class will expect to receive at instantiation time.
        Keys are variable names; values are NodeVariableInfo structs
        encoding expected distribution and shape.
        '''
        return {}

    @classmethod
    def get_local_variable_info(cls):
        '''
        Provide a dictionary of local variables that instances of this
        class will produce during instantiation.
        Keys are variable names; values are NodeVariableInfo structs
        encoding expected distribution and shape.
        '''
        return {}

    @classmethod
    def get_num_continuous_variables(self):
        '''
        Returns the number of continuous variables this class has, which
        is the sum of the products of the shapes of all of the derived
        and local variables.
        '''
        all_attr_infos = {**self.get_derived_variable_info(), **self.get_local_variable_info()}
        return sum([np.prod(info.shape) for info in all_attr_infos.values()])

    def _sanity_check_variable_dict(self, input_dict, expected_dict_of_infos):
        '''
        Given a dictionary of distributions keyed by variable names,
        checks that the included keys and corresponding shapes exactly match
        a dictionary of expected dist types + shapes keyed by the same variable names.
        '''
        for key, info in expected_dict_of_infos.items():
            assert key in input_dict.keys(), "Variable %s not in input in class %s." % (key, self.__class__.__name__)
            value = input_dict[key]
            if isinstance(value, torch.Tensor):
                got_shape = value.shape
            else:
                got_shape = input_dict[key].shape()
            assert got_shape == info.shape, "Variable %s of wrong shape: %s vs expected %s in class %s." % (key, got_shape, info.shape, self.__class__.__name__)
        for key, value in input_dict.items():
            assert key in expected_dict_of_infos.keys(), "Variable %s not expected in class %s." % (key, self.__class__.__name__)
            if isinstance(value, torch.Tensor):
                got_shape = value.shape
            else:
                got_shape = input_dict[key].shape()
            shape = expected_dict_of_infos[key].shape
            assert got_shape == shape, "Variable %s of wrong shape: %s vs expected %s in class %s." % (key, got_shape, shape, self.__class__.__name__)

    def copy_attr_dict_to_self(self, attr_dict):
        for key, value in attr_dict.items():
            if hasattr(self, key):
                # Make sure we're only updating the value.
                curr_value = getattr(self, key)
                assert isinstance(curr_value, torch.Tensor), "Class already has non-tensor attribute named %s: %s." % (key, curr_value)
                assert getattr(self, key).shape == value.shape, \
                    "Class already has an attribute named %s with different shape: %s (current) vs %s (new)." % (
                        key, str(curr_value.shape), value.shape)
            setattr(self, key, value)

    def _sample_continuous_variables_from_dict(self, dict_of_dists, observed_dict=None):
        '''
        For each key, dist pair in the dict, sample from the distribution;
        form an output dict of keys and sampled values; and for convenience,
        assign the resulting value to ourselves using the key name.
        (TODO: Is the convenience of this last one worth how messy it is?)
        '''
        out_dict = {}
        for key, dist in dict_of_dists.items():
            if observed_dict is not None and key in observed_dict.keys():
                obs = observed_dict[key]
            else:
                obs = None
            value = pyro.sample(key, dist, obs=obs)
            out_dict[key] = value
        return out_dict

    def instantiate(self, derived_variable_distributions,
                    observed_derived_variables=None,
                    observed_local_variables=None,
                    local_variable_distributions_override=None):
        '''
        Given a dictionary of derived variable distributions (matching
        the expected derived variables from `get_derived_variable_info`):
        1) Samples concrete values for the derived variables, storing
        them as correspondingly-named attributes of the class.
        2) Uses the concrete values to calculate distributions over the
        local variables (implemented by get_local_variable_dists; the local variable
        distribution set should match the shapes from `get_local_variable_info`).
        3) Samples the local variables and stores them as
        correspondingly-named attributes of the class.

        If local variable distributions are supplied, they'll be used
        rather than the class's get_local_variable_dists.

        The indirection here allows subclass implementation of
        `get_local_variable_dists` to focus on just spitting out the right
        distributions; actual sampling and distribution bookkeeping is done here.
        '''
        # Sanity-checking input.
        if self.do_sanity_checks:
            assert_dists_are_all_continuous(derived_variable_distributions)
            self._sanity_check_variable_dict(
                derived_variable_distributions,
                self.get_derived_variable_info()
            )
        self.derived_variable_distributions = derived_variable_distributions

        # Sample concrete values for the derived variables.
        with scope(prefix=self.name + "_sample_derived"):
            self.derived_variable_values = self._sample_continuous_variables_from_dict(
                self.derived_variable_distributions,
                observed_derived_variables
            )
        # TODO(gizatt) Is this too risky?
        self.copy_attr_dict_to_self(self.derived_variable_values)

        if local_variable_distributions_override is None:
            # Call instantiate implementation.
            local_variable_distributions = self.get_local_variable_dists(self.derived_variable_values)
        else:
            local_variable_distributions = local_variable_distributions_override

        # Sanity-check output.
        if self.do_sanity_checks:
            assert_dists_are_all_continuous(local_variable_distributions)
            self._sanity_check_variable_dict(
                local_variable_distributions,
                self.get_local_variable_info()
            )
        self.local_variable_distributions = local_variable_distributions

        # Sample concrete values for the local variables.
        with scope(prefix=self.name + "_sample_local"):
            self.local_variable_values = self._sample_continuous_variables_from_dict(
                self.local_variable_distributions,
                observed_local_variables
            )
        # TODO(gizatt) Is this too risky?
        self.copy_attr_dict_to_self(self.local_variable_values)

        self.instantiated = True

    def get_local_variable_dists(self, derived_variable_values):
        return {}

    @staticmethod
    def get_variable_ll_given_dicts(value_dict, dist_dict):
        total_ll = torch.tensor(0.)
        for key, value in value_dict.items():
            total_ll = total_ll + dist_dict[key].log_prob(value).sum()
        return total_ll

    def get_derived_variable_ll(self):
        assert self.instantiated
        return self.get_variable_ll_given_dicts(
            self.derived_variable_values, self.derived_variable_distributions
        )

    def get_local_variable_ll(self):
        assert self.instantiated
        return self.get_variable_ll_given_dicts(
            self.local_variable_values, self.local_variable_distributions
        )

    def get_continuous_variable_ll(self):
        assert self.instantiated
        return self.get_derived_variable_ll() + self.get_local_variable_ll()
    
    def get_derived_variable_values(self):
        assert self.instantiated
        return self.derived_variable_values

    def get_local_variable_values(self):
        assert self.instantiated
        return self.local_variable_values

    def get_all_continuous_variable_values(self):
        assert self.instantiated
        return {**self.derived_variable_values, **self.local_variable_values}

    @staticmethod
    def _flatten_tensor_dict(var_dict):
        elements = [v.flatten() for v in var_dict.values()]
        if len(elements) > 0:
            return torch.cat(elements)
        else:
            return torch.empty((0,))

    def get_derived_variables_as_vector(self):
        assert self.instantiated
        return self._flatten_tensor_dict(self.derived_variable_values)
        
    def get_local_variables_as_vector(self):
        assert self.instantiated
        return self._flatten_tensor_dict(self.local_variable_values)

    def get_all_continuous_variables_as_vector(self):
        assert self.instantiated
        return self._flatten_tensor_dict(self.get_all_continuous_variable_values())


class TerminalNode(Node):
    ''' The leafs of a generated scene tree will be terminal nodes. '''
    pass

class NonTerminalNode(Node):
    ''' Abstract interface for nonterminal nodes, which are responsible
    for sampling a set of production rules to produce new nodes.'''
    def __init__(self, **kwargs):
        self.children_sampled = False
        self.children_instantiated = False
        super().__init__(**kwargs)

    def get_maximal_child_type_list(self):
        ''' Returns a list of node types, such that the child
        set of any sampled instance of this node will be a subset
        of that list. '''
        raise NotImplementedError("Override get_maximal_child_type_list.")

    def sample_children(self, observed_child_types=None,
                        child_inclusion_dist_override=None):
        ''' Samples a list of child types for this node. '''
        # Get distribution over children.
        child_type_list = self.get_maximal_child_type_list()
        if child_inclusion_dist_override is not None:
            child_inclusion_dist = child_inclusion_dist_override
        else:
            child_inclusion_dist = self._sample_children_impl()
        if self.do_sanity_checks:
            assert child_inclusion_dist.event_shape == (len(child_type_list),), \
                "Inclusion dist %s has event_shape %s vs child list %s" % (
                    child_inclusion_dist, child_inclusion_dist.event_shape,
                    child_type_list
                )
        self.child_inclusion_dist = child_inclusion_dist

        # Sample an included child set.
        if observed_child_types is not None:
            observed_indicator = self.get_child_indicator_vector(observed_child_types)
        else:
            observed_indicator = None
        with scope(prefix=self.name + "_choose_children"):
            self.child_inclusion_values = pyro.sample(
                "sample_children_inclusion", self.child_inclusion_dist,
                obs=observed_indicator)
        # TODO: This doesn't support batching as-is.
        child_set = [
            child_type_list[k]
            for k, active in enumerate(self.child_inclusion_values.bool())
            if active
        ]
        self.children_sampled = True
        return child_set

    def _sample_children_impl(self):
        ''' Should return a distribution that, when sampled, produces
        a binary vector indicating whether each of the children in
        the maximal child list is included in the output. '''
        raise NotImplementedError("Override _sample_children_impl().")

    def get_children_ll(self):
        assert self.children_sampled
        return self.child_inclusion_dist.log_prob(self.child_inclusion_values).sum()

    def get_child_indices_into_maximal_child_list(self, observed_child_types):
        ''' Returns, for each child in children, the index into the 
        maximal child set for this node that that child was produced by.
        Errors in ambiguous cases; assumes this child set is a feasible
        full child set for this node type. '''
        raise NotImplementedError('Override get_child_indices_into_maximal_child_list().')

    def get_child_indicator_vector(self, observed_child_types):
        ''' Returns an indicator vector of which children in the
        maximal_child_list would be active for this observed child set. '''
        for t in observed_child_types:
            assert issubclass(t, Node), "Make sure you're passing node types and not nodes into this function."
        inds = self.get_child_indices_into_maximal_child_list(observed_child_types)
        ind_vec = torch.zeros(len(self.get_maximal_child_type_list()))
        ind_vec[inds.long()] = 1.
        return ind_vec

    def get_derived_variable_dists_for_children(self, child_types):
        if len(child_types) > 0:
            for child in child_types:
                if len(child.get_derived_variable_info().items()) > 0:
                    raise NotImplementedError("Override get_derived_variable_dists_for_children().")


class OrNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: will choose one
    of the available children with probabilities given by the production weights.'''

    def __init__(self, child_types, production_weights, **kwargs):
        if len(production_weights) != len(child_types):
            raise ValueError("# of production rules and weights must match.")
        if len(production_weights) == 0:
            raise ValueError("Must have nonzero number of production rules.")
        self.child_types = child_types
        self.production_weights = production_weights
        self.production_dist = dist.OneHotCategorical(probs=production_weights)
        super().__init__(**kwargs)

    def _sample_children_impl(self):
        return self.production_dist

    def get_maximal_child_type_list(self):
        return self.child_types

    def get_child_indices_into_maximal_child_list(self, observed_child_types):
        assert len(observed_child_types) == 1
        production_weights = torch.zeros(len(self.child_types))
        child = observed_child_types[0]
        inds = [k for k in range(len(self.child_types)) if self.child_types[k] == child]
        if len(inds) > 1:
            # In this case, the child matches multiple of the production types.
            # This is legal under the grammar, but is strange in the sense that
            # the two OR rules lead to functionally identically trees, making the
            # grammar ambiguous.
            # In this case, we could sample "or_sample" using the OR production weights
            # for the feasible children? I need to understand downstream users of
            # this method better to know if nondeterminism here is OK.
            raise NotImplementedError("Multiple potential children.")
        return torch.tensor(inds)

class AndNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: will choose all
    of the available children all the time.'''
    def __init__(self, child_types, **kwargs):
        self.child_types = child_types
        self.production_dist = dist.Bernoulli(torch.ones(len(self.child_types))).to_event(1)
        super().__init__(**kwargs)

    def _sample_children_impl(self):
        return self.production_dist

    def get_maximal_child_type_list(self):
        return [child for child in self.child_types]

    def get_child_indices_into_maximal_child_list(self, observed_child_types):
        assert len(observed_child_types) == len(self.child_types)
        return torch.tensor(range(len(self.child_types)))

class IndependentSetNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: has a list of
    children that can occur, and chooses each one as an independent
    Bernoulli choice..'''
    def __init__(self, child_types, production_probs, **kwargs):
        self.child_types = child_types
        self.production_probs = production_probs
        self.production_dist = dist.Bernoulli(production_probs).to_event(1)
        super().__init__(**kwargs)

    def _sample_children_impl(self):
        return self.production_dist

    def get_maximal_child_type_list(self):
        return self.child_types

    def get_child_indices_into_maximal_child_list(self, observed_child_types):
        inds = []
        for k, child_type in enumerate(self.child_types):
            matching_children_inds = [
                k for (k, observed_child_type) in enumerate(observed_child_types)
                if child_type == observed_child_type
            ]
            if len(matching_children_inds) > 1:
                # Same case as in the OR rule -- this is legal under the grammar but would
                # only happen under a weird / ambiguous grammar, I think.
                raise NotImplementedError("Child could come from multiple productions.")
            elif len(matching_children_inds) == 1:
                inds.append(k)
        return torch.tensor(inds, dtype=torch.int64)


class GeometricSetNode(NonTerminalNode):
    ''' Convenience specialization: has a single child type that can occur,
    and chooses to repeat it according to a geometric distribution, capped at
    a total number of instantiations.'''
    def __init__(self, child_type, geometric_prob, max_repeats, min_repeats=0, **kwargs):
        assert max_repeats > 0
        assert child_type is not None
        self.child_types = [child_type] * max_repeats
        self.min_repeats = min_repeats
        self.max_repeats = max_repeats
        self.geometric_prob = geometric_prob
        if geometric_prob == 0.:
            # Set up placeholder distribution that will never produce
            # children.
            self.production_dist = dist.Delta(torch.zeros(max_repeats)).to_event(1)
        else:
            self.production_dist = VectorCappedGeometricDist(geometric_prob, max_repeats)
        super().__init__(**kwargs)

    def _sample_children_impl(self):
        return self.production_dist

    def get_maximal_child_type_list(self):
        return self.child_types

    def get_child_indices_into_maximal_child_list(self, observed_child_types):
        for observed_child_type in observed_child_types:
            # All our child types should be the same.
            assert observed_child_type == self.child_types[0]
        num_active = len(observed_child_types)
        assert num_active <= self.max_repeats
        return torch.tensor(range(num_active))


class RepeatingObjectSetNode(NonTerminalNode):
    ''' Convenience specialization: has a single child type that can occur,
    and chooses to repeat it according to a specified histogram distribution.'''
    def __init__(self, child_type, repeat_probs, **kwargs):
        assert child_type is not None
        max_repeats = len(repeat_probs) - 1
        self.child_types = [child_type] * max_repeats
        self.repeat_probs = repeat_probs
        self.production_dist = LeftSidedRepeatingOnesDist(repeat_probs)
        super().__init__(**kwargs)

    def _sample_children_impl(self):
        return self.production_dist

    def get_maximal_child_type_list(self):
        return self.child_types

    def get_child_indices_into_maximal_child_list(self, observed_child_types):
        for observed_child_type in observed_child_types:
            # All our child types should be the same.
            assert observed_child_type == self.child_types[0]
        num_active = len(observed_child_types)
        return torch.tensor(range(num_active))

'''

'''