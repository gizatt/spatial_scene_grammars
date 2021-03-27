from copy import deepcopy
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.distributions.torch_distribution import (
    TorchDistribution, TorchDistributionMixin
)
from pyro.contrib.autoname import scope
import torch

from .distributions import VectorCappedGeometricDist

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
    def get_derived_variable_info(cls):
        ''' 
        Provide a dictionary of derived variables that instances of this
        class will expect to receive at instantiation time.
        Keys are variable names; values are the expected tensor shapes.
        '''
        return {}

    @classmethod
    def get_local_variable_info(cls):
        '''
        Provide a dictionary of local variables that instances of this
        class will produce during instantiation. Keys are attribute names;
        values are the expected tensor shapes.
        '''
        return {}

    @classmethod
    def get_num_continuous_variables(self):
        '''
        Returns the number of continuous variables this class has, which
        is the sum of the products of the shapes of all of the derived
        and local variables.
        '''
        all_attr_shapes = {**self.get_derived_variable_info(), **self.get_local_variable_info()}
        return sum([np.prod(shape) for shape in all_attr_shapes.values()])

    def _sanity_check_variable_dict(self, input_dict, expected_dict_of_shapes):
        '''
        Given a dictionary of distributions keyed by variable names,
        checks that the included keys and corresponding shapes exactly match
        a dictionary of expected tensor shapes keyed by the same variable names.
        '''
        for key, shape in expected_dict_of_shapes.items():
            assert key in input_dict.keys(), "Variable %s not in input in class %s." % (key, self.__class__.__name__)
            value = input_dict[key]
            if isinstance(value, torch.Tensor):
                got_shape = value.shape
            else:
                got_shape = input_dict[key].shape()
            assert got_shape == shape, "Variable %s of wrong shape: %s vs expected %s in class %s." % (key, got_shape, shape, self.__class__.__name__)
        for key, value in input_dict.items():
            assert key in expected_dict_of_shapes.keys(), "Variable %s not expected in class %s." % (key, self.__class__.__name__)
            if isinstance(value, torch.Tensor):
                got_shape = value.shape
            else:
                got_shape = input_dict[key].shape()
            shape = expected_dict_of_shapes[key]
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

    def _sample_continuous_variables_from_dict(self, dict_of_dists):
        '''
        For each key, dist pair in the dict, sample from the distribution;
        form an output dict of keys and sampled values; and for convenience,
        assign the resulting value to ourselves using the key name.
        (TODO: Is the convenience of this last one worth how messy it is?)
        '''
        out_dict = {}
        for key, dist in dict_of_dists.items():
            value = pyro.sample("key", dist)
            out_dict[key] = value
        return out_dict

    def instantiate(self, derived_variable_distributions):
        '''
        Given a dictionary of derived variable distributions (matching
        the expected derived variables from `get_derived_variable_info`):
        1) Samples concrete values for the derived variables, storing
        them as correspondingly-named attributes of the class.
        2) Uses the concrete values to calculate distributions over the
        local variables (implemented by _instantiate_impl; the local variable
        distribution set should match the shapes from `get_local_variable_info`).
        3) Samples the local variables and stores them as
        correspondingly-named attributes of the class.

        The indirection here allows subclass implementation of `_instantiate_impl`
        to focus on just spitting out the right distributions; actual sampling
        and distribution bookkeeping is done here.
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
                self.derived_variable_distributions
            )
        # TODO(gizatt) Is this too risky?
        self.copy_attr_dict_to_self(self.derived_variable_values)
        
        # Call instantiate implementation.
        local_variable_distributions = self._instantiate_impl(self.derived_variable_values)

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
                self.local_variable_distributions
            )
        # TODO(gizatt) Is this too risky?
        self.copy_attr_dict_to_self(self.local_variable_values)

        self.instantiated = True

    def _instantiate_impl(self, derived_variable_values):
        return {}

    def conditioned_instantiate(self, target_derived_variable_values, target_local_variable_values):
        # Same functionality as instantiate, but sets up Delta distributions
        # around the specified derived + local variables instead of using
        # supplies distributions.
        if self.do_sanity_checks:
            self._sanity_check_variable_dict(
                target_derived_variable_values,
                self.get_derived_variable_info()
            )
            self._sanity_check_variable_dict(
                target_local_variable_values,
                self.get_local_variable_info()
            )
        self.derived_variable_distributions = {
            key: dist.Delta(value)
            for key, value in target_derived_variable_values.items()
        }
        # Sample concrete values for the derived variables.
        with scope(prefix=self.name + "_sample_derived"):
            self.derived_variable_values = self._sample_continuous_variables_from_dict(
                self.derived_variable_distributions
            )
        self.copy_attr_dict_to_self(self.derived_variable_values)
        
        self.local_variable_distributions = {
            key: dist.Delta(value)
            for key, value in target_local_variable_values.items()
        }
        # Sample concrete values for the local variables.
        with scope(prefix=self.name + "_sample_local"):
            self.local_variable_values = self._sample_continuous_variables_from_dict(
                self.local_variable_distributions
            )
        self.copy_attr_dict_to_self(self.local_variable_values)

        self.instantiated = True

    @staticmethod
    def get_variable_ll_given_dicts(value_dict, dist_dict):
        total_ll = 0.
        for key, value in value_dict.items():
            total_ll += dist_dict[key].log_prob(value).sum()
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

    def sample_children(self):
        ''' Samples a list of children for this node. '''
        # Get distribution over children.
        child_type_list = self.get_maximal_child_type_list()
        child_inclusion_dist = self._sample_children_impl()
        if self.do_sanity_checks:
            assert child_inclusion_dist.event_shape == (len(child_type_list),), \
                "Inclusion dist %s has event_shape %s vs child list %s" % (
                    child_inclusion_dist, child_inclusion_dist.event_shape,
                    child_type_list
                )
        self.child_inclusion_dist = child_inclusion_dist

        # Sample an included child set.
        with scope(prefix=self.name + "_choose_children"):
            self.child_inclusion_values = pyro.sample(
                "sample_children_inclusion", self.child_inclusion_dist)
        # TODO: This doesn't support batching as-is.
        child_set = [
            child_type_list[k]()
            for k, active in enumerate(self.child_inclusion_values.bool())
            if active
        ]
        self.children_sampled = True
        return child_set

    def _sample_children_impl(self):
        ''' Should return a distribution that, when sampled, produces
        a binary vector indicating whether each of the children in
        the maximal child list is included in the output. '''
        raise NotImplementedError("Override _sample_children_impl.")

    def conditioned_sample_children(self, children):
        ''' Given a list of children, sets up our child inclusion
        dist as a Delta that guarantees that we'd sample that set of
        children. '''
        child_type_list = self.get_maximal_child_type_list()
        target_child_inclusion_values = self.get_child_indices_into_maximal_child_list(children).double()
        self.child_inclusion_dist = dist.Delta(target_child_inclusion_values)
        with scope(prefix=self.name + "_choose_children"):
            self.child_inclusion_values = pyro.sample(
                "sample_children_inclusion",
                self.child_inclusion_dist
            )
        self.children_sampled = True

    def get_children_ll(self):
        assert self.children_sampled
        return self.child_inclusion_dist.log_prob(self.child_inclusion_values).sum()

    def get_child_indices_into_maximal_child_list(self, children):
        ''' Returns, for each child in children, the index into the 
        maximal child set for this node that that child was produced by.
        Errors in ambiguous cases; assumes this child set is a feasible
        full child set for this node type. '''
        raise NotImplementedError('Override get_child_indices_into_maximal_child_list.')

    def instantiate_children(self, children):
        ''' Instantiates the supplied children of this node, creating
        derived variable distributions for each of them and passing those
        down into their own instantiate methods. '''
        assert self.instantiated, "Node should be instantiated before instantiating children."
        child_derived_variable_dists = self._instantiate_children_impl(children)
        assert len(child_derived_variable_dists) == len(children)
        for child, dist_dict in zip(children, child_derived_variable_dists):
            child.instantiate(dist_dict)
        self.children_instantiated = True

    def _instantiate_children_impl(self, children):
        if len(children) > 0:
            for child in children:
                if len(child.get_derived_variable_info().items()) > 0:
                    raise NotImplementedError("Override _instantiate_children_impl.")


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
        self.production_dist = dist.OneHotCategorical(probs=production_weights)
        super().__init__()

    def _sample_children_impl(self):
        return self.production_dist

    def get_maximal_child_type_list(self):
        return self.child_types

    def get_child_indices_into_maximal_child_list(self, children):
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
        return torch.tensor(inds)

class AndNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: will choose all
    of the available children all the time.'''
    def __init__(self, child_types):
        self.child_types = child_types
        self.production_dist = dist.Bernoulli(torch.ones(len(self.child_types))).to_event(1)
        super().__init__()

    def _sample_children_impl(self):
        return self.production_dist

    def get_maximal_child_type_list(self):
        return [child for child in self.child_types]

    def get_child_indices_into_maximal_child_list(self, children):
        assert len(children) == len(self.child_types)
        return torch.tensor(range(len(self.child_types)))

class IndependentSetNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: has a list of
    children that can occur, and chooses each one as an independent
    Bernoulli choice..'''
    def __init__(self, child_types, production_probs):
        self.child_types = child_types
        self.production_probs = production_probs
        self.production_dist = dist.Bernoulli(production_probs).to_event(1)
        super().__init__()

    def _sample_children_impl(self):
        return self.production_dist

    def get_maximal_child_type_list(self):
        return self.child_types

    def get_child_indices_into_maximal_child_list(self, children):
        inds = []
        for k, child_type in enumerate(self.child_types):
            matching_children_inds = [
                k for (k, child) in enumerate(children)
                if type(child) == child_type
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
    def __init__(self, child_type, geometric_prob, max_repeats):
        assert max_repeats > 0
        assert child_type is not None
        self.child_types = [child_type] * max_repeats
        self.max_repeats = max_repeats
        if geometric_prob == 0.:
            # Set up placeholder distribution that will never produce
            # children.
            self.production_dist = dist.Delta(torch.zeros(max_repeats)).to_event(1)
        else:
            self.production_dist = VectorCappedGeometricDist(geometric_prob, max_repeats)
        super().__init__()

    def _sample_children_impl(self):
        return self.production_dist

    def get_maximal_child_type_list(self):
        return self.child_types

    def get_child_indices_into_maximal_child_list(self, children):
        for child in children:
            # All our child types should be the same.
            assert type(child) == self.child_types[0]
        num_active = len(children)
        assert num_active <= self.max_repeats
        return torch.tensor(range(num_active))

'''

'''