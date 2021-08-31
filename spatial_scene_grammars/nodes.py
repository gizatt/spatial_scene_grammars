from copy import deepcopy
from collections import namedtuple
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.contrib.autoname import scope
import torch
from torch.distributions import constraints

from .torch_utils import ConstrainedParameter
from .rules import *

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
    '''
    Every node (symbol) in the grammar derives from this base.

    args:
        - tf, a 4x4 pytorch tensor representing the world-frame
          pose of this node.
        - observed: boolean, whether this node is observable.
        - physics_geometry_info: A PhysicsGeometryInfo struct
          containing physics parameters, visual and collision
          geometry, etc for this node.
        - parameters: Dictionary of parameters whose keys
          and value shapes match the default parameters declared
          for the node type.

    '''
    def __init__(self, tf, observed, physics_geometry_info, parameters={}, do_sanity_checks=True):
        # Public attributes
        self.name = node_name_store.get_name(self)
        self.tf = tf
        self.observed = observed
        self.physics_geometry_info = physics_geometry_info
        
        # Non-public attributes
        self._do_sanity_checks = do_sanity_checks
        self._rule_k = None # Bookkeeping; which rule does this correspond to in a parent?

        # Use the parameter setter to copy attrs over to self with
        # optional sanity checking and additional attribute setup.
        self._parameters = parameters
        super().__init__()

    @classmethod
    def init_with_default_parameters(cls, tf):
        return cls(tf, parameters=cls.get_default_parameters())

    @classmethod
    def get_default_parameters(cls):
        ''' Provide a dictionary of default values of parameters, keyed by
        their names, as ConstrainedParameter objects. They will be wrapped
        tracked by the containing grammar, and access by a
        get_parameter("foo") call. '''
        return {}

    @property
    def parameters(self):
        return self._parameters
    @parameters.setter
    def parameters(self, new_parameters):
        assert isinstance(new_parameters, dict)
        if self._do_sanity_checks:
            # Check that passed parameters are an exact match in terms of
            # keys and shapes to the expected param dict.
            expected_param_dict = self.get_default_parameters()
            assert set(expected_param_dict.keys()) == set(new_parameters.keys())
            for key, value in expected_param_dict.items():
                assert key in new_parameters.keys()
                assert value.get_value().shape == new_parameters[key].get_value().shape
            for key, value in new_parameters.items():
                assert key in expected_param_dict.keys()
                assert value.get_value().shape == expected_param_dict[key].get_value().shape

        self._parameters = new_parameters

        # Assign parameters to self according to their names, for convenient access.
        for key, value in attr_dict.items():
            if hasattr(self, key):
                # Make sure we're only updating the value.
                curr_value = getattr(self, key)
                assert isinstance(curr_value, torch.Tensor), "Class already has non-tensor attribute named %s: %s." % (key, curr_value)
                assert getattr(self, key).shape == value.shape, \
                    "Class already has an attribute named %s with different shape: %s (current) vs %s (new)." % (
                        key, str(curr_value.shape), value.shape)
            setattr(self, key, value)
    def get_parameter(self, param_name):
        assert param_name in self._parameters.keys()
        return self._parameters[param_name]

    # Additional accessors into the pose transform
    @property
    def translation(self):
        return self.tf[:3, 3]
    @translation.setter
    def translation(self, xyz):
        assert isinstance(xyz, torch.Tensor) and xyz.shape == (3,)
        self.tf[:3, 3] = xyz[:3]
    @property
    def rotation(self):
        return self.tf[:3, :3]
    @rotation.setter
    def rotation(self, R):
        assert isinstance(R, torch.Tensor) and R.shape == (3, 3)
        self.tf[:3, :3] = R[:3, :3]

    @property
    def rule_k(self):
        return self._rule_k
    @rule_k.setter
    def rule_k(self, rule_k):
        self._rule_k = rule_k

    def sample_children(self):
        raise NotImplementedError("Implement sample_children in subclass.")
    def score_child_set(self, children):
        raise NotImplementedError("Implemented score_child_set in subclass.")

class TerminalNode(Node):
    ''' The leafs of a generated scene tree will be terminal nodes. '''
    def sample_children(self):
        return []
    def score_child_set(self, children):
        if len(children) > 0:
            return torch.tensor(-np.inf)
        else:
            return torch.zeros(1)

class AndNode(Node):
    ''' Given a list of production rule, enacts all of them, all the time.'''
    def __init__(self, rules, **kwargs):
        assert len(rules) > 0
        assert all([isinstance(r, ProductionRule) for r in rules])
        self.rules = rules
        super().__init__(**kwargs)

    def sample_children(self):
        children = []
        for k, rule in enumerate(self.rules):
            with scope(prefix="%d" % k):
                child = rule.sample_child(self)
            child.rule_k = k
            children.append(child)
        return children

    def score_child_set(self, children):
        if len(children) != len(self.rules):
            return torch.tensor(-np.inf)
        if set([child.rule_k for child in children]) != set(range(len(self.rules))):
            return torch.tensor(-np.inf)
        for child in children:
            if self.rules[child.rule_k].child_type != type(child):
                return torch.tensor(-np.inf)
        return torch.tensor(0.)


class OrNode(Node):
    ''' Given a list of production rule, enacts one of them.'''
    def __init__(self, rules, rule_probs, **kwargs):
        assert len(rules) > 0
        assert all([isinstance(r, ProductionRule) for r in rules])
        assert isinstance(rule_probs, torch.Tensor)
        assert len(rules) == len(rule_probs)
        rule_probs = rule_probs / torch.sum(rule_probs)

        self.rules = rules
        self.rule_probs = rule_probs

        self._rule_dist = dist.Categorical(rule_probs)
        super().__init__(**kwargs)
    
    def sample_children(self):
        # Pick which child will be produced.
        child_ind = pyro.sample("OrNode_child", self._rule_dist)
        child = self.rules[child_ind].sample_child(self)
        child.rule_k = child_ind.item()
        children = [child]
        return children

    def score_child_set(self, children):
        if len(children) != 1:
            return torch.tensor(-np.inf)
        child = children[0]
        if self.rules[child.rule_k].child_type != type(child):
            return torch.tensor(-np.inf)
        return self._rule_dist.log_prob(torch.tensor(child.rule_k))


class GeometricSetNode(Node):
    ''' Given a *single production rule*, repeatedly enacts it until
    a coin flip succeeds or the maximum number of children has been added.
    Hence, it can produce [1, ..., max_children] children.

        p here is the probability of *stopping* at any given trial.'''
    def __init__(self, rule, p, max_children, **kwargs):
        assert isinstance(rule, ProductionRule)
        self.rule = rule
        self.p  = p
        self.max_children = max_children
        # Compile a Categorical dist that's equivalent to sampling
        # from a geometric distribution clamped at some max #.
        # TODO(gizatt): I'm *not* adding the extra term on the final
        # weight reflecting the total probability of a geometric dist
        # giving more than N children -- I don't have to for this to be
        # a legitimate distribution. But "geometric" in the name is a little
        # misleading, since this'll have slightly lower mean.
        self.rule_probs = torch.tensor([
            (1. - self.p) ** (k - 1) * self.p
            for k in range(self.max_children)
        ])
        self.geom_surrogate_dist = dist.Categorical(self.rule_probs)
        super().__init__(**kwargs)
    
    def sample_children(self):
        children = []
        n = pyro.sample("GeometricSetNode_n", self.geom_surrogate_dist) + 1
        assert n >= 1 and n <= self.max_children
        for k in range(n):
            with scope(prefix="%d" % k):
                child = self.rule.sample_child(self)
            child.rule_k = k
            children.append(child)
        return children

    def score_child_set(self, children):
        if len(children) == 0 or len(children) > self.max_children:
            return torch.tensor(-np.inf)
        if set([child.rule_k for child in children]) != set(range(len(children))):
            print("Set mismatch: ", [child.rule_k for child in children])
            return torch.tensor(-np.inf)
        for child in children:
            if type(child) != self.rule.child_type:
                return torch.tensor(-np.inf)
        return self.geom_surrogate_dist.log_prob(torch.tensor(len(children) - 1))