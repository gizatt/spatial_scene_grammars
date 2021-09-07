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
    def __init__(self, tf, observed, physics_geometry_info, do_sanity_checks=True,
                 params_for_rules=None):
        # Public attributes
        self.name = node_name_store.get_name(self)
        self.tf = tf
        self.observed = observed
        self.physics_geometry_info = physics_geometry_info
        
        # Non-public attributes
        self._do_sanity_checks = do_sanity_checks
        self._rule_k = None # Bookkeeping; which rule does this correspond to in a parent?
        if params_for_rules is not None:
            self._rules = self.generate_rules_with_params(params_for_rules)
        else:
            self._rules = self.generate_rules()
        assert all([isinstance(r, ProductionRule) for r in self._rules])
        super().__init__()

    @classmethod
    def get_parameter_prior(cls):
        # Returns a pyro Dist with support matching the parameter
        # space of the Node subclass being used.
        raise NotImplementedError(
            "Please implement a parameter prior in your Node subclass."
        )

    @property
    def parameters(self):
        # Should return a torch Tensor representing the current
        # parameter setting for this node. These are *not* torch
        # parameters, and this is not a Pytorch module, since the
        # Torch parameters being optimized belong to the grammar / the
        # node type, not a given instantiated node.
        raise NotImplementedError(
            "Child class should implement parameters getter. Users should"
            " never have to do this."
        )
    @parameters.setter
    def parameters(self, parameters):
        raise NotImplementedError(
            "Child class should implement parameters setter. Users should"
            " never have to do this."
        )

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

    @classmethod
    def generate_rules(cls):
        # Should return the list of ProductionRule instances that will be
        # used for the class.
        # (This is a classmethod so that a node can be queried for its
        # rule set before it's instantiated, and that the rule set can't
        # depend on instantiation details or arguments.)
        raise NotImplementedError("Please implement generate_rules in subclass.")
    
    @property
    def rule_k(self):
        return self._rule_k
    @rule_k.setter
    def rule_k(self, rule_k):
        self._rule_k = rule_k

    @property
    def rules(self):
        # Returns a list of the rules this node can take. Each
        # member will be a ProductionRule object.
        return self._rules
    
    def sample_children(self):
        raise NotImplementedError("Implement sample_children in subclass.")
    def score_child_set(self, children):
        raise NotImplementedError("Implemented score_child_set in subclass.")

class TerminalNode(Node):
    ''' The leafs of a generated scene tree will be terminal nodes. '''
    @property
    def parameters(self):
        return torch.empty(size=(0,))
    @parameters.setter
    def parameters(self, parameters):
        if parameters is not None:
            assert ValueError("TerminalNode has no parameters to set")
    @classmethod
    def get_parameter_prior(cls):
        # No params, so no prior.
        return None
    @classmethod
    def generate_rules(cls):
        return []
    @property
    def rules(self):
        return []
    def sample_children(self):
        return []
    def score_child_set(self, children):
        if len(children) > 0:
            return torch.tensor(-np.inf)
        else:
            return torch.zeros(1)

class AndNode(Node):
    ''' Given a list of production rule, enacts all of them, all the time.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert len(self.rules) > 0
    
    @property
    def parameters(self):
        return torch.empty(size=(0,))
    @parameters.setter
    def parameters(self, parameters):
        if parameters is not None:
            assert ValueError("AndNode has no parameters to set")
    @classmethod
    def get_parameter_prior(cls):
        # No params, so no prior.
        return None

    def sample_children(self):
        children = []
        for k, rule in enumerate(self.rules):
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
    def __init__(self, rule_probs, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(rule_probs, torch.Tensor)
        rule_probs = rule_probs / torch.sum(rule_probs)
        self.parameters = rule_probs

    @property
    def parameters(self):
        return self.rule_probs
    @parameters.setter
    def parameters(self, parameters):
        assert isinstance(parameters, torch.Tensor)
        assert len(parameters) == len(self.rules)
        self.rule_probs = parameters
        self._rule_dist = dist.Categorical(self.rule_probs)

    @classmethod
    def get_parameter_prior(cls):
        # Default parameter prior is a uniform distribution
        # over the simplex.
        n_rules = len(cls.generate_rules())
        return dist.Dirichlet(torch.ones(n_rules))

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
    def __init__(self, p, max_children, **kwargs):
        super().__init__(**kwargs)
        assert len(self.rules) == 1
        self.rule = self.rules[0]
        self.max_children = max_children
        self.parameters = p

    @property
    def parameters(self):
        return self.p
    @parameters.setter
    def parameters(self, parameters):
        assert isinstance(parameters, (torch.Tensor, float))
        # Compile a Categorical dist that's equivalent to sampling
        # from a geometric distribution clamped at some max #.
        # TODO(gizatt): I'm *not* adding the extra term on the final
        # weight reflecting the total probability of a geometric dist
        # giving more than N children -- I don't have to for this to be
        # a legitimate distribution. But "geometric" in the name is a little
        # misleading, since this'll have slightly lower mean.
        if isinstance(parameters, float):
            parameters = torch.tensor([parameters])
        p = parameters.reshape(1,)
        self.p  = p
        self.rule_probs = torch.empty(self.max_children)
        for k in range(self.max_children):
            self.rule_probs[k] = (1. - self.p) ** (k - 1) * self.p
        self.geom_surrogate_dist = dist.Categorical(self.rule_probs)

    @classmethod
    def get_parameter_prior(cls):
        # Default parameter prior is a uniform distribution
        # over [0, 1].
        return dist.Uniform(torch.zeros(1), torch.ones(1))

    def sample_children(self):
        children = []
        n = pyro.sample("GeometricSetNode_n", self.geom_surrogate_dist) + 1
        assert n >= 1 and n <= self.max_children
        for k in range(n):
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


class IndependentSetNode(Node):
    ''' Given a list of production rules, enacts each one independently
    according to coin flip probabilities.'''
    def __init__(self, rule_probs, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(rule_probs, torch.Tensor)
        self.parameters = rule_probs

    @property
    def parameters(self):
        return self.rule_probs
    @parameters.setter
    def parameters(self, parameters):
        assert isinstance(parameters, torch.Tensor)
        assert len(parameters) == len(self.rules)
        self.rule_probs = parameters
        self._rule_dist = dist.Bernoulli(self.rule_probs)

    @classmethod
    def get_parameter_prior(cls):
        # Default parameter prior is a uniform distribution over
        # [0, 1] for each node.
        n_rules = len(cls.generate_rules())
        return dist.Uniform(torch.zeros(n_rules), torch.ones(n_rules))

    def sample_children(self):
        children = []
        activations = pyro.sample("IndependentSetNode_n", self._rule_dist)
        for k, rule in enumerate(self.rules):
            if activations[k] > 0.5:
                child = self.rules[k].sample_child(self)
                child.rule_k = k
                children.append(child)
        return children

    def score_child_set(self, children):
        activations = torch.zeros(len(self.rules))
        for child in children:
            if child.rule_k > len(self.rules) or type(child) != self.rules[child.rule_k].child_type:
                # Mismatched child set
                return torch.tensor(-np.inf)
            elif not torch.isclose(activations[child.rule_k], torch.zeros(1)):
                # Can't produce 2 children from same rule
                return torch.tensor(-np.inf)
            else:
                activations[child.rule_k] = 1.
        return self._rule_dist.log_prob(activations).sum()