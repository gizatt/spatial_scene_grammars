import numpy as np
import pyro
import pyro.distributions as dist
import torch

import pydrake
from pydrake.all import (
    CoulombFriction,
    SpatialInertia,
    UnitInertia
)

class Node(object):
    ''' Most abstract form of a node / symbol in the grammar. '''
    def __init__(self, name):
        self.name = name


class SpatialNodeMixin(object):
    ''' Contract that a class with this mixin has a 'tf' attribute,
    which is a 4x4 tf matrix representing the node's pose in world
    frame. '''
    def __init__(self, tf):
        self.tf = tf


default_spatial_inertia = SpatialInertia(
    mass=1.0,
    p_PScm_E=np.zeros(3), G_SP_E=UnitInertia(0.01, 0.01, 0.01)
)
default_friction = CoulombFriction(0.9, 0.8)
class PhysicsGeometryNodeMixin(SpatialNodeMixin):
    '''
    Contract that this class has physics and geometry info, providing
    Drake / simulator interoperation. Implies SpatialNodeMixin (and
    calls its constructor.) 
    Enables calls to register geometry of the following types:
        - Model files (urdf/sdf), paired with a transform from the object
          local origin, the name of the root body (which gets put at that
          transform -- required if there's more than one body in the URDF),
          and optionally, the initial joint configuration of
          the model (as a dict of joint names to joint states). These
          are added to the simulated scene with the specified link
          welded (or translated, if not fixed) to the node transform.
        - Visual and collision geometry (Drake Shape types), paired with
          transforms from the object local origin and relevant color
          and friction information.
        - Clearance geometry (Drake Shape types), paired with transforms
          from the object local origin. This represents the region around
          this object that should not intersect with any other node's
          clearance geometry: e.g., the space in front of a cabinet should
          be clear so the doors can open.
    '''
    def __init__(self, tf, fixed=True, spatial_inertia=None):
        SpatialNodeMixin.__init__(self, tf)
        self.fixed = fixed
        self.model_paths = []
        self.spatial_inertia = spatial_inertia or default_spatial_inertia
        self.visual_geometry = []
        self.collision_geometry = []
        self.clearance_geometry = []
    def register_model_file(self, tf, model_path, root_body_name=None,
                            q0_dict=None):
        self.model_paths.append((tf, model_path, root_body_name, q0_dict))
    def register_geometry(self, tf, geometry, color=np.ones(4), friction=default_friction):
        # Shorthand for registering the same geometry as collision + visual.
        self.register_visual_geometry(tf, geometry, color)
        self.register_collision_geometry(tf, geometry, friction)
    def register_visual_geometry(self, tf, geometry, color=np.ones(4)):
        assert isinstance(tf, torch.Tensor) and tf.shape == (4, 4)
        assert isinstance(geometry, pydrake.geometry.Shape)
        self.visual_geometry.append((tf, geometry, color))
    def register_collision_geometry(self, tf, geometry, friction=default_friction):
        assert isinstance(tf, torch.Tensor) and tf.shape == (4, 4)
        assert isinstance(geometry, pydrake.geometry.Shape)
        self.collision_geometry.append((tf, geometry, friction))
    def register_clearance_geometry(self, tf, geometry):
        assert isinstance(tf, torch.Tensor) and tf.shape == (4, 4)
        assert isinstance(geometry, pydrake.geometry.Shape)
        self.clearance_geometry.append((tf, geometry))


class NonTerminalNode(Node):
    ''' Abstract interface for nonterminal nodes, which are responsible
    for sampling a set of production rules to produce new nodes. '''
    def sample_production_rules(self):
        ''' returns: a list of ProductionRules '''
        raise NotImplementedError()

class TerminalNode(Node):
    ''' The leafs of a generated scene tree will be terminal nodes. '''
    pass


class RootNode(NonTerminalNode):
    ''' A special node type capable of unconditionally sampling
    itself. (TODO: Does this exist for a reason? Why not just force
    a user to either make sure the intended root node can be instantiated
    with no arguments, or instantiate their own root node?) '''
    @staticmethod
    def sample():
        ''' Should return a RootNode instance. '''
        raise NotImplementedError()


class OrNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: will choose one
    of the available rules with probabilities given by the production weights. '''
    def __init__(self, name, production_rules, production_weights):
        if len(production_weights) != len(production_rules):
            raise ValueError("# of production rules and weights must match.")
        if len(production_weights) == 0:
            raise ValueError("Must have nonzero number of production rules.")
        self.production_rules = production_rules
        self.production_weights = production_weights
        self.production_dist = dist.Categorical(production_weights)
        NonTerminalNode.__init__(self, name=name)

    def sample_production_rules(self):
        active_rule = pyro.sample(self.name + "_or_sample", self.production_dist)
        return [self.production_rules[active_rule]]


class AndNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: will choose all
    of the available rules with probabilities all the time.'''
    def __init__(self, name, production_rules):
        self.production_rules = production_rules
        Node.__init__(self, name=name)

    def sample_production_rules(self):
        return self.production_rules


class CovaryingSetNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: given a set of
    production rules, can choose any combination of those rules. '''
    @staticmethod
    def build_init_weights(num_production_rules, production_weights_hints = {},
                           remaining_weight = 1.):
        '''
            Helper to generate a valid set of production weights for this node type.
    
            Args:
                num_production_rules: The total number of production rules.
                production_weights_hints: A dict keyed by tuples of ints (representing
                    the set of production rules, as indexes into the rule list), with
                    float values.
                remaining_weight: The weight to assign to each unspecified rule combination.

            Returns:
                A valid, normalized init_weights vector with 2^(num_production_rules)
                entries as a torch tensor.
        '''
        assert(remaining_weight >= 0.)
        num_combinations = 2**num_production_rules
        init_weights = torch.ones(num_combinations).double() * remaining_weight
        for hint in production_weights_hints.keys():
            val = production_weights_hints[hint]
            assert(val >= 0.)
            combination_index = 0
            for index in hint:
                assert(isinstance(index, int) and index >= 0 and
                       index < num_production_rules)
                combination_index += 2**index
            init_weights[combination_index] = val
        init_weights /= torch.sum(init_weights)
        return init_weights
        
    def __init__(self, name, production_rules, init_weights):
        ''' Make a categorical distribution over
           every possible combination of production rules
           that could be active, with a separate weight
           for each combination. (2^n weights!)'''
        self.production_rules = production_rules
        self.exhaustive_set_weights = init_weights
        self.production_dist = dist.Categorical(
            logits=torch.log(self.exhaustive_set_weights / (1. - self.exhaustive_set_weights)))
        NonTerminalNode.__init__(self, name=name)

    def sample_production_rules(self):
        # Select out the appropriate rules
        selected_rules = pyro.sample(
            self.name + "_exhaustive_set_sample",
            self.production_dist)
        output = []
        for k, rule in enumerate(self.production_rules):
            if (selected_rules >> k) & 1:
                output.append(rule)
        return output


class IndependentSetNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: given a set of
    production rules, can activate each rule as an independent Bernoulli
    trial (with specified probabilities of activation). '''
    def __init__(self, name, production_rules,
                 production_probs):
        if len(production_probs) != len(production_rules):
            raise ValueError("Must have same number of production probs "
                             "as rules.")
        self.production_probs = production_probs
        self.production_dist = dist.Bernoulli(production_probs).to_event(1)
        self.production_rules = production_rules
        NonTerminalNode.__init__(self, name=name)

    def sample_production_rules(self):
        selected_rules = pyro.sample(
            self.name + "_independent_set_sample",
            self.production_dist)
        # Select out the appropriate rules
        output = []
        for k, rule in enumerate(self.production_rules):
            if selected_rules[k] == 1:
                output.append(rule)
        return output


class GeometricSetNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: given a single production
    rule, can reapply it a number of times following a geometric distribution
    with a given repeat probability.'''
    def __init__(self, name, production_rule,
                 geometric_prob):
        self.production_dist = dist.Geometric(geometric_prob)
        self.production_rule = production_rule
        NonTerminalNode.__init__(self, name=name)

    def sample_production_rules(self):
        num_repeats = pyro.sample(
            self.name + "_geometric_set_sample",
            self.production_dist)
        # Select out the appropriate rules
        return [self.production_rule] * int(num_repeats.item())
