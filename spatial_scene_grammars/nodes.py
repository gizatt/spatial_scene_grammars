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

class Node():
    ''' Most abstract form of a node / symbol in the grammar. '''
    def __init__(self, name, **kwargs):
        self.name = name
        # Is it right to not pass **kwargs in here? Will all
        # constructors always get called? Multiple inheritance
        # is too hard...
        super().__init__()
        # Setup can only safely be done once *all* other init methods
        # have been called. Doing it after super().__init() is
        # enough to guarantee that.
        with scope(prefix=self.name):
            with scope(prefix="params"):
                self._setup(**kwargs)
    def _setup(self):
        raise NotImplementedError()

class SpatialNode(Node):
    ''' Contract that a class with this mixin has a 'tf' attribute,
    which is a 4x4 tf matrix representing the node's pose in world
    frame. '''
    def __init__(self, tf, **kwargs):
        self.tf = tf
        super().__init__(**kwargs)


default_spatial_inertia = SpatialInertia(
    mass=1.0,
    p_PScm_E=np.zeros(3), G_SP_E=UnitInertia(0.01, 0.01, 0.01)
)
default_friction = CoulombFriction(0.9, 0.8)
class PhysicsGeometryNode(SpatialNode):
    '''
    Contract that this class has physics and geometry info, providing
    Drake / simulator interoperation. Implies SpatialNode (and
    calls its constructor.) 
    Args:
        - fixed: Whether this geometry is floating in the final simulation
            (as oppposed to welded to the world).
        - spatial_inertia: Spatial inertia of the body. If None,
            will adopt a default mass of 1.0kg and 0.01x0.01x0.01 diagonal
            rotational inertia.
        - is_container: Flag whether this object will function as a
            container for other objects for the purpose of collision
            and stability checks. If so, then objects below this one
            will be isolated from collision and clearance checks for
            objects above this one, and instead only be checked against
            this object's collision geometry and this object's
            childrens' geometry. Valid for e.g. a cabinet full of
            stuff that does not interact with anything outside of
            the cabinet.

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
    def __init__(self, fixed=True, spatial_inertia=None, is_container=False, **kwargs):
        self.fixed = fixed
        self.is_container = is_container
        self.model_paths = []
        self.spatial_inertia = spatial_inertia or default_spatial_inertia
        self.visual_geometry = []
        self.collision_geometry = []
        self.clearance_geometry = []
        super().__init__(**kwargs)

    def register_model_file(self, tf, model_path, root_body_name=None,
                            q0_dict={}):
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
        assert isinstance(friction, CoulombFriction)
        self.collision_geometry.append((tf, geometry, friction))
    def register_clearance_geometry(self, tf, geometry):
        assert isinstance(tf, torch.Tensor) and tf.shape == (4, 4)
        assert isinstance(geometry, pydrake.geometry.Shape)
        self.clearance_geometry.append((tf, geometry))


class NonTerminalNode(Node):
    ''' Abstract interface for nonterminal nodes, which are responsible
    for sampling a set of production rules to produce new nodes. Designed
    for deferred registration of production rules. '''
    def __init__(self, **kwargs):
        self.rules_registered = False
        super().__init__(**kwargs)
        
    def _register_production_rules_impl(self, **kwargs):
        raise NotImplementedError()

    def register_production_rules(self, **kwargs):
        with scope(prefix=self.name + "_register"):
            self._register_production_rules_impl(**kwargs)
        self.rules_registered = True

    def _sample_production_rules_impl(self):
        raise NotImplementedError()

    def sample_production_rules(self):
        ''' returns: a list of ProductionRules '''
        assert self.rules_registered, "No call to register_production_rules for node %s" % name

        with scope(prefix=self.name):
            with scope(prefix="sample_rules"):
                return self._sample_production_rules_impl()


class TerminalNode(Node):
    ''' The leafs of a generated scene tree will be terminal nodes. '''
    pass


class OrNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: will choose one
    of the available rules with probabilities given by the production weights.'''
    def _register_production_rules_impl(self, production_rules, production_weights):
        if len(production_weights) != len(production_rules):
            raise ValueError("# of production rules and weights must match.")
        if len(production_weights) == 0:
            raise ValueError("Must have nonzero number of production rules.")
        self.production_rules = production_rules
        self.production_weights = production_weights
        self.production_dist = dist.Categorical(production_weights)

    def _sample_production_rules_impl(self):
        active_rule = pyro.sample("or_sample", self.production_dist)
        return [self.production_rules[active_rule]]


class AndNode(NonTerminalNode):
    ''' Convenience specialization of a nonterminal node: will choose all
    of the available rules with probabilities all the time.'''
    def _register_production_rules_impl(self, production_rules):
        self.production_rules = production_rules
        self.rules_registered = True

    def _sample_production_rules_impl(self):
        assert self.rules_registered, "No call to register_production_rules for node %s" % name
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

    def _register_production_rules_impl(self, production_rules, init_weights):
        ''' Make a categorical distribution over
           every possible combination of production rules
           that could be active, with a separate weight
           for each combination. (2^n weights!)'''
        self.production_rules = production_rules
        self.exhaustive_set_weights = init_weights
        self.production_dist = dist.Categorical(
            logits=torch.log(self.exhaustive_set_weights / (1. - self.exhaustive_set_weights)))
        self.rules_registered = True

    def _sample_production_rules_impl(self):
        # Select out the appropriate rules
        selected_rules = pyro.sample(
            self.name + "::exhaustive_set_sample",
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
    def _register_production_rules_impl(self, production_rules, production_probs):
        if len(production_probs) != len(production_rules):
            raise ValueError("Must have same number of production probs "
                             "as rules.")
        self.production_probs = production_probs
        self.production_dist = dist.Bernoulli(production_probs).to_event(1)
        self.production_rules = production_rules

    def _sample_production_rules_impl(self):
        selected_rules = pyro.sample(
            "independent_set_sample",
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
    def _register_production_rules_impl(
            self, production_rule_type, production_rule_kwargs, geometric_prob):
        self.production_dist = dist.Geometric(geometric_prob)
        self.production_rule_type = production_rule_type
        self.production_rule_kwargs = production_rule_kwargs

    def _sample_production_rules_impl(self):
        num_repeats = pyro.sample(
            "geometric_set_sample",
            self.production_dist)
        # Select out the appropriate rules
        out = []
        for k in range(int(num_repeats.item())):
            # Ensure they're different instantiations of the rule so
            # the scene tree is still a tree.
            out.append(self.production_rule_type(**self.production_rule_kwargs))
        return out
