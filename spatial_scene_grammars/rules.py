import torch
import pyro
from pyro.contrib.autoname import scope

from .nodes import Node

class ProductionRule(object):
    ''' Abstract interface for a production rule, which controls
    the generation of a (fixed) set of product nodes given a parent node.

    These explicitly represent the "continuous" part of a
    factorization of the grammar spec into discrete and continuous
    parts.

    To correctly create a ProductionRule subclass:
    1) Make sure your subclass has a child_types member that lists the precise
    set of child node types in the order they'll come out of sample_products.
    2) Implement _sample_products. (Not sample_products!)
    '''

    @classmethod
    def get_child_types(cls):
        if not hasattr(cls, "child_types"):
            raise ValueError("Your Rule subclass %s should explicitly list child_types"\
                             " in a static member variable." % cls.__name__)
        return cls.child_types

    def sample_products(self, parent, child_names):
        ''' 
            Wraps _sample_products in a trace, keeping track of
            the identity of the variables that are sampled by the node
            production.
        args:
            parent: the NonTerminalNode using this rule
            child_names: A list of names to pass through to children.
        returns: a list of instantiated nodes
        '''
        self.trace = pyro.poutine.trace(self._sample_products).get_trace(parent, child_names)
        if self.verify:
            # Make sure that all sampled variables are continuous.
            for key, value in self.trace.nodes.items():
                if key not in ("_RETURN", "_INPUT"):
                    if value["type"] == "sample":
                        support = value["fn"].support
                        # TODO: Expand to exhaustive list of
                        # all continuous distribution constraint / support
                        # types. I didn't see any obvious inheritence hierarchy
                        # I can take advantage of to make this easier...
                        assert isinstance(
                            support,
                            (torch.distributions.constraints._Real,
                             torch.distributions.constraints._Interval)
                        ), "Sample sites in sample_products should only sample continuous " \
                           "values: this one has support " + str(support)
        return self.trace.nodes["_RETURN"]["value"]

    def _sample_products(self, parent):
        ''' Produces a set of child nodes.
        args:
            parent: the NonTerminalNode using this rule.
        returns: a list of instantiated nodes
        '''
        raise NotImplementedError()

    def get_local_variable_names(self):
        assert hasattr(self, "trace"), "sample_products not called yet for rule " + str(self)
        return [key for key in list(self.trace.nodes.keys()) if key not in ["_RETURN", "_INPUT"]]


class EmptyProductionRule(ProductionRule):
    child_types = []
    def _sample_products(self, parent):
        assert(len(child_names) == 0)
        return []


def make_trivial_production_rule(child_type, kwargs):
    raise NotImplementedError("Involves type() or something")


class RandomRelativePoseProductionRule(ProductionRule):
    ''' Helper ProductionRule type representing random placement,
    described by a distribution on relative pose between two nodes. '''
    def __init__(self, child_type, relative_tf_sampler,  child_postfix="", **kwargs):
        ''' Args:
                child_type: Child node type. Should be a subclass of SpatialNode.
                relative_tf_sampler: callable that samples a 4x4 tf.
                kwargs: Additional arguments passed to the child.
        '''
        assert issubclass(child_type, SpatialNode)
        self.relative_tf_sampler = relative_tf_sampler
        self.child_postfix = child_postfix
        self.kwargs = kwargs
        super().__init__(child_types=[child_type])

    def _sample_products(self, parent, child_names):
        assert(isinstance(parent, SpatialNode))
        assert(len(child_names) == 1)
        new_tf = torch.mm(parent.tf, self.relative_tf_sampler())
        return [self.child_types[0](
            name=child_names[0] + self.child_postfix,
            tf=new_tf,
            **self.kwargs)]

class DeterministicRelativePoseProductionRule(RandomRelativePoseProductionRule):
    ''' Helper ProductionRule type representing
    deterministic relative offsets between two nodes that have poses.

    In practice, shells out to RandomRelativePoseProductionRule, but supplies
    a deterministic sampler. '''
    def __init__(self, child_type, relative_tf, **kwargs):
        ''' Args:
                child_type: Callable that takes `name` and `tf` args,
                    and produces a Node with SpatialNode.
                relative_tf: 4x4 torch tf matrix.
        '''
        RandomRelativePoseProductionRule.__init__(
            self,
            child_type,
            lambda: relative_tf,
            **kwargs
        )

class ComposedProductionRule(ProductionRule):
    ''' Given a set of production rules, this rule groups and enacts all of them. '''
    def __init__(self, production_rules):
        self.production_rules = production_rules
        child_types = []
        for rule in self.production_rules:
            child_types += rule.get_child_types()
        super().__init__(child_types=child_types)

    def _sample_products(self, parent, child_names):
        assert len(child_names) == len(self.child_types)
        k = 0
        children = []
        for rule_k, rule in enumerate(self.production_rules):
            k_next = k + len(rule.get_child_types())
            with scope(prefix="subprod_%d" % rule_k):
                children += rule._sample_products(parent, child_names[k:k_next])
            k = k_next
        return children