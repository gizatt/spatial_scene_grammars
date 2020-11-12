import torch

from .nodes import SpatialNodeMixin

class ProductionRule(object):
    ''' Abstract interface for a production rule, which samples
    concrete instantiations of a set of product nodes given
    a parent node.
        Production rules should always output the same number of
    nodes -- they're meant to be the "continuous" part of a
    factorization of the grammar spec into discrete and continuous
    parts. '''
    def sample_products(self, parent):
        ''' parent: the NonTerminalNode using this rule
            returns: a list of ProductionRules '''
        raise NotImplementedError()

class DeterministicRelativePoseProductionRule(ProductionRule):
    ''' Helper ProductionRule type representing
    deterministic relative offsets between two nodes that have poses. '''
    def __init__(self, child_constructor, child_name, relative_tf, **kwargs):
        ''' Args:
                child_constructor: Callable that takes `name` and `tf` args,
                    and produces a Node with SpatialNodeMixin.
                child_name: string name to be assigned to the new node.
                relative_tf: 4x4 torch tf matrix.
        '''
        self.child_constructor = child_constructor
        self.child_name = child_name
        self.relative_tf = relative_tf
        self.kwargs = kwargs

    def sample_products(self, parent):
        assert(isinstance(parent, SpatialNodeMixin))
        new_tf = torch.mm(parent.tf, self.relative_tf)
        return [self.child_constructor(name=self.child_name, tf=new_tf, **self.kwargs)]
        

