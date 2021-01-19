import numpy as np
import networkx as nx
import torch
from .nodes import TerminalNode, Node


'''

Clearly, this is gonna be an important and hard part of this system
to get right.

Unorganized thoughts:
- HMC should work for factors on continuous properties, I think?
  - How do I tell which registered constraints can be HMC-d? What if
  a node's continuous properties affect the way it'll produce children?
- Simple "existence of node" or "existence of subtree" constraints
may be an easier class than general topology constraints?

'''

class Constraint():
    '''
    To be used in combination with constraint-wrapping
    Gibbs distributions to produce factors?
    '''
    def __init__(self, lower_bound, upper_bound):
        assert isinstance(lower_bound, torch.Tensor)
        assert isinstance(upper_bound, torch.Tensor)
        assert lower_bound.shape == upper_bound.shape
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def eval(self, scene_tree):
        # Output should be torch-autodiffable from scene_tree
        # params and variables.
        raise NotImplementedError()

    def eval_violation(self, scene_tree):
        # Return (max violation, lower_violation, upper_violation) vectors
        val = self.eval(scene_tree)
        lower_violation = self.lower_bound - val
        lower_violation[torch.isinf(self.lower_bound)] = 0.
        upper_violation = val - self.upper_bound
        upper_violation[torch.isinf(self.upper_bound)] = 0.
        max_violation = torch.max(lower_violation, upper_violation)
        return max_violation, lower_violation, upper_violation


class ContinuousVariableConstraint(Constraint):
    pass

class TopologyConstraint(Constraint):
    pass
