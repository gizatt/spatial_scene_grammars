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

    def add_to_ik_prog(self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map):
        # Add this constraint to a Drake InverseKinematics object ik.
        raise NotImplementedError()
        
    def eval(self, scene_tree):
        # Output should be torch-autodiffable from scene_tree
        # params and variables. It should either match the shape
        # of lower_bound / upper_bound, or be batched with leftmost
        # dims matching the shape of lower/upper bound.
        raise NotImplementedError()

    def eval_violation(self, scene_tree):
        # Return (max violation, lower_violation, upper_violation) vectors
        val = self.eval(scene_tree)
        if self.lower_bound.shape != torch.Size() and val.shape != self.lower_bound.shape:
            lb_n_elems = len(self.lower_bound.shape)
            assert val.shape[-lb_n_elems:] == self.lower_bound.shape, (val.shape, self.lower_bound.shape)
        lower_violation = self.lower_bound - val
        lower_violation[..., torch.isinf(self.lower_bound)] = 0.
        upper_violation = val - self.upper_bound
        upper_violation[..., torch.isinf(self.upper_bound)] = 0.
        max_violation = torch.max(lower_violation, upper_violation)
        return max_violation, lower_violation, upper_violation


class PoseConstraint(Constraint):
    pass

class StructureConstraint(Constraint):
    pass

class ObjectCountConstraint(StructureConstraint):
    def __init__(self, object_type, min_count, max_count):
        self.object_type = object_type
        super().__init__(lower_bound=torch.tensor(min_count), upper_bound=torch.tensor(max_count))
    def eval(self, scene_tree):
        num = len(list(scene_tree.find_nodes_by_type(self.object_type)))
        return torch.tensor(num)

class ChildCountConstraint(StructureConstraint):
    def __init__(self, parent_type, min_count, max_count):
        self.parent_type = parent_type
        super().__init__(lower_bound=torch.tensor(min_count), upper_bound=torch.tensor(max_count))
    def eval(self, scene_tree):
        child_counts = []
        for parent_node in scene_tree.find_nodes_by_type(self.parent_type):
            child_counts.append(len(scene_tree.successors(parent_node)))
        return torch.tensor(child_counts)