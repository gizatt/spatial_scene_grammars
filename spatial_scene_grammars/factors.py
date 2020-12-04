import numpy as np
import networkx as nx
import torch
from .nodes import TerminalNode, Node

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
        raise NotImplementedError()


class ExtraSceneFactor():
    '''
    Represents a factor (a constraint or affinity function)
    on a SceneTree or subtree.

    These are constructed given a SceneTree; at construction
    time, depending on the type of constraint, I expect
    a constraint to ajopqwjop
    TODO
    These are half factory, half evaluation: at construction
    time, an InducedConstraint takes a 

    - Constraints have a lower and upper scalar or vector bound.
    - Constraints have an evaluation function that maps
    a scene tree or subtree (as a nx network) to an output scalar
    or vector value of dimensionality matching the bounds.

    TODO: This'll have to interoperate with a Gibbs-distribution-based
    energy/scoring model. Should that be part of the constraint,
    or another entity that consumes the constraint? Separation seems
    "cleaner" but more complex.
    '''
    def __init__(self):
        super().__init__()

    def eval(self, scene_subtree):
        raise NotImplementedError()

class NumberOfChildrenFactor(Constraint):
    ''' Scores the number of nodes in the given scene subtree. '''
    pass

class ClearanceConstraint(Constraint):
    ''' Extracts the PhysicsGeometryNodes in the supplied
    tree, '''

    def eval(self, scene_tree):
        builder_clearance, mbp_clearance, sg_clearance = \
            compile_scene_tree_clearance_geometry_to_mbp_and_sg(scene_tree)
        mbp_clearance.Finalize()
        diagram_clearance = builder_clearance.Build()
        diagram_context = diagram_clearance.CreateDefaultContext()
        mbp_context = diagram_clearance.GetMutableSubsystemContext(mbp_clearance, diagram_context)
        constraint = build_clearance_nonpenetration_constraint(
            mbp_clearance, mbp_context, -0.01)
        return constraint.Eval(mbp_clearance.GetPositions(mbp_context))

class PhysicalFeasibilityConstraint(Constraint):
    '''
    Given a set of 
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # copy over from kitchen code