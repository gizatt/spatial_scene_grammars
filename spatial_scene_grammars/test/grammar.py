import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from torch.nn.parameter import Parameter

from spatial_scene_grammars.torch_utils import ConstrainedParameter
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *

torch.set_default_tensor_type(torch.DoubleTensor)
pyro.enable_validation(True)

'''

Simple grammar for testing that gets basic coverage of
node types.

A -> B and C
B -> D or E
D -> G with 50% prob
C -> Geometric repetitions of F
'''

class NodeG(TerminalNode):
    def __init__(self, tf):
        super().__init__(observed=True, physics_geometry_info=None, tf=tf)

class NodeF(TerminalNode):
    def __init__(self, tf):
        super().__init__(observed=True, physics_geometry_info=None, tf=tf)

class NodeE(TerminalNode):
    def __init__(self, tf):
        super().__init__(observed=True, physics_geometry_info=None, tf=tf)

class NodeD(IndependentSetNode):
    GRule = ProductionRule(
        child_type=NodeG,
        xyz_rule=WorldBBoxRule.from_bounds(lb=torch.zeros(3), ub=torch.ones(3)),
        rotation_rule=UnconstrainedRotationRule()
    )
    def __init__(self, tf):
        super().__init__(
            rule_probs=torch.tensor([0.5]),
            observed=True,
            physics_geometry_info=None,
            tf=tf
        )
    @classmethod
    def generate_rules(cls):
        return [NodeD.GRule]

class NodeC(GeometricSetNode):
    FRule = ProductionRule(
        child_type=NodeF,
        xyz_rule=WorldBBoxRule.from_bounds(lb=torch.zeros(3), ub=torch.ones(3)),
        rotation_rule=UnconstrainedRotationRule()
    )
    def __init__(self, tf):
        super().__init__(
            p=torch.tensor(0.2),
            max_children=5,
            observed=False,
            physics_geometry_info=None,
            tf=tf
        )
    @classmethod
    def generate_rules(cls):
        return [NodeC.FRule]

class NodeB(OrNode):
    DRule = ProductionRule(
        child_type=NodeD,
        xyz_rule=AxisAlignedBBoxRule.from_bounds(lb=torch.zeros(3), ub=torch.ones(3)),
        rotation_rule=UniformBoundedRevoluteJointRule.from_bounds(axis=torch.tensor([0., 0., 1.]), lb=-1., ub=1.)
    )
    ERule = ProductionRule(
        child_type=NodeE,
        xyz_rule=AxisAlignedBBoxRule.from_bounds(lb=torch.zeros(3), ub=torch.ones(3)),
        rotation_rule=UnconstrainedRotationRule()
    )
    def __init__(self, tf):
        super().__init__(
            rule_probs=torch.tensor([0.75, 0.25]),
            observed=False,
            physics_geometry_info=None,
            tf=tf
        )
    @classmethod
    def generate_rules(cls):
        return [NodeB.DRule, NodeB.ERule]


class NodeA(AndNode):
    BRule = ProductionRule(
        child_type=NodeB,
        xyz_rule=WorldBBoxRule.from_bounds(lb=torch.zeros(3), ub=torch.ones(3)),
        rotation_rule=UnconstrainedRotationRule()
    )
    CRule = ProductionRule(
        child_type=NodeC,
        xyz_rule=WorldBBoxRule.from_bounds(lb=torch.zeros(3), ub=torch.ones(3)),
        rotation_rule=UnconstrainedRotationRule()
    )
    def __init__(self, tf):
        super().__init__(
            observed=False, physics_geometry_info=None, tf=tf
        )
    @classmethod
    def generate_rules(cls):
        return [NodeA.BRule, NodeA.CRule]


