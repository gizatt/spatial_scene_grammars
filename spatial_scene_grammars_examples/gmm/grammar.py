import torch
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.drake_interop import *

import pydrake
import pydrake.geometry as pydrake_geom
from pydrake.all import (
    RollPitchYaw,
    RigidTransform
)

'''
Gaussian mixture model. This is a little funky, since it mixes
the parameters (mode means + variances) with variables (the
sampled points), but it should work for inferring GMM parameters.

root --(rule implementing one of the mixtures)-> observed
'''

class Point(TerminalNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=True
        )

class Root(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False,
            rule_probs=torch.ones(3)
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Point,
                xyz_rule=WorldFrameGaussianOffsetRule(
                    mean=torch.zeros(3),
                    variance=torch.tensor([1.0, 1.0, 1.0])),
                rotation_rule=SameRotationRule()
            ) for k in range(3)
        ]