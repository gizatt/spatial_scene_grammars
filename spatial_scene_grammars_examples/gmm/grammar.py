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

root -> one of the mixtures -> observed
'''

class Point(TerminalNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=True
        )


class GaussianMode(AndNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False
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
            )
        ]

Modes = tuple([
    type("GaussianMode_%d" % k, (GaussianMode,), {})
    for k in range(3)
])
# Register class name in globals so we can
# pickle these types.
# https://stackoverflow.com/questions/11658511/pickling-dynamically-generated-classes
for mode in Modes:
    globals()[mode.__name__] = mode

class Root(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False,
            rule_probs=torch.ones(len(Modes))
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=mode,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ) for mode in Modes
        ]