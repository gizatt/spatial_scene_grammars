import glob
import os
from functools import lru_cache

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
Grammar describing small piles and clusters of packages.

Simple structure -- mostly here to prototype sampling code.

Root -> [AssortedStacks, AssortedBoxes]
AssortedStacks -> Stacks x N
Stack -> Boxes x N, vertically
AssortedBoxes -> Boxes x N
Box -> Reified Box Geometry
'''

eps = 1E-2

class ObjectModel(TerminalNode):
    sdf = None
    def __init__(self, tf):
        assert self.sdf is not None, "Don't instantiate ObjectModel itself; use a reified version."
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            self.sdf
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

# Use caching to try to avoid re-generating objects, which messes with pickling
# and node identity checking.
@lru_cache(maxsize=None)
def reify_models_from_folder_to_object_types(folder):
    print("Generating from folder ", folder)
    sdfs = glob.glob(os.path.join(folder, "*/box.sdf"))[:5]
    new_types = []
    for sdf in sdfs:
        new_types.append(
            type(
                "%s" % os.path.split(sdf)[0].replace("/", ":"),
                (ObjectModel,),
                {"sdf": sdf}
            )
        )
    # Make these dynamically generated types pickle-able
    # by registering them globally. Dangerous -- make sure
    # their names are unique!
    # https://stackoverflow.com/questions/11658511/pickling-dynamically-generated-classes
    for new_type in new_types:
        print(new_type.__name__, ": ", new_type)
        globals()[new_type.__name__] = new_type

    return new_types


BoxModels = reify_models_from_folder_to_object_types(
    "boxes"
)

class BoxGroup(AndNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom.register_geometry(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., -1.0])),
            pydrake_geom.Box(20., 20., 2.)
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=box_model,
                xyz_rule=AxisAlignedGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.5*k]),
                    variance=torch.tensor([0.01, 0.01, 0.01])),
                rotation_rule=WorldFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), [1., 1., 1]
                )
            )
            for k, box_model in enumerate(BoxModels)
        ]