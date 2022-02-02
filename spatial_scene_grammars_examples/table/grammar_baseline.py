import glob
import os
from functools import lru_cache

import torch
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.drake_interop import *
from spatial_scene_grammars.constraints import *


import pydrake
import pydrake.geometry as pydrake_geom
from pydrake.all import (
    RollPitchYaw,
    RigidTransform
)

''' 
Same terminal set as in grammar.py, but
without as much intermediate structure:
- PlaceSettings is replaced with PersonalPlates, Teacups,
which produce random numbers of those things.
- SteamerBottom is Terminal, and are produced from SteamerBottoms.
- SteamerTop is produced from SteamerTops.
'''

# Need full import path to match how these types are imported
# in jupyter notebooks and saved out when pickling... seems dumb,
# but don't change it.
from spatial_scene_grammars_examples.table.grammar import (
    PersonalPlate, Teacup, Teapot, ServingDish, SteamerTop, SharedDishes, SharedTeapots,
    ObjectSpacingConstraint, ObjectsOnTableConstraint
)

class PersonalPlates(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.6, max_children=4, start_at_one=True),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=PersonalPlate,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.025, 0.025, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            )
        ]

class Teacups(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.6, max_children=4, start_at_one=True),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Teacup,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.025, 0.025, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            )
        ]

class SteamerBottomBaseline(TerminalNode):
    KEEPOUT_RADIUS=0.12
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "models/misc/steamer_bottom/model.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

TabletopObjectTypesBaseline = (
    PersonalPlate, Teacup, Teapot, ServingDish, SteamerBottomBaseline, SteamerTop,
)

class SharedSteamerBottoms(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.6, max_children=8, start_at_one=True),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=SteamerBottomBaseline,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.025, 0.025, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            )
        ]

class SharedSteamerTops(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.6, max_children=6, start_at_one=True),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=SteamerTop,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.025, 0.025, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            )
        ]


class TableBaseline(IndependentSetNode):
    WIDTH = 1.25
    # Place settings + misc common dishware
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom_tf = torch.eye(4)
        geom_tf[2, 3] = -0.8
        geom.register_model_file(geom_tf, "models/misc/cafe_table/model.sdf")
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.9, 0.75, 0.8, 0.9, 0.95, 0.95]),
            physics_geometry_info=geom,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=SharedTeapots,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=SharedDishes,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=SharedSteamerTops,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=SharedSteamerBottoms,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=PersonalPlates,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=Teacups,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
        ]


# Corresponding constraint set for the grammar.
class ObjectsOnTableConstraintBaseline(PoseConstraint):
    def __init__(self):
        lb = torch.tensor([-TableBaseline.WIDTH/2.+0.15, -TableBaseline.WIDTH/2.+0.15, -0.02])
        ub = torch.tensor([TableBaseline.WIDTH/2.-0.15, TableBaseline.WIDTH/2.-0.15, 1.])
        super().__init__(
            lower_bound=lb,
            upper_bound=ub
        )
    def eval(self, scene_tree):
        tables = scene_tree.find_nodes_by_type(TableBaseline)
        xyzs = [] # in parent table frame
        for table in tables:
            # Collect table children xyz poses in table frame
            objs = [node for node in scene_tree.get_children_recursive(table)
                    if isinstance(node, TabletopObjectTypesBaseline)]
            for obj in objs:
                offset = torch.matmul(table.rotation.T, obj.translation - table.translation)
                xyzs.append(offset)
        if len(xyzs) > 0:
            return torch.stack(xyzs, axis=0)
        else:
            return torch.empty(size=(0, 3))
    def add_to_ik_prog(self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map):
        raise NotImplementedError()

class ObjectSpacingConstraintBaseline(PoseConstraint):
    # Objects all a minimum distance apart on tabletop
    def __init__(self):
        lb = torch.tensor([0.])
        ub = torch.tensor([np.inf])
        super().__init__(
            lower_bound=lb,
            upper_bound=ub
        )
    def eval(self, scene_tree):
        tables = scene_tree.find_nodes_by_type(TableBaseline)
        all_dists = []
        for table in tables:
            objs = [node for node in scene_tree.get_children_recursive(table) if isinstance(node, TabletopObjectTypesBaseline)]
            if len(objs) <= 1:
                print("no objects")
                continue
            xys = torch.stack([obj.translation[:2] for obj in objs], axis=0)
            keepout_dists = torch.tensor([obj.KEEPOUT_RADIUS for obj in objs])
            N = xys.shape[0]
            xys_rowwise = xys.unsqueeze(1).expand(-1, N, -1)
            keepout_dists_rowwise = keepout_dists.unsqueeze(1).expand(-1, N)
            xys_colwise = xys.unsqueeze(0).expand(N, -1, -1)
            keepout_dists_colwise = keepout_dists.unsqueeze(0).expand(N, -1)
            dists = (xys_rowwise - xys_colwise).square().sum(axis=-1)
            keepout_dists = (keepout_dists_rowwise + keepout_dists_colwise)

            # Set target keepout dist of steamerbottom-steamerbottom and
            # steamerbottom-steamertop pairs to 0 to allow them to
            # be on top of each other.
            filter_inds = np.array([
                k for k, obj in enumerate(objs)
                if isinstance(obj, (SteamerTop, SteamerBottomBaseline))
            ])
            x, y = np.meshgrid(filter_inds, filter_inds)
            keepout_dists[x, y] = 0.

            # Get only lower triangular non-diagonal elems
            rows, cols = torch.tril_indices(N, N, -1)
            # Make sure pairwise dists > keepout dists
            dists = (dists - keepout_dists.square())[rows, cols].reshape(-1, 1)
            all_dists.append(dists)
        if len(all_dists) > 0:
            return torch.cat(all_dists, axis=0)
        else:
            return torch.empty(size=(0, 1))
    def add_to_ik_prog(self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map):
        raise NotImplementedError()
