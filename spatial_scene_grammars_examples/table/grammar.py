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
Table -> place settings and shared dishware
Shared dishware -> Tea kettle, food plates, bamboo steamer towers
Place settings - > cup, plate, chopsticks, chair?
'''

class PersonalPlate(TerminalNode):
    KEEPOUT_RADIUS = 0.14
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "models/plates_cups_and_bowls/plates/Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring/model_simplified.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

# These are ommitted from the grammar right now as they
# cause SNOPT+simulation to struggle quite a bit due to
# their small geometry
class FirstChopstick(IndependentSetNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.0])),
            "models/misc/chopstick/model.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            rule_probs=torch.tensor([0.99]),
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=SecondChopstick,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0., 0.0, 0.0]),
                    variance=torch.tensor([0.0001, 0.0001, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1000])
                )
            )
        ]
        return rules
class SecondChopstick(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.0])),
            "models/misc/chopstick/model.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class Teacup(TerminalNode):
    KEEPOUT_RADIUS = 0.07
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "models/plates_cups_and_bowls/cups/coffee_cup_white/model_simplified.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class Teapot(TerminalNode):
    KEEPOUT_RADIUS = 0.1
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "models/plates_cups_and_bowls/cups/Threshold_Porcelain_Teapot_White/model_simplified.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class ServingDish(TerminalNode):
    KEEPOUT_RADIUS = 0.2
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "models/plates_cups_and_bowls/plates/Threshold_Dinner_Plate_Square_Rim_White_Porcelain/model_simplified.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

class Null(TerminalNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
class SteamerBottom(OrNode):
    KEEPOUT_RADIUS=0.12
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "models/misc/steamer_bottom/model.sdf"
        )
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.25, 0.4, 0.35]),
            physics_geometry_info=geom,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=SteamerBottom,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0., 0.0, 0.1]),
                    variance=torch.tensor([0.0001, 0.0001, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([10000, 10000, 1])
                )
            ),
            ProductionRule(
                child_type=SteamerTop,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0., 0.0, 0.1]),
                    variance=torch.tensor([0.0001, 0.0001, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([10000, 10000, 1])
                )
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
        ]
        return rules

class SteamerTop(TerminalNode):
    KEEPOUT_RADIUS=0.12
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0., 0.])),
            "models/misc/steamer_top/model.sdf"
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )

TabletopObjectTypes = (
    PersonalPlate, Teacup, Teapot, ServingDish, SteamerBottom, SteamerTop,
    FirstChopstick, SecondChopstick
)


class PersonalPlateAndTeacup(AndNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=PersonalPlate,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.001, 0.001, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            ),
            ProductionRule(
                child_type=Teacup,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.25, 0.0, 0.00]),
                    variance=torch.tensor([0.001, 0.005, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            )
        ]
        return rules

class PlaceSetting(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.7, 0.025, 0.025, 0.2]),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=PersonalPlateAndTeacup,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=PersonalPlate,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.001, 0.001, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            ),
            ProductionRule(
                child_type=Teacup,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.25, 0.0, 0.00]),
                    variance=torch.tensor([0.001, 0.005, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            )
            #ProductionRule(
            #    child_type=FirstChopstick,
            #    xyz_rule=ParentFrameGaussianOffsetRule(
            #        mean=torch.tensor([0.0, 0.0, 0.02]),
            #        variance=torch.tensor([0.005, 0.005, 0.0001])),
            #    rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
            #        RotationMatrix(RollPitchYaw(0., np.pi/2., 0.)), np.array([1000, 1000, 1])
            #    )
            #)
        ]
        return rules

class PlaceSettings(AndNode):
    DISTANCE_FROM_CENTER = 0.5
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
                child_type=PlaceSetting,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([-cls.DISTANCE_FROM_CENTER, 0., 0.])
                ),
                rotation_rule=SameRotationRule(offset=torch.tensor(RotationMatrix(RollPitchYaw(0., 0., 0.)).matrix()))
            ),
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([cls.DISTANCE_FROM_CENTER, 0., 0.]),
                ),
                rotation_rule=SameRotationRule(offset=torch.tensor(RotationMatrix(RollPitchYaw(0., 0., np.pi)).matrix()))
            ),
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0., cls.DISTANCE_FROM_CENTER, 0.])
                ),
                rotation_rule=SameRotationRule(offset=torch.tensor(RotationMatrix(RollPitchYaw(0., 0., -np.pi/2.)).matrix()))
            ),
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0., -cls.DISTANCE_FROM_CENTER, 0.])
                ),
                rotation_rule=SameRotationRule(offset=torch.tensor(RotationMatrix(RollPitchYaw(0., 0., np.pi/2.)).matrix()))
            )
        ]


class SharedDishes(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.5, max_children=2, start_at_one=True),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=ServingDish,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.02, 0.02, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            )
        ]

class SharedTeapots(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(p=0.5, max_children=2, start_at_one=True),
            physics_geometry_info=None,
            observed=False
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Teapot,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.01, 0.01, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            )
        ]

class SharedSteamers(RepeatingSetNode):
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
                child_type=SteamerBottom,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.025, 0.025, 0.0001])),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1000, 1000, 1])
                )
            )
        ]

class SharedStuff(IndependentSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.9, 0.6, 0.9]),
            physics_geometry_info=None,
            observed=False
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
                child_type=SharedSteamers,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule()
            ),
        ]


class Table(AndNode):
    WIDTH = 1.25
    # Place settings + misc common dishware
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom_tf = torch.eye(4)
        geom_tf[2, 3] = -0.8
        geom.register_model_file(geom_tf, "models/misc/cafe_table/model.sdf")
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True
        )
    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=PlaceSettings,
                xyz_rule=SamePositionRule(offset=torch.tensor([0.0, 0.0, 0.025])),
                rotation_rule=SameRotationRule()
            ),
            ProductionRule(
                child_type=SharedStuff,
                xyz_rule=SamePositionRule(offset=torch.tensor([0.0, 0.0, 0.025])),
                rotation_rule=SameRotationRule()
            ),
        ]


# Corresponding constraint set for the grammar.
class ObjectsOnTableConstraint(PoseConstraint):
    def __init__(self):
        lb = torch.tensor([-Table.WIDTH/2.+0.15, -Table.WIDTH/2.+0.15, -0.02])
        ub = torch.tensor([Table.WIDTH/2.-0.15, Table.WIDTH/2.-0.15, 1.])
        super().__init__(
            lower_bound=lb,
            upper_bound=ub
        )
    def eval(self, scene_tree):
        tables = scene_tree.find_nodes_by_type(Table)
        xyzs = [] # in parent table frame
        for table in tables:
            # Collect table children xyz poses in table frame
            objs = [node for node in scene_tree.get_children_recursive(table) if isinstance(node, TabletopObjectTypes)]
            for obj in objs:
                offset = torch.matmul(table.rotation.T, obj.translation - table.translation)
                xyzs.append(offset)
        if len(xyzs) > 0:
            return torch.stack(xyzs, axis=0)
        else:
            return torch.empty(size=(0, 3))
    def add_to_ik_prog(self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map):
        raise NotImplementedError()

class ObjectSpacingConstraint(PoseConstraint):
    # Objects all a minimum distance apart on tabletop
    def __init__(self):
        lb = torch.tensor([0.])
        ub = torch.tensor([np.inf])
        super().__init__(
            lower_bound=lb,
            upper_bound=ub
        )
    def eval(self, scene_tree):
        tables = scene_tree.find_nodes_by_type(Table)
        all_dists = []
        for table in tables:
            objs = [node for node in scene_tree.get_children_recursive(table) if isinstance(node, TabletopObjectTypes)
                    and not isinstance(scene_tree.get_parent(node), SteamerBottom)
                    and not isinstance(node, FirstChopstick)
                    and not isinstance(node, SecondChopstick)]
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

class TallStackConstraint(StructureConstraint):
    # The largest stack of steamers is at least 4 steamers tall.
    def __init__(self):
        lb = torch.tensor([4.])
        ub = torch.tensor([np.inf])
        super().__init__(
            lower_bound=lb,
            upper_bound=ub
        )
    def eval(self, scene_tree):
        steamers = scene_tree.find_nodes_by_type(SteamerBottom)
        tallest_stack = 0
        # For each steamer, count how many parents it has that
        # are SteamerBottoms before hitting something else.
        # This #+1 is the number of steamers in the stack.
        for steamer in steamers:
            current_steamer = steamer
            stack = 0
            while isinstance(current_steamer, SteamerBottom):
                stack += 1
                current_steamer = scene_tree.get_parent(current_steamer)
            tallest_stack = max(tallest_stack, stack)
        return torch.tensor([tallest_stack])

class NumStacksConstraint(StructureConstraint):
    # At least 3 stacks.
    def __init__(self):
        lb = torch.tensor([3.])
        ub = torch.tensor([np.inf])
        super().__init__(
            lower_bound=lb,
            upper_bound=ub
        )
    def eval(self, scene_tree):
        shared_steamers = list(scene_tree.find_nodes_by_type(SharedSteamers))
        assert len(shared_steamers) <= 1
        if len(shared_steamers) == 0:
            return torch.zeros(1)

        return torch.tensor([
            len(list(scene_tree.successors(shared_steamers[0])))
        ])
        