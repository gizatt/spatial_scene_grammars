from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import glob

import pydrake
from pydrake.all import (
    Box, RollPitchYaw, RigidTransform, Parser
)

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoname import name_count

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.tree import *
from spatial_scene_grammars.factors import *
from spatial_scene_grammars.transform_utils import *
from spatial_scene_grammars.drake_interop import *


class NonpenetrationConstraint(ContinuousVariableConstraint):
    def __init__(self, allowed_penetration_margin=0.0):
        ''' penetration_margin > 0, specifies penetration amounts we'll allow. '''
        self.allowed_penetration_margin = allowed_penetration_margin
        super().__init__(lower_bound=torch.tensor(-np.inf),
                         upper_bound=torch.tensor(0.0))

    def eval(self, scene_tree):
        # Convert into MBP/SceneGraph and find penetrating point pairs
        # using that machinery. For each penetrating pair, get the
        # penetrating points and normal in body frame, and then recompute
        # the penalty score using (pb - pa)^n directly, applying transformation
        # into world frame using torch tensors from the scene tree to get
        # the continuous parameters (partially) autodiffable from torch.

        # This will NOT work for dynamically sized objects, since
        # only the transform gradients make it through.

        # TODO: Caching? In HMC, we rebuild the same tree
        # repeatedly. A fast scene-tree-topology-hash check
        # could enable caching here.

        builder, mbp, sg, _, body_id_to_node_map = \
            compile_scene_tree_to_mbp_and_sg(scene_tree)
        mbp.Finalize()
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()

        # Get colliding point pairs and their normal info
        # using the SceneGraph.
        mbp_context = diagram.GetMutableSubsystemContext(mbp, diagram_context)
        sg_context = diagram.GetMutableSubsystemContext(sg, diagram_context)
        query_object = sg.get_query_output_port().Eval(sg_context)
        point_pairs = query_object.ComputePointPairPenetration()
        total_score = torch.tensor([0.])
        for point_pair in point_pairs:
            assert point_pair.depth > 0
            if point_pair.depth <= self.allowed_penetration_margin:
                continue
            model_id_A, body_id_A = resolve_sg_proximity_id_to_mbp_id(
                sg, mbp, point_pair.id_A)
            model_id_B, body_id_B = resolve_sg_proximity_id_to_mbp_id(
                sg, mbp, point_pair.id_B)

            # Transform the points on the objects and the normal vector
            # into the object frames
            tf_WA = mbp.CalcRelativeTransform(
                mbp_context, frame_A=mbp.world_frame(),
                frame_B=mbp.get_body(body_id_A).body_frame())
            tf_WB = mbp.CalcRelativeTransform(
                mbp_context, frame_A=mbp.world_frame(),
                frame_B=mbp.get_body(body_id_B).body_frame())
            p_ACa = torch.tensor(tf_WA.inverse().multiply(point_pair.p_WCa))
            p_BCb = torch.tensor(tf_WB.inverse().multiply(point_pair.p_WCb))
            nhat_BA_B = torch.tensor(tf_WB.inverse().rotation().multiply(point_pair.nhat_BA_W))
            
            # Retrieve their corresponding node TFs and compute error term
            tf_WA_torch = body_id_to_node_map[body_id_A].tf
            p_WCa_torch = torch.matmul(tf_WA_torch[:3, :3], p_ACa) + tf_WA_torch[:3, 3]
            tf_WB_torch = body_id_to_node_map[body_id_B].tf
            p_WCb_torch = torch.matmul(tf_WB_torch[:3, :3], p_BCb) + tf_WB_torch[:3, 3]
            nhat_BA_W_torch = torch.matmul(tf_WB_torch[:3, :3], nhat_BA_B)
            depth = torch.sum((p_WCb_torch - p_WCa_torch) * nhat_BA_W_torch)

            # Sanity-check that we did things right.
            assert np.abs(point_pair.depth - depth.item()) < 1E-6
            total_score += depth*10.0 # TODO Why the scaling? It's tied to my factor definition
                                      # for evaluation constraints with HMC -- currently errors
                                      # on the ~0.01 scale get reasonable probability. This
                                      # makes sure contact is actually really tightly handled.

        return total_score
        
            