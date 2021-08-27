from copy import deepcopy
from collections import namedtuple
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.contrib.autoname import scope
import torch
from torch.distributions import constraints

from pytorch3d.transforms.rotation_conversions import (
    quaternion_to_matrix, matrix_to_quaternion,
    axis_angle_to_matrix, quaternion_to_axis_angle
)

from .distributions import UniformWithEqualityHandling
from .torch_utils import ConstrainedParameter

import pydrake
from pydrake.all import (
    CoulombFriction,
    SpatialInertia,
    UnitInertia
)


class ProductionRule():
    '''
        Rule by which a child node's pose is derived
        from its parent.

        Produced by mixing together two sub-rules that
            control the xyz and rotation relationships.

        Groups together a few interfaces that share the
        same undlerying math:
            - Sample the child given the parent.
            - Compute the log-prob of the child given the parent.
            - Add mathematical programming constraints reflecting
              the child being feasible w.r.t. the parent.
            - Add mathematical programming costs reflecting the
              score of the child relative to the parent.        
    '''

    def __init__(self, child_type, xyz_rule, rotation_rule):
        assert isinstance(xyz_rule, XyzProductionRule)
        assert isinstance(rotation_rule, RotationProductionRule)
        self.child_type = child_type
        self.xyz_rule = xyz_rule
        self.rotation_rule = rotation_rule

    def sample_child(self, parent):
        xyz = self.xyz_rule.sample_xyz(parent)
        rotmat = self.rotation_rule.sample_rotation(parent)
        tf = torch.empty(4, 4)
        tf[:3, :3] = rotmat[:, :]
        tf[:3, 3] = xyz[:]
        tf[3, :] = torch.tensor([0., 0., 0., 1.])
        return self.child_type(tf=tf)

    def score_child(self, parent, child):
        return (
            self.xyz_rule.score_child(parent, child) +
            self.rotation_rule.score_child(parent, child)
        )


## XYZ Production rules
class XyzProductionRule():
    '''
        Instructions for how to produce a child's position
        from the parent.
    '''
    def __init__(self):
        pass
    def sample_xyz(self, parent):
        raise NotImplementedError()
    def score_child(self, parent, child):
        raise NotImplementedError()


class WorldBBoxRule(XyzProductionRule):
    ''' Child xyz is uniformly chosen in [lb, ub] in world frame,
        without relationship to the parent.'''
    def __init__(self, lb, ub):
        assert isinstance(lb, torch.Tensor) and lb.shape == (3,)
        assert isinstance(ub, torch.Tensor) and ub.shape == (3,)
        self.lb = lb
        self.ub = ub
        self.xyz_dist = UniformWithEqualityHandling(lb, ub)
        super().__init__()

    def sample_xyz(self, parent):
        return self.xyz_dist.rsample()

    def score_child(self, parent, child):
        return self.xyz_dist.log_prob(child.translation).sum()


class AxisAlignedBBoxRule(XyzProductionRule):
    ''' Child xyz is parent xyz + a uniform offset in [lb, ub]
        in world frame.

        TODO(gizatt) Add support for lb = ub; this requires
        special wrapping around Uniform to handle the equality
        cases as Delta distributions. '''
    def __init__(self, lb, ub):
        assert isinstance(lb, torch.Tensor) and lb.shape == (3,)
        assert isinstance(ub, torch.Tensor) and ub.shape == (3,)
        self.lb = lb
        self.ub = ub
        self.xyz_offset_dist = UniformWithEqualityHandling(lb, ub)
        super().__init__()

    def sample_xyz(self, parent):
        return parent.translation + self.xyz_offset_dist.rsample()

    def score_child(self, parent, child):
        xyz_offset = child.translation - parent.translation
        return self.xyz_offset_dist.log_prob(xyz_offset).sum()


## Rotation production rules
class RotationProductionRule():
    '''
        Instructions for how to produce a child's position
        from the parent.
    '''
    def __init__(self):
        pass
    def sample_rotation(self, parent):
        raise NotImplementedError()
    def score_child(self, parent, child):
        raise NotImplementedError()


class UnconstrainedRotationRule(RotationProductionRule):
    '''
        Child rotation is randomly chosen from all possible
        rotations with no relationship to parent.
    '''
    def __init__(self):
        super().__init__()

    def sample_rotation(self, parent):
        # Sample random unit quaternion and convert to rotation.
        random_quat = torch.zeros(4)
        sample_dist = dist.Normal(torch.zeros(4), torch.ones(4))
        while torch.norm(random_quat, 2) < 1E-3:
            # Repeat until we get a nonzero quaternion (to catch
            # the off chance we sample all zeros).
            random_quat = sample_dist.rsample()
        random_quat = random_quat / torch.norm(random_quat, p=2)
        R = quaternion_to_matrix(random_quat.unsqueeze(0))[0, ...]
        return R

    def score_child(self, parent, child):
        # Score is uniform over SO(3). I'm picking a random
        # quaternion from (half)* surface of the 4D hypersphere
        # and converting to a rotation matrix; so density is
        # 1 / (half area of 4D unit hypersphere).
        # Area of 4D unit hypersphere is (2 * pi^2 * R^3)
        # -> 1 / pi^2
        print("TODO: CHECK ME FOR CORRECTNESS")
        return np.log(1. / np.pi**2)


class UniformBoundedRevoluteJointRule(RotationProductionRule):
    '''
        Child rotation is randomly chosen uniformly from a bounded
        range of angles around a revolute joint axis about the parent.
    '''
    def __init__(self, axis, lb, ub):
        assert isinstance(axis, torch.Tensor) and axis.shape == (3,)
        assert isinstance(lb, float) and isinstance(ub, float) and lb <= ub
        self.axis = axis
        self.lb = lb
        self.ub = ub
        self._angle_dist = UniformWithEqualityHandling(lb, ub)

    def sample_rotation(self, parent):
        angle = self._angle_dist.rsample()
        angle_axis = self.axis * angle
        R_offset = axis_angle_to_matrix(angle_axis.unsqueeze(0))[0, ...]
        R = torch.matmul(parent.rotation, R_offset)
        return R

    def _recover_relative_angle_axis(self, parent, child):
        # Recover angle-axis
        relative_R = torch.matmul(torch.transpose(parent.rotation, 0, 1), child.rotation)

        axis_angle = quaternion_to_axis_angle(matrix_to_quaternion(relative_R))
        angle = torch.norm(axis_angle, p=2)
        axis = axis_angle / angle
        if torch.abs(angle) > 0 and not torch.allclose(axis,  self.axis, atol=1E-4, rtol=1E-4):
            if torch.allclose(-axis, self.axis, atol=1E-4, rtol=1E-4):
                # Flip axis and angle to make them match
                axis = -axis
                angle = -angle
            else:
                # No saving this; axis doesn't match.
                raise ValueError("Child illegalal rotated from parent: %s vs %s" % (axis, self.axis))
        return angle, axis
        
    def score_child(self, parent, child):
        print("TEST ME")
        if (self.ub - self.lb) >= 2. * np.pi:
            # Uniform rotation in 1D base case
            return np.log(1. / (2. * np.pi))
        angle, axis = self._recover_relative_angle_axis(parent, child)
        return self._angle_dist.log_prob(angle)
