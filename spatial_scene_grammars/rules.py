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

    def score_child(self, parent, child, verbose=False):
        xyz_part = self.xyz_rule.score_child(parent, child)
        rot_part = self.rotation_rule.score_child(parent, child)
        if verbose:
            print("XYZ: ", xyz_part.item())
            print("Rot: ", rot_part.item())
        return xyz_part + rot_part

    def get_parameter_prior(self):
        # Returns a tuple of the parameter prior dicts for the
        # xyz and rotation rules.
        # TODO(gizatt) Reorganization of ProductionRule definitions
        # might allow this to be a classmethod, which would match how
        # Node definitions are set up.
        return (
            self.xyz_rule.get_parameter_prior(),
            self.rotation_rule.get_parameter_prior()
        )
    @property
    def parameters(self):
        # Returns a tuple of the parameters dicts for the
        # xyz and rotation rules.
        return (
            self.xyz_rule.parameters,
            self.rotation_rule.parameters
        )

    @parameters.setter
    def parameters(self, parameters):
        assert isinstance(parameters, (tuple, list)) and len(parameters) == 2
        self.xyz_rule.parameters = parameters[0]
        self.rotation_rule.parameters = parameters[1]


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

    @classmethod
    def get_parameter_prior(cls):
        # Should return a dict of pyro Dists with supports matching the parameter
        # space of the Rule subclass being used.
        raise NotImplementedError("Child rule should implement parameter priors.")
    @property
    def parameters(self):
        # Should return a dict of torch Tensors representing the current
        # parameter settings for this node. These are *not* torch
        # parameters, and this is not a Pytorch module, since the
        # Torch parameters being optimized belong to the grammar / the
        # node type, not a given instantiated node.
        raise NotImplementedError(
            "Child class should implement parameters getter. Users should"
            " never have to do this."
        )
    @parameters.setter
    def parameters(self, parameters):
        raise NotImplementedError(
            "Child class should implement parameters setter. Users should"
            " never have to do this."
        )

class SamePositionRule(XyzProductionRule):
    ''' Child Xyz is identically parent xyz. '''
    def __init__(self):
        super().__init__()

    def sample_xyz(self, parent):
        return parent.translation
    def score_child(self, parent, child):
        return torch.tensor(0.)

    @classmethod
    def get_parameter_prior(cls):
        return {}
    @property
    def parameters(self):
        return {}
    @parameters.setter
    def parameters(self, parameters):
        if len(parameters.keys()) > 0:
            raise ValueError("SamePositionRule has no parameters.")


class WorldBBoxRule(XyzProductionRule):
    ''' Child xyz is uniformly chosen in [center, width] in world frame,
        without relationship to the parent.'''
    @classmethod
    def from_bounds(cls, lb, ub):
        assert all(ub >= lb)
        return cls(center=(ub + lb) / 2., width=ub - lb)
    def __init__(self, center, width):
        assert isinstance(center, torch.Tensor) and center.shape == (3,)
        assert isinstance(width, torch.Tensor) and width.shape == (3,)
        self.parameters = {"center": center, "width": width}
        
        super().__init__()

    def sample_xyz(self, parent):
        return pyro.sample("WorldBBoxRule_xyz", self.xyz_dist)
    def score_child(self, parent, child):
        return self.xyz_dist.log_prob(child.translation).sum()

    @classmethod
    def get_parameter_prior(cls):
        # Default prior is a unit Normal for center,
        # and Uniform-distributed width on some reasonable range.
        return {
            "center": dist.Normal(torch.zeros(3), torch.ones(3)),
            "width": dist.Uniform(torch.zeros(3), torch.ones(3)*10.)
        }
    @property
    def parameters(self):
        # Parameters of a BBoxRule is the *center* and *width*,
        # since those are easiest to constrain: width should be
        # >= 0.
        return {
            "center": self.center,
            "width": self.width
        }
    @parameters.setter
    def parameters(self, parameters):
        self.center = parameters["center"]
        self.width = parameters["width"]
        self.xyz_dist = UniformWithEqualityHandling(self.lb, self.ub)
    @property
    def lb(self):
        return self.center - self.width / 2.
    @property
    def ub(self):
        return self.center + self.width / 2.


class AxisAlignedBBoxRule(WorldBBoxRule):
    ''' Child xyz is parent xyz + a uniform offset in [lb, ub]
        in world frame.

        TODO(gizatt) Add support for lb = ub; this requires
        special wrapping around Uniform to handle the equality
        cases as Delta distributions.'''
    def sample_xyz(self, parent):
        offset = pyro.sample("AxisAlignedBBoxRule_xyz", self.xyz_dist)
        return parent.translation + offset
    def score_child(self, parent, child):
        xyz_offset = child.translation - parent.translation
        return self.xyz_dist.log_prob(xyz_offset).sum()


class AxisAlignedGaussianOffsetRule(XyzProductionRule):
    ''' Child xyz is diagonally-Normally distributed relative to parent in world frame.'''
    def __init__(self, mean, variance):
        assert isinstance(mean, torch.Tensor) and mean.shape == (3,)
        assert isinstance(variance, torch.Tensor) and variance.shape == (3,)
        self.parameters = {"mean": mean, "variance": variance}
        super().__init__()

    def sample_xyz(self, parent):
        return parent.translation + pyro.sample("AxisAlignedGaussianOffsetRule_xyz", self.xyz_dist)
    def score_child(self, parent, child):
        return self.xyz_dist.log_prob(child.translation - parent.translation).sum()

    @classmethod
    def get_parameter_prior(cls):
        return {
            "mean": dist.Normal(torch.zeros(3), torch.ones(3)),
            "variance": dist.Uniform(torch.zeros(3)+1E-6, torch.ones(3)*10.) # TODO: Inverse gamma is better
        }
    @property
    def parameters(self):
        return {
            "mean": self.mean,
            "variance": self.variance
        }
    @parameters.setter
    def parameters(self, parameters):
        self.mean = parameters["mean"]
        self.variance = parameters["variance"]
        self.xyz_dist = dist.Normal(self.mean, self.variance)


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

    @classmethod
    def get_parameter_prior(cls):
        # Should return a dict of pyro Dists with supports matching the parameter
        # space of the Rule subclass being used.
        raise NotImplementedError("Child rule should implement parameter priors.")
    @property
    def parameters(self):
        # Should return a dict of torch Tensors representing the current
        # parameter settings for this node. These are *not* torch
        # parameters, and this is not a Pytorch module, since the
        # Torch parameters being optimized belong to the grammar / the
        # node type, not a given instantiated node.
        raise NotImplementedError(
            "Child class should implement parameters getter. Users should"
            " never have to do this."
        )
    @parameters.setter
    def parameters(self, parameters):
        raise NotImplementedError(
            "Child class should implement parameters setter. Users should"
            " never have to do this."
        )


class SameRotationRule(RotationProductionRule):
    ''' Child Xyz is identically parent xyz. '''
    def __init__(self):
        super().__init__()

    def sample_rotation(self, parent):
        return parent.rotation
    def score_child(self, parent, child):
        return torch.tensor(0.)

    @classmethod
    def get_parameter_prior(cls):
        return {}
    @property
    def parameters(self):
        return {}
    @parameters.setter
    def parameters(self, parameters):
        if len(parameters.keys()) > 0:
            raise ValueError("SameRotationRule has no parameters.")

class UnconstrainedRotationRule(RotationProductionRule):
    '''
        Child rotation is randomly chosen from all possible
        rotations with no relationship to parent.
    '''
    def __init__(self):
        super().__init__()

    def sample_rotation(self, parent):
        # Sample random unit quaternion via
        # http://planning.cs.uiuc.edu/node198.html (referencing
        # Honkai's implementation in Drake), and convert to rotation
        # matrix.
        # TODO(gizatt) I've chosen the uniform bounds so that
        # the density of the uniform at any sample point matches the
        # expected density for sampling from SO(3); then I rescale
        # back down to unit-interval. Can I justify this by
        # demonstrating that this scaling makes log abs det Jacobian 1,
        # so I'm effectively counteracting the rescaling this whole
        # transformation is applying?
        # Expected density = 1 / pi^2
        # Actual density 1/(max)^3
        # -> Ub should be = pi^(2/3)
        ub = torch.ones(3) * np.pi**(2./3.)
        u = pyro.sample("UnconstrainedRotationRule_u", dist.Uniform(torch.zeros(3), ub))/ub
        random_quat = torch.tensor([
            torch.sqrt(1. - u[0]) * torch.sin(2. * np.pi * u[1]),
            torch.sqrt(1. - u[0]) * torch.cos(2. * np.pi * u[1]),
            torch.sqrt(u[0]) * torch.sin(2. * np.pi * u[2]),
            torch.sqrt(u[0]) * torch.cos(2. * np.pi * u[2])
        ])
        assert torch.isclose(random_quat.square().sum(), torch.ones(1))
        R = quaternion_to_matrix(random_quat.unsqueeze(0))[0, ...]
        return R

    def score_child(self, parent, child):
        # Score is uniform over SO(3). I'm picking a random
        # quaternion from (half)* surface of the 4D hypersphere
        # and converting to a rotation matrix; so density is
        # 1 / (half area of 4D unit hypersphere).
        # Area of 4D unit hypersphere is (2 * pi^2 * R^3)
        # -> 1 / pi^2
        # Agrees with https://marc-b-reynolds.github.io/quaternions/2017/11/10/AveRandomRot.html
        # TODO(gizatt) But probably doesn't agree with forward sample?
        return torch.log(torch.ones(1) / np.pi**2)

    @classmethod
    def get_parameter_prior(cls):
        return {}
    @property
    def parameters(self):
        return {}
    @parameters.setter
    def parameters(self, parameters):
        if len(parameters.keys()) > 0:
            raise ValueError("RotationProductionRule has no parameters.")


def recover_relative_angle_axis(parent, child, target_axis, zero_angle_width=1E-2, allowed_axis_diff=10. * np.pi/180.):
    # Recover angle-axis relationship between a parent and child.
    # Thrrows if we can't find a rotation axis between the two within
    # requested diff of our expected axis.
    relative_R = torch.matmul(torch.transpose(parent.rotation, 0, 1), child.rotation)
    axis_angle = quaternion_to_axis_angle(matrix_to_quaternion(relative_R))
    angle = torch.norm(axis_angle, p=2)

    # *Why* is this tolerance so high? This is ridiculous
    if angle <= zero_angle_width:
        return torch.tensor(0.), target_axis

    axis = axis_angle / angle
    axis_misalignment = torch.acos(torch.clip((axis * target_axis).sum(), -1., 1.))
    if torch.abs(angle) > 0 and axis_misalignment >= np.pi/2.:
        # Flipping axis will give us a close axis.
        axis = -axis
        angle = -angle

    axis_misalignment = torch.acos((axis * target_axis).sum()).item()
    if axis_misalignment >= allowed_axis_diff:
        # No saving this; axis doesn't match.
        raise ValueError("Parent %s, Child %s: " % (parent, child),
                         "Child illegal rotated from parent: %s vs %s, error of %f deg" % (axis, target_axis, axis_misalignment * 180./np.pi))
    return angle, axis

class UniformBoundedRevoluteJointRule(RotationProductionRule):
    '''
        Child rotation is randomly chosen uniformly from a bounded
        range of angles around a revolute joint axis about the parent.
    '''
    @classmethod
    def from_bounds(cls, axis, lb, ub):
        assert ub >= lb
        return cls(axis, (ub+lb)/2., ub-lb)
    def __init__(self, axis, center, width):
        assert isinstance(axis, torch.Tensor) and axis.shape == (3,)
        assert isinstance(center, (float, torch.Tensor)) and isinstance(width, (float, torch.Tensor)) and width >= 0 and width <= 2. * np.pi
        if isinstance(center, float):
            center = torch.tensor(center)
        if isinstance(width, float):
            width = torch.tensor(width)
        # Axis is *not* a parameter; making it a parameter
        # would require implementing a prior distribution and constraints over
        # the 3D unit ball.
        self.axis = axis
        self.axis = self.axis / torch.norm(self.axis)
        self.parameters = {
            "center": center,
            "width": width
        }
    def sample_rotation(self, parent):
        angle = pyro.sample("UniformBoundedRevoluteJointRule_theta", self._angle_dist)
        angle_axis = self.axis * angle
        R_offset = axis_angle_to_matrix(angle_axis.unsqueeze(0))[0, ...]
        R = torch.matmul(parent.rotation, R_offset)
        return R
        
    def score_child(self, parent, child, allowed_axis_diff=10. * np.pi/180.):
        if (self.ub - self.lb) >= 2. * np.pi:
            # Uniform rotation in 1D base case
            return torch.log(torch.ones(1) / (2. * np.pi))
        angle, axis = recover_relative_angle_axis(parent, child, target_axis=self.axis, allowed_axis_diff=allowed_axis_diff)
        # Correct angle to be within 2pi of both LB and UB -- which should be possible,
        # since ub - lb is <= 2pi.
        while angle < self.lb - 2.*np.pi or angle < self.ub - 2*np.pi:
            angle += 2.*np.pi
        while angle > self.ub + 2.*np.pi or angle > self.ub + 2.*np.pi:
            angle -= 2.*np.pi
        return self._angle_dist.log_prob(angle)

    @classmethod
    def get_parameter_prior(cls):
        # Default prior is a unit Normal for center,
        # and Uniform-distributed width on some reasonable range.
        return {
            "center": dist.Normal(torch.zeros(1), torch.ones(1)),
            "width": dist.Uniform(torch.zeros(1), torch.ones(1)*np.pi*2.)
        }
    @property
    def parameters(self):
        return {
            "center": self.center,
            "width": self.width
        }
    @parameters.setter
    def parameters(self, parameters):
        self.center = parameters["center"]
        self.width = parameters["width"]
        self._angle_dist = UniformWithEqualityHandling(self.lb, self.ub)
    @property
    def lb(self):
        return self.center - self.width / 2.
    @property
    def ub(self):
        return self.center + self.width / 2.


class GaussianChordOffsetRule(RotationProductionRule):
    ''' Placeholder '''
    def __init__(self, axis, loc, concentration):
        assert isinstance(axis, torch.Tensor) and axis.shape == (3,)
        assert isinstance(loc, (float, torch.Tensor))
        assert isinstance(concentration, (float, torch.Tensor)) and concentration >= 0
        if isinstance(concentration, float):
            concentration = torch.tensor(concentration)
        if isinstance(loc, float):
            loc = torch.tensor(loc)
        
        # Axis is *not* a parameter; making it a parameter
        # would require implementing a prior distribution and constraints over
        # the 3D unit ball.
        self.axis = axis
        self.axis = self.axis / torch.norm(self.axis)
        self.parameters = {
            "concentration": concentration,
            "loc": loc
        }
    
    def sample_rotation(self, parent):
        angle = pyro.sample("GaussianChordOffsetRule_theta", self._angle_dist)
        angle_axis = self.axis * angle
        R_offset = axis_angle_to_matrix(angle_axis.unsqueeze(0))[0, ...]
        R = torch.matmul(parent.rotation, R_offset)
        return R

        
    def score_child(self, parent, child, allowed_axis_diff=10. * np.pi/180.):
        angle, axis = recover_relative_angle_axis(parent, child, target_axis=self.axis, allowed_axis_diff=allowed_axis_diff)
        # Fisher distribution should be able to handle arbitrary +/-2pis.
        return self._angle_dist.log_prob(angle)

    @classmethod
    def get_parameter_prior(cls):
        # Default prior is a unit Normal for center,
        # and Uniform-distributed width on some reasonable range.
        return {
            "loc": dist.Normal(torch.zeros(1), torch.ones(1)),
            "concentration": dist.Uniform(torch.zeros(1)+1E-6, torch.ones(1)*np.pi*2.) # TODO: inverse gamma is better
        }
    @property
    def parameters(self):
        return {
            "concentration": self.concentration,
            "loc": self.loc
        }
    @parameters.setter
    def parameters(self, parameters):
        self.concentration = parameters["concentration"]
        self.loc = parameters["loc"]
        self._angle_dist = dist.VonMises(loc=self.loc, concentration=self.concentration)

