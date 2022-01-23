from copy import deepcopy
from collections import namedtuple
import logging
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

from .drake_interop import drake_tf_to_torch_tf, torch_tf_to_drake_tf
from .distributions import UniformWithEqualityHandling, BinghamDistribution
from .torch_utils import ConstrainedParameter

torch.set_default_tensor_type(torch.DoubleTensor)

import pydrake
from pydrake.all import (
    AddBilinearProductMcCormickEnvelopeSos2,
    AngleAxis,
    AngleAxis_,
    Expression,
    CoulombFriction,
    RigidTransform,
    RotationMatrix,
    RotationMatrix_,
    RollPitchYaw,
    SpatialInertia,
    UnitInertia,
    Variable
)

# Add a new rule checklist:
#  1) Make your new rule subclass.
#    1a) Fill in required virtual methods.
#    1b) Optionally provide encode_cost / encode_constraint if you plan on
#        parsing scenes with this rule type in the grammar.
#  2) Update do_fixed_structure_mcmc in sampling.py to know how to perturb
#      your rule type. (TODO: This functionality should be rolled into the rule
#      def'n...)

SiteValue = namedtuple("SiteValue", ["fn", "value"])

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

    def sample_child(self, parent, child=None):
        # Can optionally supply a child to re-sample; otherwise,
        # we make a new one. The child is pre-created so we can scope
        # using its generated name.
        if child is None:
            child = self.child_type(tf=torch.eye(4))
        with scope(prefix=child.name):
            xyz = self.xyz_rule.sample_xyz(parent)
            rotmat = self.rotation_rule.sample_rotation(parent)
        tf = torch.empty(4, 4)
        tf[:3, :3] = rotmat[:, :]
        tf[:3, 3] = xyz[:]
        tf[3, :] = torch.tensor([0., 0., 0., 1.])
        child.tf = tf
        return child

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

    def get_site_values(self, parent, child):
        # Given a parent and child, return a dictionary
        # of the pyro sample sites and SiteValue structs
        # (distribution + sampled values info)
        # that would lead to that parent child pair.
        # (Used for reconstructing traces for conditioning models
        # to match optimization-derived scene trees, and used
        # for scoring.)
        return {
            **self.xyz_rule.get_site_values(parent, child),
            **self.rotation_rule.get_site_values(parent, child)
        }
        raise NotImplementedError()

    def encode_constraint(self, prog, xyz_optim_params, rot_optim_params, parent, child, max_scene_extent_in_any_dir):
        ''' Given a MathematicalProgram prog, parameter dictionaries for
        the xyz and rotation rules matching their parameters (but possibly valued
        by decision variables or fixed values for those parameters), and parent and
        child nodes that have been given R_optim and t_optim decision variable members,
        encodes the constraints implied by this rule into the optimization.

        Returns the return value of encoding the rotation rule (i.e. whether the rotation
        was fully constrained by the constraints).'''
        self.xyz_rule.encode_constraint(prog, xyz_optim_params, parent, child, max_scene_extent_in_any_dir)
        return self.rotation_rule.encode_constraint(prog, rot_optim_params, parent, child, max_scene_extent_in_any_dir)

    def encode_cost(self, prog, xyz_optim_params, rot_optim_params, active, parent, child, max_scene_extent_in_any_dir):
        ''' Given a MathematicalProgram prog, parameter dictionaries for
        the xyz and rotation rules matching their parameters (but possibly valued
        by decision variables or fixed values for those parameters), a binary variable
        indicating whether the child is active, and parent and child nodes that have
        been given R_optim and t_optim decision variable members, encode the negative log
        probability of this rule given that parent and child into prog, s.t. the encoded
        cost is the -ll of the rule if active, and 0 if inactive.

        Active will be a decision variable if the prog is going to be a MICP, and
        True if we're in an NLP context. TODO(gizatt) Refactor these cost+constraint
        encoders in a more sensible way with a less obtuse interface. '''
        self.xyz_rule.encode_cost(prog, xyz_optim_params, active, parent, child, max_scene_extent_in_any_dir)
        self.rotation_rule.encode_cost(prog, rot_optim_params, active, parent, child, max_scene_extent_in_any_dir)

## XYZ Production rules
class XyzProductionRule():
    '''
        Instructions for how to produce a child's position
        from the parent.
    '''
    def __init__(self, fix_parameters=False):
        self.fix_parameters = fix_parameters
    def sample_xyz(self, parent):
        raise NotImplementedError()
    def score_child(self, parent, child):
        raise NotImplementedError()
    def get_site_values(self, parent, child):
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

    def encode_constraint(self, prog, optim_params, parent, child, max_scene_extent_in_any_dir):
        ''' Given a MathematicalProgram prog, parameter dictionaries for this
        rule type, and parent and child nodes that have been given R_optim and
        t_optim decision variable members, encodes the constraints implied by
        this rule into the optimization program. '''
        raise NotImplementedError()
    def encode_cost(self, prog, optim_params, active, parent, child, max_scene_extent_in_any_dir):
        ''' Given a MathematicalProgram prog, parameter dictionaries for this
        rule type, and parent and child nodes that have been given R_optim and
        t_optim decision variable members, adds the negative log probability
        of this rule given the parent and child to the program.'''
        raise NotImplementedError()

class SamePositionRule(XyzProductionRule):
    ''' Child Xyz is identically parent xyz. '''
    def __init__(self, offset=torch.zeros(3), **kwargs):
        self.offset = offset
        super().__init__(**kwargs)

    def sample_xyz(self, parent):
        return parent.translation + torch.matmul(parent.rotation, self.offset)
    def score_child(self, parent, child):
        if torch.allclose(
                parent.translation + torch.matmul(parent.rotation, self.offset),
                child.translation, atol=1E-3):
            return torch.tensor(0.)
        logging.warning("SamePositionRule mismatch (%s->%s): %s vs %s" % (parent, child, parent.translation + torch.matmul(parent.rotation, self.offset), child.translation))
        return torch.tensor(-np.inf)
    def get_site_values(self, parent, child):
        return {}

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

    def encode_constraint(self, prog, optim_params, parent, child, max_scene_extent_in_any_dir):
        # Constrain child translation to be equal to parent translation.
        t = parent.R_optim.dot(self.offset.detach().numpy())
        for k in range(3):
            prog.AddLinearEqualityConstraint(child.t_optim[k] == parent.t_optim[k] + t[k])
    def encode_cost(self, prog, optim_params, active, parent, child, max_scene_extent_in_any_dir):
        pass

class WorldFrameBBoxRule(XyzProductionRule):
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
        return pyro.sample("WorldFrameBBoxRule_xyz", self.xyz_dist)
    def score_child(self, parent, child):
        return self.xyz_dist.log_prob(child.translation).sum()
    def get_site_values(self, parent, child):
        return {"WorldFrameBBoxRule_xyz": SiteValue(self.xyz_dist, child.translation)}

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

    def encode_constraint(self, prog, optim_params, parent, child, max_scene_extent_in_any_dir):
        # Child translation should be within the translation bounds in
        # world frame.
        lb_world = optim_params["center"] - optim_params["width"]/2.
        ub_world = optim_params["center"] + optim_params["width"]/2.
        # X should be within a half-bound-width of the centerl.
        for k in range(3):
            prog.AddLinearConstraint(child.t_optim[k] >= lb_world[k])
            prog.AddLinearConstraint(child.t_optim[k] <= ub_world[k])
    def encode_cost(self, prog, optim_params, active, parent, child, max_scene_extent_in_any_dir):
        lb_world = optim_params["center"] - optim_params["width"]/2.
        ub_world = optim_params["center"] + optim_params["width"]/2.
        # log prob = 1 / width on each axis
        total_ll = -sum(ub_world - lb_world)
        return -total_ll * active


class WorldFrameBBoxOffsetRule(WorldFrameBBoxRule):
    ''' Child xyz is parent xyz + a uniform offset in [lb, ub]
        in world frame.

        TODO(gizatt) Add support for lb = ub; this requires
        special wrapping around Uniform to handle the equality
        cases as Delta distributions.'''
    def sample_xyz(self, parent):
        offset = pyro.sample("WorldFrameBBoxOffsetRule_xyz", self.xyz_dist)
        return parent.translation + offset
    def score_child(self, parent, child):
        xyz_offset = child.translation - parent.translation
        return self.xyz_dist.log_prob(xyz_offset).sum()
    def get_site_values(self, parent, child):
        return {"WorldFrameBBoxOffsetRule_xyz": SiteValue(self.xyz_dist, child.translation - parent.translation)}

    def encode_constraint(self, prog, optim_params, parent, child, max_scene_extent_in_any_dir):
        # Child translation should be within the translation bounds in
        # world frame.
        lb_world = optim_params["center"] - optim_params["width"]/2. + parent.t_optim
        ub_world = optim_params["center"] + optim_params["width"]/2. + parent.t_optim
        # X should be within a half-bound-width of the centerl.
        for k in range(3):
            prog.AddLinearConstraint(child.t_optim[k] >= lb_world[k])
            prog.AddLinearConstraint(child.t_optim[k] <= ub_world[k])
    # Uses encode_cost implementation in WorldFrameBBoxRule


class WorldFrameGaussianOffsetRule(XyzProductionRule):
    ''' Child xyz is diagonally-Normally distributed relative to parent in world frame.'''
    def __init__(self, mean, variance, **kwargs):
        assert isinstance(mean, torch.Tensor) and mean.shape == (3,)
        assert isinstance(variance, torch.Tensor) and variance.shape == (3,)
        self.parameters = {"mean": mean, "variance": variance}
        super().__init__(**kwargs)

    def sample_xyz(self, parent):
        return parent.translation + pyro.sample("WorldFrameGaussianOffsetRule_xyz", self.xyz_dist)
    def score_child(self, parent, child):
        return self.xyz_dist.log_prob(child.translation - parent.translation).sum()
    def get_site_values(self, parent, child):
        return {"WorldFrameGaussianOffsetRule_xyz": SiteValue(self.xyz_dist, child.translation - parent.translation)}

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
        self.xyz_dist = dist.Normal(self.mean, torch.sqrt(self.variance))

    def encode_constraint(self, prog, optim_params, parent, child, max_scene_extent_in_any_dir):
        pass
    def encode_cost(self, prog, optim_params, active, parent, child, max_scene_extent_in_any_dir):
        mean = optim_params["mean"]
        covar = np.diag(optim_params["variance"])
        inverse_covariance = np.linalg.inv(covar)
        covariance_det = np.linalg.det(covar)
        log_normalizer = np.log(np.sqrt( ((2. * np.pi) ** 3) * covariance_det))
        xyz_offset = (child.t_optim - (parent.t_optim + mean))
        
        # Introduce an xyz offset slack, which is forced to be equal to the
        # real xyz offset when active, and allowed to vary freely otherwise.
        # When inactive, the xyz offset slack will achieve an optimal cost of 0
        # by taking a value of 0.
        if isinstance(active, bool):
            assert active is True
            total_ll = -0.5 * (xyz_offset.transpose().dot(inverse_covariance).dot(xyz_offset)) - log_normalizer
        else:
            xyz_offset_slack = prog.NewContinuousVariables(3)
            inactive = 1. - active
            for i in range(3):
                prog.AddLinearConstraint(xyz_offset[i] - xyz_offset_slack[i] <= 2. * inactive * max_scene_extent_in_any_dir)
                prog.AddLinearConstraint(xyz_offset[i] - xyz_offset_slack[i] >= -2. * inactive * max_scene_extent_in_any_dir)
                prog.AddLinearConstraint(xyz_offset_slack[i] <= active*2.*max_scene_extent_in_any_dir)
                prog.AddLinearConstraint(xyz_offset_slack[i] >= -active*2.*max_scene_extent_in_any_dir)
            total_ll = -0.5 * ((xyz_offset_slack).transpose().dot(inverse_covariance).dot(xyz_offset_slack)) - log_normalizer * active
        prog.AddQuadraticCost(-total_ll)


class ParentFrameGaussianOffsetRule(WorldFrameGaussianOffsetRule):
    ''' Child xyz is diagonally-Normally distributed relative to parent in the
    parent's frame. Parsing this in a MIP requires extra work to handle some bilinear
    terms arising from rotating the offsets out of the parent frame.'''
    def sample_xyz(self, parent):
        return parent.translation + torch.matmul(parent.rotation, pyro.sample("ParentFrameGaussianOffsetRule_xyz", self.xyz_dist))
    def score_child(self, parent, child):
        return self.xyz_dist.log_prob(
            torch.matmul(parent.rotation.T, child.translation - parent.translation)
        ).sum()
    def get_site_values(self, parent, child):
        offset_in_parent_frame = torch.matmul(parent.rotation.T, child.translation - parent.translation)
        return {"ParentFrameGaussianOffsetRule_xyz": SiteValue(self.xyz_dist, offset_in_parent_frame)}

    def encode_constraint(self, prog, optim_params, parent, child, max_scene_extent_in_any_dir):
        pass
    def encode_cost(self, prog, optim_params, active, parent, child, max_scene_extent_in_any_dir):
        mean = optim_params["mean"]
        covar = np.diag(optim_params["variance"])
        inverse_covariance = np.linalg.inv(covar)
        covariance_det = np.linalg.det(covar)
        log_normalizer = np.log(np.sqrt( (2. * np.pi) ** 3 * covariance_det))

        # The offset in parent frame is what is penalized with
        # a normal penalty. This term has bilinear terms:
        #   parent.R_optim.T.dot(child.t_optim) + 
        #   parent.R_optim.T.dot(-parent.t_optim)
        # If we're in a nonlinear optimization context, we're OK. If we're
        # in a MIP context, we'll handle a couple of special cases in which
        # we can still express this cost in a reasonable way.
        if isinstance(active, bool):
            # NLP context.
            xyz_offset = parent.R_optim.T.dot(child.t_optim - (parent.t_optim + mean))
            total_ll = -0.5 * (xyz_offset.transpose().dot(inverse_covariance).dot(xyz_offset)) - log_normalizer
            total_ll = total_ll * active
            prog.AddCost(-total_ll)
        else:
            parent_R_observed = len(parent.R_equivalent_to_observed_nodes) > 0
            if parent_R_observed:
                #  - If the parent's rotation is fully observed, then we can enumerate over possible
                #    parent rotations. For any given rotation, the offset is linear in the translations.
                #    So calculate what every offset could be as a function of the parent / child translation
                #    variables, and create a slack that is constrained to be equal to the active one.
                xyz_offset = prog.NewContinuousVariables(3)
                # For all observed nodes that could determine the parent rotation...
                for observed_node in parent.R_equivalent_to_observed_nodes:
                    # By construction of the equivalent set, we will have an active correspondence
                    # if this node is active, so either xyz_offset is properly constrained, or it
                    # doesn't matter.
                    for k, (_, obs_var, obs_tf) in enumerate(observed_node.possible_observations):
                        parent_fixed_R = obs_tf.rotation().matrix()
                        observed_xyz_offset = parent_fixed_R.T.dot(child.t_optim - (parent.t_optim + mean))
                        inactive = 1. - obs_var
                        for i in range(3):
                            prog.AddLinearConstraint(observed_xyz_offset[i] - xyz_offset[i] <= inactive*max_scene_extent_in_any_dir*2.)
                            prog.AddLinearConstraint(observed_xyz_offset[i] - xyz_offset[i] >= -inactive*max_scene_extent_in_any_dir*2.)
            else: 
                #  - If the parent *rotation* is not observed, use the binary variables associated
                #    with parent.R_optim \in SO(3) to apply a piecewise approximation of this constraint.
                #    This will work even if the parent or child translations are observed, as the objective
                #    is quadratic 
                logging.warning("Hitting bad MIP case")
                assert parent.R_optim_mip_info is not None, "Parent rotation was unobserved but has no SO(3) MIP constraint info."
                binning = parent.R_optim_mip_info["rot_gen"].interval_binning()
                R_B = parent.R_optim_mip_info["B"]
                R_phi = parent.R_optim_mip_info["rot_gen"].phi()

                # Create a slack to approximate each bilinear product that shows up
                # in  parent.R.T * (child_t - (parent_t + mean)
                # We'll need to write this in terms of pre- and post-random-offset, since
                # the MIP binary variables are written on the pre-random-offset rotations:
                #    random_offset.T * parent.R_pre.T * (child_t - (parent_t + mean))

                # Mean gets added later
                xyz_offset_pre_rotation = child.t_optim - (parent.t_optim)
                xyz_offset_slack = prog.NewContinuousVariables(3, "xyz_offset_slack")
                for i in range(3):
                    prog.AddLinearEqualityConstraint(xyz_offset_pre_rotation[i] == xyz_offset_slack[i])
                
                product_terms = prog.NewContinuousVariables(3, 3, "product_terms")
                
                # Approximate the constraint w = x*y for every bilinear product
                # used to assemble the R*t bilinear terms. We reuse the subdivisions
                # for the terms of R.
                N_t_subdivisions = 3
                for j in range(3):
                    t_offset = prog.NewBinaryVariables(N_t_subdivisions)
                    t_phi = np.linspace(-max_scene_extent_in_any_dir, max_scene_extent_in_any_dir, N_t_subdivisions+1)
                    for i in range(3):
                        # Need to add one trivial binary variable since we've subdivided the translation
                        # range once.
                        # Big M needs to be the biggest error we can have between parent and child,
                        # which is twice the biggest error in a scene.
                        # Be sure to transpose R
                        AddBilinearProductMcCormickEnvelopeSos2(
                            prog, parent.R_optim_pre_offset[j, i], xyz_offset_slack[j], product_terms[i, j],
                            phi_x=R_phi,
                            phi_y=t_phi,
                            Bx = R_B[j][i],
                            By = t_offset,
                            binning=binning
                        )
                # Our approximate offset, then, is...
                xyz_offset = np.sum(product_terms, axis=0) - parent.R_optim.T.dot(mean)
                xyz_offset = parent.R_random_offset.matrix().T.dot(xyz_offset)

            # Now that we've assembled xyz_offset, apply a cost
            # if this node is active. (I'm not sure that this node being active
            # implies or is implied by the observed nodes being active; and this node's
            # translation may be constrained in strange ways due to the shared translation
            # between nodes with known-to-be-equal translation. So this is the safest
            # approach.)
            xyz_offset_slack = prog.NewContinuousVariables(3)
            inactive = 1. - active
            for i in range(3):
                prog.AddLinearConstraint(xyz_offset[i] - xyz_offset_slack[i] <= 2. * inactive * max_scene_extent_in_any_dir)
                prog.AddLinearConstraint(xyz_offset[i] - xyz_offset_slack[i] >= -2. * inactive * max_scene_extent_in_any_dir)
            total_ll = -0.5 * (xyz_offset_slack.transpose().dot(inverse_covariance).dot(xyz_offset_slack))  - log_normalizer * active
            prog.AddQuadraticCost(-total_ll)

    def get_max_score(self):
        return self.xyz_dist.log_prob(self.mean).sum()


class WorldFramePlanarGaussianOffsetRule(XyzProductionRule):
    ''' Child xyz is diagonally-Normally distributed relative to parent in world frame
        within a plane. Mean and variance should be 2D, and are sampled to produce a vector
        d = [x, y, 0]. The child_xyz <- parent_xyz + plane_transform * d'''
    def __init__(self, mean, variance, plane_transform, **kwargs):
        assert isinstance(mean, torch.Tensor) and mean.shape == (2,)
        assert isinstance(variance, torch.Tensor) and variance.shape == (2,)
        assert isinstance(plane_transform, RigidTransform)
        self.plane_transform = drake_tf_to_torch_tf(plane_transform)
        self.plane_transform_inv = drake_tf_to_torch_tf(plane_transform.inverse())
        self.parameters = {"mean": mean, "variance": variance}
        super().__init__(**kwargs)

    def sample_xyz(self, parent):
        xy_offset = pyro.sample("WorldFramePlanarGaussianOffsetRule", self.xy_dist)
        xyz_offset_homog = torch.cat([xy_offset, torch.tensor([0., 1.])])
        return parent.translation + torch.matmul(self.plane_transform, xyz_offset_homog)[:3]
    def score_child(self, parent, child):
        offset_homog = torch.cat([child.translation - parent.translation, torch.tensor([1.])])
        xy_offset = torch.matmul(self.plane_transform_inv, offset_homog)[:2]
        return self.xy_dist.log_prob(xy_offset).sum()

    def get_site_values(self, parent, child):
        offset_homog = torch.cat([child.translation - parent.translation, torch.tensor([1.])])
        xy_offset = torch.matmul(self.plane_transform_inv, offset_homog)[:2]
        return {"WorldFramePlanarGaussianOffsetRule": SiteValue(self.xy_dist, xy_offset)}

    @classmethod
    def get_parameter_prior(cls):
        return {
            "mean": dist.Normal(torch.zeros(2), torch.ones(2)),
            "variance": dist.Uniform(torch.zeros(2)+1E-6, torch.ones(2)*10.) # TODO: Inverse gamma is better
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
        self.xy_dist = dist.Normal(self.mean, torch.sqrt(self.variance))

    def encode_constraint(self, prog, optim_params, parent, child, max_scene_extent_in_any_dir):
        # Constrain that the child pose is in the appropriate plane
        # relative to the parent: i.e., that (child.xyz - parent.xyz)
        # dotted with the plane normal is zero.
        plane_tf_inv = torch_tf_to_drake_tf(self.plane_transform_inv).cast[Expression]()
        dx = child.t_optim - parent.t_optim
        dx_in_plane = plane_tf_inv.multiply(dx)
        prog.AddLinearConstraint(dx_in_plane[2] == 0.)

    def encode_cost(self, prog, optim_params, active, parent, child, max_scene_extent_in_any_dir):
        mean = optim_params["mean"]
        covar = np.diag(optim_params["variance"])
        inverse_covariance = np.linalg.inv(covar)
        covariance_det = np.linalg.det(covar)
        log_normalizer = np.log(np.sqrt( (2. * np.pi) ** 2 * covariance_det))

        inv_tf = torch_tf_to_drake_tf(self.plane_transform_inv).cast[Expression]()
        xy_offset = inv_tf.multiply(child.t_optim - parent.t_optim)[:2] - mean

        if isinstance(active, bool):
            total_ll = -0.5 * (xy_offset.transpose().dot(inverse_covariance).dot(xy_offset))  - log_normalizer
        else:
            # If inactive, allow the cost to vary freely, which will lead to a cost
            # of zero for this node. Otherwise, the cost will take the value implied
            # by xy_offset.
            xy_offset_slack = prog.NewContinuousVariables(2)
            inactive = 1. - active
            for i in range(2):
                prog.AddLinearConstraint(xy_offset[i] - xy_offset_slack[i] <= 2. * inactive * max_scene_extent_in_any_dir)
                prog.AddLinearConstraint(xy_offset[i] - xy_offset_slack[i] >= -2. * inactive * max_scene_extent_in_any_dir)
            total_ll = -0.5 * (xy_offset_slack.transpose().dot(inverse_covariance).dot(xy_offset_slack))  - log_normalizer * active
        prog.AddQuadraticCost(-total_ll)

## Rotation production rules
class RotationProductionRule():
    '''
        Instructions for how to produce a child's position
        from the parent.
    '''
    def __init__(self, fix_parameters=False):
        self.fix_parameters = fix_parameters
    def sample_rotation(self, parent):
        raise NotImplementedError()
    def score_child(self, parent, child):
        raise NotImplementedError()
    def get_site_values(self, parent, child):
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

    def encode_constraint(self, prog, optim_params, parent, child, max_scene_extent_in_any_dir):
        ''' Given a MathematicalProgram prog, parameter dictionaries for this
        rule type, and parent and child nodes that have been given R_optim and
        t_optim decision variable members, encodes the constraints implied by
        this rule into the optimization program. Returns whether the rotation
        of the child is fully constrained by the application of this rule. '''
        raise NotImplementedError()
    def encode_cost(self, prog, optim_params, active, parent, child, max_scene_extent_in_any_dir):
        ''' Given a MathematicalProgram prog, parameter dictionaries for this
        rule type, and parent and child nodes that have been given R_optim and
        t_optim decision variable members, adds the negative log probability
        of this rule given the parent and child to the program.'''
        raise NotImplementedError()


class SameRotationRule(RotationProductionRule):
    ''' Child Xyz is identically parent xyz. '''
    def __init__(self, offset=torch.eye(3), **kwargs):
        self.offset = offset
        super().__init__(**kwargs)

    def sample_rotation(self, parent):
        return torch.matmul(parent.rotation, self.offset)
    def score_child(self, parent, child):
        if torch.allclose(torch.matmul(parent.rotation, self.offset), child.rotation, atol=1E-3):
            return torch.tensor(0.)
        return torch.tensor(-np.inf)
    def get_site_values(self, parent, child):
        return {}

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

    def encode_constraint(self, prog, optim_params, parent, child, max_scene_extent_in_any_dir):
        R_offset = self.offset.detach().numpy()
        R_targ = np.dot(parent.R_optim, R_offset)
        for i in range(3):
            for j in range(3):
                prog.AddLinearEqualityConstraint(child.R_optim[i, j] == R_targ[i, j])
        # Child is fully constrained.
        return True
    def encode_cost(self, prog, optim_params, active, parent, child, max_scene_extent_in_any_dir):
        return 0.

class UnconstrainedRotationRule(RotationProductionRule):
    '''
        Child rotation is randomly chosen from all possible
        rotations with no relationship to parent.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        desired_density = 1 / np.pi ** 2.
        true_ub = torch.tensor([1., 1., 0.5])
        self.scaling = torch.tensor([1., 1., 2.]) * np.pi ** (2. / 3)
        self.u_dist = dist.Uniform(torch.zeros(3), true_ub * self.scaling)

    def sample_rotation(self, parent):
        # Sample random unit quaternion via
        # http://planning.cs.uiuc.edu/node198.html (referencing
        # Honkai's implementation in Drake), and convert to rotation
        # matrix, with one modification to ensure the
        # generated quaternions have known sign (the 3rd element
        # is always positive).
        # TODO(gizatt) I've chosen the uniform bounds so that
        # the density of the uniform at any sample point matches the
        # expected density for sampling from SO(3); then I rescale
        # back down to unit-interval. Can I justify this by
        # demonstrating that this scaling makes log abs det Jacobian 1,
        # so I'm effectively counteracting the rescaling this whole
        # transformation is applying?
        # Expected density = 1 / pi^2
        # Actual density 1 / (pi * pi * .5 pi) = 2 / pi^3
        # -> Normalize should be = pi^(2/3) / 2

        u = pyro.sample("UnconstrainedRotationRule_u", self.u_dist)/self.scaling
        random_quat = torch.tensor([
            torch.sqrt(1. - u[0]) * torch.sin(2. * np.pi * u[1]), # [0, 2pi -> -1 -> 1]
            torch.sqrt(1. - u[0]) * torch.cos(2. * np.pi * u[1]), # [0, 2pi -> -1 -> 1]
            torch.sqrt(u[0]) * torch.sin(2. * np.pi * u[2]), # [0, pi -> 0 -> 1]
            torch.sqrt(u[0]) * torch.cos(2. * np.pi * u[2])  # [0, pi -> -1, 1]
        ])
        assert torch.isclose(random_quat.square().sum(), torch.ones(1)), (u, random_quat, random_quat.square().sum())
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
    def get_site_values(self, parent, child):
        # Reverse the equations linked above to recover u's.
        quaternion = matrix_to_quaternion(child.rotation)
        if quaternion[2] < 0:
            quaternion = -quaternion
        if quaternion[2] == 0:
            # Got a weird negative-zero error once; squashing...
            quaternion[2] = torch.abs(quaternion[2])
        u1 = quaternion[2]**2 + quaternion[3]**2
        # Sanity-check that my inversion is reasonable
        assert torch.isclose(quaternion[0]**2. + quaternion[1]**2, 1. - u1)
        u3_1 = torch.atan2(quaternion[2], quaternion[3])
        u2_1 = torch.atan2(quaternion[0], quaternion[1])

        assert u3_1 >= 0 and u3_1 <= np.pi, (quaternion, u3_1)
        
        if u2_1 < 0:
            u2_1 = u2_1 + 2. * np.pi
        
        u2_1 /= (2. * np.pi)
        u3_1 /= (2. * np.pi)

        scaling = torch.tensor([1., 1., 2.]) * np.pi ** (2. / 3)
        return {"UnconstrainedRotationRule_u": SiteValue(self.u_dist, torch.stack([u1, u2_1, u3_1]) * self.scaling)}

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

    def encode_constraint(self, prog, optim_params, parent, child, max_scene_extent_in_any_dir):
        # Child rotation not fully constrained.
        return False
    def encode_cost(self, prog, optim_params, active, parent, child, max_scene_extent_in_any_dir):
        # Constant term
        return -self.score_child(parent, child).detach().item() * active

def recover_relative_angle_axis(parent, child, target_axis, zero_angle_width=1E-2, allowed_axis_diff=10. * np.pi/180.):
    # Recover angle-axis relationship between a parent and child.
    # Warns if we can't find a rotation axis between the two within
    # requested diff of our expected axis. Last return value is a boolean
    # of whether that rotation axis is close to the epxected axis.
    relative_R = torch.matmul(torch.transpose(parent.rotation, 0, 1), child.rotation)
    axis_angle = quaternion_to_axis_angle(matrix_to_quaternion(relative_R))
    angle = torch.norm(axis_angle, p=2)

    # *Why* is this tolerance so high? This is ridiculous
    if angle <= zero_angle_width:
        return torch.tensor(0.), target_axis, True

    axis = axis_angle / angle
    axis_misalignment = torch.acos(torch.clip((axis * target_axis).sum(), -1., 1.))
    if torch.abs(angle) > 0 and axis_misalignment >= np.pi/2.:
        # Flipping axis will give us a close axis.
        axis = -axis
        angle = -angle

    axis_inner_prod = torch.clip((axis * target_axis).sum(), -1+1E-9, 1-1E-9)
    axis_misalignment = np.abs(torch.acos(axis_inner_prod).item())
    if axis_misalignment >= allowed_axis_diff:
        # No saving this; axis doesn't match.
        logging.warning("Parent %s, Child %s:\nChild illegal rotated from parent: %s vs %s, error of %f deg" % (parent.name, child.name, axis, target_axis, axis_misalignment * 180./np.pi))
    return angle, axis, axis_misalignment < allowed_axis_diff

class UniformBoundedRevoluteJointRule(RotationProductionRule):
    '''
        Child rotation is randomly chosen uniformly from a bounded
        range of angles around a revolute joint axis about the parent.
    '''
    @classmethod
    def from_bounds(cls, axis, lb, ub, **kwargs):
        assert ub >= lb
        return cls(axis, (ub+lb)/2., ub-lb, **kwargs)
    def __init__(self, axis, center, width, **kwargs):
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
        super().__init__(**kwargs)

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
        angle, axis, aligned = recover_relative_angle_axis(parent, child, target_axis=self.axis, allowed_axis_diff=allowed_axis_diff)
        if not aligned:
            return -torch.tensor(np.inf)
        # Correct angle to be within 2pi of both LB and UB -- which should be possible,
        # since ub - lb is <= 2pi.
        while angle < self.lb - 2.*np.pi or angle < self.ub - 2*np.pi:
            angle += 2.*np.pi
        while angle > self.ub + 2.*np.pi or angle > self.ub + 2.*np.pi:
            angle -= 2.*np.pi
        return self._angle_dist.log_prob(angle)

    def get_site_values(self, parent, child):
        # TODO: Not exactly reverse-engineering, but hopefully close.
        theta, _, _ = recover_relative_angle_axis(parent, child, target_axis=self.axis)
        return {"UniformBoundedRevoluteJointRule_theta": SiteValue(self._angle_dist, theta)}

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

    def encode_constraint(self, prog, optim_params, parent, child, max_scene_extent_in_any_dir):
        axis = self.axis.detach().cpu().numpy()
        min_angle = (optim_params["center"] - optim_params["width"]/2.)[0]
        max_angle = (optim_params["center"] + optim_params["width"]/2.)[0]

        if isinstance(min_angle, float) and isinstance(max_angle, float):
            assert min_angle <= max_angle
            if max_angle - min_angle <= 1E-6:
                # In this case, the child rotation is exactly equal to the
                # parent rotation, so we can short-circuit.
                relative_rotation = RotationMatrix(AngleAxis(max_angle, axis)).matrix()
                target_rotation = parent.R_optim.dot(relative_rotation)
                for i in range(3):
                    for j in range(3):
                        prog.AddLinearEqualityConstraint(child.R_optim[i, j] == target_rotation[i, j])
                return True
        else:
            assert isinstance(min_angle, Expression) and isinstance(max_angle, Expression), (min_angle, max_angle)
            prog.AddLinearConstraint(min_angle <= max_angle)

        # Child rotation should be within a relative rotation of the parent around
        # the specified axis, and the axis should *not* be rotated between the
        # parent and child frames. This is similar to the revolute joint constraints
        # used by Hongkai Dai in his global IK formulation.
        # (1): The direction of the rotation axis doesn't change between
        # parent and child frames.
        # The axis is the same in both the parent and child frame
        # (see https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_revolute_joint.html).
        # Though there may be an additional offset according to the axis offset
        # in the parent and child frames.
        #axis_offset_in_parent = RigidTransform()
        #axis_offset_in_child = RigidTransform()
        parent_view_of_axis_in_world = parent.R_optim.dot(axis)
        child_view_of_axis_in_world = child.R_optim.dot(axis)
        for k in range(3):
            prog.AddLinearEqualityConstraint(
                parent_view_of_axis_in_world[k] == child_view_of_axis_in_world[k]
            )
        
        # Short-circuit if there is no rotational constraint other than axis alignment.
        if isinstance(min_angle, float) and isinstance(max_angle, float) and max_angle - min_angle >= 2.*np.pi:
            return False

        # If we're only allowed a limited rotation around this axis, apply a constraint
        # to enforce that.
        # (2): Eq(10) in the global IK paper. Following implementation in
        # https://github.com/RobotLocomotion/drake/blob/master/multibody/inverse_kinematics/global_inverse_kinematics.cc
        # First generate a vector normal to the rotation axis via cross products.

        v_c = np.cross(axis, np.array([0., 0., 1.]))
        if np.linalg.norm(v_c) <= np.sqrt(2)/2:
            # Axis is too close to +z; try a different axis.
            v_c = np.cross(axis, np.array([0., 1., 0.]))
        v_c = v_c / np.linalg.norm(v_c)
        # TODO: Hongkai uses multiple perpendicular vectors for tighter
        # bound. Why does that make it tighter? Maybe worth a try?

        # Translate into a symmetric bound by finding a rotation to
        # "center" us in the bound region, and the symmetric bound size alpha.
        # -alpha <= theta - (a+b)/2 <= alpha
        # where alpha = (b-a) / 2
        alpha = (max_angle - min_angle) / 2.
        offset_angle = (max_angle + min_angle) / 2.
        R_offset = RotationMatrix_[type(offset_angle)](
            AngleAxis_[type(offset_angle)](offset_angle, axis)).matrix()
        # |R_WC*R_CJc*v - R_WP * R_PJp * R(k,(a+b)/2)*v | <= 2*sin (α / 2) in
        # global ik code; for us, I'm assuming the joint frames are aligned with
        # the body frames, so R_CJc and R_PJp are identitiy.
        lorentz_bound = 2 * np.sin(alpha / 2.)
        vector_diff = (
            child.R_optim.dot(v_c) - 
            parent.R_optim.dot(R_offset).dot(v_c)
        )
        # TODO: Linear approx?
        prog.AddLorentzConeConstraint(np.r_[lorentz_bound, vector_diff])
        return False

    def encode_cost(self, prog, optim_params, active, parent, child, max_scene_extent_in_any_dir):
        # Uniform distribution over angles, constant term
        return torch.log(self.width).sum().detach().numpy() * active

wfbrr_warned_prior_detached = False
wfbrr_warned_params_detached = False

def add_bingham_cost(prog, R, active, M, Z, log_normalizer):
    '''
    Use the reinterpretation of the Bingham distribution objective as
    1/C(Z) * exp[ tr(Z M^T q q^T M) ] for quaternion q.
    The relevant part of the log-likelihood is tr(Z M^T q q^T M).

    q = [w x y z]
    q q^T = [
     ww wx wy wz
     wx xx xy xz
     wy xy yy yz
     wz xz yz zz
    ]

    We'll introduce intermediate variables for each term in that
    outer product, and constrain them to match the rotation matrix
    variables of the child node, which can be recovered from them
    via the standard quaternion-to-rotation-matrix conversion formula:

    R = [
        1 - 2yy - 2zz,   2xy - 2zw,      2xz + 2yw,
        2xy + 2zw,       1 - 2xx - 2zz,  2yz - 2xw,
        2xz - 2yw,       2yz + 2xw,      1 - 2xx - 2yy
    ]
    
    Then we can write that cost as linear function of the intermediate
    quaternion outer product terms.
    '''
    qqt_terms = prog.NewContinuousVariables(10)
    ww, wx, wy, wz, xx, xy, xz, yy, yz, zz = qqt_terms
    # w,x,y,z all in [-1, 1], so their products are as well.
    prog.AddBoundingBoxConstraint(-np.ones(10), np.ones(10), qqt_terms)
    # Build qqt matrix
    qqt = np.array([
        [ww, wx, wy, wz],
        [wx, xx, xy, xz],
        [wy, xy, yy, yz],
        [wz, xz, yz, zz]
    ])
    # The square terms are further obviously positive.
    prog.AddLinearConstraint(ww >= 0.)
    prog.AddLinearConstraint(xx >= 0.)
    prog.AddLinearConstraint(yy >= 0.)
    prog.AddLinearConstraint(zz >= 0.)
    # And the quaternion has unit norm.
    prog.AddLinearEqualityConstraint(ww + xx + yy + zz == 1.)

    # In dev, see sandbox/snopt_optimal_cost_replication
    if isinstance(active, bool):
        assert active is True
        # Adding *any* constraints here causes both SNOPT and IPOPT
        # to have trouble converging. But without it, the solutions
        # are still great. I guess I have enough constraints from
        # the rotation matrix?
        #prog.AddConstraint(ww * xx == wx**2.)
        #prog.AddConstraint(ww * yy == wy**2.)
        #prog.AddConstraint(ww * zz == wz**2.)

    else:
        # Approximation
        pass
        #prog.AddRotatedLorentzConeConstraint(ww, xx, wx**2.)
        #prog.AddRotatedLorentzConeConstraint(ww, yy, wy**2.)
        #prog.AddRotatedLorentzConeConstraint(ww, zz, wz**2.)
        #prog.AddRotatedLorentzConeConstraint(ww, 1., ww**2 + wx**2 + wy**2 + wz**2)

    # Enforce quaternion-bilinear-term-to-rotmat correspondence.
    prog.AddLinearEqualityConstraint(R[0, 0] == 1 - 2*yy - 2*zz)
    prog.AddLinearEqualityConstraint(R[0, 1] == 2*xy - 2*wz)
    prog.AddLinearEqualityConstraint(R[0, 2] == 2*xz + 2*wy)
    prog.AddLinearEqualityConstraint(R[1, 0] == 2*xy + 2*wz)
    prog.AddLinearEqualityConstraint(R[1, 1] == 1 - 2*xx - 2*zz)
    prog.AddLinearEqualityConstraint(R[1, 2] == 2*yz - 2*wx)
    prog.AddLinearEqualityConstraint(R[2, 0] == 2*xz - 2*wy)
    prog.AddLinearEqualityConstraint(R[2, 1] == 2*yz + 2*wx)
    prog.AddLinearEqualityConstraint(R[2, 2] == 1 - 2*xx - 2*yy)
    
    if isinstance(active, bool):
        assert active is True
        ll = np.trace(Z.dot(M.T.dot(qqt.dot(M)))) - log_normalizer
    else:
        # Total log prob based on quaternion terms. Here, I'm creating a slack
        # variable that'll take the the value of the mode of the distribution
        # when this node is inactive, and take the value corresponding to R_optim
        # otherwise. At the mode, the trace term is 0.
        mode = M[:, -1].reshape(4, 1)
        modemodeT = mode.dot(mode.T)
        # Add slack s.t. when active, qqt_slack = qqt, but otherwise,
        # qqt_slack = modemodeT.
        qqt_slack = prog.NewContinuousVariables(4, 4, "qqt_slack")
        inactive = 1. - active
        for i in range(4):
            for j in range(4):
                prog.AddLinearConstraint(modemodeT[i, j] - qqt_slack[i, j] >= -2.*active)
                prog.AddLinearConstraint(modemodeT[i, j] - qqt_slack[i, j] <= 2.*active)
                prog.AddLinearConstraint(qqt[i, j] - qqt_slack[i, j] >= -2. * (inactive))
                prog.AddLinearConstraint(qqt[i, j] - qqt_slack[i, j] <= 2. * (inactive))
        ll = np.trace(Z.dot(M.T.dot(qqt_slack.dot(M)))) - log_normalizer * active
    prog.AddLinearCost(-ll)

    return qqt


class WorldFrameBinghamRotationRule(RotationProductionRule):
    '''
    Child rotation is chosen randomly from a Bingham distribution
    describing a distribution over rotations in world frame.

    M: 4x4 "orientation" matrix. The last column is the mode (as a quaternion).
    Z: 4-element tensor of concentrations in ascending order with the last
    being 0.

    Recommended to use a helper function to assemble these from target rotation
    and RPY concentrations.
    '''
    @classmethod
    def from_rotation_and_rpy_variances(cls, rotation_mode, rpy_concentration, **kwargs):
        ''' Construct this rule to distribute the child rotation
        around the given RotationMatrix, with specified concentrations
        around each rotation axis. '''
        assert isinstance(rotation_mode, RotationMatrix)
        assert len(rpy_concentration) == 3 and [x > 0 for x in rpy_concentration]
        # Last column of M should be the mode, as a quaternion.
        # Construct the rest of M to be orthogonal.
        mode = rotation_mode.ToQuaternion()
        rot_around_x = RollPitchYaw([np.pi, 0., 0.]).ToQuaternion()
        rot_around_y = RollPitchYaw([0., np.pi, 0.]).ToQuaternion()
        rot_around_z = RollPitchYaw([0., 0., np.pi]).ToQuaternion()
        m = np.stack([
            rot_around_x.multiply(mode).wxyz(),
            rot_around_y.multiply(mode).wxyz(),
            rot_around_z.multiply(mode).wxyz(),
            mode.wxyz()
        ], axis=1)
        z = np.array(
            [-rpy_concentration[0],
             -rpy_concentration[1],
             -rpy_concentration[2],
             0.
            ]
        )
        # Reorder Z to be ascending, with M in lockstep.
        reorder_inds = np.argsort(z)
        z = z[reorder_inds]
        m = m[:, reorder_inds]
        return cls(
            torch.tensor(deepcopy(m)), torch.tensor(deepcopy(z)), **kwargs
        )


    def __init__(self, M, Z, **kwargs):
        assert isinstance(M, torch.Tensor) and M.shape == (4, 4)
        assert isinstance(Z, torch.Tensor) and Z.shape == (4,)
        # More detailed checks on contents of M and Z will be done
        # by the distribution type.
        self.parameters = {
            "M": M,
            "Z": Z
        }
        super().__init__(**kwargs)
    
    def sample_rotation(self, parent):
        quat = pyro.sample("WorldFrameBinghamRotationRule_quat", self._bingham_dist)
        R = quaternion_to_matrix(quat)
        return R

    def score_child(self, parent, child):
        quat = matrix_to_quaternion(child.rotation)
        # Flip the quaternion if its last element is negative, by convention of our
        # implementation of BinghamDistribution.
        if quat[-1] < 0:
            quat = quat * -1
        return self._bingham_dist.log_prob(quat)

    def get_site_values(self, parent, child):
        quat = matrix_to_quaternion(child.rotation)
        # Flip the quaternion if its last element is negative, by convention of our
        # implementation of BinghamDistribution.
        if quat[-1] < 0:
            quat = quat * -1
        return {"WorldFrameBinghamRotationRule_quat": SiteValue(self._bingham_dist, quat)}

    @classmethod
    def get_parameter_prior(cls):
        # TODO(gizatt) Putting priors on these parameters will be really hard;
        # M needs to be 4x4 orthogonal, and Z needs to be nonpositive in ascending
        # order with the last term 0. I don't currently consume priors for anything,
        # so I'm skipping implementing this in a nice way, and just initializing these
        # parameters "reasonably".
        global wfbrr_warned_prior_detached
        if not wfbrr_warned_prior_detached:
            logging.warning("Prior over parameters of WorldFrameBinghamRotationRule are Deltas.")
            wfbrr_warned_prior_detached = True
        return {
            "M": dist.Delta(torch.eye(4)),
            "Z": dist.Delta(torch.tensor([-1, -1, -1, 0.]))
        }
    @property
    def parameters(self):
        return {
            "M": self.M,
            "Z": self.Z
        }
    @parameters.setter
    def parameters(self, parameters):
        self.M = parameters["M"]
        self.Z = parameters["Z"]
        global wfbrr_warned_params_detached
        if not wfbrr_warned_params_detached:
            logging.warning("Detaching BinghamDistribution parameters.")
            wfbrr_warned_params_detached = True

        self._bingham_dist = BinghamDistribution(
            param_m=self.M.detach(), param_z=self.Z.detach(),
            options={"flip_to_positive_z_quaternions": True}
        )

    def encode_constraint(self, prog, optim_params, parent, child, max_scene_extent_in_any_dir):
        # Child rotation is not fully constrained.
        return False

    def encode_cost(self, prog, optim_params, active, parent, child, max_scene_extent_in_any_dir):
        R = child.R_optim
        M = optim_params["M"]
        Z = np.diag(optim_params["Z"])
        log_normalizer = torch.log(self._bingham_dist._norm_const).detach().numpy()
        add_bingham_cost(prog, R, active, M, Z, log_normalizer)


class ParentFrameBinghamRotationRule(WorldFrameBinghamRotationRule):
    '''
    Same as WorldFrameBinghamRotationRule, but the child rotation is chosen
    in parent rotation frame.
    '''
    
    def sample_rotation(self, parent):
        quat = pyro.sample("ParentFrameBinghamRotationRule_quat", self._bingham_dist)
        R = quaternion_to_matrix(quat)
        return torch.matmul(parent.rotation, R)

    def score_child(self, parent, child):
        rel_rotation = torch.matmul(parent.rotation.T, child.rotation)
        quat = matrix_to_quaternion(rel_rotation)
        # Flip the quaternion if its last element is negative, by convention of our
        # implementation of BinghamDistribution.
        if quat[-1] < 0:
            quat = quat * -1
        return self._bingham_dist.log_prob(quat)

    def get_site_values(self, parent, child):
        rel_rotation = torch.matmul(parent.rotation.T, child.rotation)
        quat = matrix_to_quaternion(rel_rotation)
        # Flip the quaternion if its last element is negative, by convention of our
        # implementation of BinghamDistribution.
        if quat[-1] < 0:
            quat = quat * -1
        return {"ParentFrameBinghamRotationRule_quat": SiteValue(self._bingham_dist, quat)}

    def encode_constraint(self, prog, optim_params, parent, child, max_scene_extent_in_any_dir):
        # No constraints here.
        pass

    def encode_cost(self, prog, optim_params, active, parent, child, max_scene_extent_in_any_dir):
        '''
        Use the same logic as the world-frame Bingham distribution objective
        that introduces intermediate variables for the quaternion bilinear
        terms and produces the cost from that. However, now, the
        rotation is not the child rotation, but the *relative rotation between parent and
        child.* So as a first step, we must approximate that term.

            delta_R = R_parent.T * R_child

        Expanding to consider the random offsets that are applied as part of the MIP
        parsing infrastructure:

            delta_R = (R_parent_pre * offset).T * R_child_pre * offset
                    = offset.T * R_parent_pre.T * R_child_pre * offset


        So we introduce delta_R = R_parent_pre * R_child_pre and apply this constraint.
        The parent and child rotations can be observed or not, and we handle each case
        separately:
        - If one of the parent or child rotations are observed, we can enumerate
        over the options they can take, constructing this cost (as a linear expression)
        for each possible setting, deactivating the costs for inactive settings.
        - If neither are observed, we enforce the bilinear relationship delta_R = R_p * R_c
        with piecewise McCormick envelopes using the binary variables used in the SO(3)
        constraints for the two rotations.
        '''
        M = optim_params["M"]
        Z = np.diag(optim_params["Z"])
        log_normalizer = torch.log(self._bingham_dist._norm_const).detach().numpy()
        
        if isinstance(active, bool):
            # NLP context; just add the nonlinear cost and run.
            deltaR = parent.R_optim.T.dot(child.R_optim)
            # (Have to create an intermediate var here; add_bingham_cost
            # checks linearity against this decision variable of some
            # internal constriants.)
            R_for_bingham_cost = prog.NewContinuousVariables(3, 3, "R_tmp")
            for i in range(3):
                for j in range(3):
                    prog.AddConstraint(R_for_bingham_cost[i, j] == deltaR[i, j])
            add_bingham_cost(prog, R_for_bingham_cost, active, M, Z, log_normalizer)
        else:
            parent_R_observed = len(parent.R_equivalent_to_observed_nodes) > 0
            child_R_observed = len(child.R_equivalent_to_observed_nodes) > 0

            if parent_R_observed:
                #  - If the parent's rotation is fully observed, then we can enumerate over possible
                #    parent rotations. For any given rotation, deltaR is a linear expression in the
                #    child rotation. So calculate it for every parent rotation and create a slack
                #    that is constrained to be equal to the active one.
                deltaR = prog.NewContinuousVariables(3, 3, "deltaR_from_parent_obs")
                # For all observed nodes that could determine the parent rotation...
                for observed_node in parent.R_equivalent_to_observed_nodes:
                    for k, (_, obs_var, obs_tf) in enumerate(observed_node.possible_observations):
                        parent_fixed_R = obs_tf.rotation().matrix()
                        observed_deltaR = parent_fixed_R.T.dot(child.R_optim)
                        inactive = 1. - obs_var
                        for i in range(3):
                            for j in range(3):
                                prog.AddLinearConstraint(observed_deltaR[i, j] - deltaR[i, j] <= inactive*2.)
                                prog.AddLinearConstraint(observed_deltaR[i, j] - deltaR[i, j] >= -inactive*2.)
            elif child_R_observed:
                #  Same logic as above case for a fixed child observation.
                deltaR = prog.NewContinuousVariables(3, 3, "deltaR_from_child_obs")
                # For all observed nodes that could determine the parent rotation...
                for observed_node in child.R_equivalent_to_observed_nodes:
                    for k, (_, obs_var, obs_tf) in enumerate(observed_node.possible_observations):
                        child_fixed_R = obs_tf.rotation().matrix()
                        observed_deltaR = parent.R_optim.T.dot(child_fixed_R)
                        inactive = 1. - obs_var
                        for i in range(3):
                            for j in range(3):
                                prog.AddLinearConstraint(observed_deltaR[i, j] - deltaR[i, j] <= inactive*2.)
                                prog.AddLinearConstraint(observed_deltaR[i, j] - deltaR[i, j] >= -inactive*2.)
            else:
                assert not parent_R_observed and not child_R_observed
                #  - If neither rotation is observed, then we need to deal with bilinear terms between
                # their rotations.
                assert parent.R_optim_mip_info is not None, "Parent rotation was unobserved but has no SO(3) MIP constraint info."
                assert child.R_optim_mip_info is not None, "Child rotation was unobserved but has no SO(3) MIP constraint info."
                binning = parent.R_optim_mip_info["rot_gen"].interval_binning()
                assert binning == child.R_optim_mip_info["rot_gen"].interval_binning()
                parent_R_B = parent.R_optim_mip_info["B"]
                parent_R_phi = parent.R_optim_mip_info["rot_gen"].phi()
                child_R_B = child.R_optim_mip_info["B"]
                child_R_phi = child.R_optim_mip_info["rot_gen"].phi()

                # Create a slack to approximate each bilinear product that shows up.
                deltaR = prog.NewContinuousVariables(3, 3, "deltaR")
                # This entity is itself a rotation, so go ahead and throw in rotation
                # constraints, with the hope that it'll push the MIP solver towards better
                # solutions.
                parent.R_optim_mip_info["rot_gen"].AddToProgram(deltaR, prog)
                for i in range(3):
                    for j in range(3):
                        # deltaR[i, j] = \sum_k parent.R.T [i, k] * child.R[k, j]
                        terms = prog.NewContinuousVariables(3, "deltaR_sum_terms_%d_%d" % (i, j))
                        prog.AddLinearEqualityConstraint(deltaR[i, j] == sum(terms))
                        for term_k in range(3):
                            # Be sure to transpose R parent
                            AddBilinearProductMcCormickEnvelopeSos2(
                                prog, parent.R_optim_pre_offset[term_k, i], child.R_optim_pre_offset[term_k, j], terms[term_k],
                                phi_x=parent_R_phi,
                                phi_y=child_R_phi,
                                Bx = parent_R_B[term_k][i],
                                By = child_R_B[term_k][j],
                                binning=binning
                            )
                # And now account for random offsets
                deltaR = parent.R_random_offset.matrix().T.dot(deltaR).dot(child.R_random_offset.matrix())
            # Now we can use that expression for deltaR to construct the bingham cost.
            add_bingham_cost(prog, deltaR, active, M, Z, log_normalizer)