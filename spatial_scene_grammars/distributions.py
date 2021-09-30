import sys
import numpy as np
import logging
import scipy.integrate as integrate
import scipy.optimize
import scipy.special

import torch
from numbers import Number
from torch.distributions import constraints
from torch.distributions.geometric import Geometric
from torch.distributions.categorical import Categorical
from torch.distributions.utils import broadcast_all
import pyro
import pyro.distributions.transforms
from pyro.distributions.torch_distribution import TorchDistribution
import pyro.distributions.constraints as pyro_constraints

class LeftSidedConstraint(constraints.Constraint):
    """
    Constrain to vectors that are 1s until they hit a 0, and then
    are all 0s.
    """
    is_discrete = True
    event_dim = 1

    def check(self, value):
        if len(value) == 0:
            return True
        is_boolean = (value == 0) | (value == 1)
        neighbor_diff = value[..., -1:] - value[..., :-1]
        return is_boolean.all(-1) & (neighbor_diff <= 0) .all(-1)


from pyro.ops.tensor_utils import safe_normalize
normalize_logabsdet_warned = False
class NormalizeWithPseudoInverse(torch.distributions.transforms.Transform):
    """
    Safely project a vector onto the sphere wrt the ``p`` norm. This avoids
    the singularity at zero by mapping to the vector ``[1, 0, 0, ..., 0]``.

    Inverse transform is not well defined as this transformation
    loses radius information; in inverse transform, we set radius to 1
    and compute log-det-Jac at that value.
    https://mc-stan.org/docs/2_27/reference-manual/unit-vector-section.html
    """

    domain = pyro_constraints.real_vector
    codomain = pyro_constraints.sphere
    bijective = True

    def __init__(self, p=2, cache_size=0):
        assert isinstance(p, Number)
        assert p >= 0
        self.p = p
        super().__init__(cache_size=cache_size)

    def __eq__(self, other):
        return type(self) == type(other) and self.p == other.p

    def _call(self, x):
        return safe_normalize(x, p=self.p)

    def _inverse(self, y):
        return y

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        global normalize_logabsdet_warned
        if not normalize_logabsdet_warned:
            logging.warning("NormalizeWithPseudoInverse LogAbsDet is super sketchy")
            normalize_logabsdet_warned = True
        return -(x*x).sum()

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return Normalize(self.p, cache_size=cache_size)


@torch.distributions.transform_to.register(pyro_constraints.sphere)
def _transform_to_sphere(constraint):
    return NormalizeWithPseudoInverse()
@torch.distributions.biject_to.register(pyro_constraints.sphere)
def _biject_to_sphere(constraint):
    return NormalizeWithPseudoInverse()



class UniformWithEqualityHandling(pyro.distributions.Uniform):
    ''' Uniform distribution, but if any of the lower bounds equal
    the upper bounds, those elements are replaced with Delta distributions,
    and modifies handling to allow upper bound to be inclusive. '''

    def __init__(self, low, high, validate_args=None, eps=1E-3):
        self.low, self.high = broadcast_all(low, high)

        # Should have been set by pyro Uniform __init__
        self._unbroadcasted_low = low
        self._unbroadcasted_high = high

        # Tolerance in bounds-checking. Very low because
        # I use SNOPT in this repo, which has 1E-6 default
        # tolerance and struggles to do better. TODO(gizatt) Need
        # a study of where I'm losing tolerance so I can tighten this.
        self.eps = eps

        if isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
            self.delta_mask = torch.isclose(torch.tensor(high) - torch.tensor(low), torch.tensor(0.))
        else:
            batch_shape = self.low.size()
            self.delta_mask = torch.isclose(high - low, high*0.)

        # Ensure we have at least slightly wide range on the inside
        # where we don't have equality.

        self.uniform_in_bounds_ll = -torch.log(self.high - self.low)
        # Have to be very selective with superclass constructors since we'll cause an
        # error in torch.distribution.Uniform.__init__
        # TorchDistributionMixin has no constructor, so we're OK not calling it
        super(torch.distributions.Uniform, self).__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            # TODO(gizatt) Can I turn this back on? It's throwing out
            # values that are within 1E-17 of the bounds
            #self._validate_sample(value)
            logging.warning("validate_sample disabled in UniformWithEqualityHandling")
        value = value.reshape(self.delta_mask.shape)
        # Handle uniform part
        in_bounds = torch.logical_and(
            value >= self.low,
            value <= self.high,
        )
        on_bounds = torch.logical_or(
            torch.isclose(value, self.low, atol=self.eps, rtol=self.eps),
            torch.isclose(value, self.high, atol=self.eps, rtol=self.eps)
        )
        ll = torch.log(torch.zeros_like(value))
        in_bounds_uniform = torch.logical_and(torch.logical_or(in_bounds, on_bounds), ~self.delta_mask)
        ll[in_bounds_uniform] = self.uniform_in_bounds_ll[in_bounds_uniform]
        ll[torch.logical_and(on_bounds, self.delta_mask)] = 0.
        return ll

    def cdf(self, value):
        print("WARN: NOT VALIDATED")
        if self._validate_args:
            self._validate_sample(value)
        result = (value - self.low) / (self.high - self.low)
        result[self.delta_mask] = (value >= self.low)[self.delta_mask]*1.
        return result.clamp(min=0, max=1)
    
    def entropy(self):
        print("WARN: NOT VALIDATED")
        diff = self.high - self.low
        diff[self.delta_mask] = 1.
        # Entropy of delta = 1. * log(1.) = 0.
        return torch.log(diff)


class LeftSidedRepeatingOnesDist(TorchDistribution):
    ''' Distribution with support over a binary vector x [1, ... 1, 0, ... 0],
    where the number of 1's is drawn from a Categorical distribution of specified
    parameters [p0 ... pN]. p0 is the prob of having 0 1's; pN is the prob of
    having all 1's.'''
    # TODO(gizatt): Implement enumerate_support if I ever want to rely on it
    # at inference time.
    arg_constraints = {}
    support = LeftSidedConstraint()

    def __init__(self, categorical_probs, validate_args=None):
        self._categorical = Categorical(categorical_probs)
        batch_shape = self._categorical.batch_shape
        event_shape = torch.Size((categorical_probs.shape[-1] - 1,))
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
    
    def sample(self, sample_shape=torch.Size()):
        ks = self._categorical.sample(sample_shape).int()
        out = torch.zeros(sample_shape + self.event_shape)
        out[..., :ks] = 1.
        return out

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        valid_entries = self.support.check(value)
        generating_k = value.sum(axis=-1)
        
        ll = self._categorical.log_prob(generating_k)
        ll[~valid_entries] = -np.inf
        return ll


class VectorCappedGeometricDist(TorchDistribution):
    ''' Distribution with support over a binary vector x of length max_repeats,
    where x is generated by taking a draw k from a geometric distribution,
    capping it at max_repeats, and activating entries up to k. '''
    # TODO(gizatt): Implement enumerate_support if I ever want to rely on it
    # at inference time.
    arg_constraints = {}
    support = LeftSidedConstraint()

    def __init__(self, geometric_prob, max_repeats, validate_args=None):
        # TODO: This could shell out to LeftSidedRepeatingOnesDist.
        self.max_repeats = max_repeats
        self._geometric = Geometric(geometric_prob)
        batch_shape = self._geometric.batch_shape
        event_shape = torch.Size((max_repeats,))
        super(VectorCappedGeometricDist, self).__init__(batch_shape, event_shape, validate_args=validate_args)
    
    def sample(self, sample_shape=torch.Size()):
        ks = self._geometric.sample(sample_shape).int()
        out = torch.zeros(sample_shape + self.event_shape)
        out[..., :ks] = 1.
        return out

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        valid_entries = self.support.check(value)
        generating_k = value.sum(axis=-1)
        
        geom_ll = self._geometric.log_prob(generating_k)
        # Fix those entries where generating k is the max size, which has extra prob
        # of geometric process generating any number above generating_k.
        geom_ll[generating_k == self.max_repeats] = torch.log((1.-self._geometric.probs)**((self.max_repeats - 1) + 1))
        # Also set infeasible entries to 0 prob.
        geom_ll[~valid_entries] = -np.inf
        return geom_ll

    
"""
Implementation of a Bingham distribution, adapted from
https://github.com/igilitschenski/deep_bingham/blob/master/bingham_distribution.py
under MIT license. Modified from that source to be a Pytorch distribution,
with a lot of functionality (particularly distribution composition)
removed.
Implements the Bingham distribution as it was proposed in:
Christopher Bingham, *"An Antipodally Symmetric Distribution on the Sphere"*,
Annals of Statistics 2(6), 1974
"""

class BinghamDistribution(TorchDistribution):
    """Implementation of the Bingham Distribution.
    We represent the Bingham distribution as
    .. math::
         f(x) = \\exp\\left( x^\\top M Z M^\\top x \\right)\\ , \\quad x\\in S^n
    The current implementation supports the 2d and 4d case of the Bingham
    Distribution (i.e. n=2 and n=4).
    Parameters
    ----------
        param_m : array of shape (dim,dim)
            Location and noise direction parameter matrix M of the Bingham
            distribution.
        param_z : array of shape (dim)
            Diagonal entries of dispersion parameter matrix Z of the Bingham
            distribution.
        options : dict
            Dictionary containing additional options that may be:
            "norm_const_mode":
                Mode of computing the normalization constant as described for
                the mode parameter of the normalization_constant method.
            "norm_const_options":
                Optional normalization constant computation options for the
                normalization_constant method. Only processed if norm_const_mode
                is provided.
    """

    # Constant indicating which dimensions are implemented.
    IMPLEMENTED_DIMENSIONS = [2, 4]
    
    arg_constraints = {}
    support = pyro_constraints.sphere

    has_rsample = False

    def __init__(self, param_m, param_z, options=dict(), validate_args=None):
        self.assert_parameters(param_m, param_z)

        self._dim = param_m.shape[0]
        self._param_m = param_m
        self._param_z = param_z
        self._mode = self._param_m[:, -1]
        
        batch_shape = torch.Size((1,))
        event_shape = self._param_z.shape

        if "norm_const_mode" in options.keys():
            nc_options = options["norm_const_options"] \
                if "norm_const_options" in options.keys() else dict()

            self._norm_const = torch.tensor(self.normalization_constant(
                param_z.numpy(), mode=options["norm_const_mode"], options=nc_options))
        else:
            self._norm_const = torch.tensor(self.normalization_constant(param_z.numpy()))

        if "flip_to_positive_z_quaternions" in options.keys():
            self._flip_output = options["flip_to_positive_z_quaternions"]
        else:
            self._flip_output = False

        #self._norm_const_deriv \
        #    = BinghamDistribution.normalization_constant_deriv(self._param_z)
        
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    ##############
    # Properties #
    ##############
    @property
    def dim(self):
        return self._dim

    @property
    def m(self):
        return self._param_m

    @property
    def mode(self):
        return self._mode

    @property
    def norm_const(self):
        return self._norm_const

    @property
    def norm_const_deriv(self):
        return self._norm_const_deriv

    @property
    def z(self):
        return self._param_z

    ##################
    # Public Methods #
    ##################
    def log_prob(self, data):
        """PDF of the Bingham Distribution
        Parameters
        ----------
        data : tensor of shape(n_points, dim)
            The samples at which the density is evaluated.
        Returns
        -------
        density : tensor of shape (n_points),
            Value of the log-pdf evaluated at each data point.
        """

        assert isinstance(data, torch.Tensor), \
            "Samples need bo be of type numpy.ndarray."

        data = data.reshape(-1, self._dim)
        assert len(data.shape) == 2 and data.shape[1] == self._dim, \
            "Sample dimension does not agree with distribution dimension."

        # Here, the Bingham distribution parametrization we use is
        # f(x) \propto exp(x^T M Z M^T x)
        full_param_matrix = torch.matmul(
            self._param_m,
            torch.matmul(
                torch.diag(self._param_z),
                self._param_m.T
            )
        )
        
        # This can be later vectorized for speed-up
        log_density = torch.empty(data.shape[0])
        for i in range(0, data.shape[0]):
            log_density[i] = (data[i] * torch.matmul(full_param_matrix, data[i])).sum()
        log_density = log_density - torch.log(self._norm_const)
        return log_density
        
    def sample(self, sample_shape=torch.Size([])):
        """Generates Bingham random samples.
        The random sampling uses a rejection method that was originally
        proposed in
        J. T. Kent, A. M. Ganeiber, K. V. Mardia, "A New Method to Simulate
        the Bingham and Related Distributions in Directional Data Analysis
        with Applications", arXiv preprint arXiv:1310.8110, 2013.
        Parameters
        ----------
        sample_shape : torch size of output array of samples to produce.
        Returns
        -------
        samples : array of shape (n_points, dim)
            Array with random samples.
        """
        if sample_shape == torch.Size([]):
            n = 1
        else:
            assert len(sample_shape) == 1
            n = sample_shape[0]

        # Use existing numpy code to do sampling, and then
        # wrap result as torch tensor. We won't get gradients
        # with respect to parameters this way; to do so, we'll
        # need to get gradients through an optimization that happens
        # here.
        param_z_np = self._param_z.numpy()
        param_m_np = self._param_m.numpy()
        
        samples = np.zeros([n, self._dim])

        a = -np.dot(
            self._param_m, np.dot(np.diag(param_z_np), param_m_np.T))

        b = scipy.optimize.fsolve(
            lambda x: np.sum(1. / (x - 2. * param_z_np)) - 1,
            1.0
        )[0]

        omega = np.eye(self._dim) + 2. * a / b
        mbstar = np.exp(-(self._dim - b) / 2.) \
            * (self._dim / b)**(self._dim / 2.)

        def fb_likelihood(x):
            return np.exp(np.dot(-x, np.dot(a, x.T)))

        def acg_likelihood(x):
            return np.dot(x, np.dot(omega, x.T)) ** (-self._dim / 2.)

        current_sample = 0
        while current_sample < n:
            candidate = np.random.multivariate_normal(
                np.zeros(self._dim), np.linalg.inv(omega), 1)
            candidate = candidate / np.linalg.norm(candidate)
            # Flip so the last element is positive if requested;
            # useful if the output is being converted into 3D rotations
            # but wants to be able to unambiguously recover the sample
            # that generated it. This is like always sampling from one of the
            # antipodally symmetric modes.
            if self._flip_output:
                if candidate[0, -1] < 0:
                    candidate = -candidate

            w = np.random.uniform()
            if w < fb_likelihood(candidate) / (mbstar *
                                               acg_likelihood(candidate)):
                samples[current_sample] = candidate
                current_sample += 1

        if sample_shape == torch.Size([]):
            return torch.tensor(samples[0])
        return torch.tensor(samples)

    ##################
    # Static Methods #
    ##################
    @staticmethod
    def assert_parameters(param_m, param_z):
        """Asserts param_m and param_z to satisfy requirements of the Bingham"""
        assert isinstance(param_m, torch.Tensor), \
            "m needs to be a torch tensor.."
        assert isinstance(param_z, torch.Tensor), \
            "z needs to be a torch tensor."
            
        assert param_m.requires_grad == False, \
            "Parameter gradients not available: %s" % param_m
        assert param_z.requires_grad == False, \
            "Parameter gradients not available: %s" % param_z

        dist_dim = param_m.shape[0]
        assert dist_dim in BinghamDistribution.IMPLEMENTED_DIMENSIONS, \
            "Not supported distribution dimension."

        # Currently we support only 2d Bingham distribution.
        assert param_m.shape == (dist_dim, dist_dim), \
            "m needs to be a square Matrix."

        assert param_z.shape == (dist_dim, ), \
            "z needs to be a vector and dimension needs to agree with m."

        # TODO: Get rid of these 2 asseritons by using properties for getting
        # and setting the location parameter m and the dispersion parameter z.
        assert param_z[-1] == 0., "Last entry of z needs to be 0."
        assert all(param_z[:-1] <= param_z[1:]), \
            "Entries of z need to be given in an ascending order."

        # Check for orthogonality of m.
        diff = torch.matmul(param_m, param_m.T) - torch.eye(dist_dim)
        assert torch.allclose(diff, torch.zeros(diff.shape)), \
            "param_m is not orthogonal."

    @staticmethod
    def fit(data, weights=None):
        """Fits a bingham distribution to given data.
        The implemented fitting procedure is based on the method of moments,
        i.e. we compute the empirical second moment of the data and numerically
        obtain the corresponding Bingham distribution parameters.
        Parameters
        ----------
        data : array of shape(n_points, 2)
            The samples at which the density is evaluated.
        Returns
        -------
        result : Bingham distribution object
        """
        assert isinstance(data, torch.Tensor), \
            "data needs to be a np.ndarray"

        bd_dim = data.shape[1]

        assert bd_dim in BinghamDistribution.IMPLEMENTED_DIMENSIONS, \
            "Not supported Bingham distribution dimensionality."

        n_samples = data.shape[0]
        if weights is None:
            weights = torch.ones(n_samples) / n_samples
        else:
            assert weights.shape == (n_samples,), weights.shape
            weights = weights / torch.sum(weights)
        second_moment = torch.matmul(data.T*weights, data)
        return BinghamDistribution.fit_to_moment(second_moment)

    @staticmethod
    def fit_to_moment(second_moment):
        """Finds a Bingham distribution with a given second moment.
        Parameters
        ----------
        second_moment : (d x d matrix)
            matrix representing second moment.
        Returns
        -------
        b : BinghamDistribution
            the MLE estimate for a Bingham distribution given the
            scatter matrix S
        """
        assert torch.allclose(second_moment, second_moment.T), \
            "second moment must be symmetric"
        bd_dim = second_moment.shape[1]

        # Break into numpy land during fitting process

        second_moment = second_moment.detach().numpy()

        (moment_eigval, bingham_location) = np.linalg.eig(second_moment)

        # Sort eigenvalues (and corresponding eigenvectors) in asc. order.
        eigval_order = np.argsort(moment_eigval)
        bingham_location = bingham_location[:, eigval_order]
        moment_eigval = moment_eigval[eigval_order]

        def mle_goal_fun(z, rhs):
            """Goal function for MLE optimizer."""

            z_param = np.append(z, 0)
            norm_const = BinghamDistribution.normalization_constant(z_param)
            norm_const_deriv \
                = BinghamDistribution.normalization_constant_deriv(z_param)

            res = (norm_const_deriv[0:(bd_dim-1)] / norm_const) \
                - rhs[0:(bd_dim-1)]
            return res

        bingham_dispersion = scipy.optimize.fsolve(
            lambda x: mle_goal_fun(x, moment_eigval), np.ones([(bd_dim-1)]))
        bingham_dispersion = np.append(bingham_dispersion, 0)

        return torch.tensor(bingham_location), torch.tensor(bingham_dispersion)

    @staticmethod
    def normalization_constant(param_z, mode="default", options=dict()):
        """Computes the Bingham normalization constant.
        Parameters
        ----------
        param_z : array of shape (dim)
            Diagonal entries of dispersion parameter matrix Z of the Bingham
            distribution.
        mode : string
            Method of computation (optional).
        options : dict
            Computation-method specific options.
        """
        # Gerhard Kurz, Igor Gilitschenski, Simon Julier, Uwe D. Hanebeck,
        # "Recursive Bingham Filter for Directional Estimation Involving 180
        # Degree Symmetry", Journal of Advances in Information
        # Fusion, 9(2):90 - 105, December 2014.

        bd_dim = param_z.shape[0]

        assert bd_dim in BinghamDistribution.IMPLEMENTED_DIMENSIONS \
            and param_z.ndim == 1, \
            "param_z needs to be a vector of supported dimension."

        # TODO Check structure of Z

        if bd_dim == 2:
            if mode == "default" or mode == "bessel":
                # Surface area of the unit sphere is a factor in the
                # normalization constant. The formula is taken from
                # https://en.wikipedia.org/wiki/N-sphere#Volume_and_surface_area
                sphere_surface_area = 2.0 * (np.pi**(bd_dim / 2.0) /
                                             scipy.special.gamma(bd_dim / 2.0))

                norm_const = (np.exp(param_z[1]) * sphere_surface_area *
                              scipy.special.iv(
                                  0, (param_z[0] - param_z[1]) / 2.0)
                              * np.exp((param_z[0] - param_z[1]) / 2.0))
                return norm_const
        elif bd_dim == 4:
            if mode == "default" or mode == "saddlepoint":
                f = BinghamDistribution.__norm_const_saddlepoint(
                    np.sort(-param_z)+1)
                f *= np.exp(1)
                return f[2]
            elif mode == "numerical":
                param_z_diag = np.diag(param_z)

                def bd_likelihood(x):
                    return np.exp(np.dot(x, np.dot(param_z_diag, x)))

                def integrand(phi1, phi2, phi3):
                    sp1 = np.sin(phi1)
                    sp2 = np.sin(phi2)
                    return bd_likelihood(np.array([
                        sp1 * sp2 * np.sin(phi3),
                        sp1 * sp2 * np.cos(phi3),
                        sp1 * np.cos(phi2),
                        np.cos(phi1)
                    ])) * (sp1 ** 2.) * sp2

                norm_const = integrate.tplquad(
                    integrand,
                    0.0, 2.0 * np.pi,  # phi3
                    lambda x: 0.0, lambda x: np.pi,  # phi2
                    lambda x, y: 0.0, lambda x, y: np.pi,  # phi1
                    **options
                )

                return norm_const[0]

        sys.exit("Invalid computation mode / dimension combination.")

    @staticmethod
    def normalization_constant_deriv(param_z, mode="default"):
        """Computes the derivatives (w.r.t. Z) of the normalization constant.
        Parameters
        ----------
        param_z : array of shape (dim)
            Diagonal entries of dispersion parameter matrix Z of the Bingham
            distribution.
        mode : string
            Method of computation (optional).
        """

        bd_dim = param_z.shape[0]
        assert bd_dim in BinghamDistribution.IMPLEMENTED_DIMENSIONS \
            and param_z.ndim == 1, \
            "param_z needs to be a vector of supported dimension."

        derivatives = np.zeros(bd_dim)
        if bd_dim == 2 and mode == "default":
            derivatives = np.zeros(2)
            z_param_diff = (param_z[0] - param_z[1]) / 2.0
            z_param_mean = (param_z[0] + param_z[1]) / 2.0
            b1 = scipy.special.iv(1, z_param_diff)
            b0 = scipy.special.iv(0, z_param_diff)
            derivatives[0] = np.pi * np.exp(z_param_mean) * (b1 + b0)
            derivatives[1] = np.pi * np.exp(z_param_mean) * (-b1 + b0)
        elif bd_dim == 4 and mode == "quad":
            def bd_deriv_likelihood(x, j):
                return x[j]**2 * np.exp(np.dot(x, np.dot(np.diag(param_z), x)))

            for i in range(0, bd_dim):
                derivatives[i] = integrate.tplquad(
                    lambda phi1, phi2, phi3:
                    bd_deriv_likelihood(np.flip(np.array([
                        np.cos(phi1),
                        np.sin(phi1) * np.cos(phi2),
                        np.sin(phi1) * np.sin(phi2) * np.cos(phi3),
                        np.sin(phi1) * np.sin(phi2) * np.sin(phi3),
                    ])), i) * (np.sin(phi1) ** 2.) * np.sin(phi2),
                    0.0, 2.0 * np.pi,  # phi3
                    lambda x: 0.0, lambda x: np.pi,  # phi2
                    lambda x, y: 0.0, lambda x, y: np.pi  # phi1
                )[0]
        else:
            if mode == "default" or mode == "saddlepoint":
                derivatives = np.zeros(bd_dim)
                for i in range(0, bd_dim):
                    modz = np.concatenate((param_z[0:i + 1],
                                           np.array([param_z[i]]),
                                           param_z[i:bd_dim + 1]))
                    t = BinghamDistribution.__norm_const_saddlepoint(
                        np.sort(-modz) + 1)
                    t *= np.exp(1) / (2 * np.pi)
                    derivatives[i] = t[2]
            else:
                sys.exit("No such computation mode.")

        return derivatives

    ##########################
    # Private Static Methods #
    ##########################
    @staticmethod
    def __xi2cgfderiv(t, dim, la, derriv):
        """Calculates first 4 derivatives of the cumulant generating function"""
        res = [0] * 4
        for i in range(dim):
            if i == derriv:
                scale = 3.0
            else:
                scale = 1.0
            res[0] += scale*0.5/(la[i]-t)
            res[1] += scale*0.5/((la[i]-t)*(la[i]-t))
            res[2] += scale*1/((la[i]-t)*(la[i]-t)*(la[i]-t))
            res[3] += scale*3/((la[i]-t)*(la[i]-t)*(la[i]-t)*(la[i]-t))
        return res

    @staticmethod
    def __find_root_newton(dim, la, min_el):
        """Root finding algorithm using Newton's Method"""
        prec = 1E-10  # Precision
        x = min_el - 0.5  # Upper bound for initial evaluation point
        i = 0
        while True:
            val = BinghamDistribution.__xi2cgfderiv(x, dim, la, -1)
            val[0] -= 1
            x += -val[0] / val[1]
            i += 1
            if not ((val[0] > prec or val[0] < -prec) and i < 1000):
                break
        return x

    @staticmethod
    def __find_multiple_roots_newton(dim, la, min_el):
        """Multiple roots finding algorithm using Newton's Method"""
        prec = 1E-10
        ubound = min_el - 0.5
        retval = [ubound] * (dim + 1)  # set starting value of Newton method
        i = 0
        while True:
            err = 0
            # Iterate over the Norm const and each partial derivative
            for j in range(dim + 1):
                v0 = 0
                v1 = 0
                for k in range(dim):
                    if k != j - 1:
                        v0 += 0.5 / (la[k] - retval[j])
                        v1 += 0.5 / ((la[k] - retval[j]) * (la[k]-retval[j]))
                    else:
                        v0 += 3 * 0.5/(la[k] - retval[j])
                        v1 += 3 * 0.5/((la[k] - retval[j]) * (la[k]-retval[j]))
                v0 -= 1  # because we want to solve K(t)=1
                err += abs(v0)
                retval[j] += -v0 / v1  # Newton iteration
            i += 1
            if not (err > prec and i < 1000):
                break
        return retval

    @staticmethod
    def __norm_const_saddlepoint(eigval, deriv=False):
        """ Saddlepoint based approximation of the normalization constant. """

        assert isinstance(eigval, np.ndarray), \
            "input needs to be of type numpy.ndarray."
        assert eigval.ndim == 1, \
            "input needs to be a vector"

        dim = eigval.shape[0]
        min_el = np.amin(eigval)
        result = np.zeros(3)
        derivatives = {}
        la = eigval
        scale_factor = 1.0
        if min_el <= 0:
            la = eigval - (min_el - 0.1)
            scale_factor = np.exp(-min_el + 0.1)
            min_el = 0.1
        if deriv:
            r = BinghamDistribution.__find_multiple_roots_newton(
                dim, la, min_el)
            hk = BinghamDistribution.__xi2cgfderiv(r[0], dim, la, -1)
            t = (1.0 / 8 * (hk[3] / (hk[1] * hk[1])) - 5.0 / 24 *
                 (hk[2] * hk[2] / (hk[1] * hk[1] * hk[1])))
            result[0] = (np.sqrt(2 * pow(np.pi, dim - 1)) * np.exp(-r[0]) /
                         np.sqrt(hk[1]) * scale_factor)

            for i in range(dim):
                result[0] /= np.sqrt(la[i] - r[0])

            result[1] = result[0] * (1 + t)
            result[2] = result[0] * np.exp(t)

            for i in range(dim):
                hk = BinghamDistribution.__xi2cgfderiv(r[i + 1], dim, la, i)

                t = (1.0 / 8 * (hk[3] / (hk[1] * hk[1])) - 5.0 / 24 *
                     (hk[2] * hk[2] / (hk[1] * hk[1] * hk[1])))
                derivatives[3*i] = (np.sqrt(2*pow(np.pi, dim+1))*np.exp(-r[i+1])
                                    / (np.sqrt(hk[1]) * 2 * np.pi) *
                                    scale_factor)
                for j in range(dim):
                    if j != i:
                        derivatives[3 * i] /= np.sqrt(la[j] - r[i + 1])
                    else:
                        derivatives[3 * i] /= pow(np.sqrt(la[j] - r[i + 1]), 3)

                derivatives[3 * i + 1] = derivatives[3 * i] * (1 + t)
                derivatives[3 * i + 2] = derivatives[3 * i] * np.exp(t)
                return result, derivatives
        else:
            r = BinghamDistribution.__find_root_newton(dim, la, min_el)
            hk = BinghamDistribution.__xi2cgfderiv(r, dim, la, -1)
            t = (1.0 / 8 * (hk[3] / (hk[1] * hk[1])) - 5.0 / 24 *
                 (hk[2] * hk[2] / (hk[1] * hk[1] * hk[1])))
            result[0] = (np.sqrt(2 * pow(np.pi, dim - 1)) * np.exp(-r) /
                         np.sqrt(hk[1]) * scale_factor)

            for i in range(dim):
                result[0] /= np.sqrt(la[i] - r)
            result[1] = result[0] * (1 + t)
            result[2] = result[0] * np.exp(t)
            return result
