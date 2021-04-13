# Based on Pyro PPL MCMC kernel.s
# See them for licenses.

import math
from collections import OrderedDict

import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.util import scalar_like
from pyro.distributions import Normal

from pyro.infer.autoguide import init_to_uniform
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model
from pyro.util import optional, torch_isnan


class RandomWalkKernel(MCMCKernel):
    r"""
    Simple random-walk MCMC kernel.

    :param model: Python callable containing Pyro primitives.
    :param potential_fn: Python callable calculating potential energy with input
        is a dict of real support parameters.
    :param float variance: Variance of the normal distribution used to select
        random walk steps.
    :param dict transforms: Optional dictionary that specifies a transform
        for a sample site with constrained support to unconstrained space. The
        transform should be invertible, and implement `log_abs_det_jacobian`.
        If not specified and the model has sites with constrained support,
        automatic transformations will be applied, as specified in
        :mod:`torch.distributions.constraint_registry`.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts. This is required if model contains
        discrete sample sites that can be enumerated over in parallel.
    :param bool jit_compile: Optional parameter denoting whether to use
        the PyTorch JIT to trace the log density computation, and use this
        optimized executable trace in the integrator.
    :param dict jit_options: A dictionary contains optional arguments for
        :func:`torch.jit.trace` function.
    :param bool ignore_jit_warnings: Flag to ignore warnings from the JIT
        tracer when ``jit_compile=True``. Default is False.
    :param callable init_strategy: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    """

    def __init__(self,
                 model=None,
                 potential_fn=None,
                 variance=1.0,
                 transforms=None,
                 max_plate_nesting=None,
                 jit_compile=False,
                 jit_options=None,
                 ignore_jit_warnings=False,
                 init_strategy=init_to_uniform):
        if not ((model is None) ^ (potential_fn is None)):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        # NB: deprecating args - model, transforms
        self.model = model
        self.transforms = transforms
        self._max_plate_nesting = max_plate_nesting
        self._jit_compile = jit_compile
        self._jit_options = jit_options
        self._ignore_jit_warnings = ignore_jit_warnings
        self._init_strategy = init_strategy
        self.potential_fn = potential_fn
        self.variance = variance
        self._reset()
        super().__init__()

    def _initialize_model_properties(self, model_args, model_kwargs):
        init_params, potential_fn, transforms, trace = initialize_model(
            self.model,
            model_args,
            model_kwargs,
            transforms=self.transforms,
            max_plate_nesting=self._max_plate_nesting,
            jit_compile=self._jit_compile,
            jit_options=self._jit_options,
            skip_jit_warnings=self._ignore_jit_warnings,
            init_strategy=self._init_strategy,
            initial_params=None,
        )
        self.potential_fn = potential_fn
        self.transforms = transforms
        self._initial_params = init_params
        self._prototype_trace = trace

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params

    def _reset(self):
        self._warmup_steps = None
        self._t = 0
        self._accept_cnt = 0
        self._mean_accept_prob = 0.
    
    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        if self.model is not None:
            self._initialize_model_properties(args, kwargs)

    def cleanup(self):
        self._reset()

    def sample(self, params):
        lp_old = -self.potential_fn(params)
        new_params = {}
        for site_name, v in params.items():
            size = v.shape
            step_dist = Normal(torch.zeros(v.shape), torch.ones(v.shape)*self.variance)
            new_params[site_name] = v + pyro.sample(
                "{}_step".format(site_name), step_dist)
        lp_new = -self.potential_fn(new_params)
        accept_prob = (lp_new - lp_old).exp().clamp(max=1.)
        u = pyro.sample("accept_thresh", dist.Uniform(0., 1.))

        # Do MH check for acceptance based on score of new params.
        if u <= accept_prob:
            self._accept_cnt += 1
            accepted = True
            params = new_params
        else:
            accepted = False  

        # Compute diagnostics.
        self._t += 1
        if self._t > self._warmup_steps:
            n = self._t - self._warmup_steps
            if accepted:
                self._accept_cnt += 1
        else:
            n = self._t

        self._mean_accept_prob += (accept_prob.item() - self._mean_accept_prob) / n

        return params
        
    def logging(self):
        return OrderedDict([
            ("acc. prob", "{:.3f}".format(self._mean_accept_prob))
        ])
