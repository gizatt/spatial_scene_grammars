import torch

class SceneGenerativeProgram(torch.nn.Module):
    '''
    Interface for a program that generates scenes via
    a generative process.

    - Samples from a distribution over environments
    described by latent variables z and observed variables
    x, dictated by parameters beta.

    - Allows evaluation of p_beta(x, z).
    '''

    def __init__(self):
        super().__init__()

    def get_default_param_dict(self):
        # Dictionary of ConstrainedParameter values.
        raise NotImplementedError()

    def forward(self, params=None):
        # Sample scene using either current parameter values (if no override
        # provided, or the specified overriden param values. The param values
        # should be a dictionary of param name : param value that matches the
        # module named_parameters() dict output.
        # A pyro.poutine.trace() that wraps this method should
        # grab all internal sample sites, and I think its log_prob_sum
        # should match the output from score().
        raise NotImplementedError()

    def score(self, **kwargs):
        # Score given scene. Input should match the output
        # of forward.
        raise NotImplementedError()
