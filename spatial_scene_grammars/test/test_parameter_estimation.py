import pytest

import torch

from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.parameter_estimation import *

from .grammar import *

from pydrake.all import SnoptSolver

torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=range(1))
def set_seed(request):
    torch.manual_seed(request.param)

## Get some quick rough coverage of SVI algo by
# running a few steps of parameter fitting on the grammar.
@pytest.mark.skipif(os.environ.get('GUROBI_PATH') is None or not SnoptSolver().available(),
                    reason='This test relies on Gurobi and SNOPT.')
def test_svi(set_seed):
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4)
    )
    observed_node_sets = [grammar.sample_tree(detach=True).get_observed_nodes() for k in range(3)]

    svi = SVIWrapper(grammar, observed_node_sets)
    # Can't do more than 1 iter of fitting in case the parameters jump
    # to a setting that makes parsing impossible (which is likely, since
    # we're only doing a very noisy few steps here.)
    svi.do_iterated_vi_fitting(major_iterations=1, minor_iterations=3, tqdm=None, num_elbo_samples=2, base_lr=0.01, clip=1.)

    # Make sure something happened + was logged
    assert len(svi.elbo_history) == 3
    assert all(torch.isfinite(torch.stack(svi.elbo_history))), svi.elbo_history