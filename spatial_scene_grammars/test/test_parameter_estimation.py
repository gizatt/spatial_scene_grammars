import pytest

import torch
import pyro

from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.parameter_estimation import *

from .grammar import *

from pydrake.all import (
    SnoptSolver
)

torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=range(1))
def set_seed(request):
    pyro.set_rng_seed(request.param)

## Get some quick rough coverage of EM algo by
# running a few steps of parameter fitting on the grammar.
@pytest.mark.skipif(os.environ.get('GUROBI_PATH') is None or not SnoptSolver().available(),
                    reason='This test relies on Gurobi and SNOPT.')
def test_em(set_seed):
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4)
    )
    observed_node_sets = [grammar.sample_tree(detach=True).get_observed_nodes() for k in range(3)]

    em = EMWrapper(grammar, observed_node_sets)
    # Can't do more than 1 iter of fitting in case the parameters jump
    # to a setting that makes parsing impossible (which is likely, since
    # we're only doing a very noisy few steps here.)
    em.do_iterated_em_fitting(em_iterations=3)

    # Make sure something happened + was logged
    assert len(em.grammar_iters) == 4 # 3 iters + original
    assert all(torch.isfinite(torch.tensor(em.log_evidence_iters).view(-1))), em.log_evidence_iters
