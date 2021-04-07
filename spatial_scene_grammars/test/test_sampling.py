import numpy as np
import unittest
import pytest

from pydrake.all import (
    PackageMap
)

import pyro
import pyro.distributions as dist
import torch

from spatial_scene_grammars.tree import *
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.sampling import *

from spatial_scene_grammars.test.grammar import *

torch.set_default_tensor_type(torch.DoubleTensor)
pyro.enable_validation(True)

root_node_type = Building
inst_dict = {"xy": dist.Delta(torch.zeros(2))}
object_count_constraint = StackedObjectCountConstraint()

@pytest.fixture(params=range(5))
def set_seed(request):
    pyro.clear_param_store()
    torch.manual_seed(request.param)

def test_rejection(set_seed):
    grammar = SceneGrammar(root_node_type)
    trees, success = sample_tree_from_root_type_with_constraints(
        root_node_type,
        inst_dict,
        constraints=[],
        backend="rejection"
    )
    assert success

    trees, success = sample_tree_from_root_type_with_constraints(
        root_node_type,
        inst_dict,
        constraints=[object_count_constraint],
        backend="rejection",
        max_num_attempts=1000
    )
    assert success
    assert len(trees) == 1
    assert eval_total_constraint_set_violation(trees[0], [object_count_constraint]) <= 0.

if __name__ == "__main__":
    pytest.main()