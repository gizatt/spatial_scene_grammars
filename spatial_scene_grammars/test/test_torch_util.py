import pytest

import torch

from spatial_scene_grammars.torch_utils import *

torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=range(10))
def set_seed(request):
    torch.manual_seed(request.param)

def test_inv_softplus(set_seed):
	x_test = torch.abs(torch.normal(0., 1., size=(100,))) + 0.01
	assert torch.allclose(inv_softplus(torch.nn.functional.softplus(x_test)), x_test)

def test_inv_sigmoid(set_seed):
	x_test = torch.clip(torch.normal(0., 1., size=(100,)), -0.99, 0.99)
	x_test_pred = inv_sigmoid(torch.sigmoid(x_test))
	assert torch.allclose(x_test_pred, x_test, atol=1E-5), x_test - x_test_pred

if __name__ == "__main__":
    pytest.main()