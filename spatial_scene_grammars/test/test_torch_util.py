import pytest

import torch

from spatial_scene_grammars.torch_utils import *
from torch.distributions import constraints

from pytorch3d.transforms.rotation_conversions import (
    euler_angles_to_matrix
)

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

def test_constrained_param():
    # Make sure calling the constraints work for empty and nonempty init_values.
    def do_tests(init_value, constraint):
        p = ConstrainedParameter(init_value, constraint=constraint)
        val = p()
        assert(torch.allclose(init_value, val))
        p.set_unconstrained_as_param(init_value)
        val = p.get_unconstrained_value()
        assert(torch.allclose(init_value, val))
        p.set(init_value)
        val = p()
        assert(torch.allclose(init_value, val))
    do_tests(torch.tensor([]), constraint=constraints.real)
    do_tests(torch.tensor([]), constraint=constraints.simplex)
    do_tests(torch.ones(10), constraint=constraints.real)
    do_tests(torch.ones(10)/10., constraint=constraints.simplex)

def test_interpolate_translation():
    t1 = torch.tensor([0., 0., 0.])
    t2 = torch.tensor([1., 1., 1])

    t = interp_translation(t1, t2, interp_factor=0.)
    assert torch.allclose(t, t1), (t, t1)

    t = interp_translation(t1, t2, interp_factor=1.)
    assert torch.allclose(t, t2), (t, t2)

    t = interp_translation(t1, t2, interp_factor=0.5)
    expected = torch.tensor([0.5, 0.5, 0.5])
    assert torch.allclose(t, expected), (t, expected)

    t = interp_translation(t2, t2, interp_factor=0.75)
    assert torch.allclose(t, t2), (t, t2)



def test_interpolate_rotation():
    R1 = euler_angles_to_matrix(torch.tensor([0.0, 0.0, 0.0]), convention="ZYX")
    R2 = euler_angles_to_matrix(torch.tensor([np.pi/2., 0., 0.]), convention="ZYX")

    R = interp_rotation(R1, R2, interp_factor=0.)
    assert torch.allclose(R, R1), (R, R1)

    R = interp_rotation(R1, R2, interp_factor=1.)
    assert torch.allclose(R, R2), (R, R2)

    R = interp_rotation(R2, R2, interp_factor=0.75)
    assert torch.allclose(R, R2), (R, R2)

    R = interp_rotation(R1, R2, interp_factor=0.5)
    expected = euler_angles_to_matrix(torch.tensor([np.pi/4., 0., 0.]), convention="ZYX")
    assert torch.allclose(R, expected), (R, expected)

    R3 = euler_angles_to_matrix(torch.tensor([np.pi, 0., 0.]), convention="ZYX")
    R = interp_rotation(R1, R3, interp_factor=0.5)
    angle_distance = torch.arccos((torch.trace(torch.matmul(R.transpose(1, 0), R3)) - 1)/2.)
    assert torch.isclose(angle_distance, torch.tensor(np.pi/2.)), (angle_distance, np.pi/2.)

    R4 = euler_angles_to_matrix(torch.tensor([3.*np.pi/2., 0., 0.]), convention="ZYX")
    R = interp_rotation(R1, R4, interp_factor=0.5)
    expected = euler_angles_to_matrix(torch.tensor([-np.pi/4., 0., 0.]), convention="ZYX")
    assert torch.allclose(R, expected), (R, expected)

if __name__ == "__main__":
    pytest.main()