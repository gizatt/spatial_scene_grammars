import pytest

import torch
import pyro

from spatial_scene_grammars.torch_utils import *
from spatial_scene_grammars.drake_interop import drake_tf_to_torch_tf
from torch.distributions import constraints

from pytorch3d.transforms.rotation_conversions import (
    euler_angles_to_matrix
)

from pydrake.all import (
    UniformlyRandomRotationMatrix,
    RandomGenerator,
    RigidTransform,
    RollPitchYaw
)

torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=range(10))
def set_seed(request):
    pyro.set_rng_seed(request.param)
    return RandomGenerator(request.param)

def test_inverse_tf(set_seed):
    # Test against drake
    t = np.random.uniform(-1, 1, size=3)
    R = UniformlyRandomRotationMatrix(set_seed)
    drake_tf = RigidTransform(p=t, R=R)
    tf = drake_tf_to_torch_tf(drake_tf)
    tf_inv = invert_torch_tf(tf)
    assert torch.allclose(tf_inv, drake_tf_to_torch_tf(drake_tf.inverse()))

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

    # Test that gradient propagate correctly after changing param value
    p = ConstrainedParameter(torch.ones(3), constraint=constraints.real)
    loss = p().square().sum()
    # grad = 2. * val
    loss.backward()
    assert torch.allclose(p.get_unconstrained_value().grad, 2.*torch.ones(3)), p.get_unconstrained_value().grad

    p.get_unconstrained_value().grad = None
    p.set(torch.ones(3)*2.)
    loss = p().square().sum()
    loss.backward()
    assert torch.allclose(p.get_unconstrained_value().grad, 2.*2.*torch.ones(3)), p.get_unconstrained_value().grad


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


def test_se3_dist(set_seed):
    # Should be zero distance
    population_1 = drake_tf_to_torch_tf(RigidTransform(p=np.zeros(3))).unsqueeze(0)
    population_2 = drake_tf_to_torch_tf(RigidTransform(p=np.zeros(3))).unsqueeze(0)
    dists = se3_dist(population_1, population_2, beta=1., eps=0)
    assert dists.shape == (1, 1) and torch.isclose(dists[0, 0], torch.tensor(0.))

    # Should be 1 distance
    population_1 = drake_tf_to_torch_tf(RigidTransform(p=np.zeros(3))).unsqueeze(0)
    population_2 = drake_tf_to_torch_tf(RigidTransform(p=np.array([0, 1, 0]))).unsqueeze(0)
    dists = se3_dist(population_1, population_2, beta=1., eps=0)
    assert dists.shape == (1, 1) and torch.isclose(dists[0, 0], torch.tensor(1.))

    # Should be pi distance
    population_1 = drake_tf_to_torch_tf(RigidTransform(p=np.zeros(3))).unsqueeze(0)
    population_2 = drake_tf_to_torch_tf(RigidTransform(p=np.zeros(3), rpy=RollPitchYaw(np.pi, 0., 0.))).unsqueeze(0)
    dists = se3_dist(population_1, population_2, beta=1., eps=0)
    assert dists.shape == (1, 1) and torch.isclose(dists[0, 0], torch.tensor(np.pi))

    # Make sure it works at scale
    M = 200
    N = 100
    population_1 = []
    for k in range(M):
        t = np.random.uniform(-1, 1, size=3)
        R = UniformlyRandomRotationMatrix(set_seed)
        tf = drake_tf_to_torch_tf(RigidTransform(p=t, R=R))
        population_1.append(tf)
    population_2 = []
    for k in range(N):
        t = np.random.uniform(-1, 1, size=3)
        R = UniformlyRandomRotationMatrix(set_seed)
        tf = drake_tf_to_torch_tf(RigidTransform(p=t, R=R))
        population_2.append(tf)

    population_1 = torch.stack(population_1)
    population_2 = torch.stack(population_2)
    dists = se3_dist(population_1, population_2, beta=1., eps=0)
    assert dists.shape == (M, N) and torch.all(torch.isfinite(dists)) and torch.all(dists >= 0)

def test_mmd_se3(set_seed):
    # Basic proof-of-life for at-scale samples
    M = 200
    N = 100
    population_1 = []
    for k in range(M):
        t = np.random.uniform(-1, 1, size=3)
        R = UniformlyRandomRotationMatrix(set_seed)
        tf = drake_tf_to_torch_tf(RigidTransform(p=t, R=R))
        population_1.append(tf)
    population_2 = []
    for k in range(N):
        t = np.random.uniform(-1, 1, size=3)
        R = UniformlyRandomRotationMatrix(set_seed)
        tf = drake_tf_to_torch_tf(RigidTransform(p=t, R=R))
        population_2.append(tf)

    population_1 = torch.stack(population_1)
    population_2 = torch.stack(population_2)
    population_1.requires_grad = True
    mmd = calculate_mmd(population_1, population_2, alphas=[0.1, 1.0, 10.0], use_se3_metric=True, beta=1.0)
    # Note: this MMD estimate can be negative. See page 729 of https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf.
    assert torch.isfinite(mmd)

    # Check grad goes through
    mmd.backward()
    assert population_1.grad is not None

if __name__ == "__main__":
    pytest.main()