import numpy as np
import pytest

from foregrounds_diffusion.flatmaps import make_gaussian_realisation

_EL = np.arange(1, 5000)
_CL = 1e-5 * _EL.astype(float) ** (-2)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def flatskymapparams():
    return [64, 64, 1.40625, 1.40625]


@pytest.fixture
def flatskymapparams_256():
    return [256, 256, 1.40625, 1.40625]


@pytest.fixture
def gaussian_patch(flatskymapparams):
    np.random.seed(0)
    return make_gaussian_realisation(flatskymapparams, _EL, _CL)


@pytest.fixture
def patch_stack(flatskymapparams):
    np.random.seed(1)
    return np.array([
        make_gaussian_realisation(flatskymapparams, _EL, _CL)
        for _ in range(16)
    ])


@pytest.fixture
def patch_stack_256(flatskymapparams_256):
    np.random.seed(2)
    return np.array([
        make_gaussian_realisation(flatskymapparams_256, _EL, _CL)
        for _ in range(16)
    ])


@pytest.fixture
def binary_map(gaussian_patch):
    return gaussian_patch > np.median(gaussian_patch)
