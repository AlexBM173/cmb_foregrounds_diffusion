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
    return np.array([make_gaussian_realisation(flatskymapparams, _EL, _CL) for _ in range(16)])


@pytest.fixture
def patch_stack_256(flatskymapparams_256):
    np.random.seed(2)
    return np.array([make_gaussian_realisation(flatskymapparams_256, _EL, _CL) for _ in range(16)])


@pytest.fixture
def binary_map(gaussian_patch):
    return gaussian_patch > np.median(gaussian_patch)


@pytest.fixture
def skewed_patch(flatskymapparams):
    """Spatially-correlated, positively-skewed patch (lognormal-type).

    Built as ``exp(g)`` from a unit-variance correlated Gaussian realisation.
    Unlike per-pixel chi-square noise, the non-Gaussianity lives in the
    correlated structure and survives bandpass filtering (real-space skew
    ≈ 5), making it a realistic non-Gaussian test field for the higher-order
    statistics, peak-count, and morphology machinery.
    """
    np.random.seed(7)
    g = make_gaussian_realisation(flatskymapparams, _EL, _CL)
    return np.exp(g / g.std())


@pytest.fixture
def skewed_patch_stack(flatskymapparams):
    """Stack of 16 positively-skewed lognormal-type patches (see skewed_patch)."""
    stack = []
    for s in range(16):
        np.random.seed(100 + s)
        g = make_gaussian_realisation(flatskymapparams, _EL, _CL)
        stack.append(np.exp(g / g.std()))
    return np.array(stack)
