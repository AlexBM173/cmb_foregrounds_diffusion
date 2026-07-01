scattering_stats — Scattering transform statistics
==================================================

``scattering_stats`` computes wavelet scattering transform (WST) coefficients
for ensembles of flat-sky patches.  Scattering coefficients capture
multi-scale non-Gaussian structure that is complementary to the power
spectrum and higher-order moments — they are sensitive to phase correlations
between scales in a way that collapsed moments are not.

.. rubric:: Scattering coefficients

- **S1** (first-order): mean modulus of wavelet coefficients per scale and
  orientation.  Captures the energy distribution across angular scales.
- **S2** (second-order): mean modulus of wavelet coefficients of modulus
  wavelet coefficients, i.e. cross-scale coupling.  The S2/S1 ratio matrix
  identifies which scale pairs the DDPM fails to reproduce.
- **C11** (scattering covariance): covariance of first-order coefficients
  across realisations.  Requires the Cheng et al. backend.

.. rubric:: Installation

The module tries to import ``scattering`` (Cheng et al. 2021) first, then
falls back to ``kymatio``.  The Cheng et al. implementation is preferred:

.. code-block:: bash

    git clone https://github.com/SihaoCheng/scattering_transform.git
    # copy scattering/ into the project root, or add to sys.path

    # Or install kymatio as a fallback:
    pip install kymatio

.. rubric:: Usage

.. code-block:: python

    from foregrounds_diffusion.scattering_stats import (
        compute_scattering_coefficients,
        scattering_summary,
    )

    # Compute S1 and S2 for a stack of maps
    J = 4    # number of dyadic scales
    L = 4    # number of orientations
    s1, s2 = compute_scattering_coefficients(maps_nhw, J=J, L=L)
    # s1: (N, J, L),   s2: (N, J, J, L)

    # Flatten to a feature vector for comparison
    features = scattering_summary(s1, s2)   # (N, J*L + J*J*L)

See :doc:`../tutorials/11_scattering_transforms` for a full comparison of
scattering coefficients between AGORA, DDPM, and Gaussian maps, including
the S2/S1 ratio matrix.

.. rubric:: API

.. automodule:: foregrounds_diffusion.scattering_stats
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
