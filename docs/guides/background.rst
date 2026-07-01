Scientific background and model overview
=========================================

This page provides the conceptual context behind the pipeline: what CIB and
tSZ foregrounds are, why they must be modelled jointly, what the flat-sky
approximation means in practice, and how a denoising diffusion model learns
to generate correlated map pairs.  It is intended for readers new to one or
more of these topics.

.. contents:: Contents
   :local:
   :depth: 2

----

The cosmic microwave background and its foregrounds
----------------------------------------------------

The cosmic microwave background (CMB) is thermal radiation from when the
Universe became transparent, roughly 380,000 years after the Big Bang.  It
arrives at Earth as a nearly isotropic blackbody spectrum at 2.725 K, with
temperature fluctuations of order 10\ :sup:`−5` K imprinted by primordial
density perturbations.  Measuring these fluctuations with high precision is
one of the primary goals of modern cosmology.

The difficulty is that the CMB signal is contaminated by **foregrounds** —
emission from astrophysical sources at similar frequencies.  At millimetre
wavelengths, the dominant extragalactic foregrounds are:

- The **Cosmic Infrared Background (CIB)**: the integrated thermal emission
  from dust-enshrouded star-forming galaxies across cosmic history.  CIB
  intensity peaks at far-infrared wavelengths (~100–200 μm) but has a
  long tail into the millimetre band.  Its angular power spectrum is roughly
  a power law, C_ℓ ∝ ℓ\ :sup:`−1`, reflecting the clustering of galaxies on
  large scales and Poisson shot noise on small scales.

- The **thermal Sunyaev–Zeldovich (tSZ) effect**: inverse Compton scattering
  of CMB photons off hot electrons in the intracluster medium (ICM) of galaxy
  clusters.  The tSZ signal has a distinctive spectral shape — it decreases
  the CMB intensity below ~217 GHz and increases it above — parameterised by
  the Compton-y parameter, which integrates the electron pressure along the
  line of sight.  The angular power spectrum is dominated by massive clusters
  and has a characteristic peak near ℓ ≈ 3000.

Both foregrounds are non-Gaussian: the CIB is skewed by bright point sources
and filamentary large-scale structure, while the tSZ is heavily skewed by
massive clusters whose profiles have extended wings.

Why model CIB and tSZ jointly?
--------------------------------

CIB and tSZ are physically correlated.  Galaxy clusters — the sources of tSZ
— also host dusty star-forming galaxies that contribute to the CIB.  The
cross-correlation between CIB and tSZ, parametrised by C_ℓ\ :sup:`CIB×tSZ`,
is non-zero and scale-dependent.  Ignoring this correlation introduces biases
in CMB component separation algorithms (e.g. ILC, cILC) that attempt to
subtract both foregrounds simultaneously.

The standard approach models CIB and tSZ as independent Gaussian random
fields.  This is inaccurate for two reasons:

1. **Non-Gaussianity**: both fields have significant higher-order statistics
   (bispectrum, trispectrum) arising from the clustering of sources and the
   mass function of clusters.
2. **Correlations**: a Gaussian model cannot reproduce the physical CIB×tSZ
   cross-correlation without an explicit covariance term, which is
   scale-dependent and simulation-dependent.

A joint generative model trained on simulations sidesteps both problems: the
DDPM learns the full joint distribution p(CIB, tSZ) implicitly, reproducing
all higher-order statistics and cross-correlations present in the training
data.

The AGORA simulation
--------------------

The training data comes from the **AGORA** suite of multi-wavelength
cosmological simulations (Omori et al. 2022), which produces full-sky
HEALPix maps of the CIB, tSZ, kSZ, and other foreground components.  AGORA
uses the MDPL2 N-body simulation with BAHAMAS sub-grid baryonic physics,
producing maps that are statistically consistent with SPT-3G, ACT, and
*Planck* observations.

The maps used here are:

- **CIB at 150 GHz** (lensed, magnified): ``agora_len_mag_cibmap_act_150ghz.fits``
  at NSIDE=8192 in Jy/sr.  Available from the Globus collection *Agora* at
  ``/components/cib/len/act/nocc/``.
- **tSZ Compton-y** (lensed): ``agora_ltszNG_bahamas80_bnd_unb_1.0e+12_1.0e+18_lensed.fits``
  at NSIDE=8192.  Available at ``/components/tsz/len/``.

See :doc:`../tutorials/02_masking` and :doc:`../tutorials/03_patch_extraction`
for how these maps are preprocessed into training patches.

The flat-sky approximation
---------------------------

Full-sky CMB analysis uses spherical harmonics, which are the natural basis
for functions on the sphere.  However, for small sky patches (≲ 10°), the
curvature of the sphere is negligible and the sky can be treated as a flat
2D plane — the **flat-sky approximation**.

Under this approximation, the 2D Fourier transform replaces spherical
harmonic decomposition.  The angular wavenumber ℓ maps to the Fourier
wavenumber ``|k| = sqrt(kx² + ky²)`` in radians\ :sup:`−1`, and the angular
power spectrum C_ℓ is estimated by azimuthal averaging of the 2D power
spectral density.

The patches used here are **6° × 6°** projected onto a 256 × 256 pixel grid
at 1.40625 arcmin/pixel.  At 6°, the flat-sky approximation introduces
angular errors of order (θ/2)² ≈ 0.3% — negligible for the foreground
statistics computed here.

The approximation breaks down for patches much larger than ~10° and for the
lowest multipoles (ℓ ≲ 100), where the curvature of the sphere becomes
significant.  It also cannot represent E/B mixing of polarisation on large
scales.  :func:`~foregrounds_diffusion.flatmaps.convert_eb_qu` handles the
E/B ↔ Q/U rotation within the flat-sky framework.

Denoising diffusion probabilistic models
-----------------------------------------

A **denoising diffusion probabilistic model (DDPM)** is a generative model
that learns to synthesise data samples from a target distribution p(x) — in
this case, pairs of CIB and tSZ patches.

.. rubric:: Forward process (adding noise)

Training defines a fixed **forward process** that gradually adds Gaussian
noise to a training patch x₀ over T timesteps::

    x_t = sqrt(ᾱ_t) · x₀ + sqrt(1 − ᾱ_t) · ε,    ε ~ N(0, I)

where ᾱ_t ∈ (0, 1) is a scheduled noise level.  At t = T = 1000, ᾱ_T ≈ 0
and x_T is almost pure Gaussian noise.

.. rubric:: Reverse process (denoising)

A neural network (here, a U-Net) is trained to predict the noise ε that was
added at timestep t, given the noisy image x_t and the timestep t itself::

    ε_θ(x_t, t) ≈ ε

This is equivalent to estimating the score function ∇ log p_t(x_t) of the
marginal distribution at time t.  Once the network is trained, new samples
are generated by starting from x_T ~ N(0, I) and iteratively applying the
learned denoising step from t = T down to t = 0 — the **reverse process**.

.. rubric:: Why it works for correlated fields

The two-channel input (CIB + tSZ stacked along the channel dimension) means
the U-Net denoises both maps simultaneously at every timestep.  The shared
attention and residual connections in the U-Net backbone learn the joint
spatial structure, so the generated CIB and tSZ patches are automatically
correlated in the same way as the training data.  No explicit cross-spectrum
constraint is imposed — the correlation emerges from the training objective.

.. rubric:: DDIM: faster sampling

Standard DDPM sampling requires T = 1000 reverse steps.  **DDIM** (Denoising
Diffusion Implicit Models; Song et al. 2020) reformulates the reverse process
as a deterministic ODE, allowing accurate sampling in as few as 50–250 steps
by skipping most intermediate timesteps.  Pass ``--sampling-timesteps 250``
to the sampling script to use DDIM without retraining:

.. code-block:: bash

    accelerate launch foregrounds_diffusion/sample.py \
      --checkpoint results/my_run/model-20.pt \
      --sampling-timesteps 250 \
      --output samples_ddim.npy

Interpreting the validation statistics
----------------------------------------

The paper validates DDPM samples against AGORA maps using several statistics.
Here is a brief guide to what each one measures physically.

.. rubric:: Power spectrum C_ℓ

The power spectrum measures how much variance in the map is contributed by
structures at each angular scale ℓ.  A match at all ℓ means the DDPM
reproduces the correct two-point statistics — i.e. it gets the right
*amplitudes* at every scale, but this is insensitive to phase structure and
non-Gaussianity.

.. rubric:: Higher-order cross-moments

The 12 cross-moments (S2, S3, S4 combinations of CIB and tSZ) test whether
the joint distribution of bandpass-filtered CIB and tSZ agrees beyond the
power spectrum:

- **S2^{ab}** (cross-variance): the covariance of CIB and tSZ in an ℓ-band,
  proportional to the cross-spectrum.  A mismatch here means the CIB×tSZ
  correlation is wrong at that scale.
- **S3** terms (skewness): sensitive to asymmetric tails — in tSZ maps,
  driven by massive clusters contributing large positive peaks.
- **S4** terms (kurtosis): sensitive to extreme-value events.  The paper
  finds the DDPM slightly under-reproduces kurtosis in the tSZ channel,
  corresponding to a deficit of very massive clusters in the samples.

.. rubric:: Minkowski functionals

The three Minkowski functionals probe the *topology* of excursion sets at
threshold ν:

- **M0(ν)**: if the DDPM curve lies above AGORA, it is generating too many
  bright pixels at that threshold — the map amplitude is over-dispersed.
- **M1(ν)**: a mismatch in perimeter indicates that cluster boundary shapes
  are too smooth or too rough relative to training data.
- **M2(ν)**: a mismatch in Euler characteristic indicates that the
  connectivity structure of bright regions is wrong — e.g. too many isolated
  clusters or too few merged structures.

.. rubric:: Minkowski tensors (anisotropy index β)

β = λ_min / λ_max ∈ [0, 1] measures how anisotropic the excursion set
boundaries are.  β = 1 means perfectly circular cluster boundaries; β < 1
means elongated or filamentary structures.  Real tSZ clusters are mildly
anisotropic (β ≈ 0.7–0.9 at intermediate thresholds) due to mergers and
filamentary infall.  A DDPM that underproduces anisotropy (β too close to 1)
is generating overly round clusters.

.. rubric:: tSZ stacking

Stacking averages cutouts centred on local SNR peaks, suppressing noise and
revealing the mean cluster profile.  A mismatch in the stacked amplitude at
a given SNR bin means the DDPM is generating clusters with a different
integrated Compton-y signal than AGORA at that mass/redshift range.  The
~19% deficit found in the paper at high SNR directly corresponds to the
under-representation of very massive clusters noted in the kurtosis analysis.

.. rubric:: Peak and minima counts

Peak counts N(ν) count local maxima above threshold ν = T/σ.  They are more
sensitive than the one-point PDF to the clustering of peaks — a random field
with the right one-point distribution can still have the wrong peak count
statistics if the spatial correlations are wrong.  Minima counts provide an
additional test of the negative tail of the distribution, probing void-like
underdense regions.

.. rubric:: Scattering transform coefficients

Wavelet scattering coefficients capture multi-scale phase correlations that
the power spectrum misses.  The S2/S1 ratio matrix identifies which pairs of
angular scales the DDPM fails to couple correctly.  Values close to 1 across
the full matrix indicate the DDPM has learned the correct multi-scale
structure; deviations identify the angular scales where the model breaks down.
