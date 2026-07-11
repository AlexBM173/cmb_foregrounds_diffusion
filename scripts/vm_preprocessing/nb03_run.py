#!/usr/bin/env python
"""NB03 — Patch extraction + normalisation (script version of
docs/tutorials/03_patch_extraction.ipynb).

Low-pass filters the masked maps (lmax capped at 3*nside-1), extracts
6x6 deg 256x256 gnomonic patches, z-scores both channels, generates the
correlated Gaussian baseline, and saves the training .npy files with the
filenames pipeline/train.py expects.

Fixes vs the notebook: no pip-install line, VM paths, OUT_DIR defined
before first use, lmax=6143 (not 7000) for NSIDE=2048, correlated Gaussian
realisations use qu_or_eb="eb" (the "qu" default would run the second field
through an E/B->Q/U polarisation conversion, corrupting the tSZ baseline),
headless matplotlib.

Run:  python nb03_run.py 2>&1 | tee logs/nb03.log
"""

from pathlib import Path

import astropy.units as u
import healpy as hp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foregrounds_diffusion.flatmaps import make_gaussian_realisation, map2cl
from foregrounds_diffusion.preprocessing import FlatCutter, get_patch_centers

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
PROJECT_ROOT = Path.home() / "cmb_foregrounds_diffusion"

PATCH_DEG = 6.0  # patch side length in degrees
NRES = 256  # pixels per side
STEP_DEG = 6.0  # patch centre spacing in degrees (st6 in filenames)
GAL_CUT = 20.0  # Galactic-plane exclusion half-width in degrees
PTSRC = 2  # point-source threshold label (mJy)

CIB_MASKED_FITS = PROJECT_ROOT / "data" / "cib_150_masked.fits"
TSZ_MASKED_FITS = PROJECT_ROOT / "data" / "tsz_150_masked.fits"
OUT_DIR = PROJECT_ROOT / "data" / "low_pass" / f"{PTSRC}mJy"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

flatskymapparams = [NRES, NRES, PATCH_DEG * 60.0 / NRES, PATCH_DEG * 60.0 / NRES]
print(f"Pixel resolution: {flatskymapparams[2]:.4f} arcmin")

np.random.seed(42)  # make_gaussian_realisation uses np.random
rng = np.random.default_rng(seed=42)

# -------------------------------------------------------------------------
# 1. Load masked maps and low-pass filter
# -------------------------------------------------------------------------
cib_map = hp.read_map(CIB_MASKED_FITS)
tsz_map = hp.read_map(TSZ_MASKED_FITS)
nside = hp.get_nside(cib_map)
lmax_filter = min(7000, 3 * nside - 1)  # 6143 at NSIDE=2048; no power above it
print(f"Loaded maps at NSIDE={nside}; low-pass filtering at lmax={lmax_filter} ...", flush=True)

cib_alm = hp.map2alm(cib_map, lmax=lmax_filter)
cib_lp = hp.alm2map(cib_alm, nside=nside)
del cib_alm
tsz_alm = hp.map2alm(tsz_map, lmax=lmax_filter)
tsz_lp = hp.alm2map(tsz_alm, nside=nside)
del tsz_alm, cib_map, tsz_map

# Zero negative CIB pixels (Gibbs ringing around masked regions)
n_neg = (cib_lp < 0).sum()
print(f"Zeroing {n_neg:,} negative CIB pixels ({100 * n_neg / len(cib_lp):.2f}%)")
cib_lp[cib_lp < 0] = 0.0

# -------------------------------------------------------------------------
# 2. Extract patches
# -------------------------------------------------------------------------
centers = get_patch_centers(
    gal_cut=GAL_CUT * u.deg, step_size=STEP_DEG * u.deg, pole_cut=PATCH_DEG * u.deg
)
print(f"Patch centres (|b| > {GAL_CUT} deg, step {STEP_DEG} deg): {len(centers)}", flush=True)

cutter = FlatCutter(ang_x=PATCH_DEG * u.deg, ang_y=PATCH_DEG * u.deg, xres=NRES, yres=NRES)

cib_maps, tsz_maps = [], []
for k, (lon, lat) in enumerate(centers):
    patch = cutter.rotate_to_pole_and_interpolate(lon, lat, [cib_lp, tsz_lp])
    cib_maps.append(patch[:, :, 0:1])
    tsz_maps.append(patch[:, :, 1:2])
    if (k + 1) % 100 == 0:
        print(f"  [{k + 1}/{len(centers)}] patches extracted", flush=True)

cib_maps = np.stack(cib_maps)  # (N, 256, 256, 1) channels-last
tsz_maps = np.stack(tsz_maps)
print(f"Extracted {len(cib_maps)} patches, shape {cib_maps.shape}")

# Remove patches that are entirely inside masked regions
valid = np.array([p.max() > 0 for p in cib_maps])
n_removed = int((~valid).sum())
print(f"Removing {n_removed} all-zero patches ({100 * n_removed / len(valid):.1f}%)")
cib_maps = cib_maps[valid]
tsz_maps = tsz_maps[valid]

# -------------------------------------------------------------------------
# 3. Normalise: global z-score, BOTH channels (resolved convention)
# -------------------------------------------------------------------------
cib_mean, cib_std = cib_maps.mean(), cib_maps.std()
tsz_mean, tsz_std = tsz_maps.mean(), tsz_maps.std()
cib_norm = ((cib_maps - cib_mean) / cib_std).astype(np.float32)
tsz_norm = ((tsz_maps - tsz_mean) / tsz_std).astype(np.float32)
del cib_maps, tsz_maps
print(
    f"norm_params: cib_mean={cib_mean:.6e} cib_std={cib_std:.6e} "
    f"tsz_mean={tsz_mean:.6e} tsz_std={tsz_std:.6e}"
)

# -------------------------------------------------------------------------
# 4. Mean spectra -> correlated Gaussian baseline
# -------------------------------------------------------------------------
LMIN, LMAX, LBIN = 300, 7000, 100
n_spec = min(200, len(cib_norm))
print(f"Measuring mean spectra from {n_spec} patches ...", flush=True)

cl_cib_arr, cl_tsz_arr, cl_cross_arr = [], [], []
for i in range(n_spec):
    cib_m = cib_norm[i, :, :, 0]
    tsz_m = tsz_norm[i, :, :, 0]
    el, cl_c = map2cl(flatskymapparams, cib_m, binsize=LBIN, minbin=LMIN, maxbin=LMAX)
    _, cl_t = map2cl(flatskymapparams, tsz_m, binsize=LBIN, minbin=LMIN, maxbin=LMAX)
    _, cl_x = map2cl(flatskymapparams, cib_m, tsz_m, binsize=LBIN, minbin=LMIN, maxbin=LMAX)
    cl_cib_arr.append(cl_c)
    cl_tsz_arr.append(cl_t)
    cl_cross_arr.append(cl_x)

cl_cib_mean = np.mean(cl_cib_arr, axis=0)
cl_tsz_mean = np.mean(cl_tsz_arr, axis=0)
cl_cross_mean = np.mean(cl_cross_arr, axis=0)
print(f"Mean CIB C_ell range: {cl_cib_mean.min():.3e} - {cl_cib_mean.max():.3e}")
print(f"Mean tSZ C_ell range: {cl_tsz_mean.min():.3e} - {cl_tsz_mean.max():.3e}")

print(f"Generating {len(cib_norm)} correlated Gaussian realisations ...", flush=True)
gaussian_maps = []
for _ in range(len(cib_norm)):
    sim = make_gaussian_realisation(
        flatskymapparams,
        el,
        cl=cl_cib_mean,
        cl2=cl_tsz_mean,
        cl12=cl_cross_mean,
        qu_or_eb="eb",  # scalar fields — "qu" would EB->QU-rotate field 2
    )
    gaussian_maps.append(sim[:2])
gaussian_arr = np.asarray(gaussian_maps, dtype=np.float32)  # (N, 2, H, W)
print(f"Gaussian baseline shape: {gaussian_arr.shape}")

# -------------------------------------------------------------------------
# 5. Save training arrays (filenames must match pipeline/train.py)
# -------------------------------------------------------------------------
np.save(OUT_DIR / f"CIB_map_150GHz_{NRES}_st6_zscore_{PTSRC}mJy_lp.npy", cib_norm)
np.save(OUT_DIR / f"tSZ3_map_150GHz_{NRES}_st6_zscore_{PTSRC}mJy_lp.npy", tsz_norm)
np.save(OUT_DIR / f"norm_params_{PTSRC}mJy.npy", np.array([cib_mean, cib_std, tsz_mean, tsz_std]))
np.save(OUT_DIR / f"gaussian_cib_tsz_{PTSRC}mJy_lp.npy", gaussian_arr)
print(f"Saved 4 arrays to {OUT_DIR}")

# -------------------------------------------------------------------------
# 6. Diagnostics (Gate B sanity checks)
# -------------------------------------------------------------------------
print(
    f"\nCIB patches — shape {cib_norm.shape}  min {cib_norm.min():.4f}  "
    f"max {cib_norm.max():.4f}  mean {cib_norm.mean():.4f}  std {cib_norm.std():.4f}"
)
print(
    f"tSZ patches — shape {tsz_norm.shape}  min {tsz_norm.min():.4f}  "
    f"max {tsz_norm.max():.4f}  mean {tsz_norm.mean():.4f}  std {tsz_norm.std():.4f}"
)

zero_frac_cib = np.array([(p == 0).mean() for p in cib_norm[:, :, :, 0]])
print(f"Patches with >10% zero pixels (CIB): {(zero_frac_cib > 0.1).sum()}")

N_SHOW = 9
idx = rng.choice(len(cib_norm), size=min(N_SHOW, len(cib_norm)), replace=False)
fig, axes = plt.subplots(2, len(idx), figsize=(len(idx) * 3, 6))
for col, i in enumerate(idx):
    axes[0, col].imshow(cib_norm[i, :, :, 0], cmap="magma", vmin=-3, vmax=6)
    axes[0, col].set_title(f"CIB {i}", fontsize=8)
    axes[0, col].axis("off")
    axes[1, col].imshow(tsz_norm[i, :, :, 0], cmap="magma", vmin=-5, vmax=5)
    axes[1, col].set_title(f"tSZ {i}", fontsize=8)
    axes[1, col].axis("off")
plt.suptitle("Sample z-score-normalised training patches")
plt.tight_layout()
plt.savefig(PLOT_DIR / "sample_patches.png", dpi=150, bbox_inches="tight")
print(f"Wrote {PLOT_DIR}/sample_patches.png")
print("NB03 done.")
