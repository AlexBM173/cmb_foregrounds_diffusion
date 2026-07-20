#!/usr/bin/env python
"""NB02 — Masking (script version of docs/tutorials/02_masking.ipynb).

Applies the 2 mJy point-source mask at NSIDE=8192, inpaints, degrades to
NSIDE=2048, applies the apodised M500c >= 3e14 cluster mask, converts to uK,
and writes the two masked FITS maps consumed by nb03_run.py.

Fixes vs the notebook: no pip-install shell line, VM paths (auto-located),
catalogue filename matches NB01's actual output, sequential CIB/tSZ
processing with del/gc to fit in 64 GB RAM, GATE prints for the
mask-fraction sanity checks.

Run:  python nb02_run.py 2>&1 | tee logs/nb02.log
"""

import gc
import os
from pathlib import Path

import healpy as hp
import numpy as np

from foregrounds_diffusion.masking import (
    get_apodised_mdpl2_cluster_mask,
    get_point_source_mask_in_healpix,
    inpaint_masked_regions,
)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
PROJECT_ROOT = Path.home() / "cmb_foregrounds_diffusion"

FREQ = 150.0  # GHz
PTSRC_THRESH_MJY = 2.0  # mJy threshold at 150 GHz
# Cluster-mask threshold in FILE units. Resolved 2026-07-04: the catalogue's
# totm500 is M_sun/h, so the paper's 3e14 M_sun cut is 2.03e14 in file units.
# This is the corrected default; the 3e14 value over-cuts (see GATE 2 below).
M500C_THRESHOLD = float(os.environ.get("M500C_THRESHOLD", "2.03e14"))
NSIDE_IN = 8192
NSIDE_OUT = 2048
# uK per unit Compton-y at 150 GHz: g(x) * T_CMB
TSZ_SPECTRAL_FACTOR = -0.952 * 2.7255e6
CONV_K_TO_MJY_PER_SR = 375.876  # MJy/sr per K (dB/dT) at 150 GHz

CIB_NAME = "agora_len_mag_cibmap_act_150ghz.fits"
TSZ_NAME = "agora_ltszNG_bahamas80_bnd_unb_1.0e+12_1.0e+18_lensed.fits"
HALO_CAT = Path(
    os.environ.get(
        "HALO_CAT",
        PROJECT_ROOT / "data" / "halo_catalogue" / "halo_catalogue_m500gt1e13.npz",
    )
)

OUT_DIR = PROJECT_ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_file(name):
    """Locate a data file under the usual VM directories."""
    for root in (PROJECT_ROOT / "data", Path.home() / "agora_data", Path.home()):
        hits = sorted(root.rglob(name)) if root.exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under {PROJECT_ROOT}/data, ~/agora_data or ~")


CIB_FITS = find_file(CIB_NAME)
TSZ_FITS = find_file(TSZ_NAME)
print(f"CIB map : {CIB_FITS}")
print(f"tSZ map : {TSZ_FITS}")
print(f"Halo cat: {HALO_CAT}")
print(f"Cluster-mask threshold: {M500C_THRESHOLD:.3e} (file units, M_sun/h)")
assert HALO_CAT.exists(), f"halo catalogue missing: {HALO_CAT}"

rng = np.random.default_rng(seed=42)
npix_in = hp.nside2npix(NSIDE_IN)

# -------------------------------------------------------------------------
# 1. CIB: load, convert Jy/sr -> MJy/sr, build point-source mask
# -------------------------------------------------------------------------
print(f"\nLoading CIB at NSIDE={NSIDE_IN} ({npix_in:,} pixels, float32) ...", flush=True)
cib = hp.read_map(CIB_FITS, dtype=np.float32)
assert len(cib) == npix_in, f"CIB NSIDE mismatch: {hp.get_nside(cib)}"
print(f"CIB — min {cib.min():.4e}  max {cib.max():.4e}  std {cib.std():.4e} (Jy/sr)")

cib *= 1e-6  # Jy/sr -> MJy/sr, in place

ptsrc_pixels = get_point_source_mask_in_healpix(
    freq=FREQ,
    hmap_Mjy_per_sr=cib,
    threshold_mjy_freq0=PTSRC_THRESH_MJY,
    freq0=150.0,
)
ptsrc_mask = np.ones(npix_in, dtype=np.float32)
ptsrc_mask[ptsrc_pixels] = 0.0
ptsrc_frac = 100.0 * (ptsrc_mask == 0).mean()
del ptsrc_pixels
gc.collect()
print(
    f"\nGATE 1 — point-source mask removes {ptsrc_frac:.3f}% of sky (paper expectation: < 1%)",
    flush=True,
)

# -------------------------------------------------------------------------
# 2. Inpaint point sources at NSIDE=8192 BEFORE degrading (order matters:
#    degrading first would spread the holes over 4x larger angular scales),
#    then degrade to NSIDE=2048. CIB first, then tSZ, freeing as we go.
# -------------------------------------------------------------------------
print("Inpainting + degrading CIB ...", flush=True)
cib = inpaint_masked_regions(cib, ptsrc_mask, rng=rng)
cib_2048 = hp.ud_grade(cib, nside_out=NSIDE_OUT).astype(np.float32)
del cib
gc.collect()

print("Loading, inpainting + degrading tSZ ...", flush=True)
tsz = hp.read_map(TSZ_FITS, dtype=np.float32)
assert len(tsz) == npix_in, f"tSZ NSIDE mismatch: {hp.get_nside(tsz)}"
print(f"tSZ — min {tsz.min():.4e}  max {tsz.max():.4e}  std {tsz.std():.4e} (Compton-y)")
tsz = inpaint_masked_regions(tsz, ptsrc_mask, rng=rng)
tsz_2048 = hp.ud_grade(tsz, nside_out=NSIDE_OUT).astype(np.float32)
del tsz, ptsrc_mask
gc.collect()

# -------------------------------------------------------------------------
# 3. Cluster mask at NSIDE=2048 (built at output resolution — cheaper).
#    Note: paper quotes masking radii of 3-5 x theta_500c; the code uses a
#    flat 3x (howmanythetaforclusters=3).
# -------------------------------------------------------------------------
print("Building apodised cluster mask ...", flush=True)
cluster_mask = get_apodised_mdpl2_cluster_mask(
    nside=NSIDE_OUT,
    halo_cat_fname=str(HALO_CAT),
    m500c_threshold=M500C_THRESHOLD,
    howmanythetaforclusters=3,
    apodise=True,
)
cluster_frac = 100.0 * (1.0 - cluster_mask.mean())
print(f"\nGATE 2 — cluster mask removes {cluster_frac:.2f}% of sky.")
print("         Expect ~1% with the resolved default (2.03e14 file units,")
print("         m500gt1e13 catalogue): masses are M_sun/h, cut at the paper's")
print("         3e14 M_sun. A ~3-4% cut means the 3e14 file-unit over-cut is")
print("         in effect (wrong catalogue/threshold) — stop and check.\n", flush=True)

cib_masked = inpaint_masked_regions(cib_2048 * cluster_mask, cluster_mask, rng=rng)
tsz_masked = inpaint_masked_regions(tsz_2048 * cluster_mask, cluster_mask, rng=rng)
del cib_2048, tsz_2048, cluster_mask
gc.collect()

# -------------------------------------------------------------------------
# 4. Convert to uK and save
# -------------------------------------------------------------------------
cib_uK = (cib_masked / CONV_K_TO_MJY_PER_SR * 1e6).astype(np.float32)
tsz_uK = (tsz_masked * TSZ_SPECTRAL_FACTOR).astype(np.float32)
print(f"CIB — min {cib_uK.min():.4e}  max {cib_uK.max():.4e} uK")
print(f"tSZ — min {tsz_uK.min():.4e}  max {tsz_uK.max():.4e} uK")

hp.write_map(OUT_DIR / "cib_150_masked.fits", cib_uK, overwrite=True)
hp.write_map(OUT_DIR / "tsz_150_masked.fits", tsz_uK, overwrite=True)
print(f"\nSaved masked maps to {OUT_DIR}/cib_150_masked.fits and tsz_150_masked.fits")
print("NB02 done.")
