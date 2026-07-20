#!/usr/bin/env python
"""NB03b — 4-channel patch extraction + normalisation.

Script version of docs/tutorials/03_patch_extraction.ipynb extended from 2 to
4 channels (CIB, tSZ, kSZ, kappa_CMB). Loads the four masked FITS maps
(CIB/tSZ from the original run, kSZ/kappa from nb02b), low-pass filters each,
extracts 6x6 deg 256x256 gnomonic patches at IDENTICAL centres, z-scores each
channel independently, and saves training .npy files plus an 8-entry
norm_params array.

Every patch is extracted at the same centre for all four channels and the
same all-zero-CIB rejection is applied to all four, so the four per-channel
arrays are row-aligned: patch i of CIB, tSZ, kSZ and kappa are the same
6x6 deg region of sky. train.py stacks them into (N, 4, 256, 256).

Per-channel choices:
  * CIB   — positive-definite intensity; negative pixels from Gibbs ringing
            around masked bright sources are zeroed (as in the 2-channel run).
  * tSZ   — pure decrement at 150 GHz (Compton-y >= 0 times a negative spectral
            factor), so the physical field is one-sided negative; positive
            pixels from Gibbs ringing are zeroed (mirror of the CIB treatment).
  * kSZ   — two-sided (sign = line-of-sight velocity direction); sign kept.
            DO NOT zero negatives — that would delete half the signal.
  * kappa — two-sided (over/under-densities); sign kept. DO NOT zero negatives.

Normalisation is per-channel z-score (mean 0, std 1 per channel). z-scoring is
a per-channel linear map, so it preserves the cross-channel correlations the
model must learn while putting the four very different amplitudes (kappa ~ 0.01,
kSZ/tSZ ~ uK) on a common scale.

The correlated-Gaussian baseline that the 2-channel nb03 produced is NOT
generated here: a faithful 4-field baseline needs the full 4x4 cross-spectrum
matrix, which make_gaussian_realisation (2-field) does not support. It is an
*evaluation* baseline, not a training input, so it is deferred to the
evaluation refactor.

Run:  python nb03b_extract_4ch.py 2>&1 | tee logs/nb03b.log
"""

from pathlib import Path

import astropy.units as u
import healpy as hp
import numpy as np

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

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "low_pass" / f"{PTSRC}mJy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (label, masked-FITS name, output .npy name, clip)
# clip controls how low-pass filter artefacts (Gibbs ringing) are handled:
#   "zero_neg" — CIB is a non-negative intensity; zero negative pixels.
#   "zero_pos" — tSZ at 150 GHz is a pure decrement (Compton-y >= 0 times a
#                negative spectral factor), so the physical field is <= 0; zero
#                positive pixels to keep it one-sided negative.
#   None       — kSZ and kappa are genuinely two-sided; keep both signs.
CHANNELS = [
    (
        "CIB",
        "cib_150_masked.fits",
        f"CIB_map_150GHz_{NRES}_st6_zscore_{PTSRC}mJy_lp.npy",
        "zero_neg",
    ),
    (
        "tSZ",
        "tsz_150_masked.fits",
        f"tSZ3_map_150GHz_{NRES}_st6_zscore_{PTSRC}mJy_lp.npy",
        "zero_pos",
    ),
    ("kSZ", "ksz_150_masked.fits", f"kSZ_map_150GHz_{NRES}_st6_zscore_{PTSRC}mJy_lp.npy", None),
    ("kappa", "kappa_masked.fits", f"kappa_map_{NRES}_st6_zscore_{PTSRC}mJy_lp.npy", None),
]

flatskymapparams = [NRES, NRES, PATCH_DEG * 60.0 / NRES, PATCH_DEG * 60.0 / NRES]
print(f"Pixel resolution: {flatskymapparams[2]:.4f} arcmin")

# -------------------------------------------------------------------------
# 1. Load each masked map and low-pass filter (lmax capped at 3*nside-1).
# -------------------------------------------------------------------------
maps_lp = []
for label, fits_name, _, clip in CHANNELS:
    path = DATA_DIR / fits_name
    if not path.exists():
        raise SystemExit(
            f"missing {path} — run nb02_run.py (CIB/tSZ) and nb02b_mask_ksz_kappa.py "
            "(kSZ/kappa) first."
        )
    m = hp.read_map(path)
    nside = hp.get_nside(m)
    lmax_filter = min(7000, 3 * nside - 1)  # 6143 at NSIDE=2048; no power above it
    print(f"[{label}] NSIDE={nside}, low-pass at lmax={lmax_filter} ...", flush=True)
    alm = hp.map2alm(m, lmax=lmax_filter)
    m_lp = hp.alm2map(alm, nside=nside)
    del alm, m

    if clip == "zero_neg":
        n_neg = int((m_lp < 0).sum())
        print(f"[{label}] zeroing {n_neg:,} negative pixels ({100 * n_neg / len(m_lp):.2f}%)")
        m_lp[m_lp < 0] = 0.0
    elif clip == "zero_pos":
        n_pos = int((m_lp > 0).sum())
        print(f"[{label}] zeroing {n_pos:,} positive pixels ({100 * n_pos / len(m_lp):.2f}%)")
        m_lp[m_lp > 0] = 0.0
    else:
        print(f"[{label}] two-sided field — both signs kept")
    maps_lp.append(m_lp)

# -------------------------------------------------------------------------
# 2. Extract patches at identical centres for all four channels.
# -------------------------------------------------------------------------
centers = get_patch_centers(
    gal_cut=GAL_CUT * u.deg, step_size=STEP_DEG * u.deg, pole_cut=PATCH_DEG * u.deg
)
print(f"\nPatch centres (|b| > {GAL_CUT} deg, step {STEP_DEG} deg): {len(centers)}", flush=True)

cutter = FlatCutter(ang_x=PATCH_DEG * u.deg, ang_y=PATCH_DEG * u.deg, xres=NRES, yres=NRES)

patches = [[] for _ in CHANNELS]  # one list per channel
for k, (lon, lat) in enumerate(centers):
    # Extract each channel INDEPENDENTLY (one single-map call each). Passing the
    # whole list at once would (with spin2=True) trigger the spin-2 (Q/U)
    # parallactic rotation on the LAST TWO maps — correct for a polarisation/
    # shear pair, but these are four spin-0 SCALAR fields. That rotation
    # cross-mixes the last pair (kSZ, kappa); with kSZ ~50 uK and kappa ~0.6 the
    # kSZ leakage swamps kappa (~80sigma spikes). Per-field spin2=False calls
    # give pure bounded bilinear interpolation per scalar field.
    for c, m in enumerate(maps_lp):
        patches[c].append(cutter.rotate_to_pole_and_interpolate(lon, lat, m, spin2=False))
    if (k + 1) % 100 == 0:
        print(f"  [{k + 1}/{len(centers)}] patches extracted", flush=True)

stacks = [np.stack(p) for p in patches]  # each (N, 256, 256, 1) channels-last
print(f"Extracted {len(stacks[0])} patches per channel, shape {stacks[0].shape}")

# -------------------------------------------------------------------------
# 3. Drop patches that fall entirely inside masked regions. The criterion is
#    CIB.max() > 0 (same as the 2-channel run) applied to ALL channels, so the
#    four arrays stay row-aligned.
# -------------------------------------------------------------------------
cib_stack = stacks[0]
valid = np.array([p.max() > 0 for p in cib_stack])
n_removed = int((~valid).sum())
print(f"Removing {n_removed} all-zero (CIB) patches ({100 * n_removed / len(valid):.1f}%)")
stacks = [s[valid] for s in stacks]

# -------------------------------------------------------------------------
# 4. Per-channel z-score. norm_params ordering matches CHANNELS:
#    [cib_mean, cib_std, tsz_mean, tsz_std, ksz_mean, ksz_std, kappa_mean, kappa_std]
# -------------------------------------------------------------------------
norm_params = []
normed = []
for (label, _, out_name, _), s in zip(CHANNELS, stacks):
    mean, std = float(s.mean()), float(s.std())
    norm_params += [mean, std]
    sn = ((s - mean) / std).astype(np.float32)
    normed.append(sn)
    print(
        f"[{label}] mean={mean:.6e} std={std:.6e} -> "
        f"min {sn.min():.3f} max {sn.max():.3f} mean {sn.mean():.3f} std {sn.std():.3f}"
    )

# -------------------------------------------------------------------------
# 5. Save training arrays + 8-entry norm_params.
# -------------------------------------------------------------------------
for (label, _, out_name, _), sn in zip(CHANNELS, normed):
    np.save(OUT_DIR / out_name, sn)
    print(f"[{label}] saved {OUT_DIR / out_name}  shape {sn.shape}")

np.save(OUT_DIR / f"norm_params_{PTSRC}mJy.npy", np.array(norm_params, dtype=np.float64))
print(
    f"\nSaved norm_params ({len(norm_params)} entries) -> {OUT_DIR / f'norm_params_{PTSRC}mJy.npy'}"
)
print("Order: [cib_mean, cib_std, tsz_mean, tsz_std, ksz_mean, ksz_std, kappa_mean, kappa_std]")
print("\nNB03b done. Upload with:")
print(f"  gcloud storage cp {OUT_DIR}/*.npy gs://cmb-diffusion-artifacts-alexbm173/patches/v5_4ch/")
