#!/usr/bin/env python
"""NB02b — Masking for the kSZ and CMB-lensing (kappa) channels.

Extends the 2-channel masking (nb02_run.py) to the two NEW channels of the
4-channel run (CIB, tSZ, kSZ, kappa_CMB). It rebuilds the *same* point-source
and apodised cluster masks used for CIB/tSZ so all four channels share an
identical masked footprint, then applies them to the kSZ and kappa maps and
writes ``ksz_150_masked.fits`` and ``kappa_masked.fits``.

Why the shared footprint matters: the DDPM learns the joint distribution of
the four channels *including their cross-correlations*. If channel A has a
pixel inpainted (noise) while channel B keeps its real value there, the model
sees a spurious A-B decorrelation at that pixel. Masking the identical pixels
in all four channels avoids this.

The CIB/tSZ masked FITS from the original run are REUSED as-is — this script
does not touch them, so the two validated channels stay bit-for-bit unchanged.
Only the point-source mask (rebuilt from the raw CIB map) and the cluster mask
(rebuilt from the halo catalogue) are recomputed; both are deterministic
functions of the same thresholds, so the footprint matches the original run.

Design decisions (revisit these if the science calls for it):
  * Point-source mask: rebuilt from the raw CIB map at NSIDE=8192 at the same
    2 mJy threshold. kSZ/kappa have no IR point sources, but the same <1% of
    sky is inpainted in them for footprint consistency.
  * Cluster mask: the same apodised M500c mask, applied to kSZ AND kappa.
    NOTE: kappa has REAL signal at cluster centres (clusters are the lenses),
    so masking clusters in kappa removes genuine convergence-cluster
    correlation. We mask anyway for footprint consistency with tSZ/CIB. Set
    MASK_CLUSTERS_IN_KAPPA=0 to keep clusters in kappa instead.
  * kSZ units: the kSZ effect is an achromatic blackbody temperature shift.
    If the FITS is already in uK it is used as-is; if it is dimensionless
    (dT/T) set KSZ_INPUT_UNIT=dimensionless to scale by T_CMB. The header
    TUNIT is printed — CONFIRM it before trusting the default.
  * kappa units: dimensionless convergence — no unit conversion.
  * kappa FITS may hold 3 columns (kappa, gamma1, gamma2) from raytracing;
    only the convergence column (KAPPA_FIELD, default 0) is read. VERIFY the
    column order in the header (`fitsheader <file> | grep TTYPE`) before
    trusting field=0 — if TTYPE1 is a shear component this will be wrong.

Inputs (searched under ~/cmb_foregrounds_diffusion/data, ~/agora_data, ~):
  * raw CIB FITS  (for the point-source mask)      — CIB_FILE  glob
  * raw kSZ FITS                                   — KSZ_FILE  glob
  * raw kappa FITS (raytrace convergence)          — KAPPA_FILE glob
  * halo catalogue .npz                            — HALO_CAT  (env, as nb02)

Run:  python nb02b_mask_ksz_kappa.py 2>&1 | tee logs/nb02b.log
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
# Repo root: derived from this script's location so it works from any checkout
# (e.g. an RDS clone), overridable via the PROJECT_ROOT env var.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

FREQ = 150.0  # GHz (kSZ is achromatic; label kept for filename parity)
PTSRC_THRESH_MJY = 2.0  # mJy threshold at 150 GHz (same as CIB/tSZ)
# Cluster-mask threshold in FILE units. Verified 2026-07-04: totm500 is
# M_sun/h, so the paper's 3e14 M_sun cut is 2.03e14 in file units. This MUST
# match the value used for the CIB/tSZ masked FITS (rerun_chain.sh: 2.03e14).
M500C_THRESHOLD = float(os.environ.get("M500C_THRESHOLD", "2.03e14"))
NSIDE_IN = 8192
NSIDE_OUT = 2048
T_CMB_UK = 2.7255e6  # uK, for dimensionless (dT/T) -> uK if needed

MASK_CLUSTERS_IN_KAPPA = os.environ.get("MASK_CLUSTERS_IN_KAPPA", "1") != "0"
KSZ_INPUT_UNIT = os.environ.get("KSZ_INPUT_UNIT", "uK").lower()  # uK | dimensionless
KAPPA_FIELD = int(os.environ.get("KAPPA_FIELD", "0"))  # column holding convergence

# File globs (override with env vars once the exact AGORA names are known).
CIB_FILE = os.environ.get("CIB_FILE", "agora_len_mag_cibmap_act_150ghz.fits")
KSZ_FILE = os.environ.get("KSZ_FILE", "*ksz*lensed*.fits")
KAPPA_FILE = os.environ.get("KAPPA_FILE", "*cmbkappa*.fits")
HALO_CAT = Path(
    os.environ.get(
        "HALO_CAT",
        PROJECT_ROOT / "data" / "halo_catalogue" / "halo_catalogue_m500gt1e13.npz",
    )
)

OUT_DIR = PROJECT_ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_file(pattern):
    """Locate a data file (name or glob) under the usual VM directories."""
    for root in (PROJECT_ROOT / "data", Path.home() / "agora_data", Path.home()):
        hits = sorted(root.rglob(pattern)) if root.exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"no match for {pattern!r} under {PROJECT_ROOT}/data, ~/agora_data or ~.\n"
        "Transfer it from Globus (or `gcloud storage cp gs://cmb-diffusion-artifacts-"
        "alexbm173/raw/... ~/agora_data/`) and retry."
    )


CIB_FITS = find_file(CIB_FILE)
KSZ_FITS = find_file(KSZ_FILE)
KAPPA_FITS = find_file(KAPPA_FILE)
print(f"CIB (for ptsrc mask): {CIB_FITS}")
print(f"kSZ map             : {KSZ_FITS}")
print(f"kappa map           : {KAPPA_FITS}  (reading field={KAPPA_FIELD})")
print(f"Halo cat            : {HALO_CAT}")
print(f"Cluster-mask threshold: {M500C_THRESHOLD:.3e} (file units, M_sun/h)")
assert HALO_CAT.exists(), f"halo catalogue missing: {HALO_CAT}"

rng = np.random.default_rng(seed=42)
npix_in = hp.nside2npix(NSIDE_IN)

# -------------------------------------------------------------------------
# 1. Rebuild the point-source mask from the raw CIB map (deterministic given
#    the map + 2 mJy threshold, so it matches the CIB/tSZ run's footprint).
#    We only need CIB to *build the mask*; its values are then discarded.
# -------------------------------------------------------------------------
print(f"\nLoading CIB at NSIDE={NSIDE_IN} to rebuild the 2 mJy mask ...", flush=True)
cib = hp.read_map(CIB_FITS, dtype=np.float32)
assert len(cib) == npix_in, f"CIB NSIDE mismatch: {hp.get_nside(cib)} (expected {NSIDE_IN})"
cib *= 1e-6  # Jy/sr -> MJy/sr, as in nb02

ptsrc_pixels = get_point_source_mask_in_healpix(
    freq=FREQ,
    hmap_Mjy_per_sr=cib,
    threshold_mjy_freq0=PTSRC_THRESH_MJY,
    freq0=150.0,
)
ptsrc_mask = np.ones(npix_in, dtype=np.float32)
ptsrc_mask[ptsrc_pixels] = 0.0
ptsrc_frac = 100.0 * (ptsrc_mask == 0).mean()
del cib, ptsrc_pixels
gc.collect()
print(f"GATE 1 — point-source mask removes {ptsrc_frac:.3f}% of sky (expect < 1%)", flush=True)

# -------------------------------------------------------------------------
# 2. Build the apodised cluster mask at NSIDE=2048 (same as nb02).
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
print(f"GATE 2 — cluster mask removes {cluster_frac:.2f}% of sky (expect ~3-4%)", flush=True)


# -------------------------------------------------------------------------
# 3. Helper: apply the shared footprint to one channel and write a masked
#    FITS, mirroring nb02's order exactly (ptsrc-inpaint at 8192 BEFORE
#    degrading — degrading first would spread the holes over 4x the area —
#    then degrade to 2048, then cluster-inpaint).
# -------------------------------------------------------------------------
def mask_channel(fits_path, out_name, *, field=0, to_uK=1.0, apply_cluster=True, label=""):
    print(f"\n[{label}] loading {fits_path.name} (field={field}) ...", flush=True)
    m = hp.read_map(fits_path, field=field, dtype=np.float32)
    if len(m) != npix_in:
        # Some products ship at a different NSIDE; bring to NSIDE_IN so the
        # 8192 point-source footprint applies pixel-for-pixel.
        print(f"[{label}] NSIDE={hp.get_nside(m)} != {NSIDE_IN}; ud_grading to {NSIDE_IN}")
        m = hp.ud_grade(m, nside_out=NSIDE_IN).astype(np.float32)
    print(f"[{label}] raw  min {m.min():.4e}  max {m.max():.4e}  std {m.std():.4e}")

    m = inpaint_masked_regions(m, ptsrc_mask, rng=rng)
    m_2048 = hp.ud_grade(m, nside_out=NSIDE_OUT).astype(np.float32)
    del m
    gc.collect()

    if apply_cluster:
        m_2048 = inpaint_masked_regions(m_2048 * cluster_mask, cluster_mask, rng=rng)
    else:
        print(f"[{label}] cluster mask SKIPPED (MASK_CLUSTERS_IN_KAPPA=0)")

    m_uK = (m_2048 * to_uK).astype(np.float32)
    del m_2048
    gc.collect()
    print(f"[{label}] masked min {m_uK.min():.4e}  max {m_uK.max():.4e} (out units)")

    out_path = OUT_DIR / out_name
    hp.write_map(out_path, m_uK, overwrite=True)
    print(f"[{label}] wrote {out_path}")
    del m_uK
    gc.collect()


# -------------------------------------------------------------------------
# 4. kSZ — achromatic temperature shift. Convert to uK if dimensionless.
# -------------------------------------------------------------------------
_, ksz_hdr = hp.read_map(KSZ_FITS, h=True)
ksz_tunit = dict(ksz_hdr).get("TUNIT1", "unknown")
print(f"\nkSZ header TUNIT1 = {ksz_tunit!r}; KSZ_INPUT_UNIT={KSZ_INPUT_UNIT} — CONFIRM these agree")
ksz_to_uK = T_CMB_UK if KSZ_INPUT_UNIT == "dimensionless" else 1.0
mask_channel(
    KSZ_FITS,
    "ksz_150_masked.fits",
    field=0,
    to_uK=ksz_to_uK,
    apply_cluster=True,
    label="kSZ",
)

# -------------------------------------------------------------------------
# 5. kappa_CMB — dimensionless convergence, no unit conversion. Cluster
#    masking is optional here (see MASK_CLUSTERS_IN_KAPPA note above).
# -------------------------------------------------------------------------
mask_channel(
    KAPPA_FITS,
    "kappa_masked.fits",
    field=KAPPA_FIELD,
    to_uK=1.0,
    apply_cluster=MASK_CLUSTERS_IN_KAPPA,
    label="kappa",
)

print("\nNB02b done — kSZ and kappa masked maps written to", OUT_DIR)
print("Next: python nb03b_extract_4ch.py")
