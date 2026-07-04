"""Corrected NB01: filter halo lightcone slices to M500c >= 1e13 (file units).

The tutorial notebook assumed a single (N, 8) array per slice; the real
files store 17 named 1-D arrays (totra, totm500, ...). This script keeps
the 8 keys get_mdpl2_halo_cat() reads and saves a filtered .npz.

The 1e13 floor (matching the original HPC workflow) makes the catalogue a
superset supporting any downstream science threshold. NB01 verified
2026-07-04 that totm500 is in M_sun/h: the paper's 3e14 M_sun cluster-mask
cut corresponds to 2.03e14 in file units, applied in nb02_run.py via the
M500C_THRESHOLD env var.
"""

import glob
from pathlib import Path

import numpy as np

HALO_DIR = "/home/alexbm173/agora_data"
M500C_THRESHOLD = 1e13
OUT_PATH = (
    Path.home() / "cmb_foregrounds_diffusion/data/halo_catalogue/halo_catalogue_m500gt1e13.npz"
)
KEYS = ["totra", "totdec", "totz", "totm200", "totm500", "totvlos", "totvtht", "totvphi"]

slice_files = sorted(glob.glob(f"{HALO_DIR}/haloslc_rot_*.npz"))
print(f"Found {len(slice_files)} slice files", flush=True)

kept = {k: [] for k in KEYS}
total = 0
for i, fpath in enumerate(slice_files, 1):
    d = np.load(fpath)
    mask = d["totm500"] >= M500C_THRESHOLD
    total += mask.size
    for k in KEYS:
        kept[k].append(d[k][mask].astype(np.float64))
    print(
        f"[{i}/{len(slice_files)}] {Path(fpath).name}: {mask.size:,} halos, kept {mask.sum():,}",
        flush=True,
    )

catalogue = {k: np.concatenate(v) for k, v in kept.items()}
n = catalogue["totra"].size
print(f"\nScanned {total:,} halos; kept {n:,} with M500c >= {M500C_THRESHOLD:.0e} (file units)")
print(f"z range:     {catalogue['totz'].min():.3f} - {catalogue['totz'].max():.3f}")
print(f"M500c range: {catalogue['totm500'].min():.2e} - {catalogue['totm500'].max():.2e}")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT_PATH, **catalogue)
print(f"Saved -> {OUT_PATH}")
