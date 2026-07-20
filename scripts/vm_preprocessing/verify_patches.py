#!/usr/bin/env python
"""Pre-training artefact gate for extracted patches.

Run this after nb03_run.py / nb03b_extract_4ch.py and BEFORE launching any
training. It denormalises the saved z-scored arrays back to physical units and
asserts the invariants the spin-2 extraction bug used to violate:

  * CIB is an intensity  -> min >= 0   (clip "zero_neg")
  * tSZ at 150 GHz is a pure decrement -> max <= 0   (clip "zero_pos")
  * every channel is finite (no NaN / inf)

It also prints per-channel moments so a human can eyeball the distribution
(the corrupted v4 set had e.g. tSZ kurtosis 15.4 vs the true 49.4).

Exit status is non-zero if any hard check fails, so it can gate a SLURM chain:

    python scripts/vm_preprocessing/verify_patches.py --data-dir data/low_pass/2mJy \
        && sbatch scripts/slurm/train.sh
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew

# (filename glob stem, human label, expected sign): sign is +1 for one-sided
# non-negative, -1 for one-sided non-positive, 0 for genuinely two-sided.
CHANNELS = [
    ("CIB_map_*", "CIB", +1),
    ("tSZ3_map_*", "tSZ", -1),
    ("kSZ_map_*", "kSZ", 0),
    ("kappa_map_*", "kappa", 0),
]
# Small tolerance for float round-trip through z-score (physical units ~ O(1-100) uK).
SIGN_TOL = 1e-3


def _find(data_dir: Path, stem: str):
    hits = sorted(p for p in data_dir.glob(f"{stem}.npy") if "norm_params" not in p.name)
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--ptsrc", default=2, type=int, help="point-source label for norm_params")
    args = ap.parse_args()

    d = args.data_dir
    npf = d / f"norm_params_{args.ptsrc}mJy.npy"
    if not npf.exists():
        print(f"FAIL: norm_params not found: {npf}", file=sys.stderr)
        return 2
    norm = np.load(npf)
    print(f"norm_params ({len(norm)} entries): {np.array2string(norm, precision=4)}")

    failures = []
    ci = 0  # channel index into the [mean, std, mean, std, ...] norm vector
    checked = 0
    for stem, label, sign in CHANNELS:
        f = _find(d, stem)
        if f is None:
            continue  # channel absent (e.g. 2-field run has no kSZ/kappa)
        if ci + 2 > len(norm):
            failures.append(f"{label}: norm_params too short for this channel")
            break
        mean, std = float(norm[ci]), float(norm[ci + 1])
        ci += 2
        checked += 1

        z = np.load(f, mmap_mode="r")
        phys = np.asarray(z, dtype=np.float64) * std + mean  # denormalise

        n_bad = int((~np.isfinite(phys)).sum())
        pmin, pmax = float(phys.min()), float(phys.max())
        print(
            f"\n[{label}] {f.name}  shape {z.shape}\n"
            f"    physical: min {pmin:+.3f}  max {pmax:+.3f}  "
            f"mean {phys.mean():+.3f}  std {phys.std():.3f}\n"
            f"    skew {skew(phys, axis=None):+.3f}  "
            f"kurtosis {kurtosis(phys, axis=None):+.3f}  non-finite {n_bad}"
        )
        if n_bad:
            failures.append(f"{label}: {n_bad} non-finite pixels")
        if sign > 0 and pmin < -SIGN_TOL:
            n = int((phys < -SIGN_TOL).sum())
            failures.append(
                f"{label}: expected >= 0 but min {pmin:.3f} "
                f"({n:,} px = {100 * n / phys.size:.4f}% negative) -- clip not applied or spin-2 leak"
            )
        if sign < 0 and pmax > SIGN_TOL:
            n = int((phys > SIGN_TOL).sum())
            failures.append(
                f"{label}: expected <= 0 but max {pmax:.3f} "
                f"({n:,} px = {100 * n / phys.size:.4f}% positive) -- clip not applied or spin-2 leak"
            )

    if checked == 0:
        print("FAIL: no channel arrays found in", d, file=sys.stderr)
        return 2

    print("\n" + "=" * 60)
    if failures:
        print("ARTEFACT CHECK FAILED:", file=sys.stderr)
        for msg in failures:
            print("  -", msg, file=sys.stderr)
        return 1
    print(f"ARTEFACT CHECK PASSED ({checked} channels): one-sidedness + finiteness OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
