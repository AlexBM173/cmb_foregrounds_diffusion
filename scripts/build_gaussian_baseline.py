#!/usr/bin/env python
"""Build the correlated-Gaussian baseline for a C-channel run.

Reads the z-scored per-channel patch ``.npy`` files (from
``scripts/vm_preprocessing/nb03b_extract_4ch.py`` or notebook 03), measures the
full C×C mean auto/cross power-spectrum matrix, draws a Gaussian realisation
matched to it, and saves ``gaussian_<C>field_<ptsrc>mJy_lp.npy`` alongside the
patches. This is the baseline ``pipeline/evaluate.py`` loads as the ``gaussian``
source — the reference that isolates genuinely non-Gaussian structure.

The baseline is generated in **z-score space** (like the patches), matching the
DDPM samples before denormalisation. Each field is spectrum-matched to the
measured matrix; per-field std is ≈1 by construction.

Usage:
    python scripts/build_gaussian_baseline.py \
        --patches-dir data/low_pass/2mJy --channels 4
"""

import argparse
from pathlib import Path

import numpy as np

from foregrounds_diffusion.flatmaps import make_correlated_gaussian_fields
from foregrounds_diffusion.moments import measure_cross_spectrum_matrix

# Canonical per-channel patch filenames (mirror nb03b / pipeline.train). kappa
# is achromatic — no frequency tag. (Consolidation into one shared table is a
# Tier-1 evaluation-refactor item; see docs/nfield_evaluation_design.md.)
CHANNEL_FILE_TEMPLATES = [
    "CIB_map_150GHz_{res}_st6_zscore_{ptsrc}mJy_lp.npy",
    "tSZ3_map_150GHz_{res}_st6_zscore_{ptsrc}mJy_lp.npy",
    "kSZ_map_150GHz_{res}_st6_zscore_{ptsrc}mJy_lp.npy",
    "kappa_map_{res}_st6_zscore_{ptsrc}mJy_lp.npy",
]


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--patches-dir", required=True, help="Directory holding the channel .npy files")
    p.add_argument("--channels", type=int, default=4, choices=[2, 4])
    p.add_argument("--ptsrc", type=int, default=2, help="Point-source threshold label (mJy)")
    p.add_argument("--res", type=int, default=256, help="Patch resolution in pixels")
    p.add_argument("--patch-deg", type=float, default=6.0, help="Patch side length in degrees")
    p.add_argument("--lmin", type=float, default=300)
    p.add_argument("--lmax", type=float, default=7000)
    p.add_argument("--binsize", type=float, default=100)
    p.add_argument(
        "--n-spec",
        type=int,
        default=None,
        help="Cap patches used for the spectrum measurement (default: all)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for the measurement")
    args = p.parse_args(argv)

    patches_dir = Path(args.patches_dir)
    templates = CHANNEL_FILE_TEMPLATES[: args.channels]
    channels = []
    for t in templates:
        f = patches_dir / t.format(res=args.res, ptsrc=args.ptsrc)
        if not f.is_file():
            raise SystemExit(f"channel patch file not found: {f}")
        arr = np.load(f)
        arr = arr[..., 0] if arr.ndim == 4 else arr  # (N, H, W, 1) -> (N, H, W)
        channels.append(arr)
    n_patches = channels[0].shape[0]
    print(f"loaded {args.channels} channels, {n_patches} patches each, from {patches_dir}")

    dx = args.patch_deg * 60.0 / args.res
    mapparams = [args.res, args.res, dx, dx]

    n_spec = min(args.n_spec, n_patches) if args.n_spec else n_patches
    el, cl_matrix = measure_cross_spectrum_matrix(
        [c[:n_spec] for c in channels],
        mapparams,
        lmin=args.lmin,
        lmax=args.lmax,
        binsize=args.binsize,
        n_jobs=args.n_jobs,
    )
    print(
        f"measured {args.channels}x{args.channels} spectrum matrix over {n_spec} patches "
        f"({len(el)} ell-bins, {el[0]:.0f}-{el[-1]:.0f})"
    )

    rng = np.random.default_rng(args.seed)
    baseline = make_correlated_gaussian_fields(
        mapparams, el, cl_matrix, n_realisations=n_patches, rng=rng
    )  # (N, C, H, W)

    out = patches_dir / f"gaussian_{args.channels}field_{args.ptsrc}mJy_lp.npy"
    np.save(out, baseline.astype(np.float32))
    print(f"saved baseline {baseline.shape} -> {out}")
    print(
        "  per-channel std (z-score space): matches the patch std only insofar as "
        f"[{args.lmin:.0f}, {args.lmax:.0f}] captures the power; <1 if the field has "
        "significant power below lmin."
    )
    for i in range(args.channels):
        print(f"  channel {i}: std {baseline[:, i].std():.3f}")


if __name__ == "__main__":
    main()
