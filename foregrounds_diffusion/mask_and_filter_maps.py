#!/usr/bin/env python
import argparse
import os

import healpy as hp
import numpy as np


def load_combined_mask(mask_paths, nside):
    if not mask_paths:
        return None
    combined = None
    for path in mask_paths:
        mask_map = hp.read_map(path, memmap=True, verbose=False)
        if hp.get_nside(mask_map) != nside:
            raise ValueError(f"Mask NSIDE mismatch for {path}")
        mask_map = np.asarray(mask_map, dtype=np.float64)
        combined = mask_map if combined is None else combined * mask_map
    return combined


def inpaint_masked(map_data, mask, threshold):
    if mask is None:
        return map_data
    masked = mask < threshold
    if not np.any(masked):
        return map_data
    unmasked_vals = map_data[~masked]
    mean_val = np.mean(unmasked_vals)
    std_val = np.std(unmasked_vals)
    filled = map_data.copy()
    filled[masked] = np.random.normal(mean_val, std_val, size=np.sum(masked))
    return filled


def bandpass_filter(map_data, ell_min, ell_max, lmax):
    alm = hp.map2alm(map_data, lmax=lmax)
    ells = np.arange(lmax + 1)
    band = (ells >= ell_min) & (ells <= ell_max)
    flt = np.where(band, 1.0, 0.0)
    alm_filtered = hp.almxfl(alm, flt)
    return hp.alm2map(alm_filtered, nside=hp.get_nside(map_data))


def process_map(
    input_path,
    output_path,
    mask_paths,
    ell_min,
    ell_max,
    lmax,
    inpaint,
    mask_threshold,
):
    map_data = hp.read_map(input_path, memmap=True, verbose=False)
    map_data = np.asarray(map_data, dtype=np.float64)
    nside = hp.get_nside(map_data)

    combined_mask = load_combined_mask(mask_paths, nside)
    if combined_mask is not None:
        map_data = map_data * combined_mask

    if inpaint:
        map_data = inpaint_masked(map_data, combined_mask, mask_threshold)

    filtered = bandpass_filter(map_data, ell_min, ell_max, lmax)
    hp.write_map(output_path, filtered, overwrite=True, dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mask full-sky maps, apply a bandpass filter, and save results."
    )
    parser.add_argument("inputs", nargs="+", help="Input map FITS files.")
    parser.add_argument(
        "--mask", action="append", default=[], help="Mask FITS file (can be repeated)."
    )
    parser.add_argument("--ell-min", type=int, default=0, help="Bandpass low cutoff.")
    parser.add_argument("--ell-max", type=int, default=7000, help="Bandpass high cutoff.")
    parser.add_argument("--lmax", type=int, default=13000, help="Max ell for alm.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to input directory.",
    )
    parser.add_argument(
        "--suffix",
        default="masked_bandpass",
        help="Suffix for output filenames.",
    )
    parser.add_argument(
        "--inpaint",
        action="store_true",
        help="Inpaint masked pixels with Gaussian noise before filtering.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.999,
        help="Mask threshold for inpainting (mask < threshold).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ell_min > args.ell_max:
        raise ValueError("--ell-min must be <= --ell-max")

    for input_path in args.inputs:
        base = os.path.basename(input_path)
        name, _ext = os.path.splitext(base)
        out_dir = args.output_dir or os.path.dirname(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{name}_{args.suffix}.fits")
        process_map(
            input_path,
            output_path,
            args.mask,
            args.ell_min,
            args.ell_max,
            args.lmax,
            args.inpaint,
            args.mask_threshold,
        )


if __name__ == "__main__":
    main()
