#!/usr/bin/env python
import argparse
import glob
import os

import healpy as hp
import numpy as np


def bandpass_filter(map_data, ell_min, ell_max, lmax):
    alm = hp.map2alm(map_data, lmax=lmax)
    ells = np.arange(lmax + 1)
    band = (ells >= ell_min) & (ells <= ell_max)
    flt = np.where(band, 1.0, 0.0)
    alm_filtered = hp.almxfl(alm, flt)
    return hp.alm2map(alm_filtered, nside=hp.get_nside(map_data))


def inpaint_masked(map_data, mask, threshold):
    masked = mask < threshold
    if not np.any(masked):
        return map_data
    unmasked_vals = map_data[~masked]
    mean_val = np.mean(unmasked_vals)
    std_val = np.std(unmasked_vals)
    filled = map_data.copy()
    filled[masked] = np.random.normal(mean_val, std_val, size=np.sum(masked))
    return filled


def build_point_source_mask(map_data, sigma_threshold):
    sigma = np.std(map_data)
    mask = np.ones(map_data.shape, dtype=np.float64)
    inds = np.where(np.abs(map_data) > sigma_threshold * sigma)[0]
    mask[inds] = 0.0
    return mask


def build_cluster_mask(
    map_data,
    sigma_threshold,
    base_radius_arcmin,
    max_radius_factor,
):
    sigma = np.std(map_data)
    threshold = sigma_threshold * sigma
    mask = np.ones(map_data.shape, dtype=np.float64)

    peak_inds = np.where(np.abs(map_data) > threshold)[0]
    if len(peak_inds) == 0:
        return mask

    nside = hp.get_nside(map_data)
    for idx in peak_inds:
        scale = np.abs(map_data[idx]) / threshold
        scale = min(max_radius_factor, max(1.0, scale))
        radius_arcmin = base_radius_arcmin * scale
        radius_rad = np.radians(radius_arcmin / 60.0)
        vec = hp.pix2vec(nside, idx)
        disc = hp.query_disc(nside, vec, radius_rad)
        mask[disc] = 0.0
    return mask


def process_map(
    input_path,
    output_dir,
    ell_min,
    ell_max,
    lmax,
    point_sigma,
    cluster_sigma,
    cluster_radius_arcmin,
    cluster_max_radius_factor,
    inpaint,
    mask_threshold,
):
    map_data = hp.read_map(input_path, memmap=True, verbose=False)
    map_data = np.asarray(map_data, dtype=np.float64)

    point_mask = build_point_source_mask(map_data, point_sigma)
    cluster_mask = build_cluster_mask(
        map_data,
        cluster_sigma,
        cluster_radius_arcmin,
        cluster_max_radius_factor,
    )
    combined_mask = point_mask * cluster_mask

    masked_map = map_data * combined_mask
    if inpaint:
        masked_map = inpaint_masked(masked_map, combined_mask, mask_threshold)

    filtered_map = bandpass_filter(masked_map, ell_min, ell_max, lmax)

    base = os.path.basename(input_path)
    name, _ext = os.path.splitext(base)
    os.makedirs(output_dir, exist_ok=True)

    mask_path = os.path.join(output_dir, f"{name}_mask.fits")
    masked_path = os.path.join(output_dir, f"{name}_masked.fits")
    filtered_path = os.path.join(output_dir, f"{name}_masked_bandpass.fits")

    hp.write_map(mask_path, combined_mask, overwrite=True, dtype=np.float64)
    hp.write_map(masked_path, masked_map, overwrite=True, dtype=np.float64)
    hp.write_map(filtered_path, filtered_map, overwrite=True, dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create masks from maps, apply bandpass filter, and save maps."
    )
    parser.add_argument(
        "--input-dir",
        default="/home/alex/projects/cmb_foregrounds_diffusion/data",
        help="Directory containing input .fits maps.",
    )
    parser.add_argument(
        "--pattern",
        default="*.fits",
        help="Glob pattern for input maps.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for masks and filtered maps.",
    )
    parser.add_argument("--ell-min", type=int, default=0, help="Bandpass low cutoff.")
    parser.add_argument("--ell-max", type=int, default=7000, help="Bandpass high cutoff.")
    parser.add_argument("--lmax", type=int, default=13000, help="Max ell for alm.")
    parser.add_argument(
        "--point-sigma",
        type=float,
        default=10.0,
        help="Sigma threshold for point-source masking.",
    )
    parser.add_argument(
        "--cluster-sigma",
        type=float,
        default=10.0,
        help="Sigma threshold for cluster masking.",
    )
    parser.add_argument(
        "--cluster-radius-arcmin",
        type=float,
        default=10.0,
        help="Base cluster mask radius in arcmin.",
    )
    parser.add_argument(
        "--cluster-max-radius-factor",
        type=float,
        default=5.0,
        help="Maximum scaling factor for cluster radius.",
    )
    parser.add_argument(
        "--inpaint",
        action="store_true",
        help="Inpaint masked pixels with Gaussian noise.",
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

    output_dir = args.output_dir or args.input_dir
    pattern = os.path.join(args.input_dir, args.pattern)
    inputs = sorted(glob.glob(pattern))
    if not inputs:
        raise FileNotFoundError(f"No input maps found for {pattern}")

    for input_path in inputs:
        process_map(
            input_path,
            output_dir,
            args.ell_min,
            args.ell_max,
            args.lmax,
            args.point_sigma,
            args.cluster_sigma,
            args.cluster_radius_arcmin,
            args.cluster_max_radius_factor,
            args.inpaint,
            args.mask_threshold,
        )


if __name__ == "__main__":
    main()
