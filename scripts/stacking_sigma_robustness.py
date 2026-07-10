#!/usr/bin/env python
"""Robustness check: cluster stacking under per-map σ instead of a fixed Agora σ.

``run_evaluate`` pins the stacking SNR threshold to the Agora maps' mean and
std (``snr_ref_mean``/``snr_ref_std``), so every source is thresholded at the
same *physical* depth.  The obvious alternative — thresholding each source at
its own per-map σ — is not comparable across sources: the DDPM under-disperses
(σ_tSZ ≈ 1.91 µK vs Agora's 2.16 µK), so a per-map threshold sits at a lower
absolute temperature for the DDPM and inflates its cluster counts.

This script recomputes the three stacking statistics with that reference
removed, caching them under ``stats/stacking_permap/``, and prints both
conventions side by side.  The fixed-σ numbers remain the headline result; this
exists so the write-up can state what the per-map convention would have shown.

    python scripts/stacking_sigma_robustness.py --config config/v5_4ch.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import load_config
from pipeline.evaluate import (
    _TSZ_STACKING_STATS,
    STATISTIC_REGISTRY,
    _mapparams,
    load_sources,
)
from pipeline.rundir import RunDir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run = RunDir(root=cfg.run_dir())
    sources, norm_params, channel_labels, _ = load_sources(cfg, run)
    mapparams = _mapparams(cfg)

    permap_dir = run.stats / "stacking_permap"
    permap_dir.mkdir(parents=True, exist_ok=True)

    tsz = sources["agora"][channel_labels.index("tsz")]
    ref_std = float(tsz.std())
    print(f"Agora tSZ reference σ = {ref_std:.4f} µK (fixed-σ convention)")
    for src, maps in sources.items():
        m = maps[channel_labels.index("tsz")]
        print(
            f"  {src:>8}: global σ {float(m.std()):.4f} µK, mean per-map σ "
            f"{float(np.mean([p.std() for p in m])):.4f} µK"
        )

    for name in _TSZ_STACKING_STATS:
        if name not in cfg.evaluation.statistics:
            continue
        # No snr_ref_* keys → compute() falls back to each source's own mean and
        # per-map σ inside select_snr_pixels.
        params = dict(cfg.evaluation.params.get(name, {}))
        params.pop("snr_ref_mean", None)
        params.pop("snr_ref_std", None)

        stat = STATISTIC_REGISTRY[name](
            params,
            cfg.evaluation.n_jobs,
            mapparams,
            noise=None,
            norm_params=norm_params,
            channel_labels=channel_labels,
        )
        permap = {src: stat.compute_or_load(permap_dir, src, m) for src, m in sources.items()}
        fixed = {}
        for src in sources:
            f = run.stats / f"{name}__{src}.npz"
            if f.exists():
                fixed[src] = np.load(f, allow_pickle=True)

        print(f"\n=== {name} — peaks/map (DDPM/Agora ratio) ===")
        print(
            f"{'bin':>8} {'fixed-σ agora':>15} {'fixed-σ ddpm':>14} {'ratio':>7} "
            f"{'per-map agora':>15} {'per-map ddpm':>14} {'ratio':>7}"
        )
        for smin, smax in params["snr_bins"]:
            label = stat._bin_label(smin, smax)
            row = [label]
            for res in (fixed, permap):
                rates = {}
                for src in ("agora", "ddpm"):
                    if src not in res or f"n_{label}" not in res[src]:
                        rates[src] = float("nan")
                        continue
                    r = res[src]
                    n_maps = int(r["n_maps"]) if "n_maps" in r else len(sources[src][0])
                    rates[src] = int(r[f"n_{label}"]) / n_maps
                ratio = rates["ddpm"] / rates["agora"] if rates["agora"] else float("nan")
                row += [f"{rates['agora']:.2f}", f"{rates['ddpm']:.2f}", f"{ratio:.2f}"]
            print(
                f"{row[0]:>8} {row[1]:>15} {row[2]:>14} {row[3]:>7} "
                f"{row[4]:>15} {row[5]:>14} {row[6]:>7}"
            )

    print(f"\nper-map-σ caches written to {permap_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
