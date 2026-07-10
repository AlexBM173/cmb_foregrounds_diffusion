#!/usr/bin/env python
"""Map galleries: Agora vs the DDPM, Agora vs the Gaussian baseline, and contact
sheets of every Agora patch.

Writes into the 4-field run's ``plots/``:

``2f_map_gallery.png``
    Agora and the 2-field (v4) DDPM, CIB and tSZ.  Both come from the v4 patch
    set — the data that model was actually trained on.
``4f_map_gallery.png``
    Agora and the 4-field (v5) DDPM, all of CIB / tSZ / kSZ / kappa.
``agora_vs_gaussian.png``
    One patch per field, Agora beside the Gaussian baseline, sharing a colour
    scale per field — the visual statement of what a Gaussian random field is
    missing.
``agora_all_patches_<field>.png`` (supplementary, one per field)
    Contact sheet of ALL the original Agora patches (the full extraction, not
    just the held-out split), at a single colour scale per field.

Within a *set* (Agora, DDPM) every row shows the SAME patch indices, so a column
is one patch seen through each field.  Indices differ between sets: the sources
have different lengths and no patch-by-patch correspondence.

Every row of a given field shares one colour scale, fixed by the Agora patches
of that field, so "the Gaussian tSZ has no clusters" is legible directly from
the images rather than from the colourbars.

    python scripts/plot_map_gallery.py               # galleries + contact sheets
    python scripts/plot_map_gallery.py --skip-all-patches
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import load_config
from pipeline.evaluate import (
    CHANNEL_META,
    _apply_style,
    _channel_display,
    _finish,
    load_sources,
)
from pipeline.rundir import RunDir

# Signed fields get a diverging map centred on zero; the rest are one-sided.
_DIVERGING = {"ksz", "kappa"}
_UNITS = {"cib": r"$\mu$K", "tsz": r"$\mu$K", "ksz": r"$\mu$K", "kappa": "dimensionless"}


def _load(config_path):
    cfg = load_config(config_path)
    run = RunDir(root=cfg.run_dir())
    sources, norm_params, labels, _ = load_sources(cfg, run)
    return cfg, run, sources, norm_params, labels


def _scale(field, ref):
    """(vmin, vmax, cmap) for a field, fixed by the Agora reference patches.

    Limits are clipped at a few sigma about the mean rather than at extreme
    percentiles: the tSZ decrements and CIB point sources are so far out in the
    tail that percentile limits crush all the diffuse structure into one colour.
    """
    mu, sd = float(ref.mean()), float(ref.std())
    if field in _DIVERGING:
        v = 3.0 * sd
        return -v, v, "RdBu_r"
    if field == "tsz":
        # decrement: clusters live in the negative tail, so reverse the ramp
        # and let them read as the bright end against a dark background
        return mu - 3.5 * sd, mu + 1.5 * sd, "cividis_r"
    return mu - 2.5 * sd, mu + 3.0 * sd, "cividis"


def _gallery(rows, scales, out, suptitle, deg, n):
    """Grid of len(rows) x n patches; rows are (set_name, field, maps, indices)."""
    _apply_style()
    import matplotlib.pyplot as plt

    ext = [-deg / 2, deg / 2, -deg / 2, deg / 2]
    fig, axes = plt.subplots(
        len(rows), n, figsize=(3.2 * n + 1.6, 3.2 * len(rows)), layout="constrained"
    )
    axes = np.atleast_2d(axes)
    last = len(rows) - 1
    for r, (set_name, field, maps, idx) in enumerate(rows):
        vmin, vmax, cmap = scales[field]
        for c in range(n):
            ax = axes[r, c]
            im = ax.imshow(maps[c], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", extent=ext)
            ax.set_title(f"patch {int(idx[c])}", fontsize=9)
            ax.tick_params(labelsize=7)
            # axis labels on the figure edges only: panels sharing one geometry
            # do not need N copies of the same label
            if r == last:
                ax.set_xlabel(r"$\theta_x$ [deg]", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(
                    f"{set_name}\n{_channel_display(field)}\n" + r"$\theta_y$ [deg]",
                    fontsize=9,
                    linespacing=1.5,
                )
            else:
                ax.tick_params(labelleft=False)
        cb = fig.colorbar(im, ax=list(axes[r]), fraction=0.021, pad=0.01)
        cb.set_label(f"{_channel_display(field)} [{_UNITS[field]}]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    _finish(fig, out, suptitle, tight=False)
    print(f"[gallery] wrote {out}")
    for set_name, field, _, idx in rows:
        print(f"    {set_name:>16} / {field:<6} patches {list(map(int, idx))}")


def _agora_vs_gaussian(sources, labels, fields, scales, out, deg, idx_a, idx_g):
    """One patch per field: Agora beside the Gaussian baseline, shared scale."""
    _apply_style()
    import matplotlib.pyplot as plt

    ext = [-deg / 2, deg / 2, -deg / 2, deg / 2]
    fig, axes = plt.subplots(len(fields), 2, figsize=(8.6, 3.3 * len(fields)), layout="constrained")
    axes = np.atleast_2d(axes)
    cols = [("Agora", sources["agora"], idx_a), ("Gaussian", sources["gaussian"], idx_g)]
    for r, field in enumerate(fields):
        vmin, vmax, cmap = scales[field]
        for c, (src_name, maps, idx) in enumerate(cols):
            ax = axes[r, c]
            im = ax.imshow(
                maps[labels.index(field)][idx],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                extent=ext,
            )
            ax.set_title(f"{src_name} — patch {int(idx)}", fontsize=10)
            ax.tick_params(labelsize=7)
            if r == len(fields) - 1:
                ax.set_xlabel(r"$\theta_x$ [deg]", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(
                    f"{_channel_display(field)}\n" + r"$\theta_y$ [deg]",
                    fontsize=10,
                    linespacing=1.5,
                )
            else:
                ax.tick_params(labelleft=False)
        cb = fig.colorbar(im, ax=list(axes[r]), fraction=0.045, pad=0.02)
        cb.set_label(f"{_channel_display(field)} [{_UNITS[field]}]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    _finish(fig, out, "Agora vs the Gaussian baseline", tight=False)
    print(f"[gallery] wrote {out}")


def _load_all_agora(cfg, labels):
    """Every original Agora patch, denormalised to physical units, (C, N, H, W).

    ``load_sources`` returns only the held-out split; the supplementary contact
    sheets want the full extraction, so read the patch files directly.
    """
    patches_dir = Path(cfg.data.patches_dir)
    ptsrc, res = int(cfg.preprocessing.point_source_mjy), cfg.data.res
    norm = np.load(patches_dir / f"norm_params_{ptsrc}mJy.npy").ravel()
    out = []
    for i, key in enumerate(labels):
        template = next(m[3] for m in CHANNEL_META if m[0] == key)
        arr = np.squeeze(np.load(patches_dir / template.format(res=res, ptsrc=ptsrc)))
        out.append(arr.astype(np.float64) * norm[2 * i + 1] + norm[2 * i])
    return np.stack(out, axis=0)


def _contact_sheet(maps, field, vmin, vmax, cmap, out, ncols=27):
    """Thumbnail of every patch of one field, on a single colour scale."""
    _apply_style()
    import matplotlib.pyplot as plt

    n = maps.shape[0]
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(0.62 * ncols + 1.2, 0.62 * nrows + 1.0), layout="constrained"
    )
    axes = np.atleast_2d(axes)
    for k, ax in enumerate(axes.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        if k >= n:
            ax.set_axis_off()
            continue
        im = ax.imshow(maps[k], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        for s in ax.spines.values():
            s.set_linewidth(0.2)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01)
    cb.set_label(f"{_channel_display(field)} [{_UNITS[field]}]", fontsize=9)
    _finish(fig, out, f"All {n} Agora patches — {_channel_display(field)}", tight=False)
    print(f"[gallery] wrote {out}  ({n} patches)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-4ch", default="config/v5_4ch.yaml")
    ap.add_argument("--config-2ch", default="config/v4_eval.yaml")
    ap.add_argument("--n-patches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-all-patches", action="store_true")
    args = ap.parse_args()

    cfg4, run4, src4, _, lab4 = _load(args.config_4ch)
    _, _, src2, _, lab2 = _load(args.config_2ch)
    if "ddpm" not in src4 or "ddpm" not in src2:
        raise SystemExit("both runs need DDPM samples in <run>/samples/")

    rng = np.random.default_rng(args.seed)
    n, deg = args.n_patches, cfg4.data.patch_deg

    def pick(maps, k=None):
        return rng.choice(maps.shape[1], size=k or n, replace=False)

    def scales_from(agora, labels, fields, idx):
        return {f: _scale(f, agora[labels.index(f)][idx]) for f in fields}

    def _patch_title(model):
        return f"Random {deg:g}$^\\circ\\times${deg:g}$^\\circ$ patches for the {model} model"

    # --- 2-field gallery: v4 Agora + v4 DDPM (the data that model trained on)
    a2, d2 = pick(src2["agora"]), pick(src2["ddpm"])
    rows2 = [("Agora", f, src2["agora"][lab2.index(f)][a2], a2) for f in lab2]
    rows2 += [("DDPM (2-field)", f, src2["ddpm"][lab2.index(f)][d2], d2) for f in lab2]
    _gallery(
        rows2,
        scales_from(src2["agora"], lab2, lab2, a2),
        run4.plots / "2f_map_gallery.png",
        _patch_title("two-field"),
        deg,
        n,
    )

    # --- 4-field gallery: v5 Agora + v5 DDPM, all four fields
    a4, d4 = pick(src4["agora"]), pick(src4["ddpm"])
    rows4 = [("Agora", f, src4["agora"][lab4.index(f)][a4], a4) for f in lab4]
    rows4 += [("DDPM (4-field)", f, src4["ddpm"][lab4.index(f)][d4], d4) for f in lab4]
    sc4 = scales_from(src4["agora"], lab4, lab4, a4)
    _gallery(
        rows4,
        sc4,
        run4.plots / "4f_map_gallery.png",
        _patch_title("four-field"),
        deg,
        n,
    )

    # --- Agora vs Gaussian, one patch per field, shared per-field scale
    _agora_vs_gaussian(
        src4,
        lab4,
        lab4,
        sc4,
        run4.plots / "agora_vs_gaussian.png",
        deg,
        int(pick(src4["agora"], 1)[0]),
        int(pick(src4["gaussian"], 1)[0]),
    )

    # --- supplementary: contact sheet of every original Agora patch, per field
    if not args.skip_all_patches:
        all_agora = _load_all_agora(cfg4, lab4)
        for i, field in enumerate(lab4):
            vmin, vmax, cmap = _scale(field, all_agora[i])
            _contact_sheet(
                all_agora[i],
                field,
                vmin,
                vmax,
                cmap,
                run4.plots / f"agora_all_patches_{field}.png",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
