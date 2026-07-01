"""
Baseline profiling sweep for Phase 2.

Sweeps N (maps), H (map side), T (thresholds), B (ℓ-bands) for the key
evaluation functions and saves results to results/profiling/baseline.npz.

Usage:
    python scripts/run_profiling_baseline.py
"""

import cProfile
import pstats
import timeit
import tracemalloc
import io
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foregrounds_diffusion.flatmaps import (
    get_lpf_hpf, make_gaussian_realisation, map2cl,
)
from foregrounds_diffusion.moments import compute_cross_moments, compute_summed_moments
from foregrounds_diffusion.morphology import compute_minkowski_tensors
from foregrounds_diffusion.peak_counts import compute_peak_minima_counts

# ---------------------------------------------------------------------------
# Profiling harness
# ---------------------------------------------------------------------------

def profile_fn(fn, *args, n_repeat=3, **kwargs):
    times = timeit.repeat(lambda: fn(*args, **kwargs), number=1, repeat=n_repeat)
    tracemalloc.start()
    fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        'time_median_s': float(sorted(times)[n_repeat // 2]),
        'time_min_s':    float(min(times)),
        'peak_mem_mb':   peak / 1024 ** 2,
    }


def sweep(fn, dim_values, make_args, n_repeat=3, label='N'):
    """Sweep a single input dimension and record timing + memory."""
    times, mems = [], []
    for val in dim_values:
        args, kwargs = make_args(val)
        r = profile_fn(fn, *args, n_repeat=n_repeat, **kwargs)
        times.append(r['time_median_s'])
        mems.append(r['peak_mem_mb'])
        print(f"  {label}={val:>5}  time={r['time_median_s']:.3f}s  mem={r['peak_mem_mb']:.1f}MB")
    return np.array(times), np.array(mems)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_EL = np.arange(1, 5000)
_CL = 1e-5 * _EL.astype(float) ** (-2)


def make_maps(N, H):
    np.random.seed(42)
    params = [H, H, 1.40625, 1.40625]
    return (
        np.array([make_gaussian_realisation(params, _EL, _CL) for _ in range(N)]),
        params,
    )


def make_bp_filters(params, B):
    edges = np.linspace(200, 7000, B + 1)
    return [get_lpf_hpf(params, (edges[i], edges[i + 1]), filter_type=2)
            for i in range(B)]


# ---------------------------------------------------------------------------
# §2.3  N-scaling sweeps  (H=W=64, fixed other dims)
# ---------------------------------------------------------------------------

N_VALUES = [1, 5, 10, 50, 100]
H_FIXED  = 64
T_FIXED  = 25
B_FIXED  = 4
params64 = [H_FIXED, H_FIXED, 1.40625, 1.40625]
thresholds_fixed = np.linspace(-2, 2, T_FIXED)
bp_fixed = make_bp_filters(params64, B_FIXED)

print("=" * 60)
print("N-scaling: compute_minkowski_tensors  (H=64, T=25)")
print("=" * 60)
maps_cache = {N: make_maps(N, H_FIXED)[0] for N in N_VALUES}
mink_N_times, mink_N_mems = sweep(
    compute_minkowski_tensors,
    N_VALUES,
    lambda N: ((maps_cache[N], lambda x: x, thresholds_fixed), {}),
    label='N',
)

print()
print("=" * 60)
print("N-scaling: compute_cross_moments  (H=64, B=4)")
print("=" * 60)
cross_N_times, cross_N_mems = sweep(
    compute_cross_moments,
    N_VALUES,
    lambda N: ((maps_cache[N], maps_cache[N], bp_fixed), {}),
    label='N',
)

print()
print("=" * 60)
print("N-scaling: map2cl  (H=64)")
print("=" * 60)
map2cl_N_times, map2cl_N_mems = [], []
for N in N_VALUES:
    maps = maps_cache[N]
    def _fn(maps=maps):
        for m in maps:
            map2cl(params64, m)
    r = profile_fn(_fn, n_repeat=3)
    map2cl_N_times.append(r['time_median_s'])
    map2cl_N_mems.append(r['peak_mem_mb'])
    print(f"  N={N:>5}  time={r['time_median_s']:.3f}s  mem={r['peak_mem_mb']:.1f}MB")
map2cl_N_times = np.array(map2cl_N_times)
map2cl_N_mems  = np.array(map2cl_N_mems)

_thresh_peaks  = np.linspace(-1, 5, 20)
_thresh_minima = np.linspace(-5, 1, 20)

print()
print("=" * 60)
print("N-scaling: compute_peak_minima_counts  (H=64)")
print("=" * 60)
peak_N_times, peak_N_mems = sweep(
    compute_peak_minima_counts,
    N_VALUES,
    lambda N: ((maps_cache[N], _thresh_peaks, _thresh_minima), {}),
    label='N',
)

# ---------------------------------------------------------------------------
# §2.3  H-scaling sweeps  (N=10, fixed other dims)
# ---------------------------------------------------------------------------

H_VALUES = [32, 64, 128, 256]
N_FIXED  = 10

print()
print("=" * 60)
print("H-scaling: compute_minkowski_tensors  (N=10, T=25)")
print("=" * 60)
mink_H_times, mink_H_mems = [], []
for H in H_VALUES:
    maps, params = make_maps(N_FIXED, H)
    thresh = np.linspace(-2, 2, T_FIXED)
    r = profile_fn(compute_minkowski_tensors, maps, lambda x: x, thresh, n_repeat=3)
    mink_H_times.append(r['time_median_s'])
    mink_H_mems.append(r['peak_mem_mb'])
    print(f"  H={H:>4}  time={r['time_median_s']:.3f}s  mem={r['peak_mem_mb']:.1f}MB")
mink_H_times = np.array(mink_H_times)
mink_H_mems  = np.array(mink_H_mems)

print()
print("=" * 60)
print("H-scaling: map2cl  (N=10)")
print("=" * 60)
map2cl_H_times, map2cl_H_mems = [], []
for H in H_VALUES:
    maps, params = make_maps(N_FIXED, H)
    def _fn(maps=maps, params=params):
        for m in maps:
            map2cl(params, m)
    r = profile_fn(_fn, n_repeat=3)
    map2cl_H_times.append(r['time_median_s'])
    map2cl_H_mems.append(r['peak_mem_mb'])
    print(f"  H={H:>4}  time={r['time_median_s']:.3f}s  mem={r['peak_mem_mb']:.1f}MB")
map2cl_H_times = np.array(map2cl_H_times)
map2cl_H_mems  = np.array(map2cl_H_mems)

# ---------------------------------------------------------------------------
# §2.3  T-scaling: compute_minkowski_tensors  (N=10, H=64)
# ---------------------------------------------------------------------------

T_VALUES = [5, 10, 25, 50, 100]
maps10, _ = make_maps(N_FIXED, H_FIXED)

print()
print("=" * 60)
print("T-scaling: compute_minkowski_tensors  (N=10, H=64)")
print("=" * 60)
mink_T_times, mink_T_mems = sweep(
    compute_minkowski_tensors,
    T_VALUES,
    lambda T: ((maps10, lambda x: x, np.linspace(-2, 2, T)), {}),
    label='T',
)

# ---------------------------------------------------------------------------
# §2.3  B-scaling: compute_cross_moments  (N=10, H=64)
# ---------------------------------------------------------------------------

B_VALUES = [2, 4, 8, 16]

print()
print("=" * 60)
print("B-scaling: compute_cross_moments  (N=10, H=64)")
print("=" * 60)
cross_B_times, cross_B_mems = sweep(
    compute_cross_moments,
    B_VALUES,
    lambda B: ((maps10, maps10, make_bp_filters(params64, B)), {}),
    label='B',
)

# ---------------------------------------------------------------------------
# Fit power-law slopes
# ---------------------------------------------------------------------------

from scipy.stats import linregress

def fit_slope(xs, ys):
    mask = np.array(ys) > 0
    if mask.sum() < 2:
        return float('nan')
    slope, *_ = linregress(np.log(np.array(xs)[mask]), np.log(np.array(ys)[mask]))
    return slope

slopes = {
    'mink_vs_N':    fit_slope(N_VALUES, mink_N_times),
    'cross_vs_N':   fit_slope(N_VALUES, cross_N_times),
    'map2cl_vs_N':  fit_slope(N_VALUES, map2cl_N_times),
    'peak_vs_N':    fit_slope(N_VALUES, peak_N_times),
    'mink_vs_H':    fit_slope(H_VALUES, mink_H_times),
    'map2cl_vs_H':  fit_slope(H_VALUES, map2cl_H_times),
    'mink_vs_T':    fit_slope(T_VALUES, mink_T_times),
    'cross_vs_B':   fit_slope(B_VALUES, cross_B_times),
}

print()
print("=" * 60)
print("Power-law slopes (log-log fit; 1.0 = linear scaling)")
print("=" * 60)
for k, v in slopes.items():
    print(f"  {k:<25}  slope = {v:.2f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

os.makedirs('results/profiling', exist_ok=True)
np.savez(
    'results/profiling/baseline.npz',
    N_values=N_VALUES, H_values=H_VALUES, T_values=T_VALUES, B_values=B_VALUES,
    mink_N_times=mink_N_times, mink_N_mems=mink_N_mems,
    cross_N_times=cross_N_times, cross_N_mems=cross_N_mems,
    map2cl_N_times=map2cl_N_times, map2cl_N_mems=map2cl_N_mems,
    peak_N_times=peak_N_times, peak_N_mems=peak_N_mems,
    mink_H_times=mink_H_times, mink_H_mems=mink_H_mems,
    map2cl_H_times=map2cl_H_times, map2cl_H_mems=map2cl_H_mems,
    mink_T_times=mink_T_times, mink_T_mems=mink_T_mems,
    cross_B_times=cross_B_times, cross_B_mems=cross_B_mems,
    **{f'slope_{k}': v for k, v in slopes.items()},
)
print()
print("Results saved to results/profiling/baseline.npz")
