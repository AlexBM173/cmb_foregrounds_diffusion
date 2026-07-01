import numpy as np

from foregrounds_diffusion.flatmaps import cl2map, map2cl


def test_power_spectrum_roundtrip():
    """cl2map → map2cl recovers input spectrum within 20% (median over bins)."""
    np.random.seed(99)
    mapparams = [256, 256, 1.40625, 1.40625]
    el_in = np.arange(1, 10001)
    cl_in = 1e-6 * el_in.astype(float) ** (-2)

    cls = []
    for _ in range(20):
        m = cl2map(mapparams, cl_in, el_in)
        _, cl_out = map2cl(mapparams, m, minbin=300, maxbin=5000)
        cls.append(cl_out)
    el_out, _ = map2cl(mapparams, m, minbin=300, maxbin=5000)

    mean_cl = np.mean(cls, axis=0)
    cl_ref = np.interp(el_out, el_in, cl_in)
    valid = cl_ref > 0
    frac_err = np.abs(mean_cl[valid] - cl_ref[valid]) / cl_ref[valid]
    assert np.median(frac_err) < 0.20
