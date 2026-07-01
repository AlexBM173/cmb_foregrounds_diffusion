import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from foregrounds_diffusion.plot_style import WONG, apply


def test_wong_palette_length():
    assert len(WONG) == 8


def test_wong_palette_are_hex_strings():
    for c in WONG:
        assert c.startswith("#")
        assert len(c) == 7


def test_apply_returns_wong_palette():
    result = apply()
    assert result == WONG


def test_apply_sets_font_family():
    apply()
    assert mpl.rcParams["font.family"] == ["serif"]


def test_apply_sets_figure_size():
    apply(fig_width_pt=246.0, n_cols=1)
    w, h = mpl.rcParams["figure.figsize"]
    assert w == pytest.approx(246.0 / 72.27, abs=0.01)
    assert h == pytest.approx(w / ((1 + 5**0.5) / 2), abs=0.01)


def test_apply_n_cols_scales_width():
    apply(fig_width_pt=246.0, n_cols=1)
    w1 = mpl.rcParams["figure.figsize"][0]
    apply(fig_width_pt=246.0, n_cols=2)
    w2 = mpl.rcParams["figure.figsize"][0]
    assert w2 == pytest.approx(2 * w1, abs=0.01)


def test_apply_sets_prop_cycle_to_wong():
    apply()
    cycle_colors = [c["color"] for c in mpl.rcParams["axes.prop_cycle"]]
    assert cycle_colors == WONG


@pytest.fixture(autouse=True)
def restore_rcparams():
    """Reset matplotlib rcParams after each test."""
    orig = mpl.rcParams.copy()
    yield
    mpl.rcParams.update(orig)
    plt.close("all")
