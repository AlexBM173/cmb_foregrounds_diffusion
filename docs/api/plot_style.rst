plot_style — Publication-quality Matplotlib style
=================================================

``plot_style`` sets a consistent Matplotlib style for all paper figures.
Call :func:`apply` once at the top of each figure notebook before creating
any axes.

.. rubric:: Wong colour palette

All figures use the `Wong (2011) <https://www.nature.com/articles/nmeth.1618>`_
eight-colour palette, which is optimised for colour-blind accessibility:

.. code-block:: python

    from foregrounds_diffusion.plot_style import WONG

    # WONG[0] = black, WONG[1] = orange, WONG[2] = sky blue, ...
    ax.plot(x, y, color=WONG[1])

.. rubric:: Setting figure size

Pass the journal column width in points.  Common values:

.. list-table::
   :header-rows: 1

   * - Journal
     - Single column
     - Double column
   * - MNRAS
     - 246 pt
     - 510 pt
   * - ApJ
     - 242 pt
     - 513 pt

.. code-block:: python

    from foregrounds_diffusion.plot_style import apply

    apply(fig_width_pt=246.0)            # MNRAS single column
    apply(fig_width_pt=510.0, n_cols=2)  # MNRAS double column, 2-panel figure

.. rubric:: Save convention

Always save as both PDF (primary, lossless) and PNG (backup, 300 dpi):

.. code-block:: python

    fig.savefig(FIGURES_DIR / "fig01_cib_power_spectrum.pdf")
    fig.savefig(FIGURES_DIR / "fig01_cib_power_spectrum.png", dpi=300)

.. rubric:: API

.. automodule:: foregrounds_diffusion.plot_style
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
