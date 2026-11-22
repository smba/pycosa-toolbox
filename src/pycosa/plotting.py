#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting utilities for comparing distributions of measurements, e.g., the
performance of configurations with a group of options enabled vs. disabled.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def _kde_curve(dist: np.ndarray, bandwidth: float, n_points: int = 256):
    """Returns points of the kernel density estimate of ``dist`` as
    ``(x, y)`` arrays."""
    dist = np.asarray(dist, dtype=float)
    lo, hi = dist.min(), dist.max()
    if lo == hi:
        # degenerate distribution: widen the range a little
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.1 * (hi - lo)
    x = np.linspace(lo - pad, hi + pad, n_points)
    return x, gaussian_kde(dist, bw_method=bandwidth)(x)


def mirrored_histogram(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    label_a: str,
    label_b: str,
    bandwith: float = 0.05,
    figsize: Tuple[float, float] = (6, 2.5),
    export_name: Optional[str] = None,
    xlabel: Optional[str] = None,
    medians: bool = True,
    legend: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots the kernel density estimates of two distributions mirrored at the
    x-axis (``dist_a`` above, ``dist_b`` below).

    Returns the axes. If ``export_name`` is given, the figure is saved there;
    otherwise, if no ``ax`` was passed in, the figure is shown.
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_a, y_a = _kde_curve(dist_a, bandwith)
    x_b, y_b = _kde_curve(dist_b, bandwith)

    ax.plot(x_a, y_a, color="C0")
    ax.fill_between(x_a, y_a, color="C0", alpha=0.8, label=label_a)
    ax.plot(x_b, -y_b, color="C1")
    ax.fill_between(x_b, -y_b, color="C1", alpha=0.8, label=label_b)

    if medians:
        # only draw the medians in the respective half of the plot
        ymin, ymax = ax.get_ylim()
        y_zero = abs(ymin) / (ymax - ymin)
        ax.axvline(
            np.median(dist_a),
            color="black",
            linewidth=0.8,
            label="median",
            ymin=y_zero,
            ymax=1,
        )
        ax.axvline(np.median(dist_b), color="black", linewidth=0.8, ymin=0, ymax=y_zero)

    # make y ticks positive on both halves
    ticks = ax.get_yticks()
    ax.set_yticks(ticks)
    ax.set_yticklabels([round(float(abs(tick)), 2) for tick in ticks])

    ax.set_ylabel("Density")
    ax.axhline(0, color="black", linewidth=0.5)

    if legend:
        ax.legend()

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if export_name is not None:
        fig.savefig(export_name, bbox_inches="tight")
    elif created:
        plt.show()

    return ax


if __name__ == "__main__":
    pass
