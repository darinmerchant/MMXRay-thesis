"""tools/plot_chili_parity.py — per-row parity for the two regression targets.

    python tools/plot_chili_parity.py                      # THE DELIVERABLE
    python tools/plot_chili_parity.py --schemes probe      # one scheme, before the rest land

Appendix expansion of Figure 2's regression half. Figure 2 compresses each cell into a
bar; this shows the 360 test points that bar was computed from, so a reader can see HOW
an R² is earned or lost — shrinkage toward the mean, a bent trend, a cloud that has
stopped tracking the target at all. Its companion is `plot_chili_confusion.py`, which
does the same for the two categorical targets off the same store.

LAYOUT. Rows are targets, columns are (scheme x fit channel) with the two channels of
one scheme ADJACENT, because "what did this scheme lose to the instrument shift" is the
comparison Figure 2 exists to make and it should be a glance, not a scan across the page.

WITHIN A ROW ALL SIX PANELS SHARE ONE PAIR OF LIMITS, and that is load-bearing rather
than tidy: per-panel autoscaling would rescale each cloud to its own spread, which makes
a collapsed prediction range look identical to a good one. The identity line is then the
same line in all six.

COLOUR IS THE SCHEME AND NOTHING ELSE. The repo's convention is that the fit channel is
never a hue (`plot_chili_four_arms.py` spends a hatch on it); here it is a column, so no
hue is spent on it either and `ARM_COLORS` keeps meaning exactly what it means elsewhere.

The annotated R² is RECOMPUTED from the plotted points by `chili_predictions.load`,
which raises if it disagrees with the value the run recorded.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from analysis import paperstyle
from analysis.paperstyle import ARM_COLORS, INK, MUTED, REFLINE
from tools.chili_predictions import REGRESSION, columns, load, row_label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schemes", nargs="+", default=None,
                    help="subset of baseline/probe/fine-tuned (default: all three)")
    ap.add_argument("--stem", default="chili_parity")
    args = ap.parse_args()

    cols = columns(args.schemes)
    data = load([t[0] for t in REGRESSION], args.schemes)
    paperstyle.use()

    ncol, nrow = len(cols), len(REGRESSION)
    # Panels are square (`aspect="equal"`), so the figure HEIGHT is DERIVED from the
    # panel width the column count leaves. Hardcoding it instead makes matplotlib
    # centre the row band inside a taller axes area, which is dead space that grows
    # with every column removed — the `--schemes` subsets looked broken because of it.
    # LEFT is wider here than in the confusion figure: this one stacks THREE rotated
    # things in the left margin — the row header, the "predicted (Å)" label and the
    # tick labels — where that one only has a 4-character ylabel. At 0.105 the header
    # printed on top of the axis label.
    # WSPACE/HSPACE are sized to fit TICK LABELS ON EVERY PANEL (see the tick block
    # below), not merely to separate the frames: the gaps have to hold a label that
    # overhangs each panel by ~half its width horizontally and a full line vertically.
    # RIGHT stops short of 1.0 for the same reason — the last column's rightmost tick
    # overhangs the axes and ran off the page at 0.995.
    LEFT, RIGHT, WSPACE, HSPACE = 0.155, 0.982, 0.34, 0.62
    HEAD_IN, FOOT_IN = 0.46, 0.46
    panel = (RIGHT - LEFT) * paperstyle.WIDTH / (ncol + (ncol - 1) * WSPACE)
    height = HEAD_IN + FOOT_IN + panel * (nrow + (nrow - 1) * HSPACE)
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(paperstyle.WIDTH, height))
    fig.subplots_adjust(left=LEFT, right=RIGHT, wspace=WSPACE, hspace=HSPACE,
                        top=1 - HEAD_IN / height, bottom=FOOT_IN / height)

    for r, (target, title, unit) in enumerate(REGRESSION):
        # One pair of limits for the whole row — see the docstring. Built from the
        # union of truth and prediction so a cloud that overshoots is not clipped
        # out of frame, which would hide exactly the failure the panel is there for.
        vals = np.concatenate([np.concatenate([data[(s, p, target)]["y_true"],
                                               data[(s, p, target)]["y_pred"]])
                               for s, _, p, _ in cols])
        lo, hi = float(vals.min()), float(vals.max())
        pad = 0.04 * (hi - lo)
        lim = (lo - pad, hi + pad)

        for c, (scheme, key, proto, chan) in enumerate(cols):
            ax = axes[r][c]
            d = data[(scheme, proto, target)]
            ax.plot(lim, lim, ls=(0, (4, 2)), lw=0.8, color=REFLINE, zorder=1)
            ax.scatter(d["y_true"], d["y_pred"], s=3.0, alpha=0.45, linewidths=0,
                       color=ARM_COLORS[key], zorder=2)
            ax.set(xlim=lim, ylim=lim)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            # R² INSIDE the panel: at this size an axis title would collide with the
            # column header, and the number belongs to the cloud, not to the column.
            ax.text(0.05, 0.95, f"{d['score']:.2f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=6.5, color=INK,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8))
            if r == 0:
                ax.set_title(chan, fontsize=7.5, color=MUTED, pad=3)
            # TICK LABELS ON EVERY PANEL. An earlier version carried them on the
            # leftmost column only, on the argument that the shared row limits make the
            # other five redundant. They are redundant and it still read badly: the two
            # rows differ twentyfold in range, and a reader checking a cloud in column 6
            # against the identity had to track back across five panels to learn what
            # the axis was. The axis NAMES stay on the leftmost panel, where repeating
            # them six times would be clutter with no such excuse.
            ax.tick_params(labelsize=6)
            if c == 0:
                # NO LOCAL fontsize (2026-08-25): these take `paperstyle`'s
                # `axes.labelsize` like every other axis label in the paper. The rest
                # of this figure keeps its reduced scale on purpose — 6 pt ticks and a
                # 7.5 pt row header — because a 2x6 grid of parity panels cannot carry
                # the main figures' sizes. The AXIS NAMES are the exception: they are
                # drawn once, on the leftmost panel only, so there is room for them,
                # and they are what a reader crossing figures matches on.
                ax.set_ylabel(f"predicted ({unit})")
                ax.set_xlabel(f"true ({unit})")

        # Row header, placed from the row's realized geometry rather than a guessed
        # figure coordinate, so it stays centred if the grid is redrawn at another size.
        box = axes[r][0].get_position()
        fig.text(0.008, (box.y0 + box.y1) / 2, row_label(title), rotation=90,
                 ha="left", va="center", fontsize=7.5, color=INK, linespacing=1.05)

    # Scheme header spanning that scheme's two channel columns, in the scheme's own hue —
    # the only place the hue is named, since the panels themselves carry no legend.
    for i in range(0, ncol, 2):
        scheme, key = cols[i][0], cols[i][1]
        span = (axes[0][i].get_position().x0 + axes[0][min(i + 1, ncol - 1)].get_position().x1) / 2
        fig.text(span, 1 - 0.15 / height, scheme, ha="center", va="center",
                 fontsize=8.5, color=ARM_COLORS[key])

    paperstyle.save(fig, args.stem, sub="appendix")


if __name__ == "__main__":
    main()
