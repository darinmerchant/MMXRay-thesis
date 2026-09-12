"""tools/plot_chili_confusion.py — per-row confusion for the two categorical targets.

    python tools/plot_chili_confusion.py                    # THE DELIVERABLE
    python tools/plot_chili_confusion.py --schemes probe    # one scheme, before the rest land

Appendix expansion of Figure 2's categorical half, and the companion of
`plot_chili_parity.py` — same store, same schemes, same two fit channels, same column
order. A weighted F1 says how much of the label a scheme recovers; it cannot say WHICH
classes it confuses, and the two failure modes that matter here look identical in the
aggregate: a scheme that spreads error across neighbouring classes and a scheme that has
collapsed onto the modal class both land near the modal floor.

CELLS ARE ROW-NORMALIZED (recall), NOT COUNTS, and that is the whole point of the
figure. Both targets are imbalanced — coordination number is 55% one class, oxidation
state 42% (Figure 8) — so a count matrix is dominated by the class with the most rows and
the collapse reads as "one bright cell", which is also what a correct prediction of the
modal class looks like. Under row normalization every row sums to 1 regardless of its
support, so a collapse becomes one bright COLUMN cutting across every row, which nothing
else in the figure can imitate.

NO IN-CELL NUMBERS. Six columns at text width leaves the 5-class matrix ~0.12 in per
cell, which needs ~4 pt type — below any print floor. `paperstyle.SEQUENTIAL`'s own
note sanctions the fallback ("Print counts beside the matrix instead"), so magnitude is
carried by the ramp against a fixed 0-1 scale shared by all twelve panels, and the exact
per-class values live in the accompanying table.

The annotated weighted F1 is RECOMPUTED from the plotted rows by
`chili_predictions.load`, which raises if it disagrees with what the run recorded.
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
from analysis.paperstyle import ARM_COLORS, INK, MUTED, SEQUENTIAL
from tools.chili_predictions import (CATEGORICAL, classes_of, columns, load,
                                     row_label)


def pretty(v):
    """Class values are floats because oxidation state carries 8/3. Render the
    thirds as the fraction the paper prints, everything else as a plain integer."""
    return "8/3" if abs(v - 8 / 3) < 1e-3 else f"{v:g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schemes", nargs="+", default=None,
                    help="subset of baseline/probe/fine-tuned (default: all three)")
    ap.add_argument("--stem", default="chili_confusion")
    args = ap.parse_args()

    cols = columns(args.schemes)
    data = load([t[0] for t in CATEGORICAL], args.schemes)
    paperstyle.use()

    ncol, nrow = len(cols), len(CATEGORICAL)
    # Panels are square (`aspect="equal"`), so the figure HEIGHT is DERIVED from the
    # panel width the column count leaves. Hardcoding it instead makes matplotlib
    # centre the row band inside a taller axes area, which is dead space that grows
    # with every column removed — the `--schemes` subsets looked broken because of it.
    #
    # THE GAPS ARE IN INCHES, NOT PANEL FRACTIONS, for the same reason. Every panel
    # carries its own furniture — a ylabel between columns, tick labels + xlabel + wF1
    # between rows — and that furniture is a fixed size in points however wide the
    # panels are. As fractions, a `--schemes` subset at two columns has 2 in panels and
    # opens a 1.4 in canyon around a 0.4 in label stack. Solving `WSPACE = COL_IN/panel`
    # for `panel` is what makes the subsets and the deliverable use the same margins.
    LEFT, RIGHT = 0.105, 0.995
    COL_IN, ROW_IN = 0.24, 0.44     # ylabel; tick labels + xlabel + wF1
    HEAD_IN, FOOT_IN = 0.46, 0.86
    panel = ((RIGHT - LEFT) * paperstyle.WIDTH - (ncol - 1) * COL_IN) / ncol
    WSPACE, HSPACE = COL_IN / panel, ROW_IN / panel
    height = HEAD_IN + FOOT_IN + panel * nrow + (nrow - 1) * ROW_IN
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(paperstyle.WIDTH, height))
    fig.subplots_adjust(left=LEFT, right=RIGHT, wspace=WSPACE, hspace=HSPACE,
                        top=1 - HEAD_IN / height, bottom=FOOT_IN / height)

    for r, (target, title) in enumerate(CATEGORICAL):
        # The class vocabulary is a property of the TARGET, not of a panel: taken over
        # every scheme at once so all six matrices are the same shape and the same axis
        # order, which is what lets one row be read across.
        classes = classes_of(
            np.concatenate([data[(s, p, target)]["y_true"] for s, _, p, _ in cols]),
            np.concatenate([data[(s, p, target)]["y_pred"] for s, _, p, _ in cols]))
        code = {v: i for i, v in enumerate(classes)}
        k = len(classes)

        for c, (scheme, key, proto, chan) in enumerate(cols):
            ax = axes[r][c]
            d = data[(scheme, proto, target)]
            m = np.zeros((k, k))
            for t, p in zip(d["y_true"].tolist(), d["y_pred"].tolist()):
                m[code[t], code[p]] += 1
            # Row-normalized; a row with no support would divide by zero, and stays 0.
            support = m.sum(axis=1, keepdims=True)
            m = np.divide(m, support, out=np.zeros_like(m), where=support > 0)

            ax.imshow(m, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="equal",
                      interpolation="nearest")
            ax.grid(False)
            ax.set_xticks(range(k)), ax.set_yticks(range(k))
            # EVERY panel is labelled, x ticks included. The alternative — furniture on
            # column 0 only — asks the reader to count cells back to the leftmost panel
            # to name the column a bright cell sits in, which is the one thing this
            # figure exists to be read for.
            ax.set_xticklabels([pretty(v) for v in classes], fontsize=5.5)
            # THE AXIS NAMES TAKE THIS FIGURE'S REDUCED SCALE (7 pt), unlike
            # plot_chili_parity's. That figure keeps `axes.labelsize` because it names
            # its axes ONCE, on the leftmost panel, where there is room. Named on all
            # twelve panels, 8.5 pt type is wider than the 0.6 in panel it belongs to
            # and the labels outweigh the matrices.
            ax.set_xlabel("predicted", labelpad=2, fontsize=7)
            ax.set_ylabel("true", labelpad=2, fontsize=7)
            # Below the xlabel, not in its slot: the panel names its own axis now. The
            # drop is in INCHES off the axes edge — it clears tick labels and an
            # xlabel, both of which are a fixed size — so it stays put when a subset
            # changes the panel size. In axes fractions it slid down the page.
            box = ax.get_position()
            fig.text((box.x0 + box.x1) / 2, box.y0 - 0.29 / height,
                     f"wF1 {d['score']:.2f}",
                     ha="center", va="top", fontsize=6.5, color=INK)
            if r == 0:
                ax.set_title(chan, fontsize=7.5, color=MUTED, pad=3)
            # Y TICK LABELS STAY ON COLUMN 0. The class order is identical across a row
            # and repeating it six times costs panel width, which the matrices need
            # more than the reader does — the ylabel says which axis it is.
            if c != 0:
                ax.set_yticklabels([])
            else:
                ax.set_yticklabels([pretty(v) for v in classes], fontsize=5.5)
            ax.tick_params(length=0)

        box = axes[r][0].get_position()
        fig.text(0.008, (box.y0 + box.y1) / 2, row_label(title), rotation=90,
                 ha="left", va="center", fontsize=7.5, color=INK, linespacing=1.05)

    for i in range(0, ncol, 2):
        scheme, key = cols[i][0], cols[i][1]
        span = (axes[0][i].get_position().x0 + axes[0][min(i + 1, ncol - 1)].get_position().x1) / 2
        fig.text(span, 1 - 0.15 / height, scheme, ha="center", va="center",
                 fontsize=8.5, color=ARM_COLORS[key])

    # The footer stacks bottom-up in INCHES off the figure edge: the bar's tick labels
    # (0.08), the bar (0.20), its caption (0.30). Laying it out in figure fractions
    # instead clipped the colorbar's tick labels off the page, because the fraction a
    # fixed 0.09 in label needs changes with every row count.
    mid = (axes[-1][0].get_position().x0 + axes[-1][-1].get_position().x1) / 2

    cax = fig.add_axes([mid - 0.15, 0.20 / height, 0.30, 0.055 / height])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQUENTIAL,
                                            norm=plt.Normalize(0, 1)),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=6, length=2, color=MUTED, pad=1.5)
    cb.ax.set_title("share of true class", fontsize=6.5, color=MUTED, pad=2.5)
    cb.outline.set_visible(False)

    paperstyle.save(fig, args.stem, sub="appendix")


if __name__ == "__main__":
    main()
