"""tools/plot_rruff_confusion.py — per-class confusion for Figure 2's four arms.

    python tools/plot_rruff_confusion.py            # THE DELIVERABLE

Appendix expansion of `analysis/plot_crystal_ladder.py` (Figure 2, crystal system from
the full RRUFF labeled set), and the RRUFF twin of `tools/plot_chili_confusion.py` —
same ramp, same row normalization, same no-numbers-in-cells rule, so a reader who has
read one reads the other without re-learning anything.

WHAT THE BARS CANNOT SAY. Figure 2 orders four arms by weighted F1 and nothing more. Two
very different failures land at the same height: an arm that spreads its error across
neighbouring systems, and an arm that has quietly collapsed onto the modal class. The
second would make the whole figure meaningless — `supervised` sits at 0.209 against a
0.125 modal floor, close enough that "it has learned to say cubic" is a live reading —
and no aggregate on that figure can rule it out. These matrices can, and do: every arm
spreads its predictions across all seven systems, with predicted-share tracking support
(cubic 0.22-0.25 against a 0.22 prior). Nothing here has collapsed.

WHAT THEY SHOW INSTEAD. Pretraining's gain is CONCENTRATED, not uniform. From
`supervised` to `finetuned` the weighted F1 goes 0.209 -> 0.357, and almost all of it is
cubic (recall 0.35 -> 0.65) and trigonal (0.22 -> 0.41); orthorhombic does not move
(0.29 -> 0.30). Both small classes stay near zero in every pretrained arm — triclinic has
10 rows and hexagonal 6, and hexagonal is the one class the RAW signal reads better
(0.22) than any trained arm.

CELLS ARE ROW-NORMALIZED (recall), NOT COUNTS, for the reason `plot_chili_confusion`
gives at length: RRUFF's crystal systems run 6 rows (hexagonal) to 33 (cubic), so a
count matrix is dominated by the big classes and a collapse onto the modal class looks
exactly like correct prediction of it. Under row normalization a collapse is one bright
COLUMN cutting every row, which nothing else in the figure imitates.

NO IN-CELL NUMBERS, same as the CHILI twin: four panels at text width leave ~0.15 in per
cell, which needs ~4 pt type. Magnitude is the ramp against a fixed 0-1 scale shared by
all four panels; the exact per-class values go in the accompanying table.

THE THREE REPEATS ARE POOLED, not averaged. Each repeat is a COMPLETE out-of-fold pass
over all 148 minerals, so pooling gives 444 (true, predicted) pairs per arm and every
mineral contributes three times — once per fold assignment. Averaging three separately
normalized matrices would weight a repeat's rare-class row (hexagonal: 6 rows) the same
as its cubic row (33), which is not what "share of the true class" means. The annotated
wF1 stays the MEAN over repeats, because that is the number Figure 2's bar is.

⚠️ **THE TWO SOURCES ENCODE PREDICTIONS DIFFERENTLY, AND THE LOADER DOES NOT TRUST
EITHER.** `analysis/finetune_cv.py` banks `oof_per_repeat` as 0-based CODES into
`config["classes"]`, while `analysis/probe_cv.py` banks CLASS VALUES under the same key
name (see that scorer's note, and docs/TRAPS.md). Read naively, the gradient arms appear
to predict cubic exactly zero times and to score 10% accuracy — under the modal floor.
`load` therefore decodes per source and then RECOMPUTES the weighted F1 from the vectors
it is about to plot, raising unless it reproduces the aggregate that run recorded. That
check is what makes the panels trustworthy: a matrix that disagreed with its own bar
would be a figure arguing against Figure 2 in the same document.

⚠️ **THE PROBE PANEL IS A LOCAL RE-RUN AND ITS wF1 IS 0.395, WHERE FIGURE 2's BAR SAYS
0.399.** Figure 2 reads `analysis/out/rruff_mpfull_full.json`, a hand-assembled aggregate
with no writer in this repo, no config block, and — per RESULTS.md — produced in another
session on another machine. It stores no per-row predictions, so this figure cannot come
from it. The re-run
(`analysis/out/rruff_probe_cv_xrd_crystal_matched_preds.json`) is the same checkpoint at
epoch 300, the same 5x3 matched-geometry protocol, and it reproduces the RAW arm and the
modal floor BIT-EXACTLY and the probe's second repeat to four decimals; the gap is 0.005
and 0.007 on the other two repeats, i.e. a handful of borderline minerals flipping under
float differences in the encoder's forward pass. The caption must carry this. If the two
are ever reconciled, the bar is the thing to recompute — RESULTS.md's own rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import paperstyle
from analysis.paperstyle import ARM_COLORS, INK, MUTED, SEQUENTIAL

REPO = Path(__file__).resolve().parent.parent
TARGET = "target_crystal_system"

#: The one carrier, matching `analysis/plot_crystal_ladder.py`'s MODEL. Both pretrained
#: arms are this checkpoint and no other — the single-carrier decision (RESULTS.md Q8).
MODEL = "vicreg_cov50"

#: IUCr order, from `data/builders/rruff.py`. Class VALUES are 1..7 in this order, and
#: the codes `finetune_cv` banks are 0-based indices into the same sequence.
SYSTEMS = ("triclinic", "monoclinic", "orthorhombic", "tetragonal",
           "trigonal", "hexagonal", "cubic")

#: FOUR LETTERS, ON BOTH AXES. A 7-class matrix at a quarter of WIDTH leaves ~0.15 in
#: per cell and "orthorhombic" is 0.36 in at 5.5 pt, so the full names cannot be x tick
#: labels. They would fit down the y axis of the leftmost panel, and that is exactly the
#: trap: one figure would then name its rows and its columns in two different
#: vocabularies for one set of seven classes. The caption spells them out once.
SHORT = ("tric", "mono", "orth", "tetr", "trig", "hexa", "cubi")

#: (panel label, ARM_COLORS key) in Figure 2's own left-to-right order — ascending by
#: F1, which is the paper's narrative ordering. A reader meets the four arms in one
#: sequence in both figures or the appendix is answering a question nobody asked.
ARMS = (("supervised", "supervised"), ("raw", "raw"),
        ("finetuned", "finetuned"), ("probe", "probe"))

#: Where each arm's per-row predictions live, and how that file encodes them.
#:
#: `codes` — `analysis/finetune_cv.py`: `oof_per_repeat` holds 0-based indices into
#:     `config["classes"]`, while `y_true` holds class VALUES. The two are not on the
#:     same footing and nothing in the file says so.
#: `values` — `analysis/probe_cv.py` since 2026-09-09: both lists hold class values,
#:     with `classes` beside them, so the cell decodes itself.
#:
#: The probe and raw arms come from ONE file because one `probe_cv` run produces both:
#: the raw baseline is scored in the same pass, on the same folds, before any encoder
#: is loaded. That is also why raw reproduces bit-exactly — it never touches the GPU.
SOURCES = {
    "supervised": ("runs/xrd/rruff_cv/sup_cnn_crystal_system.json", ("supervised",), "codes"),
    "finetuned": (f"runs/xrd/rruff_cv/ft_cnn_{MODEL}_mpfull_crystal_system.json",
                  ("finetuned",), "codes"),
    "raw": ("analysis/out/rruff_probe_cv_xrd_crystal_matched_preds.json",
            ("raw_baseline", TARGET), "values"),
    "probe": ("analysis/out/rruff_probe_cv_xrd_crystal_matched_preds.json",
              ("models", "*", "tasks", TARGET), "values"),
}


def dig(doc, path):
    """Walk `path` through `doc`; a `"*"` segment takes the single key at that level."""
    for key in path:
        if key == "*":
            if len(doc) != 1:
                raise SystemExit(f"expected exactly one model in the probe file, "
                                 f"found {len(doc)}: {list(doc)}")
            key = next(iter(doc))
        doc = doc[key]
    return doc


def load(arm):
    """`(y_true, oof_per_repeat, wF1)` for one arm, as CLASS VALUES, verified.

    The verification is the point and is not optional: the two sources encode
    `oof_per_repeat` differently (see the module docstring), so a decode that is wrong
    produces a plausible matrix rather than an error. Recomputing the weighted F1 from
    the decoded vectors and demanding it reproduce what the run banked catches that,
    and catches any future change to either scorer's format at the point of drawing.
    """
    path, keys, encoding = SOURCES[arm]
    doc = json.loads((REPO / path).read_text())
    cell = dig(doc, keys)
    # `finetune_cv` files carry the arm's scores one level up from y_true/ids; the
    # probe_cv ones carry everything in the cell. `dig` has already landed on whichever
    # holds the metrics, so y_true is taken from the cell if present and the doc if not.
    y_true = np.array(cell.get("y_true", doc.get("y_true")), dtype=float)
    banked = cell["f1_per_repeat"]
    oof = np.array(cell["oof_per_repeat"], dtype=float)
    if encoding == "codes":
        classes = np.array(doc["config"]["classes"], dtype=float)
        oof = classes[oof.astype(int)]

    got = [f1_score(y_true, p, average="weighted", zero_division=0) for p in oof]
    if not np.allclose(got, banked, atol=2e-4):
        raise SystemExit(
            f"{arm}: weighted F1 recomputed from the stored per-row predictions "
            f"{[round(v, 4) for v in got]} does not reproduce the aggregate the run "
            f"recorded {[round(v, 4) for v in banked]}. The vectors and the score "
            f"disagree — most likely {path} changed how it encodes `oof_per_repeat` "
            f"(this loader expects '{encoding}'); do not draw the matrix until they do.")
    return y_true, oof, float(np.mean(banked))


def matrix(y_true, oof):
    """Row-normalized 7x7 over all repeats pooled. Rows with no support stay 0."""
    m = np.zeros((len(SYSTEMS), len(SYSTEMS)))
    for preds in oof:
        for t, p in zip(y_true, preds):
            m[int(t) - 1, int(p) - 1] += 1
    support = m.sum(axis=1, keepdims=True)
    return np.divide(m, support, out=np.zeros_like(m), where=support > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stem", default="rruff_crystal_confusion")
    ap.add_argument("--table", action="store_true",
                    help="print the per-class recall table this figure's caption "
                         "points at, and exit")
    args = ap.parse_args()

    data = {label: load(key) for label, key in ARMS}
    if args.table:
        print(recall_table(data))
        return

    paperstyle.use()
    ncol = len(ARMS)
    # Square panels, so the HEIGHT is derived from the width four columns leave — the
    # same arithmetic as `plot_chili_confusion`, and for the same reason: hardcoding a
    # height centres the row inside a taller axes band and opens dead space. The gaps
    # are in INCHES because the furniture between panels (y tick labels on column 0, an
    # xlabel and a wF1 line under every panel) is a fixed size in points.
    LEFT, RIGHT = 0.075, 0.995
    COL_IN = 0.20                      # between panels: no ylabel except on column 0
    HEAD_IN, FOOT_IN = 0.30, 0.92      # arm name; tick labels + xlabel + wF1 + colorbar
    panel = ((RIGHT - LEFT) * paperstyle.WIDTH - (ncol - 1) * COL_IN) / ncol
    height = HEAD_IN + FOOT_IN + panel
    fig, axes = plt.subplots(1, ncol, squeeze=False,
                             figsize=(paperstyle.WIDTH, height))
    fig.subplots_adjust(left=LEFT, right=RIGHT, wspace=COL_IN / panel,
                        top=1 - HEAD_IN / height, bottom=FOOT_IN / height)

    for c, (label, key) in enumerate(ARMS):
        ax = axes[0][c]
        y_true, oof, wf1 = data[label]
        ax.imshow(matrix(y_true, oof), cmap=SEQUENTIAL, vmin=0, vmax=1,
                  aspect="equal", interpolation="nearest")
        ax.grid(False)
        ax.set_xticks(range(len(SYSTEMS)))
        ax.set_yticks(range(len(SYSTEMS)))
        # EVERY panel carries its x ticks. The alternative — furniture on column 0 only
        # — asks the reader to count cells back across three panels to name the column
        # a bright cell sits in, which is the one thing this figure exists to be read
        # for. `plot_chili_confusion` settled the same question the same way.
        ax.set_xticklabels(SHORT, fontsize=5.0, rotation=90)
        ax.set_yticklabels(SHORT if c == 0 else [], fontsize=5.0)
        ax.set_xlabel("predicted", labelpad=2, fontsize=7)
        if c == 0:
            ax.set_ylabel("true", labelpad=2, fontsize=7)
        ax.tick_params(length=0)
        # The arm name in its own hue, so the panel keys itself against Figure 2's bars
        # without a legend. `raw`'s violet is the re-stepped one — a dark label on a
        # dark fill is what cost the old hex its place (paperstyle).
        ax.set_title(label, fontsize=8.5, color=ARM_COLORS[key], pad=3)
        # In INCHES off the axes edge, clearing the rotated tick labels and the xlabel,
        # both fixed in points — in axes fractions this slid with the panel size.
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, box.y0 - 0.44 / height, f"wF1 {wf1:.3f}",
                 ha="center", va="top", fontsize=6.5, color=INK)

    mid = (axes[0][0].get_position().x0 + axes[0][-1].get_position().x1) / 2
    cax = fig.add_axes([mid - 0.15, 0.18 / height, 0.30, 0.055 / height])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=plt.Normalize(0, 1)),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=6, length=2, color=MUTED, pad=1.5)
    cb.ax.set_title("share of true class", fontsize=6.5, color=MUTED, pad=2.5)
    cb.outline.set_visible(False)

    paperstyle.save(fig, args.stem, sub="appendix")
    print(recall_table(data))


def recall_table(data):
    """Per-class recall for all four arms — the exact values the cells encode.

    The other half of the no-numbers-in-cells decision: the ramp carries magnitude on
    the figure, and anything a reader needs to QUOTE is here. Support is printed once,
    because it is a property of the 148 minerals and not of an arm.
    """
    y_true = data["supervised"][0]
    rows = []
    for i, name in enumerate(SYSTEMS):
        c = i + 1
        support = int((y_true == c).sum())
        cells = []
        for label, _ in ARMS:
            yt, oof, _ = data[label]
            cells.append(np.mean([(p[yt == c] == c).mean() for p in oof]))
        rows.append(f"        {name:<13s} & {support:>3d} & "
                    + " & ".join(f"{v:.2f}" for v in cells) + r" \\")
    heads = " & ".join(label for label, _ in ARMS)
    return rf"""% --- generated by tools/plot_rruff_confusion.py --table ---
\begin{{table}}[h]
    \centering
    \caption{{Per-class recall behind the confusion matrices, over the three pooled
    out-of-fold repeats. Support is the number of the 148 minerals in each system and
    is a property of the labeled set, not of a scheme. The two smallest classes carry
    almost no weight in the weighted F1 and correspondingly little evidence: one
    hexagonal mineral is 0.17 of that row.}}
    \label{{tab:rruff_confusion}}
    \begin{{tabular}}{{lr{"c" * len(ARMS)}}}
        \toprule
        Crystal system & $n$ & {heads} \\
        \midrule
{chr(10).join(rows)}
        \bottomrule
    \end{{tabular}}
\end{{table}}"""


if __name__ == "__main__":
    main()
