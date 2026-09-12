"""tools/plot_appendix_labels.py — appendix A.3.3: the four targets' label distributions.

    /opt/anaconda3/envs/mmxray/bin/python -m tools.plot_appendix_labels

A 2x2 grid: the two REGRESSION targets on the top row as step histograms, the two
CATEGORICAL targets on the bottom as bars of each class's share. The split by metric
IS the layout, so a reader never has to check which panel is scored by R2 and which
by weighted F1.

EVERY PANEL PLOTS SHARE. The regression row used to plot COUNT, which made the two
rows read as two different quantities stacked on one figure and cost the reader a
translation between them. It is now share of the labelled samples in each of `BINS`
equal-width bins, so all four panels answer "what fraction of the corpus is here"
and ONE figure-level y label covers them. Note the bin dependence that buys: a
histogram bar scales with 1/BINS, so its height is comparable to a categorical bar
only at a stated bin count, which is why `BINS` is a named constant.

The limits are per ROW, not shared across all four (`sharey="row"`). A bin's share
tops out near 0.16 and a modal class's near 0.55; one axis for both would leave the
regression row using a quarter of its panel height to say the same thing.

ALL 3180 SAMPLES POOLED — no split breakdown. The question this figure answers is
"what do the labels look like", and one series per panel answers it without the
reader decoding a colour key first. The per-split view (whether the formula-disjoint
split skewed a class share) is a different question; A.3.1's counts cover it, and
`analysis/target_distributions.py` draws it per split if it is ever needed.

ONE HUE, slot 1 of the documented palette, the same colour in every panel, because
nothing here is distinguished BY colour. Identity comes from the panel title, and on
the categorical row that title is also the x axis: the ticks are the classes
themselves, so a label under them only restates the title in other words.

NO n ON THE FIGURE. Every panel's count is printed to stdout instead. It was tried
as a corner annotation and then as a title suffix, and in both places it read as a
caption fragment on a figure whose panels otherwise carry nothing but their name.

NOTHING IS DRAWN OVER THE BARS, AND SO THERE IS NO LEGEND. Two decorations were
tried on the categorical panels and both were removed:

  * A dashed REFLINE at the MODAL CLASS'S SHARE. It re-drew the tallest bar's height
    as a line across the panel — a quantity already readable off the y axis — in the
    exact vocabulary Figures 2 and 3 use for the modal FLOOR, which is a weighted F1
    in score units and a different number entirely. It bought a legend key to
    disambiguate itself and nothing else. NOTHING ON THIS FIGURE IS A MODAL FLOOR.
  * An annotated `modal wF1`. Score units on a share axis, and it was the POOLED
    floor — 0.385 for coordination number — while the paper quotes the fit-split
    modal class scored on TEST, 0.418. A figure printing a number the text
    contradicts is worse than a figure printing none.

Both quantities are still computed and written to stdout on every run, side by side,
so a caption can quote the split-based pair without re-deriving it. The computation
is
`sklearn.metrics.f1_score(average="weighted")` on the fit-modal convention — the
same call `analysis/downstream_eval.py` makes.

THE COORDINATION-NUMBER PANEL HAS FEWER ROWS than the other three — 2915 against
3180 — and since the figure no longer says so, THE CAPTION MUST. Only the Spinel prototype disagrees across its metal
sites, which leaves 265 samples unlabelled for that target alone. Per split that is
2310 train / 340 test, which is the source of the "2310" in section 3.3 that
otherwise reads as an inconsistency.

WHY A NEW MODULE rather than a `--paper` flag on `analysis/target_distributions.py`:
that one discovers every `target_*` column itself and adapts its panel count when a
target is added, which is what makes it useful as a data sanity check. This one is
pinned to the paper's four targets, in the paper's layout, in the paper's style.
One module cannot be both without the generic half acquiring a hardcoded list.

Reads the registry only — no checkpoint, no run directory, nothing trained.

NO `import torch` GUARD HERE, deliberately, unlike `tools/plot_appendix_views.py`.
That guard exists because reaching pandas before torch segfaults (`docs/TRAPS.md` ->
the import-order bug), and it only bites a process that imports torch at all. This
module never does: no simulator, no `core.*`, no checkpoint. Verified to run clean
without it. Add the guard back the moment this file imports anything from `core`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from analysis import paperstyle as ps

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data" / "downstream" / "chili" / "chili_registry.parquet"

#: The paper's four, in the order Figure 2 and Table 4 use them.
REGRESSION = [("target_np_size", "Nanoparticle size", "diameter (Å)"),
              ("target_mo_bond", "M–O distance", "M–O distance (Å)")]
#: No x label on these two: the tick labels ARE the classes, and "neighbours" under
#: 4/6/8 or "formal metal charge" under 1/2/8-3/4/6 restates the title in other words.
CATEGORICAL = [("target_cn", "Coordination number"),
               ("target_oxidation", "Oxidation state")]

#: Bins on the regression row. Load-bearing now that the row plots SHARE rather than
#: count: a bar's height is its bin's share of the samples, so it scales with 1/BINS
#: and is only comparable to the categorical row's bars at a stated bin count.
BINS = 40


def _class_label(v: float) -> str:
    """8/3 is a stoichiometric average, not a valence — print it as the fraction."""
    return "8/3" if abs(v - 8 / 3) < 1e-6 else f"{v:g}"


def _modal_floor(y: pd.Series) -> tuple[float, float, float]:
    """(modal value, its share, the weighted F1 a constant predictor of it scores).

    Classes are factorized first because oxidation's 8/3 is a float and sklearn
    would otherwise read the column as continuous and refuse it.
    """
    codes, uniq = pd.factorize(y.round(6), sort=True)
    modal = pd.Series(codes).value_counts().idxmax()
    floor = f1_score(codes, np.full(len(codes), modal), average="weighted", zero_division=0)
    return float(uniq[modal]), float((codes == modal).mean()), float(floor)


def _paper_floor(df: pd.DataFrame, target: str) -> float:
    """The floor Figures 2 and 3 use: the TRAIN modal class scored on TEST.

    Reported to stdout, never drawn — this figure is pooled, and mixing a
    split-dependent number into it is exactly the confusion the docstring warns of.
    """
    d = df[["split", target]].dropna()
    codes, _ = pd.factorize(d[target].round(6), sort=True)
    d = d.assign(code=codes)
    modal = d.loc[d.split == "train", "code"].value_counts().idxmax()
    ev = d.loc[d.split == "test", "code"]
    return float(f1_score(ev, np.full(len(ev), modal), average="weighted", zero_division=0))


def _nice_top(m: float) -> tuple[float, float]:
    """(axis top, tick step) clearing `m` with ~10% headroom, on a round step.

    The two rows carry the SAME quantity on very different ranges — a bin's share
    of 3180 samples tops out near 0.16, a modal class's share near 0.55 — so the
    limit is computed rather than shared, and computed rather than hardcoded so a
    registry change cannot silently clip a bar off the top of its panel.
    """
    for step in (0.02, 0.05, 0.1, 0.2):
        top = float(np.ceil(m * 1.10 / step) * step)
        if top / step <= 6:  # more than six gridless ticks is a ruler, not an axis
            return top, step
    raise ValueError(f"no round step fits a maximum of {m}")


def _set_row(axes, values: list[np.ndarray]) -> None:
    """One y limit and one tick set across a row, from that row's own maximum."""
    top, step = _nice_top(max(float(v.max()) for v in values))
    axes[0].set_ylim(0, top)
    axes[0].set_yticks(np.arange(0, top + step / 2, step))


def build(registry: Path, out_stem: str, sub: str | None) -> None:
    ps.use()
    cols = ["split"] + [t for t, *_ in REGRESSION + CATEGORICAL]  # split: stdout only
    df = pd.read_parquet(registry, columns=cols)

    # sharey="row", not True: both rows are SHARE, but a histogram bin's share and a
    # class's share differ ~3.5x in range, and one axis across both would leave the
    # top row using a quarter of its panel height.
    fig, axes = plt.subplots(2, 2, figsize=(ps.WIDTH, 4.3), sharey="row")
    hue = ps.SLOTS[0]  # one series everywhere; identity comes from the title

    # SHARE, NOT COUNT, on the regression row. Count is what made the two rows read as
    # two different quantities; share is the same quantity the categorical row already
    # plots, so one figure-level y label now covers all four panels. It is share PER
    # BIN and therefore depends on BINS — halve the bin count and every bar doubles —
    # which is why the bin count is a named constant and belongs in the caption.
    row0 = []
    for ax, (target, title, xlabel) in zip(axes[0], REGRESSION):
        v = df[target].dropna()
        heights, _, _ = ax.hist(v, bins=BINS, color=hue, histtype="step", linewidth=1.4,
                                weights=np.full(len(v), 1 / len(v)))
        row0.append(heights)
        ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel(xlabel)
        # The regression row's only stdout line. It exists because n left the figure:
        # without it these two panels' counts are recorded nowhere the caption can
        # reach, and 3180 vs the categorical row's 2915 is the point of quoting them.
        print(f"  {target:18} n={len(v)} min={v.min():.3g} max={v.max():.3g}", flush=True)
    _set_row(axes[0], row0)

    row1 = []
    for ax, (target, title) in zip(axes[1], CATEGORICAL):
        v = df[target].dropna().round(6)
        classes = sorted(v.unique())
        share = np.array([float((v == c).mean()) for c in classes])
        ax.bar(np.arange(len(classes)), share, 0.6, color=hue, zorder=3)
        row1.append(share)

        modal, modal_share, floor = _modal_floor(v)  # stdout only; nothing drawn
        ax.set_xticks(np.arange(len(classes)), [_class_label(c) for c in classes])
        ax.set_title(title, loc="left", pad=3)
        print(f"  {target:18} n={len(v)} modal={_class_label(modal)} "
              f"share={modal_share:.3f} pooled_wF1={floor:.3f} "
              f"paper_wF1(train-modal on test)={_paper_floor(df, target):.3f}", flush=True)
    _set_row(axes[1], row1)

    # One label for four panels, because after the count -> share change all four
    # panels finally carry the same quantity. Per-axis labels would repeat one word
    # four times and cost the left column the width to do it.
    fig.supylabel("share of labeled samples", x=0.015)

    # NO LEGEND, and nothing to put in one: every panel is one hue, identity comes
    # from the title, and each bar's height is read straight off the shared y axis.
    ps.layout(fig, rect=(0.015, 0, 1, 1))
    ps.save(fig, out_stem, sub=sub)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=str(REGISTRY))
    ap.add_argument("--out", default="appendix_labels")
    ap.add_argument("--sub", default="appendix")
    args = ap.parse_args()
    build(Path(args.registry), args.out, args.sub)


if __name__ == "__main__":
    main()
