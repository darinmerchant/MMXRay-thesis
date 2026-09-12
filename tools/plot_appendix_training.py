"""tools/plot_appendix_training.py — appendix A.2.3: pretraining curves.

    /opt/anaconda3/envs/mmxray/bin/python -m tools.plot_appendix_training

Four panels in a 2x2 GRID over the 300 pretraining epochs of
`cnn_vicreg_mpfull_final` (the checkpoint every headline figure uses), read
straight from the run's `metrics.jsonl`: total training loss and the three VICReg
terms on the top row, learning rate and gradient norm below — the same grid shape
as the A.3.3 labels figure. A stacked column (the A.1.5 layout) and a 1x4 row
were both built first and replaced by request; 2x2 is the compromise between
them — half the column's page height, twice the row's panel width, so titles and
the term legend fit without abbreviation.

CANVAS HEIGHT IS SET BY THE APPENDIX 2x2 PANEL HEIGHT, NOT BY THIS FIGURE ALONE.
These are placed at natural size, so the declared height IS the printed height
(`paperstyle.save`), and the other 2x2 appendix figures draw 1.1 in panels. At the
original 3.6 in this figure's panels came to 1.52 in — the same canvas height as
the invariance-decay figure, but only because that one spends its bottom 0.7 in on
a legend, so on the page these panels read a third taller than every neighbor.
2.8 in puts them at 1.07/1.12 in and the whole figure 0.8 in shorter. The cost is
the learning-rate panel dropping to two y ticks (0.5, 1.0), which leaves the 1.4
warmup peak unlabeled; the peak belongs in the caption rather than in a hand-set
locator that no other panel here needs.

ONE LOSS PANEL, NOT TWO. The run logs exactly one loss series: the epoch-mean of
the weighted VICReg total. There is no validation loss to draw beside it — the MP
pretraining splits are (1.0, 0, 0), which A.2.4 states outright — and nothing
per-batch was retained. A second "mean training loss" panel would replot the same
numbers under a second name.

THE PER-TERM PANEL IS AN EXACT DECOMPOSITION of the panel above it. Terms are drawn
as WEIGHTED contributions — 25·inv, 25·var, 50·cov, the run's configured weights —
which is the `VICREG_COLORS` convention in `analysis/paperstyle.py`, and here the
three curves sum to the total to within float noise (max abs gap 0.002 over 300
epochs, checked against this run before this module was written). Raw unweighted
terms would show each term's own dynamics on a more equal footing but would not
sum to anything the reader has seen.

THE GRADIENT NORM IS THE PRE-CLIP NORM. `core/train.py` logs the return value of
`clip_grad_norm_`, which is the total norm measured BEFORE clipping to the
configured max of 5.0 — that is why the whole curve sits at 8-42, far above the
clip: every step of this run was clipped, and the panel title says "pre-clip"
so a reader cannot mistake the curve for the effective update size.

WHY THE GRADIENT CURVE HAS GAPS, and why they are left visible. On 15 of 300
epochs the AMP loss-scaler skipped at least one step (10 at epoch 1 while the
scale was still searching, then one per scale-doubling attempt as `amp_scale`
climbed 64 -> 2048), and a skipped step's non-finite gradient poisons that epoch's
mean, logged as NaN. Matplotlib breaks the line there. Interpolating across the
gaps would fabricate 15 values and erase the only visible trace of the scaler
doing its job; the caption can say what the gaps are.

SINGLE-SERIES PANELS ARE INK, NOT A HUE. `VICREG_COLORS` spends slots 1-3 (blue /
orange / aqua) on the term panel, so a blue total-loss line one panel up would
read as "the invariance term again" — the exact collision the paperstyle
docstring's slots-are-per-figure rule exists to prevent WITHIN a figure. Loss, LR
and gradient norm are each the only series on their axis, so they need no
identity colour at all; they take INK and the term panel keeps the only legend
(mandatory there: the aqua slot sits below the 3:1 contrast gate and must never
ride on colour alone).

LR IS PLOTTED IN UNITS OF 1e-4 with the unit in the axis label, rather than
letting matplotlib hang a "1e-4" offset above the axis — at 8 epochs of warmup to
1.4e-4 and a decay to 1e-5, every value shares that exponent, and the offset text
collides with the panel title in a stack this tight.

NOT A DUPLICATE OF `analysis/train_curves.py`. That module is the 3x3 diagnostic
comparing every run of a sweep directory; this one is the paper figure for the
single production checkpoint, in the appendix style. They agree on the weighted-
contribution convention because both read the same docstringed rule.

Reads only json + numpy + matplotlib — none of the torch-before-pandas import
trap applies (docs/TRAPS.md). Runtime ~2 s.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis import paperstyle as ps

REPO = Path(__file__).resolve().parents[1]

#: The pretraining run behind every headline figure (RESULTS.md). The per-term
#: weights below are read from this run's config.yaml, not re-derived.
DEFAULT_RUN = REPO / "runs" / "pdf" / "sweep" / "2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372"

#: (metrics.jsonl key, legend label, configured weight). The weights are the
#: run's sim_weight / var_weight / cov_weight; drawn weighted, the three curves
#: sum to the logged total — see the docstring.
TERMS = [
    ("inv_loss", "invariance ×25", 25.0),
    ("var_loss", "variance ×25", 25.0),
    ("cov_loss", "covariance ×50", 50.0),
]

TERM_COLORS = {"inv_loss": ps.VICREG_COLORS["inv"],
               "var_loss": ps.VICREG_COLORS["var"],
               "cov_loss": ps.VICREG_COLORS["cov"]}


def load_metrics(run_dir: Path) -> dict[str, np.ndarray]:
    """metrics.jsonl -> one array per key. NaNs pass through (grad_norm keeps its gaps)."""
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


def build(run_dir: Path, out_stem: str, sub: str | None) -> None:
    ps.use()
    m = load_metrics(run_dir)
    epoch = m["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(ps.WIDTH, 2.8), sharex=True)
    (ax_loss, ax_terms), (ax_lr, ax_grad) = axes

    ax_loss.plot(epoch, m["loss"], lw=1.2, color=ps.INK)
    ax_loss.set_title("training loss (weighted total)", loc="left", pad=3)

    for key, label, weight in TERMS:
        ax_terms.plot(epoch, weight * m[key], lw=1.2, color=TERM_COLORS[key], label=label)
    ax_terms.set_title("weighted loss terms", loc="left", pad=3)
    ax_terms.legend(loc="upper right", handlelength=1.2, labelspacing=0.3)

    ax_lr.plot(epoch, m["lr"] * 1e4, lw=1.2, color=ps.INK)
    ax_lr.set_title("learning rate (×10⁻⁴)", loc="left", pad=3)

    ax_grad.plot(epoch, m["grad_norm"], lw=1.2, color=ps.INK)
    ax_grad.set_title("pre-clip grad norm (epoch mean)", loc="left", pad=3)

    n_gaps = int(np.isnan(m["grad_norm"]).sum())
    print(f"  {len(epoch)} epochs; grad-norm gaps at {n_gaps} epochs "
          f"(AMP scaler skips); final loss {m['loss'][-1]:.2f}")

    for ax in (ax_lr, ax_grad):
        ax.set_xlim(epoch[0], epoch[-1])
        ax.set_xticks([0, 100, 200, 300])
        ax.set_xlabel("epoch")
    ps.layout(fig)
    ps.save(fig, out_stem, sub=sub)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN,
                    help="pretraining run directory holding metrics.jsonl")
    ap.add_argument("--out", default="appendix_training")
    ap.add_argument("--sub", default="appendix", help="subdirectory under figures/")
    args = ap.parse_args()
    build(args.run, args.out, args.sub)


if __name__ == "__main__":
    main()
