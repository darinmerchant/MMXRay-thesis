"""analysis/plot_invariance_decay.py — draw the invariance decay curves.

    python -m analysis.plot_invariance_decay --filter '*transformer_vicreg_mpfull_final*'

Reads `<run>/invariance_decay.json` and draws one panel per instrument parameter:
displacement of the representation against the value of that parameter, with the same
measurement on the raw signal beside it.

Separate from the measurement per the `aug_sweep.py` / `plot_aug_sweep.py` split — that
module writes the numbers and stays plot-free, so a figure change never risks a
measurement.

THE Y AXIS IS LOG, and it is what makes this figure work at all. In-range displacements
are ~1e-4 to 1e-2 while the between-material scale is ~0.6, so on a linear axis the
entire result sits indistinguishably on zero — which is why the reference repo's version
had to spend its width on the extrapolation tails to show anything. On a log axis the
in-range flatness is legible AND the tails stay on the same panel: no broken axis, no
inset, no second row. The two reference lines become well-separated horizontals, so
"the latent curve rides the floor, two decades below the ceiling" reads at a glance.

PANELS SHARE THE Y AXIS on purpose. The four parameters move the representation by very
different amounts, and per-panel autoscaling would draw them all as similar-looking
curves — hiding the comparison the figure exists to make.

THE SHADED BAND IS THE PRODUCTION AUGMENTATION RANGE, and the caption says it is also
how CHILI is augmented. That is what makes it the region that explains the downstream
results rather than a training detail: inside it, the figure is the mechanism behind the
shifted-eval numbers; outside it, the figure makes a separate and weaker claim about
generalizing past what was trained on.

No random-init series: see `analysis/invariance_decay.py` on why input space is the
baseline that separates "discarded that axis" from "the axis never moved the signal".
Light mode only — this renders a static PNG for a paper, not a themed page.
"""

from __future__ import annotations

import argparse
import json
from fnmatch import fnmatch
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Categorical slots 1 and 2 of the validated default palette, in fixed order and the
#: same assignment `analysis/plot_latent_theta.py` uses.
LATENT_COLOUR = "#2a78d6"
INPUT_COLOUR = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"

SERIES = (("latent", "Latent space", LATENT_COLOUR),
          ("input", "Input space — raw G(r)", INPUT_COLOUR))

#: Cosine distance is non-negative, but a perfectly invariant rung lands at ~1e-16 and
#: float noise can put it marginally below zero. Clip before the log transform rather
#: than letting matplotlib drop the point silently.
FLOOR = 1e-6

LABELS = {"uiso": "$U_{iso}$  (Å²)", "qmax": "$Q_{max}$  (Å⁻¹)",
          "qbroad": "$Q_{broad}$", "qmin": "$Q_{min}$  (Å⁻¹)"}


def draw(rec: dict, out_path: Path) -> None:
    params = list(rec["parameters"])
    fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params), 3.4),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, p in zip(axes, params):
        d = rec["parameters"][p]
        x = np.array(d["levels"])
        lo, hi = d["production_range"]
        ax.axvspan(lo, hi, color="#4a9d6a", alpha=0.10, linewidth=0, zorder=0)

        for key, label, colour in SERIES:
            q = d[key]
            p25, p50, p75 = (np.clip(np.array(q[k]), FLOOR, None) for k in ("p25", "p50", "p75"))
            ax.fill_between(x, p25, p75, color=colour, alpha=0.18, linewidth=0)
            ax.plot(x, p50, color=colour, linewidth=1.6, label=label, zorder=3)

        for key, style in (("within_material", (0, (4, 3))), ("between_material", (0, (1, 2)))):
            ax.axhline(d["reference_latent"][key], color=MUTED, linewidth=1.0,
                       linestyle=style, zorder=1)

        ax.set_yscale("log")
        ax.set_xlabel(LABELS.get(p, p), fontsize=9, color=INK)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c3")

    axes[0].set_ylabel("cosine displacement from the in-range mean", fontsize=9, color=MUTED)

    # Reference lines are identical across panels (shared y), so they are labelled once,
    # on the last panel, rather than repeated four times or pushed into the legend where
    # they would read as two more series.
    last, d = axes[-1], rec["parameters"][params[-1]]
    for key, text in (("within_material", "two views of one material"),
                      ("between_material", "two different materials")):
        y = d["reference_latent"][key]
        last.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"),
                      xytext=(4, 0), textcoords="offset points",
                      fontsize=7.5, color=MUTED, va="center", annotation_clip=False)

    # Inside the first panel, not a figure-level legend: `loc="outside upper left"`
    # reserves the same strip the suptitle occupies and the two overlap. The first
    # panel's lower-left is empty on a log axis — every curve turns up long before
    # reaching it — so the legend costs no data.
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK,
                   handlelength=1.7, borderaxespad=0.6)
    fig.suptitle("Sweeping the instrument barely moves the representation",
                 fontsize=12, color=INK, ha="left", x=0.005, y=1.04)

    fig.text(0.005, -0.10,
             f"{rec['n_materials']} materials from {rec['registry']}, the corpus this encoder "
             f"pretrained on. Line = median over materials, band = interquartile range.\n"
             f"Shaded = the augmentation range used in pretraining, which is also how the "
             f"downstream CHILI evaluation set is shifted. Encoder: {rec['run']} (n=1).",
             fontsize=7.5, color=MUTED, ha="left")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs/pdf/sweep")
    parser.add_argument("--filter", default="*", help="glob over run-dir names")
    parser.add_argument("--out", default="figures/invariance_decay.png")
    args = parser.parse_args()

    found = [p for p in sorted(Path(args.runs).glob("*/invariance_decay.json"))
             if fnmatch(p.parent.name, args.filter)]
    if not found:
        raise SystemExit("no invariance_decay.json found — run `python -m analysis.invariance_decay`")
    if len(found) > 1:
        raise SystemExit(f"{len(found)} runs match {args.filter!r}; this figure draws ONE encoder "
                         f"— narrow the filter:\n  " + "\n  ".join(p.parent.name for p in found))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    draw(json.loads(found[0].read_text()), out)


if __name__ == "__main__":
    main()
