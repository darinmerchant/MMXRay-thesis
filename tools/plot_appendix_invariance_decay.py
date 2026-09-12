"""tools/plot_appendix_invariance_decay.py — appendix A.6.1: displacement under a sweep.

    /opt/anaconda3/envs/mmxray/bin/python -m tools.plot_appendix_invariance_decay

Four panels in a 2x2 GRID, one per instrument parameter, over the 16-rung ladders
`analysis/invariance_decay.py` measured for `cnn_vicreg_mpfull_final` — the
checkpoint every headline figure uses. Each panel draws the cosine displacement of
h from that structure's mean embedding over the production range, against the same
measurement on min-max-normalized G(r), with the production range shaded — and EACH
CURVE IS DIVIDED BY ITS OWN SPACE'S NEGATIVE-PAIR DISTANCE, so the y axis reads
"fraction of the way to a different structure" in both spaces at once.

THE NORMALIZATION IS THE POINT OF THIS REVISION (2026-08-27), not a styling choice.
The raw version compared cosine distances across two spaces with different global
scales — latent negative pair 0.072 against input 0.134, positive pairs 0.00099
against 0.0155 — so "the blue curve rides below the orange one" conflated the claim
being made (the encoder is invariant) with a fact about the embedding (the latent
space is globally tighter). Dividing each curve by its own space's identity scale
makes the two commensurable for the first time:

  - 1.0 is "as far as a different structure", the SAME ceiling in both spaces, drawn
    once in REFLINE ink;
  - each space keeps its own positive-pair floor (latent ~0.014, input ~0.115 of its
    identity scale), so the floors are now two lines, not one;
  - the in-range latent advantage reads as the defensible 5-19x, not the raw 9-35x
    that normalizing by the positive pair would collapse to ~0.6-2.3x;
  - the out-of-range inversion STRENGTHENS: at qmax = 5 the latent curve reaches 0.92
    of identity against the input curve's 0.29.

THE POSITIVE-PAIR FLOORS TAKE THE SERIES HUE, a deliberate deviation from the
references-take-REFLINE rule (`analysis/paperstyle.py`). There are two floors, one
per space, and REFLINE ink on both could not say which belongs to which curve — the
hue is doing identification here, not decoration. They stay dashed and thin so they
cannot be mistaken for a third and fourth series, and the one reference that really
is shared — identity at 1.0 — keeps REFLINE.

THE BAND IS +-1 GEOMETRIC STANDARD DEVIATION, i.e. `median x/ gsd` where
`gsd = exp(sd(log d))`, and BOTH halves of that are deliberate.

  - **Geometric, not arithmetic.** `mean - sd` is measured negative on 13 of 16 `qmin`
    latent rungs and 12 of 16 `qmin` input rungs (`analysis/invariance_decay.py`
    records `mean`/`sd` so this is checkable, not asserted). A distance cannot be
    negative and a log axis cannot draw it, so an arithmetic band would clip to the
    floor on the panel carrying the out-of-range inversion and read as five decades of
    spread that are not there. The geometric band is multiplicative, cannot reach zero,
    and is symmetric on the axis these distances are actually spread evenly on.
    Dividing by the per-space constant commutes with `x/ gsd`, so the band is
    normalized by normalizing its centre — `gsd` itself is already dimensionless.
  - **Centred on the MEDIAN, not on the geometric mean it was computed about.** That
    costs at most 6% — measured `gmean / median` runs 0.945 to 1.06 across all eight
    curves, which is ~0.025 of a decade and invisible here — and it buys the figure's
    line being EXACTLY the number A.6.1's table and RESULTS.md quote. One centre
    through the whole thesis is worth more than the third decimal of an estimator name.

⚠️ `analysis/plot_invariance_decay.py` — the 22-checkpoint diagnostic over the same
JSONs — STILL DRAWS INTERQUARTILE. It is left alone deliberately: it is not a paper
figure, changing it would silently restate every archived summary panel, and its own
docstring argues the IQR case. If the two are ever shown side by side, their bands mean
different things.

READS THE JSON, MEASURES NOTHING. `<run>/invariance_decay.json` already holds the
quantile curves; this module is the appendix-styled twin of
`analysis/plot_invariance_decay.py` and shares its input exactly, so the two cannot
disagree about a number. The ladder lives at `runs/pdf/latent/invariance` (the
JSON's recorded `runs_latent` path is where it sat when first measured) and is not
needed here — which is the whole point of the measure/plot split this repo keeps
everywhere.

THE Y AXIS IS LOG, and it is what makes the figure work. In-range values run 1e-4
to 2e-2 of the identity scale while the out-of-range tail reaches 0.9 of it: on a
linear axis the entire in-range result — the half A.6.1 actually licenses — collapses
onto zero, and the panel spends its whole height on the extrapolation tail. Log keeps
both on one axis with no broken scale, no inset, no second row.

PANELS SHARE THE Y AXIS on purpose. The four parameters move the representation by
very different amounts; per-panel autoscaling would draw them as four similar-looking
curves and hide the comparison the figure exists to make.

2x2, NOT 1x4 — it is an appendix figure and page height is the cheap axis here. A row
of four gives each panel ~1.1 in, which is narrower than its own x tick labels want
(see TICKS) and squeezes the log decades into a strip; the grid doubles panel width for
one extra row of height, and is the shape `plot_appendix_training` and `..._labels`
already use. sharey still spans all four, not just each row.

THE TWO REFERENCE LINES ARE KEYED IN THE LEGEND, NOT ANNOTATED BESIDE THEM. The
diagnostic version writes them past the right spine, which is exactly the decoration
overhang `paperstyle.save` warns about — it widens the tight crop and rescales every
type size in LaTeX. `plot_chili_four_arms` solved the same problem the same way.

THE REFERENCE KEYS CARRY NO NUMBERS, by request — they name the pair types and the
axis supplies the value. One residual trap: every reference here is
`invariance_decay.json`'s own draw (positive 0.00099 latent / 0.0155 input, negative
0.072 / 0.134), NOT `latent_theta.json`'s 0.00085 / 0.0686 "81x" pair that the drafted
A.6.1 prose quotes — a different estimator on a different sample. A caption quoting
81x beside this figure describes lines other than the ones drawn.

POSITIVE / NEGATIVE PAIR, not within-/between-material. These are the quantities the
contrastive objective is defined on — the positive pair is literally what VICReg pulls
together — so naming them after the loss says why they are the right reference scales,
where "within-material" only says how they were computed. `analysis/invariance_decay.py`
keeps the computational names in its JSON keys; the rename is figure-facing only.

`between_material` differs by panel in the 4th decimal (0.07244-0.07258) because it is
re-drawn per ladder; `within_material` is identical across all four, since one
`refs.npz` serves every parameter. Each panel draws its own value.

NO SUPTITLE AND NO CAPTION BLOCK — the appendix set carries its caption in LaTeX. The
provenance the diagnostic version prints into the figure goes to stdout instead, which
is `plot_appendix_training`'s convention.

Reads only json + numpy + matplotlib — none of the torch-before-pandas import trap
applies (docs/TRAPS.md). Runtime ~1 s.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from analysis import paperstyle as ps

REPO = Path(__file__).resolve().parents[1]

#: The pretraining run behind every headline figure (RESULTS.md), and the one A.6.1
#: names. Its `invariance_decay.json` was written by the s36 all-checkpoint pass.
DEFAULT_RUN = REPO / "runs" / "pdf" / "sweep" / "2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372"

#: Cosine distance is non-negative, but a perfectly invariant rung lands near float
#: noise and can come out marginally below zero. Clip before the log transform rather
#: than letting matplotlib drop the point without saying so.
FLOOR = 1e-6

SERIES = (("latent", "latent space, $h$", ps.SPACE_COLORS["latent"]),
          ("input", r"input space, $\nu(G(r))$", ps.SPACE_COLORS["input"]))

#: Panel order follows A.6.1's table, not the JSON's insertion order, so a reader
#: moving between the two never has to re-find a parameter.
PARAMS = ("uiso", "qmax", "qbroad", "qmin")

LABELS = {"uiso": "$U_{iso}$ (Å²)", "qmax": "$Q_{max}$ (Å⁻¹)",
          "qbroad": "$Q_{broad}$", "qmin": "$Q_{min}$ (Å⁻¹)"}

#: X ticks are SPELLED OUT, not left to a locator. This is what forced the 2x2: in a
#: 1x4 row `MaxNLocator(4)` picked 0.025/0.050/0.075/0.100 for `uiso`, four 5-character
#: labels overlapping into a smear at 1.1 in of panel. The grid's ~2.4 in panels would
#: survive the locator, but the count that fits still depends on label width — which no
#: locator setting expresses — so the ticks stay chosen rather than negotiated.
#: `qbroad` stops at 0.10 though its ladder runs to 0.15: in the grid the left column's
#: last tick label and the right column's first sit ~0.02 in apart, and `0.15` against
#: `qmin`'s `0` touched. Dropping a label at the axis END costs nothing a reader needs;
#: dropping `qmin`'s `0` would cost the lower bound of a range that starts there.
TICKS = {"uiso": [0.0, 0.05, 0.10], "qmax": [10, 20, 30, 40],
         "qbroad": [0.0, 0.05, 0.10], "qmin": [0, 1, 2, 3, 4]}

#: Dash patterns for the two reference lines. Both take REFLINE ink: neither is a
#: series, and the house rule is that a reference never spends a hue.
REF_STYLE = {"within_material": (0, (4, 3)), "between_material": (0, (1, 2))}


def build(run_dir: Path, out_stem: str, sub: str | None) -> None:
    ps.use()
    rec = json.loads((run_dir / "invariance_decay.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(ps.WIDTH, 3.5), sharey=True)
    flat = axes.ravel()
    handles: list = []

    for ax, p in zip(flat, PARAMS):
        d = rec["parameters"][p]
        x = np.array(d["levels"])
        lo, hi = d["production_range"]
        ax.axvspan(lo, hi, color=ps.GRID, zorder=0)

        for key, label, colour in SERIES:
            q, refs = d[key], d[f"reference_{key}"]
            neg = refs["between_material"]
            p50 = np.clip(np.array(q["p50"]), FLOOR, None) / neg
            gsd = np.array(q["gsd"])
            ax.fill_between(x, p50 / gsd, p50 * gsd, color=colour, alpha=0.20, linewidth=0)
            line, = ax.plot(x, p50, color=colour, linewidth=1.2, zorder=3, label=label)
            ax.axhline(refs["within_material"] / neg, color=colour, linewidth=0.9,
                       linestyle=REF_STYLE["within_material"], zorder=1)
            if ax is flat[0]:
                handles.append(line)

        # Identity: both spaces' negative pairs land here by construction, so the
        # ceiling really is one shared line, and it keeps REFLINE.
        ax.axhline(1.0, color=ps.REFLINE, linewidth=0.9,
                   linestyle=REF_STYLE["between_material"], zorder=1)

        ax.set_yscale("log")
        ax.set_xlabel(LABELS[p])
        ax.set_xlim(x.min(), x.max())
        ax.set_xticks(TICKS[p])

        in_range = np.array(d["in_range"], dtype=bool)
        lat = np.array(d["latent"]["p50"]) / d["reference_latent"]["between_material"]
        inp = np.array(d["input"]["p50"]) / d["reference_input"]["between_material"]
        print(f"  {p:7} of identity: latent in {lat[in_range].max():.4f} "
              f"out {lat[~in_range].max():.3f} | input in {inp[in_range].max():.4f} "
              f"out {inp[~in_range].max():.3f} | in-range latent advantage "
              f"{inp[in_range].max() / lat[in_range].max():.1f}x")

    # The band is load-bearing — every claim in A.6.1 is "inside it" or "outside it" —
    # so it takes a legend key rather than living only in the caption.
    band = plt.Rectangle((0, 0), 1, 1, facecolor=ps.GRID, edgecolor="none",
                         label="pretraining augmentation range")

    # One tuple handle for the two per-space floors: the legend swatch shows both
    # dashes side by side, which is the only honest key for a reference that exists
    # once per space rather than once per figure.
    handles += [
        tuple(plt.Line2D([], [], color=colour, lw=0.9, ls=REF_STYLE["within_material"])
              for _, _, colour in SERIES),
        plt.Line2D([], [], color=ps.REFLINE, lw=0.9, ls=REF_STYLE["between_material"],
                   label="negative pair"),
        band,
    ]
    labels = ["positive pair, each space" if isinstance(h, tuple) else h.get_label()
              for h in handles]

    flat[0].set_ylim(3e-5, 2.2)
    # The y ticks are chosen, not left to the log locator, for one reason: the
    # locator labels the odd decades and skips 10^0, and 10^0 is the entire point
    # of the normalized axis — the negative pair sits at exactly 1 by construction.
    # Even decades keep the same label density and put "1" on the axis, written as
    # "1" rather than "10^0" so the dotted line visibly IS the unit.
    flat[0].set_yticks([1e-4, 1e-2, 1], ["$10^{-4}$", "$10^{-2}$", "1"])
    # supylabel, not a per-axis label: sharey spans all four panels, so the quantity
    # belongs to the figure rather than to the left column that happens to carry the
    # tick labels.
    # "negative pair" is a defined key one inch below the label, so the label can
    # lean on it: the full quantity ("cosine displacement ÷ negative-pair distance")
    # is spelled once, in the caption, not down the whole left edge.
    fig.supylabel("displacement / negative pair", x=0.015)

    # `ncol=3` fills COLUMN-MAJOR, so this handle order lands as
    # (spaces | pair references | band) — three groups, two rows, one read.
    # The anchor and the rect reserve must name the SAME band. Anchoring below the
    # canvas (y < 0) while `rect` also reserves a strip above it stacks two gaps: the
    # reserve empties a band the legend then declines to sit in, and `save` crops to
    # the legend anyway, so the blank ships. Both are 0.11 here.
    # columnspacing 1.1: the tuple handle widened the middle column and the whole
    # row must clear WIDTH, or `save`'s crop reports an overfull \hbox.
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=3,
               handlelength=1.6, columnspacing=0.9, borderaxespad=0.0,
               bbox_to_anchor=(0.5, 0.005),
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})
    ps.layout(fig, rect=(0.015, 0.11, 1, 1))
    ps.save(fig, out_stem, sub=sub)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN,
                    help="run directory holding invariance_decay.json")
    ap.add_argument("--out", default="appendix_invariance_decay")
    ap.add_argument("--sub", default="appendix", help="subdirectory under figures/")
    args = ap.parse_args()
    print(f"  {args.run.name}, epoch {json.loads((args.run / 'invariance_decay.json').read_text())['epoch']}")
    build(args.run, args.out, args.sub)


if __name__ == "__main__":
    main()
