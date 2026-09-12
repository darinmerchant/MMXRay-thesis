"""analysis/plot_invariance_hist.py — where view-invariance comes from, as histograms.

    python -m analysis.plot_invariance_hist \
        --run runs/pdf/sweep/2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372 \
        --block data/banked/material_registry__n32__55ad1ff1

One encoder, two rows. The TOP row shows what the measurement is taken over: one
example positive pair (two views of one material, staggered traces) and one example
negative pair (two different materials), each the pair whose input-space cosine
distance sits nearest the median of its arm — representative by construction, not
picked by eye. The BOTTOM row is three panels, each a pair of cosine-distance
histograms over the SAME materials and the SAME banked view pairs. Panel 1 measures
the signals the encoder is fed, panel 2 a random-init copy of the encoder, panel 3
the trained checkpoint. The figure's claim is the Q4 finding in distribution form:
invariance is learned, not architectural — the random encoder separates positive from
negative pairs only about as well as the raw signal does, the trained one an order of
magnitude further. (Two-row layout, positive/negative nomenclature, one color pair
across the three dataset figures, and no in-figure AUROC: 2026-09-09, Darin's
redesign — the overlap statistics are still computed and printed, the text quotes
them, the figure carries the distributions.)

WHAT IS MEASURED, exactly:
  positive pair   `1 - cos(A_i, B_i)`, one distance per material — A/B are the banked
                  view pair `analysis.latent.load_views` draws, seeded per material by
                  `global_index`, so this figure and the Q4 scalars see identical pairs.
  negative pair   `1 - cos(A_i, A_j)` over ALL i<j pairs of view A — view A alone, so
                  every distance crosses materials and never views, the same convention
                  `analysis/latent.py` uses for uniformity. With n=4096 that is ~8.4M
                  pairs; the count mismatch is handled by plotting each histogram as
                  fraction-of-pairs, not raw counts.

On the unit sphere `1 - cos = ||u - v||^2 / 2`, so panel 3's two histograms are the
distributions whose means the Q4 table reports as alignment and (via the Gaussian
kernel) uniformity — this figure adds no new quantity, it un-collapses two expectations.

THE X AXIS IS LOG AND SHARED, AND THE RATIO IS DRAWN, NOT INFERRED. Two earlier cuts
each hid half the result. A bare log axis resolves every panel (PDF medians, centered:
9.2e-2/1.01 input, 7.0e-2/1.00 random, 8.6e-3/1.02 trained) but renders the trained
encoder's 118x separation as a modest sideways shift — panel 3 read as panel 1 moved
left. A linear axis makes panel 3 dramatic but squashes the positive-pair tails that
carry the comparison. The keep-both answer: log axis for the structure, plus a
bracket in each panel spanning the two medians, labelled with their ratio — on a log
axis that bracket's LENGTH is the log-ratio, so "x118 dwarfs x11" is geometry rather
than arithmetic left to the reader (the arrow is dropped under half a decade, where
its heads overlap into an ✗). Distances are computed in float64 and floored at FLOOR,
where float noise can put a true 0 marginally negative.

Representation is `h`, never `z` — the repo-wide convention (`h` is what every probe and
finetune reads). VICReg enforces invariance on `z`, so panel 3 UNDERSTATES the trained
separation; that is the conservative direction. Panel 1 is the NORMALIZED signal,
exactly what `encode()` is fed (the read-path convention), not the raw physical one.
Signal normalization makes every entry non-negative, and a ReLU encoder's `h` is
non-negative too — which is why every distance here is computed on mean-centered
rows (see `cosine_distances`): uncentered, the shared positive mean packed each
panel into one cone and the random-init panel read as collapse (docs/TRAPS.md,
2026-09-03; the uncentered medians live in RESULTS.md Q4).

BOTH MODALITIES RUN THROUGH THIS FILE, and the only things that differ are the panel-1
label and the example row's axes — the measurement is a cosine distance between two
rows and does not care what the rows mean. Which normalization those rows carry is
`analysis.latent.load_views`'s job and is dispatched on the block's `modality` (min-max
for PDF, max-norm for XRD); this module must not second-guess it, so the labels are
keyed by the block's `modality`, never inferred from the run name. XRD panel 1 is
therefore max-normalized I(2θ) on the [10, 80]° grid.

Inherited caveats, all from `analysis/latent.py` and worth a sentence in the paper's
caption: measured on training data (no held-out MP exists); the sample comes from
whichever contiguous shards are on disk, so it is biased by registry ordering; this
checkpoint pretrained on full-MP and is measured here on an MP-20 subset of its
corpus; the random-init control runs under `.eval()` with initial BatchNorm stats (an
untrained encoder under the same protocol, not a batch-statistics estimate of one);
n=1, no pretraining seeds.

Compute and plot live in one script, unlike the aug_sweep / invariance_decay split:
those bank numbers that many figures re-read, while everything here re-derives from
checkpoint + block + seeds in under a minute, so a cached intermediate would only be
one more file to keep honest.

`import torch` comes first on purpose: importing pandas or `core.*` ahead of it
segfaults with no traceback (exit 139) — see docs/ENVIRONMENT.md.
"""

from __future__ import annotations

import torch  # MUST precede pandas / core.* — see module docstring

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analysis import paperstyle as ps
from analysis.downstream_eval import embed, load_encoder
from analysis.latent import load_views
from core.train import build_model, pick_device

__all__ = ["cosine_distances", "median_examples", "draw_example"]

#: The one pair-type color key of the three invariance figures — see paperstyle.
POS_COLOR, NEG_COLOR = ps.PAIR_COLORS["pos"], ps.PAIR_COLORS["neg"]
POS_TINT, NEG_TINT = ps.PAIR_TINTS["pos"], ps.PAIR_TINTS["neg"]

#: Vertical stagger between the two traces of an example pair, in units of the
#: normalized signal — above 1.0 so the traces never touch (1.1 leaves a 0.1 gap;
#: 0.7 overlapped, revised same day). Staggered rather than overlaid (2026-09-09,
#: Darin): overlaid, the positive pair's second view hid under the first — which
#: is the finding, but not a visible drawing of it.
STAGGER = 1.1

#: Below this a cosine distance is "identical at input precision" (the banked signals
#: are float32); float error can also put a true 0 marginally negative. Clipped into
#: the leftmost log bin rather than silently dropped by plt.hist.
FLOOR = 1e-7

#: Panel-1 title per modality — what `encode()` is actually fed. Keyed by the block's
#: own `modality`, so a mislabelled figure would need a mis-banked block.
#: 2026-09-03: em dashes out and the thesis caption register in (chapter 4 rule),
#: so the panel title and the caption below it name the same three states.
INPUT_LABEL = {"pdf": "Input signal, G(r)", "xrd": "Input signal, I(2θ)"}

#: The example row's (x label, y label) per modality — the axes the traces live on.
EXAMPLE_AXIS = {"pdf": ("r (Å)", "G(r), normalized"),
                "xrd": ("2θ (deg)", "I(2θ), normalized")}


def _style() -> None:
    """The plot_pdf_aug_effect.py style: large type, editable text, no chartjunk."""
    plt.rcParams.update({
        "pdf.fonttype": 42,          # TrueType, not outlines — stays editable in Illustrator
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    })


def cosine_distances(A: torch.Tensor, B: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Positive-pair and negative-pair cosine distances for one representation.

    `A`, `B` are `(n, d)` — the two views of each of n materials. Returns
    `(same, diff)`: `1 - cos(A_i, B_i)` per material (length n), and
    `1 - cos(A_i, A_j)` over all i<j (length n(n-1)/2), clipped at FLOOR.

    Each view set is mean-centered by its own sample mean before normalization,
    the same map `analysis.latent._unit` applies (2026-09-03) — without it a ReLU
    encoder's shared positive mean packs every row into one narrow cone and the
    random-init panel reads as collapse. Centering keeps the distances invariant
    to scaling and to a constant offset of the representation.
    """
    A64, B64 = A.to(torch.float64), B.to(torch.float64)
    UA = torch.nn.functional.normalize(A64 - A64.mean(dim=0), dim=1)
    UB = torch.nn.functional.normalize(B64 - B64.mean(dim=0), dim=1)
    same = 1.0 - (UA * UB).sum(dim=1)
    iu = torch.triu_indices(len(UA), len(UA), offset=1)
    diff = 1.0 - (UA @ UA.T)[iu[0], iu[1]]
    return (np.maximum(same.numpy(), FLOOR), np.maximum(diff.numpy(), FLOOR))


def median_examples(A: torch.Tensor, B: torch.Tensor) -> dict:
    """The two example pairs the top row draws, picked at the median of each arm.

    Positive: the material whose two-view input distance is nearest the median of
    the positive arm — traces `(A_i, B_i)`. Negative: the (i, j) pair of view-A
    signals nearest the median of the negative arm — view A alone, the same
    convention `cosine_distances` measures.
    """
    same, diff = cosine_distances(A, B)
    i = int(np.argmin(np.abs(same - np.median(same))))
    iu = torch.triu_indices(len(A), len(A), offset=1)
    k = int(np.argmin(np.abs(diff - np.median(diff))))
    j1, j2 = int(iu[0, k]), int(iu[1, k])
    return {
        "pos": [A[i].numpy(), B[i].numpy()],
        "neg": [A[j1].numpy(), A[j2].numpy()],
        "pos_dist": float(same[i]), "neg_dist": float(diff[k]),
    }


def pair_stats(same: np.ndarray, diff: np.ndarray) -> dict:
    """Separation of the two distance distributions, beyond the median ratio.

    Computed and printed, NOT drawn (2026-09-09) — the text quotes these, the
    figure carries the distributions.

    auroc      probability a random negative-pair distance exceeds a random
               positive-pair one (ties counted half) — threshold-free, 1.0 is
               perfect separation. Computed exactly by binary search on the sorted
               diff arm, not by sampling.
    tail_frac  fraction of positive pairs whose distance exceeds the 10th
               percentile of the negative-pair distances — the overlap the
               median ratio cannot see: how much of the positive-pair mass sits
               inside the bulk of the negative-pair distribution. 0 is perfect.
    """
    d = np.sort(diff)
    lo = np.searchsorted(d, same, side="left")
    hi = np.searchsorted(d, same, side="right")
    auroc = float((len(d) - hi + 0.5 * (hi - lo)).mean() / len(d))
    p10 = float(np.quantile(diff, 0.10))
    return {"auroc": auroc, "tail_frac": float((same > p10).mean()), "diff_p10": p10}


def draw_example(ax, x: np.ndarray, traces: list, dist: float,
                 color: str, tint: str, title: str, *, paper: bool = False) -> None:
    """One example pair: two staggered traces in one hue, the second in its lighter
    tint, with the pair's cosine distance — the number the bottom row's x axis
    measures — in the corner."""
    trace_lw, note_size = (0.7, 7) if paper else (1.6, 15)
    ax.plot(x, np.asarray(traces[1]) - STAGGER, color=tint, lw=trace_lw)
    ax.plot(x, np.asarray(traces[0]), color=color, lw=trace_lw)
    ax.annotate(f"cosine distance {dist:.2f}", xy=(0.98, 0.97),
                xycoords="axes fraction", ha="right", va="top",
                fontsize=note_size, color=ps.INK if paper else "#0b0b0b")
    ax.set_xlim(x[0], x[-1])
    ax.set_yticks([])
    ax.set_title(title, loc="left", pad=4)
    ax.spines[["top", "right"]].set_visible(False)


def _bracket(ax, same: np.ndarray, diff: np.ndarray, *, paper: bool = False) -> None:
    """Median ticks and a <-> arrow between them, labelled with their ratio.

    Everything sits in the top ~15% of the axes, which `draw` reserves as headroom —
    at data heights the random panel's peak would collide with the arrow.
    """
    ms, md = float(np.median(same)), float(np.median(diff))
    tick_lw, arrow_lw, label_size, head = (1.2, 0.9, 8, 4.5) if paper else (2.2, 1.8, 22, 11)
    for m, color in ((ms, POS_COLOR), (md, NEG_COLOR)):
        ax.axvline(m, ymin=0.85, ymax=0.92, color=color, linewidth=tick_lw)
    # Every panel gets its arrow (Darin, 2026-09-09 — an earlier cut dropped it
    # under half a decade). What keeps a short arrow from collapsing into an ✗ of
    # overlapped heads is the small mutation_scale, not a drop rule.
    ax.annotate("", xy=(md, 0.885), xytext=(ms, 0.885),
                xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="<->", linewidth=arrow_lw, color="#0b0b0b",
                                mutation_scale=head, shrinkA=0, shrinkB=0))
    r = md / ms
    ax.annotate(f"$\\times${r:.0f}" if r >= 10 else f"$\\times${r:.1f}",
                xy=(np.sqrt(ms * md), 0.925), xycoords=("data", "axes fraction"),
                ha="center", fontsize=label_size)


def draw(panels: list[tuple[str, np.ndarray, np.ndarray]], out_path: Path,
         *, examples: dict, x: np.ndarray, modality: str, paper: bool = False) -> None:
    """The two-row figure: example pairs on top, the distance histograms below."""
    lo = min(v.min() for _, s, d in panels for v in (s, d))
    hi = max(v.max() for _, s, d in panels for v in (s, d))
    bins = np.geomspace(lo, hi, 61)

    if paper:
        # Thesis width via paperstyle (ps.use() ran in main).
        fig = plt.figure(figsize=(ps.WIDTH, 3.6))
        hist_lw, title_size = 1.2, 8.5
    else:
        # Scaled alongside the aug_effect canvas (8x5 per histogram panel).
        fig = plt.figure(figsize=(8.0 * len(panels), 11.0))
        hist_lw, title_size = 2.0, None
    # The middle gridspec row is a SPACER: tight_layout only guarantees decorations
    # don't collide, and with the x-axis labels alone the two rows sat nearly flush.
    gs = fig.add_gridspec(3, 2 * len(panels), height_ratios=(0.85, 0.14, 1.0))
    half = len(panels)
    ex_axes = [fig.add_subplot(gs[0, :half]), fig.add_subplot(gs[0, half:])]
    # The histogram panels SHARE the y axis and only the left one is labeled:
    # per-panel autoscale put three different "fraction of pairs" scales side by
    # side, and the middle panels' tick labels collided with their neighbors.
    axes = [fig.add_subplot(gs[2, 0:2])]
    axes += [fig.add_subplot(gs[2, 2 * i:2 * i + 2], sharey=axes[0])
             for i in range(1, len(panels))]

    xlabel, ylabel = EXAMPLE_AXIS[modality]
    draw_example(ex_axes[0], x, examples["pos"], examples["pos_dist"],
                 POS_COLOR, POS_TINT, "positive pair", paper=paper)
    draw_example(ex_axes[1], x, examples["neg"], examples["neg_dist"],
                 NEG_COLOR, NEG_TINT, "negative pair", paper=paper)
    for ax in ex_axes:
        ax.set_xlabel(xlabel)
        if paper:
            ax.tick_params(axis="both", length=0)
    ex_axes[0].set_ylabel(ylabel)

    for ax, (title, same, diff) in zip(axes, panels):
        for vals, color in ((diff, NEG_COLOR), (same, POS_COLOR)):
            w = np.full(len(vals), 1.0 / len(vals))
            ax.hist(vals, bins=bins, weights=w, histtype="stepfilled",
                    color=color, alpha=0.25, linewidth=0)
            ax.hist(vals, bins=bins, weights=w, histtype="step",
                    color=color, linewidth=hist_lw)

        _bracket(ax, same, diff, paper=paper)
        ax.set_xscale("log")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel("cosine distance")
        ax.spines[["top", "right"]].set_visible(False)
        if paper:
            ax.tick_params(axis="both", length=0)
            ax.xaxis.grid(False)

    # Headroom for the bracket band, ONCE, after every panel has autoscaled the
    # shared axis — inside the loop each panel would restretch the others' 1.5x.
    axes[0].set_ylim(0.0, axes[0].get_ylim()[1] * 1.5)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].set_ylabel("fraction of pairs")
    # FILLED swatches, matching plot_pdf_invariance's legend — the step-histogram
    # handles rendered as hollow rectangles and the two figures disagreed.
    from matplotlib.patches import Patch
    swatch = lambda c: Patch(facecolor=c, edgecolor=c, alpha=0.35, lw=1.1)
    axes[0].legend([swatch(POS_COLOR), swatch(NEG_COLOR)],
                   ["positive pairs", "negative pairs"], frameon=False,
                   loc="center left", handlelength=1.7,
                   **({"fontsize": 6.5} if paper else {}))

    if paper:
        ps.layout(fig)  # without it the default margins shrink the ink well below WIDTH
        ps.save(fig, out_path.stem)
    else:
        fig.tight_layout()
        fig.savefig(out_path)
        print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run dir whose ckpt_best.pt is drawn")
    parser.add_argument("--block", required=True,
                        help="banked block dir; whichever shard_*.npy are present are used")
    parser.add_argument("--n-materials", type=int, default=4096,
                        help="materials sampled from the shards present; the Q4 sample")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--control-seed", type=int, default=0,
                        help="torch seed for the random-init control weights")
    parser.add_argument("--device", default="auto", help='"auto" | "cpu" | "cuda"')
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default="figures/invariance_hist.pdf")
    parser.add_argument("--paper", action="store_true",
                        help="thesis render: paperstyle sizing, ps.save under figures/ "
                             "with --out's stem; without it, the diagnostic canvas")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, cfg, _ = load_encoder(Path(args.run) / "ckpt_best.pt", device)
    A, B, info = load_views(args.block, args.n_materials, args.sample_seed)
    print(f"{info['block']}: shards {info['shards_present']} -> {info['n_materials']} of "
          f"{info['n_materials_available']} materials, encoder {cfg.encoder} on {device}",
          flush=True)

    torch.manual_seed(args.control_seed)
    control = build_model(cfg).to(device).eval()
    for p in control.parameters():
        p.requires_grad_(False)

    modality = info.get("modality", "pdf")
    if modality not in INPUT_LABEL:
        raise SystemExit(f"{info['block']}: modality {modality!r} has no panel-1 label — "
                         f"add one to INPUT_LABEL rather than drawing an unlabeled axis")

    reps = [
        (INPUT_LABEL[modality], A, B),
        ("Untrained encoder", embed(control, A, device, args.batch_size),
         embed(control, B, device, args.batch_size)),
        ("Pretrained encoder", embed(model, A, device, args.batch_size),
         embed(model, B, device, args.batch_size)),
    ]
    panels = [(title, *cosine_distances(HA, HB)) for title, HA, HB in reps]
    examples = median_examples(A, B)

    # Effective rank on view A of each representation — latent.py's pinned estimator
    # (participation ratio over covariance eigenvalues, centered unnormalized rows),
    # so this column and the Q4 rank table cannot use two definitions of "rank".
    from analysis.latent import participation_ratio, spectrum
    for (title, HA, _HB), (_t, same, diff) in zip(reps, panels):
        st = pair_stats(same, diff)
        print(f"  {title:<24} median same {np.median(same):.2e}  "
              f"median diff {np.median(diff):.2e}  ratio {np.median(diff) / np.median(same):.1f}x  "
              f"AUROC {st['auroc']:.4f}  same>p10(diff) {100 * st['tail_frac']:.1f}%  "
              f"eff.rank {participation_ratio(spectrum(HA)):.1f}", flush=True)

    if args.paper:
        ps.use()
    else:
        _style()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    draw(panels, out, examples=examples, x=np.load(Path(args.block) / "r_axis.npy"),
         modality=modality, paper=args.paper)


if __name__ == "__main__":
    main()
