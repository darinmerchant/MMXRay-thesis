"""PDF paper, FIGURE 2 — invariance as variance.

    python -m analysis.plot_pdf_invariance              # draws from the banked JSON
    python -m analysis.plot_pdf_invariance --recompute  # re-embeds first (~2.5 min, CPU)

THE CLAIM THIS FIGURE CARRIES. A positive pair — two views of one material drawn from
`PDF_RANGES` — lands in nearly the same place, while a negative pair — two different
materials — does not, and the gap is built by pretraining, not by the architecture.
All distances are on mean-centered rows (2026-09-03, via `cosine_distances` — see
docs/TRAPS.md on the uncentered mean artifact): the random-init encoder separates at
14×, about the input signal's own 11×, i.e. random features roughly preserve the
input's separation; the pretrained encoder separates at 118×, an order of magnitude
beyond both.

TWO ROWS (2026-09-09, Darin's redesign, applied to all three dataset figures). The
top row shows what the histograms measure: one example positive pair and one example
negative pair, as overlaid G(r) traces. Each example is the pair whose input-space
cosine distance sits NEAREST THE MEDIAN of its arm — representative by construction,
not picked by eye. The bottom row is the three distance histograms (input signal,
untrained encoder, pretrained encoder). The same redesign renamed the series to
positive/negative pairs, dropped the per-panel AUROC/tail annotation (the statistics
are still computed, banked and printed — they are quoted in the text, the figure
carries the distributions), and unified the colors: positive = blue, negative =
orange (`ps.SLOTS` 1-2), the same pair in all three dataset figures. The previous
cut colored each panel's histogram by the representation it measured, which was
redundant with the panel title and made the three figures read as three keys.

ONE CNN EXEMPLAR, `cnn_vicreg_mpfull_final` — the encoder that carries the thesis,
fixed rather than selected. (`cnn_vicreg_cov1` held the sweep's highest separation
under the pre-centering convention, 415×, and is deliberately NOT used: it is the run
with the known directional `np_size` failure, so drawing it would flatter the figure
with the one checkpoint whose geometry does not predict its transfer.)

⚠️ THE SECOND MEASUREMENT IS NO LONGER DRAWN HERE (2026-09-03, Darin's call). This
figure used to carry a second row: pooled out-of-fold R² of a linear head reading each
instrument parameter back out of the frozen embedding — the harder test, since a
nuisance can hide in a low-variance direction this figure would never see. That row
drew all 22 sweep checkpoints, which the thesis does not define, so it was first cut to
the carrier alone (`--carrier-only`) and then cut entirely. **The measurement still
stands and still belongs beside this figure**: it lives in `analysis/plot_theta_probe.py`
(all 22 checkpoints) and in the thesis as `tab:theta-decode` (the carrier's twelve
numbers). Its headline is that `qmin` is readable from the raw signal at 0.571, survives
an untrained encoder at 0.247, and is driven to zero by pretraining, while `uiso` is
kept — see `RESULTS.md` → Q4 → the θ-probe section.

⚠️ THE HEADLINE THE FIGURE DOES NOT SHOW, AND THE CAPTION MUST. Out of range the
invariance INVERTS — the encoder extrapolates worse than doing nothing (`qmin` 22/22
checkpoints, `qmax` 19/22, `uiso` 17/22, `qbroad` 3/22; RESULTS.md Q4 extension pt 4).
Everything here is IN-RANGE draws only. The one-sentence form is **you get the
invariance you simulate, exactly over the band you simulated it** — never "invariance
generalizes". `figures/invariance_decay.pdf` is the panel that shows it, and it was
left out of this figure by a scope decision, not because it is unresolved.

⚠️ MEASURED ON THE PRETRAINING CORPUS. MP splits are (1.0, 0.0, 0.0), so no held-out
MP materials exist and none of these distances is a generalization statement.

WHAT IS CACHED AND WHY. The figure re-embeds 4,096 materials through three encoders, a
~2.5-minute CPU job, and the layout was iterated on far more often than that. So the
compute banks what the figure actually draws — bin edges, per-panel weighted counts,
medians, and (since the 2026-09-09 redesign) the four example traces — into one JSON.
The distances themselves are NOT cached: the negative arm is ~8.4 M pairs per panel,
which is a 200 MB file to avoid a 150-second job. `--recompute` is the only way to
refresh the histograms, so a stale cache is a deliberate act rather than an accident.
The example traces need no encoder, so a cache written before the redesign is
back-filled in place on the next draw rather than forcing the full recompute.

Distances come from `analysis/plot_invariance_hist.cosine_distances`, imported rather
than re-derived, so the single-encoder figure and this one cannot drift apart.

`import torch` comes first on purpose: importing pandas or `core.*` ahead of it
segfaults with no traceback (exit 139) — see docs/ENVIRONMENT.md.
"""
from __future__ import annotations

import torch  # MUST precede pandas / core.* — see module docstring

import argparse
import json

import numpy as np

import matplotlib.pyplot as plt
from analysis import paperstyle as ps

CACHE = ps.REPO / "analysis/out/pdf_invariance_hist.json"

RUN = ps.REPO / "runs/pdf/sweep/2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372"
BLOCK = ps.REPO / "data/banked/material_registry__n32__55ad1ff1"
N_MATERIALS = 4096

#: One color per PAIR TYPE, identical across the three dataset figures — see
#: paperstyle. Tints are the second trace of each example pair.
POS_COLOR, POS_TINT = ps.PAIR_COLORS["pos"], ps.PAIR_TINTS["pos"]
NEG_COLOR, NEG_TINT = ps.PAIR_COLORS["neg"], ps.PAIR_TINTS["neg"]

#: Capitalized and centered by set_title's defaults, matching the two figures
#: drawn by plot_invariance_hist (2026-09-09, Darin — they were lowercase, left).
PANEL_TITLES = ("Input signal, G(r)", "Untrained encoder", "Pretrained encoder")


def example_pairs() -> dict:
    """The top row's example pairs — `median_examples` over this figure's sample,
    as JSON-ready lists plus the block's r axis, banked into the cache."""
    from analysis.latent import load_views
    from analysis.plot_invariance_hist import median_examples

    A, B, _info = load_views(str(BLOCK), N_MATERIALS, 0)
    ex = median_examples(A, B)
    return {
        "r": np.load(BLOCK / "r_axis.npy").tolist(),
        "pos": [t.tolist() for t in ex["pos"]],
        "neg": [t.tolist() for t in ex["neg"]],
        "pos_dist": ex["pos_dist"], "neg_dist": ex["neg_dist"],
    }


def compute() -> dict:
    """Re-embed, histogram, and bank what the figure draws. See the module docstring."""
    from analysis.downstream_eval import embed, load_encoder
    from analysis.latent import load_views, participation_ratio, spectrum
    from analysis.plot_invariance_hist import cosine_distances, pair_stats
    from core.train import build_model, pick_device

    device = pick_device("auto")
    model, cfg, _ = load_encoder(RUN / "ckpt_best.pt", device)
    A, B, info = load_views(str(BLOCK), N_MATERIALS, 0)
    print(f"{info['block']}: {info['n_materials']} of {info['n_materials_available']} "
          f"materials, encoder {cfg.encoder} on {device}", flush=True)

    torch.manual_seed(0)
    control = build_model(cfg).to(device).eval()
    for p in control.parameters():
        p.requires_grad_(False)

    views = [(A, B)]
    for enc in (control, model):
        views.append((embed(enc, A, device, 256), embed(enc, B, device, 256)))
    pairs = [cosine_distances(HA, HB) for HA, HB in views]

    lo = min(v.min() for s, d in pairs for v in (s, d))
    hi = max(v.max() for s, d in pairs for v in (s, d))
    bins = np.geomspace(lo, hi, 61)
    out = {
        "run": RUN.name, "block": info["block"], "modality": info.get("modality", "pdf"),
        "n_materials": int(info["n_materials"]), "sample_seed": 0, "control_seed": 0,
        "bins": bins.tolist(), "panels": [], "examples": example_pairs(),
    }
    for title, (same, diff), (HA, _HB) in zip(PANEL_TITLES, pairs, views):
        out["panels"].append({
            "title": title,
            # Weighted to FRACTION OF PAIRS, not counts: n=4,096 positive distances
            # against ~8.4 M negative ones, so raw counts would put the two
            # histograms four decades apart on the y axis.
            "same": (np.histogram(same, bins=bins)[0] / len(same)).tolist(),
            "diff": (np.histogram(diff, bins=bins)[0] / len(diff)).tolist(),
            "same_median": float(np.median(same)), "diff_median": float(np.median(diff)),
            "n_same": int(len(same)), "n_diff": int(len(diff)),
            # Overlap stats + effective rank (view A, latent.py's pinned estimator),
            # banked because the raw distances are not (see the caching note above).
            # NOT drawn since 2026-09-09 — quoted in the text, printed below.
            **pair_stats(same, diff),
            "eff_rank": participation_ratio(spectrum(HA)),
        })
        p = out["panels"][-1]
        print(f"  {title:<24} median same {p['same_median']:.2e}  "
              f"diff {p['diff_median']:.2e}  ratio {p['diff_median'] / p['same_median']:.1f}x  "
              f"AUROC {p['auroc']:.4f}  same>p10(diff) {100 * p['tail_frac']:.1f}%  "
              f"eff.rank {p['eff_rank']:.1f}")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    print(f"banked {CACHE.relative_to(ps.REPO)}")
    return out


def draw_hist(ax, panel: dict, bins: np.ndarray) -> None:
    """One panel: the two pair-distance histograms and the bracket measuring them apart."""
    # EACH CURVE IS SCALED TO ITS OWN PEAK. The banked numbers are fractions of pairs,
    # but the two arms hold 4,096 and ~8.4 M pairs over very different supports, so the
    # negative-pair spike is 6x the positive-pair hump in two panels — plotted
    # as-is it either clips flat or squashes the hump it is meant to be compared with.
    # The claim is where the distributions SIT and how far apart their medians are,
    # neither of which peak-scaling touches; heights were never comparable across arms.
    for key, c in (("diff", NEG_COLOR), ("same", POS_COLOR)):
        h = np.array(panel[key]) / max(panel[key])
        ax.stairs(h, bins, fill=True, color=c, alpha=0.22, lw=0, zorder=3)
        ax.stairs(h, bins, color=c, lw=1.1, zorder=4)

    # THE RATIO IS DRAWN, NOT LEFT TO THE READER. On a log axis the bracket's LENGTH
    # is the log-ratio, so "×122 dwarfs ×2.5" is geometry rather than arithmetic. A
    # linear axis would collapse the random-init panel to one spike at zero and lose
    # exactly the overlap that makes it the control.
    ms, md = panel["same_median"], panel["diff_median"]
    for m, c in ((ms, POS_COLOR), (md, NEG_COLOR)):
        ax.axvline(m, ymin=0.80, ymax=0.88, color=c, lw=1.4)
    # Every panel gets its arrow, with the same small heads as
    # plot_invariance_hist._bracket — mutation_scale is what keeps a short arrow
    # from collapsing into an ✗ of overlapped heads, not a drop rule.
    ax.annotate("", xy=(md, 0.84), xytext=(ms, 0.84),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=ps.INK,
                                mutation_scale=4.5, shrinkA=0, shrinkB=0))
    r = md / ms
    ax.annotate(f"×{r:.0f}" if r >= 10 else f"×{r:.1f}", xy=(np.sqrt(ms * md), 0.90),
                xycoords=("data", "axes fraction"), ha="center", fontsize=7.5,
                color=ps.INK)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute", action="store_true",
                    help="re-embed and refresh the banked JSON (~2.5 min)")
    args = ap.parse_args()

    if args.recompute or not CACHE.exists():
        hist = compute()
    else:
        hist = json.loads(CACHE.read_text())
        print(f"drawn from {CACHE.relative_to(ps.REPO)} ({hist['run']}) "
              f"— pass --recompute to refresh")
        if "examples" not in hist:  # pre-redesign cache: back-fill without re-embedding
            hist["examples"] = example_pairs()
            CACHE.write_text(json.dumps(hist))
            print(f"banked example pairs into {CACHE.relative_to(ps.REPO)}")
    bins = np.array(hist["bins"])

    ps.use()
    fig = plt.figure(figsize=(ps.WIDTH, 3.6))
    # The middle gridspec row is a SPACER: tight_layout only guarantees decorations
    # don't collide, and with the r-axis labels alone the two rows sat nearly flush.
    gs = fig.add_gridspec(3, 6, height_ratios=(0.85, 0.14, 1.0))
    ex_axes = [fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])]
    hist_axes = [fig.add_subplot(gs[2, 2 * i:2 * i + 2]) for i in range(3)]

    from analysis.plot_invariance_hist import draw_example
    ex = hist["examples"]
    r = np.array(ex["r"])
    draw_example(ex_axes[0], r, ex["pos"], ex["pos_dist"], POS_COLOR, POS_TINT,
                 "positive pair", paper=True)
    draw_example(ex_axes[1], r, ex["neg"], ex["neg_dist"], NEG_COLOR, NEG_TINT,
                 "negative pair", paper=True)
    for ax in ex_axes:
        ax.set_xlabel("r (Å)")
        ax.tick_params(axis="both", length=0)
    ex_axes[0].set_ylabel("G(r), normalized")

    # Titles come from PANEL_TITLES, not the cache: the banked "title" strings predate
    # two wording passes and refreshing them would cost a full --recompute.
    for ax, panel, title in zip(hist_axes, hist["panels"], PANEL_TITLES):
        draw_hist(ax, panel, bins)
        ax.set_xscale("log")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xticks([1e-6, 1e-4, 1e-2, 1e0])
        ax.set_ylim(0, 1.45)  # curves are peak-scaled in draw_hist; headroom holds the bracket
        ax.set_yticks([])
        ax.set_title(title, color=ps.INK, pad=4)
        ax.minorticks_off()  # the log minor ticks survive length=0, which is major-only
        ax.tick_params(axis="x", length=0, labelsize=6.5)
    hist_axes[0].set_ylabel("density, peak-scaled")
    hist_axes[1].set_xlabel("cosine distance")

    # The color key is uniform across panels now, so it lives INSIDE the first
    # histogram panel (the band under the bracket is free since the stats line left)
    # rather than in a reserved figure-coordinate band.
    from matplotlib.patches import Patch
    swatch = lambda c: Patch(facecolor=c, edgecolor=c, alpha=0.35, lw=1.1)
    hist_axes[0].legend([swatch(POS_COLOR), swatch(NEG_COLOR)],
                        ["positive pairs", "negative pairs"],
                        loc="center left", fontsize=6.5, handlelength=1.4,
                        borderaxespad=0.2, labelspacing=0.3)

    ps.layout(fig)
    ps.save(fig, "pdf_fig2_invariance")

    for p in hist["panels"]:
        extra = (f"  AUROC {p['auroc']:.4f}  same>p10(diff) {100 * p['tail_frac']:.1f}%"
                 f"  eff.rank {p['eff_rank']:.1f}") if "auroc" in p else ""
        print(f"  {p['title']:<24} same {p['same_median']:.2e}  diff {p['diff_median']:.2e}"
              f"  ratio {p['diff_median'] / p['same_median']:.1f}x{extra}")


if __name__ == "__main__":
    main()
