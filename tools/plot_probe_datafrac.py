"""tools/plot_probe_datafrac.py — the labelled-set ladder, all three arms on AUGMENTED data.

    python tools/plot_probe_datafrac.py            # provenance block above the axes
    python tools/plot_probe_datafrac.py --paper    # paper-ready + LaTeX caption

Three arms, four targets. **Every arm fits AUGMENTED labels and is scored on the
same augmented test split**, so the only thing that varies down a column is what the
encoder is and whether it moves:

    supervised end-to-end   random init, encoder trains   runs/pdf/datafrac/ + runs/pdf/supervised/
    frozen encoder + probe  pretrained, encoder FROZEN    <run>/probe_frac<f>_seed<42-46>.json
    finetuned + head        pretrained, encoder trains    runs/pdf/datafrac/ + runs/pdf/datafrac_full/

EVERY POINT IS THE MEAN OF SEEDS 42/43/44 (2026-08-27). The spread is reported in the
appendix from `--table`, not drawn as whiskers. See limit 1 for what the seeds do and
do not vary, and DATAFRAC_FULL for why the fine-tuned scheme's full-data cells moved
out of `runs/pdf/sweep/`.

The gradient arms hold five rungs (126–2530); **the probe continues alone down three
more, to 13 labeled rows (2026-08-21)** — see PROBE_FRACS for why the tail is
probe-only, why n = 5 is excluded, and the single-draw caveat that grows as n
shrinks.

**THIS REPLACED THE CLEAN-FIT LADDER (2026-08-19), AND IT IS A DIFFERENT CLAIM.**
The previous version of this figure drew two arms across the off-diagonal — a
frozen probe fit on CLEAN against scratch fit on AUGMENTED — and its claim was
*pretraining substitutes for paying the re-simulation cost*. Holding every arm at
fit-augmented gives up that claim and buys a cleaner one: **at equal labels, on
identical rows, what does pretraining buy?** No arm is label-disadvantaged, so a
reader does not have to hold two axes of difference at once. The clean-fit numbers
are not lost — they are banked in `RESULTS.md` Q2 and are what
`tools/plot_chili_four_arms.py` still draws at full data.

⚠️ **The probe-vs-finetune ordering INVERTS between the two framings, and that is a
real finding, not a plotting artifact.** On the clean fit the frozen probe beat
gradient finetuning on 3 of 4 targets (`RESULTS.md` Q1 → the four-arm table). On the
augmented fit it wins the two classification targets at every rung and LOSES both
regression targets at every rung. Freezing is what protects a readout that never saw
the shift; once the readout does see it, the encoder's extra capacity pays. Do not
quote the four-arms sentence next to this figure without that distinction.

WHY THESE FOUR TARGETS — same reasoning as the four-arms figure: `target_mo_bond` is
r = 0.999 with `target_mean_bond` so plotting both double-counts one task, and
`target_metal` separates nothing. Regression reads R², classification weighted F1
with the modal-class floor drawn as a reference line (never a hue).

MATCHING, VERIFIED FROM THE JSONs AT DRAW TIME rather than assumed:
- **Same rows, all three arms, every rung.** `analysis/downstream_eval.py
  --n-train-frac` draws through `analysis.finetune.subsample_train_mask` — a function
  of `(mask, frac, seed)` alone — over the same `load_probe_data` row order, so the
  probe fits the IDENTICAL rows the two gradient arms trained on. The per-target
  labelled counts are asserted equal across all three at every rung and the figure
  refuses to draw otherwise.
- **One scratch arm across two config stems.** frac<1 comes from `runs/pdf/datafrac/`
  under `cnn_infonce_mpfull_final_lr0.001` (a naming artifact of the datafrac
  redirect, not a pretrained cell — `pretrained` is asserted false); frac=1.0 from
  `runs/pdf/supervised/` under `cnn_lr0.001`. Both are checked equal on encoder,
  `latent_dim` and every field in `GRADIENT_INVARIANTS` rather than by name. The
  finetuned arm spans `runs/pdf/datafrac/` and `runs/pdf/sweep/` and is checked the
  same way.
- **Same registry, same test split**, asserted per source.
- **No test leakage.** The scratch LR was val-selected with test never read
  (`RESULTS.md` Q3); the probe has no hyperparameter to select, its ridge λ coming
  from GCV on the fit rows alone.

⚠️ **FOUR LIMITS, all stated in the caption rather than fixed:**
1. **EVERY POINT IS THE MEAN OF THREE SEEDS (42/43/44) SINCE 2026-08-27**, and the
   spread is reported in the appendix (`--table`) rather than drawn. What the seeds
   vary is the head initialisation, the batch order and the subsample draw — NOT the
   encoder: the pretraining run behind both pretrained schemes is still n = 1, so
   nothing here resolves variation in the representation itself.
   The spread is wildly non-uniform and the appendix has to say so: negligible at
   full data (every scheme ≤ 0.03) and largest at the BASELINE'S SWITCH-ON RUNG,
   where the seeds disagree about whether the network trained at all — `np_size` at
   n = 253 is 0.001 / 0.891 / 0.830, an sd of 0.500. The mean there sits in the gap
   between two outcomes rather than at a typical one, which is a property of
   averaging a step function near its step, not a defect in the cells.
2. **THE LR ASYMMETRY NOW CUTS THE OTHER WAY.** Scratch got a five-point LR search;
   the finetuned arm is the untuned incumbent `1e-5 / 1e-4` with a random head. Under
   the old clean-fit framing that asymmetry biased AGAINST pretraining, which made it
   safe to leave. Here every arm sees the same data, so it flatters scratch instead
   and the finetuned line is a LOWER BOUND. `runs/pdf/lpft/` has the LR-matched
   recipe at full data only; a rung-by-rung LP-FT arm does not exist.
3. **NOT a gradual crossover.** Scratch is at chance until a target-dependent
   threshold and then switches on abruptly (`mo_bond` goes −0.099 → 0.848 in one
   rung). The claim is *"scratch cannot train below its threshold and either
   pretrained arm can"*, never "smoothly crossing curves". A caption that omits this
   is a wrong caption.
4. **Ridge at EVERY rung**, so the probe curve is one protocol. Required below
   n_train = latent_dim = 256, i.e. the 5% and 10% rungs, where plain OLS
   interpolates silently.

COLOR. `analysis.paperstyle.ARM_COLORS`, and the arm→hue mapping is the SAME as the
four-arms figure — `supervised` indigo, `probe` teal, `finetuned` green — so a reader
carrying the key over from that figure is not re-taught it. Here all three slots of
that ramp are drawn at once, which is what the ramp was ordered for (increasing use
of pretraining). The low-contrast `probe` and `finetuned` slots' stated label
obligation is discharged twice over since 2026-08-21: distinct marker shapes on the
lines themselves (o/s/^, the chem figures' scheme encoding — see ARMS) and the
legend's line+marker handles.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator

from analysis import paperstyle
from analysis.paperstyle import ARM_COLORS, INK, MODAL, MUTED
from analysis.ssl_vs_scratch import CEILING_FRACTION, metric_key

REPO = Path(__file__).resolve().parent.parent

RUN_STEM = "2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372"
PRETRAIN_RUN = REPO / "runs/pdf/sweep" / RUN_STEM
DATAFRAC = REPO / "runs/pdf/datafrac"
SUPERVISED = REPO / "runs/pdf/supervised"

#: EVERY POINT IS THE MEAN OF THESE THREE SEEDS (2026-08-27, user). The figure was
#: seed 42 alone until then; the spread is reported in the appendix rather than as
#: whiskers, because at a 1.0 in panel height with three low-contrast lines the bars
#: cost more legibility than they buy. `--table` emits the per-seed values and sd
#: that the appendix quotes, from these same JSONs, so the two cannot drift.
#:
#: ⚠️ THE MEAN OF THREE IS NOT ONE OF THE THREE, unlike the median this briefly used.
#: That is the point: a median marker at n = 3 is a particular run's number relabelled
#: as a summary. It also means these values will NOT match `RESULTS.md`, which quotes
#: 3-seed medians; a number lifted from here into that file must be re-derived.
SEEDS = (42, 43, 44)

#: The fine-tuned scheme's frac=1.0 cells. Its seed-42 point used to be read from
#: `PRETRAIN_RUN/finetune.json` — one file per checkpoint, with no seed in the path,
#: so seeds 43/44 could not be written beside it. All three were re-run into this tree
#: (job 5503644, 2026-08-27) rather than splicing one 2026-07 point against two new
#: ones: `docs/TRAPS.md` puts fixed-seed GPU re-runs at a median 9.4% in val loss,
#: which a mixed-batch triple would carry as if it were seed spread. The re-run
#: reproduced the old cell to within 0.0011 on every panel target, from different
#: early-stopping epochs — so the splice would have been survivable, and is still not
#: worth making when the alternative costs seven minutes.
DATAFRAC_FULL = REPO / "runs/pdf/datafrac_full"
FULL_FT_CELL = f"{RUN_STEM}_enc1e-05_head0.0001"

#: Scratch run-name prefixes: frac<1 in DATAFRAC, frac=1.0 in SUPERVISED. Two
#: different config stems for one arm — see the module docstring.
SCRATCH_STEM = "cnn_infonce_mpfull_final_lr0.001"
FULL_STEM = "cnn_lr0.001"

#: The one protocol this figure draws. Fit on augmented, score on augmented — for
#: all three arms, which is the whole point of the 2026-08-19 rewrite.
PROTOCOL = "aug_aug"

#: Rung -> the literal string in the probe JSON filename. Explicit rather than
#: formatted: `f"{1.0:g}"` is "1" while the cell on disk is "1.0", and a near-miss
#: there reads as a missing run.
#:
#: FRACS are the rungs all three schemes share. PROBE_FRACS extends the probe alone
#: down to 13 labeled rows (2026-08-21): below n = 64 the two gradient schemes
#: cannot form a single training batch (`batch_size=64` with `drop_last=True`,
#: which BatchNorm forces — analysis/finetune.py), so the tail is not a choice to
#: drop them, it is the mechanical floor of end-to-end training under this
#: figure's own GRADIENT_INVARIANTS; the dotted vertical rule draws that floor.
#: 0.002 (n = 5) exists on disk and is excluded:
#: that is the probe's breakdown point (M–O straddles zero across draws, cn within
#: noise of the modal floor) — a sentence for the text, not a data point.
#: ⚠️ The tail rungs are ONE draw, like every other point, and down there the draw
#: itself is a real variance source: over the five seeds banked on disk
#: (`probe_frac<f>_seed<42-46>.json`, 2026-08-21), M–O R² at n = 51 ranges
#: 0.30–0.78. A mean ± sd band over those draws was built and rejected
#: (2026-08-21, user call) in favour of plain points; the caption carries the
#: caveat instead, and the sweep stays on disk for whoever wants the spread.
#: 0.0277 is the BATCH-FLOOR RUNG (2026-08-21): the smallest single fraction where
#: every target still forms one training batch — cn lands on exactly 64 labeled
#: rows, the other targets on 70 — so the gradient lines terminate AT their floor
#: instead of hanging at 126.
#: ⚠️ ITS SEED-42 CELLS WERE RUN LOCALLY ON CPU, seeds 43/44 on cluster GPUs (job
#: 5503644): config-identical, and the fingerprint assertion checks that, but the
#: hardware differs within this one rung, which matters at the fixed-seed re-run
#: noise level in docs/TRAPS.md. Seed 42 was not re-run because that would have
#: overwritten a measured cell.
#: ⚠️ **A CLAIM MADE HERE FROM SEED 42 ALONE DID NOT SURVIVE THE OTHER TWO
#: (corrected 2026-08-27).** This note used to say that "even the FINE-TUNED scheme
#: sits exactly ON the cn modal floor (0.418)". That is seed 42's cell; seeds 43/44
#: reach 0.722 and 0.626, so the scheme's mean is 0.589 and the floor-touch was one
#: draw, not the rung's behaviour. WHAT DOES HOLD, on all three seeds and exactly, is
#: the BASELINE's: 0.418 in every seed, because predicting the modal class is what a
#: network that has learned nothing evaluates to. The probe holds 0.83 here.
FRACS = (0.0277, 0.05, 0.1, 0.25, 0.5, 1.0)
PROBE_FRACS = (0.005, 0.01, 0.02) + FRACS
FRAC_TAG = {0.005: "0.005", 0.01: "0.01", 0.02: "0.02", 0.0277: "0.0277",
            0.05: "0.05", 0.1: "0.1", 0.25: "0.25", 0.5: "0.5", 1.0: "1.0"}

#: Round reference ticks on ONE shared x-axis (2026-08-19; previously each panel
#: labelled its own exact counts on every row). The points sit at each panel's TRUE
#: labelled counts — `target_cn` runs 12-2310 where the others run 13-2530 — so
#: the ticks are a scale, not per-panel claims, and the bottom row can label for
#: all four. The caption states the cn range the ticks no longer do. Extended left
#: with the probe tail (2026-08-21); fewer, wider-spaced ticks than before because
#: the axis now spans 2.3 decades in the same panel width.
XTICKS = (25, 100, 250, 1000, 2500)

#: (target, panel title, is_classification). Row 1 regression, row 2
#: classification — the same order and the same four targets as the four-arms
#: figure, so the two can be read side by side.
PANELS = (
    ("target_np_size", "Nanoparticle size", False),
    ("target_mo_bond", "M–O distance", False),
    ("target_cn", "Coordination number", True),
    ("target_oxidation", "Oxidation state", True),
)

#: (label, source key, colour). Colours are the four-arms figure's, by ARM, and the
#: order here is that figure's ramp: no pretraining, frozen, finetuned. The labels
#: are the unified probe-figure names (2026-08-19) — "baseline" / "probe" /
#: "fine-tuned" in every figure that draws these arms — so the caption carries what
#: they stop saying (same rows, same augmented channel, encoder init/freezing the
#: only contrast).
#: Marker shapes are the chem figures' scheme encoding (plot_chem_shift_interaction:
#: baseline o, probe s, fine-tuned ^), adopted 2026-08-21 so a reader moving between
#: the two carries one scheme→marker mapping — and the low-contrast colour slots are
#: now relieved in-panel, not only in the legend.
ARMS = (
    ("baseline", "scratch", ARM_COLORS["supervised"], "o"),
    ("probe", "probe", ARM_COLORS["probe"], "s"),
    ("fine-tuned", "finetune", ARM_COLORS["finetuned"], "^"),
)

#: Fields that must agree across all five cells of a gradient arm for them to be one
#: arm. `CNNEncoder`'s geometry depends on nothing beyond `latent_dim`, so these pin
#: the network; the rest pin the optimization.
GRADIENT_INVARIANTS = ("lr_enc", "lr_head", "epochs", "batch_size", "patience")

#: Per-ROW y-axis (2026-08-21; previously one shared (-0.18, 1.06) for all four).
#: The rows are different metrics — R² above, weighted F1 below — so sharing was a
#: convenience, and at the compressed 3.0-inch height it made the regression curves
#: pay ~15% of their span for a sub-−0.1 region only the classification row's
#: layout needed. Each row now spans its own content: regression still reaches
#: below 0 because the scratch arm measures −0.099 and R² = 0 (predicting the test
#: mean) must stay an interior line, not an edge; classification's floor sits just
#: under the oxidation modal baseline at 0.245.
#:
#: ⚠️ **`panel(ylim=...)` OVERRIDES BOTH, and one caller does.** Per-metric limits are
#: right for THIS figure's 2x2, where a metric owns a row. On the 1x4 of
#: `tools/plot_chili_combined.py` the two would sit side by side in one row, i.e. two
#: y axes mid-row, so that caller passes `YLIM_REG` — the union — for all four. Only
#: `YLIM_REG` can be the shared one: it is the one that reaches R² = 0 and below.
#: The measured fixed-seed re-run noise floor (docs/TRAPS.md), quoted by the
#: appendix table so a reader is not invited to read differences below it.
NOISE_FLOOR = "$0.0143$"

YLIM_REG = (-0.13, 1.02)
YLIM_CLF = (0.16, 1.02)


def load_probe(seed):
    """`(target, frac) -> (score, n_train)`, plus the JSONs, from the probe rungs."""
    out, docs = {}, {}
    for frac in PROBE_FRACS:
        p = PRETRAIN_RUN / f"probe_frac{FRAC_TAG[frac]}_seed{seed}.json"
        if not p.exists():
            raise SystemExit(
                f"missing {p} — run:\n  python -m analysis.downstream_eval "
                f"--runs runs/pdf/sweep --filter '*cnn_vicreg_mpfull_final*' --ridge \\\n"
                f"    --target {' '.join(t for t, _, _ in PANELS)} \\\n"
                f"    --n-train-frac {FRAC_TAG[frac]} --subsample-seed {seed} "
                f"--out-name probe_frac{FRAC_TAG[frac]}_seed{seed}.json")
        d = json.loads(p.read_text())
        if not d.get("ridge"):
            raise SystemExit(f"{p}: not a ridge probe — this curve is ridge at every rung")
        if d.get("n_train_frac") != frac:
            raise SystemExit(f"{p}: n_train_frac={d.get('n_train_frac')} != {frac}")
        if frac < 1.0 and d.get("subsample_seed") != seed:
            raise SystemExit(f"{p}: subsample_seed={d.get('subsample_seed')} != {seed}")
        for target, _title, _clf in PANELS:
            if PROTOCOL not in d["tasks"][target]:
                raise SystemExit(f"{p}: no {PROTOCOL} protocol for {target}")
            out[(target, frac)] = (d["tasks"][target][PROTOCOL][metric_key(target)],
                                   d["n_labelled"][target]["train"])
        docs[frac] = d
    return out, docs


def load_gradient(seed, pretrained):
    """`(target, frac) -> (score, n_train, hit_ceiling)`, plus the JSONs, for one
    end-to-end arm — scratch (`pretrained=False`) or finetuned (`True`).

    Each arm spans two trees (the rungs below full data live under `runs/pdf/datafrac/`),
    so the cells are checked to be one network and one optimization; a mismatch raises
    rather than drawing a line whose full-data point came from another setup.
    """
    out, docs, ref = {}, {}, None
    for frac in FRACS:
        if pretrained:
            p = (DATAFRAC_FULL / f"seed{seed}" / FULL_FT_CELL if frac == 1.0
                 else DATAFRAC / f"{RUN_STEM}_frac{FRAC_TAG[frac]}_seed{seed}")
        else:
            p = (SUPERVISED / f"{FULL_STEM}_seed{seed}" if frac == 1.0
                 else DATAFRAC / f"{SCRATCH_STEM}_seed{seed}_frac{FRAC_TAG[frac]}_seed{seed}")
        f = p / "finetune.json"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        d = json.loads(f.read_text())
        if d["pretrained"] != pretrained:
            raise SystemExit(f"{f}: pretrained={d['pretrained']}, expected {pretrained}")
        cfg = d["finetune_cfg"]
        if cfg["seed"] != seed:
            raise SystemExit(f"{f}: seed={cfg['seed']} != {seed}")
        # The full-data cells predate `--n-train-frac` and omit the key; every other
        # cell must carry its own rung, or the run is not the one the path claims.
        if cfg.get("n_train_frac", 1.0) != frac:
            raise SystemExit(f"{f}: n_train_frac={cfg.get('n_train_frac')} != {frac}")
        fingerprint = ((d["encoder"], d.get("latent_dim"))
                       + tuple(cfg[k] for k in GRADIENT_INVARIANTS))
        if ref is None:
            ref = (fingerprint, f)
        elif fingerprint != ref[0]:
            raise SystemExit(
                f"cells are not one arm:\n  {ref[1]}: {ref[0]}\n  {f}: {fingerprint}")
        ceiling = cfg["epochs"]
        for target, _title, _clf in PANELS:
            if PROTOCOL not in d["tasks"][target]:
                raise SystemExit(f"{f}: no {PROTOCOL} protocol for {target}")
            out[(target, frac)] = (d["tasks"][target][PROTOCOL][metric_key(target)],
                                   d["n_labelled"][target]["train"],
                                   d["best_epoch"][target]["aug"] >= CEILING_FRACTION * ceiling)
        docs[frac] = d
    return out, docs, ref[0]


def tex_lr(x):
    """`1e-05` -> `10^{-5}`. A caption that prints Python float repr is a caption
    nobody would typeset by hand, and `%g` gives `1e-05` for exactly the learning
    rates this figure uses."""
    e = round(__import__("math").log10(x))
    if 10.0 ** e != x:
        raise SystemExit(f"tex_lr: {x} is not a power of ten")
    return rf"10^{{{e}}}"


def seed_table(per_seed, sources, stat="mean"):
    """The appendix table for this figure, in the thesis's own layout.

    Schemes as rows, rungs as columns, one block per target — and `stat` chooses
    whether the cells are the means the figure plots or their standard deviations.
    TWO TABLES OF ONE SHAPE, not one table of paired cells: nine columns already run
    `\\small`, and `0.888 (0.500)` in every one of them overruns the text block. A
    reader compares a value to its neighbour along a row, which the split preserves.

    ⚠️ THE COLUMN HEADS ARE `target_np_size`'s LABELLED COUNTS, for all four blocks.
    `target_cn` masks CHILI's mixed-coordination structures and runs 12--2310 where
    the others run 13--2530, so its row is filed one column left of its true n. That
    is the figure's own shared-axis convention (XTICKS) and the caption states it;
    it is not a rounding.
    """
    ns = [sources["probe"][(PANELS[0][0], f)][1] for f in PROBE_FRACS]
    tail = len(PROBE_FRACS) - len(FRACS)          # probe-only rungs, left of the floor
    body = []
    for k, (target, title, _clf) in enumerate(PANELS):
        if k:
            body.append(r"    \midrule")
        body.append(rf"    \multicolumn{{{len(ns) + 1}}}{{l}}{{\emph{{{title}}}}} \\")
        for label, source, _c, _m in ARMS:
            fr = PROBE_FRACS if source == "probe" else FRACS
            cells = ["---"] * (0 if source == "probe" else tail)
            for f in fr:
                ys = [per_seed[sd][source][(target, f)][0] for sd in SEEDS]
                v = statistics.stdev(ys) if stat == "sd" else statistics.fmean(ys)
                # `$-$` not a hyphen: a minus sign in text mode is a hyphen, and the
                # baseline's sub-zero cells are the ones a reader must not misread.
                cells.append((rf"$-${abs(v):.3f}" if v < 0 else f"{v:.3f}"))
            body.append(f"    {label} & " + " & ".join(cells) + r" \\")
    heads = " & ".join(str(n) for n in ns)
    what = ("Standard deviations over the three seeds behind Table~\\ref{tab:probe_datafrac}"
            if stat == "sd" else
            "Exact numbers behind Figure \\ref{fig:labels}")
    extra = (r"""Each entry is the sample standard deviation at $n{=}3$ and is
  correspondingly poorly determined; differences below the $0.0143$ fixed-seed re-run
  noise floor should not be read in either direction. The two largest entries are the
  baseline's at its switch-on rung, where the seeds disagree about whether the network
  trained at all, so the mean opposite them sits between two outcomes rather than at a
  typical one. The probe's spread comes from the subsample draw alone, since a
  closed-form fit on fixed rows is deterministic, while the two end-to-end schemes also
  carry head-initialisation and batch-order variation."""
             if stat == "sd" else
             r"""Entries below the batch floor of 64 exist only for the probe. Each entry
  is the mean over three seeds, which vary the head initialisation, the batch order and
  the labeled subsample draw; standard deviations are in
  Table~\ref{tab:probe_datafrac_sd}. The pretraining run behind the probe and
  fine-tuned schemes is itself $n{=}1$, so no entry resolves encoder variation. Column
  heads are nanoparticle size's labeled counts; coordination number masks
  mixed-coordination structures and runs 12--2310 against the others' 13--2530.""")
    label = "tab:probe_datafrac_sd" if stat == "sd" else "tab:probe_datafrac"
    return rf"""% --- generated by tools/plot_probe_datafrac.py --table{' --sd' if stat == 'sd' else ''} ---
\begin{{table}} [h]
  \centering
  \small
  \caption{{{what}. {extra}}}
  \label{{{label}}}
  \begin{{tabular}}{{l{"r" * len(ns)}}}
    \toprule
    & \multicolumn{{{len(ns)}}}{{c}}{{labeled training samples}} \\
    \cmidrule(lr){{2-{len(ns) + 1}}}
    scheme & {heads} \\
    \midrule
{chr(10).join(body)}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def caption(probe_docs, scratch_docs, finetune_docs):
    """LaTeX caption for `--paper`, built from the JSONs so it cannot drift.

    Everything the three short scheme labels stop saying has to be here: that all
    three are supervised on identical rows AND on the same augmented channel, so the
    contrast is initialisation-and-freezing alone; that the probe's tail below 126 is
    alone because end-to-end training cannot batch there, not because the comparison
    was dropped; and the four limits — single seed (which on the tail means the draw
    itself), the LR asymmetry that now runs the other way, ridge throughout, and
    above all that this is a THRESHOLD not a smooth crossover. A caption that drops the last one is not
    shorter, it is wrong (see the module docstring).
    """
    full = probe_docs[1.0]
    n_full = full["n_labelled"]["target_np_size"]["train"]
    n_low = probe_docs[FRACS[0]]["n_labelled"]["target_np_size"]["train"]
    n_tail = probe_docs[PROBE_FRACS[0]]["n_labelled"]["target_np_size"]["train"]
    cn_full = full["n_labelled"]["target_cn"]["train"]
    cn_low = probe_docs[PROBE_FRACS[0]]["n_labelled"]["target_cn"]["train"]
    lr = scratch_docs[1.0]["finetune_cfg"]["lr_enc"]
    ft = finetune_docs[1.0]["finetune_cfg"]
    n_seeds = len(SEEDS)
    seed_list = ", ".join(str(x) for x in SEEDS)
    return rf"""% --- generated by tools/plot_probe_datafrac.py --paper ---
\caption{{%
  \textbf{{Given the same augmented labels, both pretrained schemes train at budgets
  where the supervised model is at chance.}}
  All three schemes are trained \emph{{supervised}} on the same seeded subsets of the
  {n_full} CHILI training examples --- at every shared point the three fit
  \emph{{identical}} rows, down to {n_low} --- and all are evaluated on the same
  {full['n_test']} augmented test examples.
  Below 64 labeled rows an end-to-end scheme cannot form a single training batch
  (batch size 64, dropped-last for BatchNorm), so the tail down to {n_tail} is the
  probe alone --- a mechanical floor of gradient training, marked by the dotted
  vertical rule, not a dropped comparison; the smallest shared point sits at that
  floor itself (64 rows for coordination number, one batch exactly).
  \textbf{{Every scheme is fit on the augmented channel:}} baseline, probe and
  fine-tuned all see the same augmented spectra, so the three differ only in
  initialisation and in whether the encoder moves.
  Coordination number masks CHILI's mixed-coordination structures, so that panel
  runs over {cn_low}--{cn_full} training and
  {full['n_labelled']['target_cn']['test']} test examples; every point sits at
  its panel's true labeled count, with the shared axis ticks at round reference
  values.
  \emph{{Baseline}} trains a randomly initialised CNN
  (learning rate ${tex_lr(lr)}$, selected on validation loss over five points).
  The other two start from an encoder pretrained with VICReg on simulated PDFs of the
  full Materials Project (\texttt{{cnn\_vicreg\_mpfull\_final}}, disjoint from CHILI):
  \emph{{probe}} holds it fixed and fits a ridge head in closed form,
  while \emph{{fine-tuned}} trains encoder and head jointly
  (${tex_lr(ft['lr_enc'])}$ / ${tex_lr(ft['lr_head'])}$).
  Regression panels are R\textsuperscript{{2}}, classification panels weighted F1
  with the modal-class baseline dashed.
  \textbf{{The gap is a training threshold, not a smooth crossover:}} the baseline
  sits at or below chance until a target-dependent labeled-set size and then
  switches on abruptly (M--O distance moves from $-0.10$ to $0.85$ in a single rung),
  so the result reads as \emph{{a randomly initialised network cannot train below its
  threshold where a pretrained one can}}, rather than as crossing curves.
  Freezing and finetuning split the targets: the probe leads on both
  classification targets at every shared rung, the fine-tuned encoder on both
  regression targets.
  Every point is the mean of {n_seeds} seeds ({seed_list}), which vary the head
  initialisation, the batch order and the labeled subsample draw; the pretraining run
  behind the two pretrained schemes is itself $n{{=}}1$, so no point resolves
  variation in the encoder. Per-seed values and standard deviations are tabulated in
  the appendix. The spread is far from uniform: it is negligible at full data and
  largest at the baseline's switch-on rung, where seeds disagree about whether the
  network has trained at all, so the mean there sits between two outcomes rather than
  at a typical one. On the probe's smallest rungs the
  subsample draw itself is a substantial variance source, so the tail points carry
  more uncertainty than the line's right-hand end. The fine-tuned scheme carries the untuned
  incumbent learning rate rather than a search of its own, so that line is a lower
  bound. The probe uses ridge at every rung, so its curve is one protocol throughout.
}}"""


def panel(ax, sources, target, title, is_clf, scratch_docs, show_xlabel, show_y,
          xlim, xticks=XTICKS, show_title=True, floor_label_x=0.82,
          floor_label_rotation=0, ylim=None, show_ylabel=None):
    # These three exist for ONE caller, `tools/plot_chili_combined.py` (2026-08-24),
    # where these panels are a quarter of WIDTH rather than a half: five tick labels
    # collide at that width, the target name is already the column title carried by
    # the bar row above, and `floor_label_x` is a MULTIPLE of the floor on a log axis,
    # so the clearance it buys the rotated label halves with the panel. None of the
    # three has any other use here.
    # `ylim` OVERRIDES THE PER-METRIC DEFAULT, for one caller: see YLIM_REG/YLIM_CLF.
    lo_hi = ylim or (YLIM_CLF if is_clf else YLIM_REG)
    # x is the panel's TRUE labelled count per rung (asserted identical across the
    # three schemes at the shared rungs in main), so points land at their real n
    # against the round XTICKS. The probe's axis is longer: its tail runs below the
    # gradient schemes' batch floor, alone by mechanics rather than by choice.
    for label, source, colour, marker in ARMS:
        src = sources[source]
        fr = PROBE_FRACS if source == "probe" else FRACS
        ns = [src[(target, f)][1] for f in fr]
        ys = [src[(target, f)][0] for f in fr]
        ax.plot(ns, ys, color=colour,
                marker=marker, markeredgecolor="white", markeredgewidth=0.8,
                label=label, clip_on=False, zorder=3)

    # The gradient schemes' mechanical floor: below one batch of labeled rows
    # (`batch_size=64`, `drop_last=True` for BatchNorm — analysis/finetune.py) they
    # cannot train at all, which is why their lines stop and the probe's does not.
    # DOTTED, not dashed: dashed is this figure's modal-floor vocabulary, and the
    # rule marks a mechanism, never a measured value — hence also no scheme colour.
    # Labelled once, in the first panel; read from the JSON so it cannot drift.
    floor_n = scratch_docs[1.0]["finetune_cfg"]["batch_size"]
    # NEAR-BLACK, not MUTED (2026-08-25, review), matching the modal floor below and
    # `plot_chili_four_arms`'s MODAL_LINE. `paperstyle`'s REFLINE/MODAL grey is tuned
    # to RECEDE against white page, which is what made it hard to see once the grid
    # came out. The TOKEN is deliberately not darkened at source: `MODAL` is also used
    # as a SERIES colour in `analysis/plot_pdf_invariance.py`, so re-inking it there
    # would change data, not furniture. The dash vocabulary is unchanged and still
    # load-bearing — DOTTED marks a mechanism, dashed a measured value.
    ax.axvline(floor_n, color=INK, linewidth=0.8, linestyle=(0, (1, 1.6)), zorder=1)
    if target == PANELS[0][0]:
        # HORIZONTAL, and right-aligned so it ends just before the rule (2026-08-25,
        # review). It was rotated 90° when it sat beside the line, which is the usual
        # answer for a vertical rule on a crowded axis — but this axis is LOG and the
        # rule is at n = 64 against a left limit of ~11, so the band to its left is
        # 31% of the panel, about 0.76 in. "batch floor" at 7 pt is 0.45 in and fits
        # in it flat, with room to spare. The band is empty in this panel: the only
        # curve reaching it is the probe's, which is at y = 0.46 by n = 13, well
        # above a label pinned to the axis floor.
        # `floor_label_rotation` EXISTS BECAUSE THE LABEL DOES NOT ALWAYS FIT FLAT.
        # Horizontal is the paper figure's form (2026-08-25, review) and works at
        # half-WIDTH: the band left of the rule is 31% of the panel, ~0.76 in, against
        # 0.45 in of text. At the QUARTER-width of `plot_chili_combined` that band is
        # ~0.34 in and the flat label runs off the left spine, so that caller keeps
        # the rotated form. Anchoring follows the rotation — flat text ends just
        # before the rule, rotated text starts just after it.
        ax.text(floor_n * floor_label_x, lo_hi[0] + 0.03, "batch floor",
                rotation=floor_label_rotation,
                ha="left" if floor_label_rotation else "right",
                va="bottom", fontsize=7, color=INK)

    if is_clf:
        # Same reference line as the four-arms figure, and it is load-bearing here:
        # the supervised arm sits exactly ON it at the two lowest rungs of `cn`,
        # which is what "has learned nothing yet" looks like.
        floor = scratch_docs[1.0]["tasks"][target][PROTOCOL]["f1_baseline"]
        assert floor > lo_hi[0], "modal floor fell off the classification axis"
        # Same ink, weight and dash as `plot_chili_four_arms`'s MODAL_LINE, so the
        # modal floor is one recognisable object across both figures that draw it.
        ax.axhline(floor, color=INK, linewidth=1.1, linestyle=(0, (3, 2)), zorder=1)
        # Right-aligned: at the low rungs the supervised line sits ON this floor
        # (`cn` is exactly 0.418 at both), so a left-aligned label collides with
        # the very coincidence the line is drawn to show.
        ax.text(sources["probe"][(target, 1.0)][1], floor + 0.035, "modal",
                ha="right", va="bottom", fontsize=7, color=INK)

    ax.set_xscale("log")
    ax.set_xticks(list(xticks))
    # One shared axis, labelled on the bottom row only — see XTICKS for why that
    # is now sound despite the cn panel's smaller counts.
    ax.set_xticklabels([f"{t:d}" for t in xticks] if show_xlabel
                       else [""] * len(xticks))
    # A log axis re-labels its minor ticks even after set_xticks, which at this
    # width collides with the decade labels.
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(*xlim)
    ax.set_ylim(*lo_hi)
    # THE TICK SET FOLLOWS THE LIMITS, NOT THE METRIC. Which ticks fit is a property
    # of the range drawn: 0 is a tick wherever the axis reaches it, and is not one
    # where it does not. Keying this off `is_clf` instead worked only while the two
    # metrics implied the two ranges, which `ylim` is precisely there to break.
    ticks = (0, 0.25, 0.5, 0.75, 1.0) if lo_hi[0] < 0 else (0.25, 0.5, 0.75, 1.0)
    # All four labelled: this list used to read ("0.25", "0.5", "", "1"), i.e. it
    # labelled 0.25 and blanked 0.75 on the same axis, which is arbitrary.
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks] if show_y else [""] * len(ticks))
    if show_xlabel:
        ax.set_xlabel("labeled training samples")
    # THE METRIC IS A Y-AXIS LABEL, NOT A CORNER TITLE (2026-08-25, review). It used to
    # be a right-aligned second `set_title` in MUTED at 8 pt, which put the name of the
    # quantity as far from its own axis as the panel allows and greyed it out on top of
    # that. It is now `set_ylabel`, so it takes `axes.labelsize` and the black
    # `axes.labelcolor` that `paperstyle.use` sets, like every other axis label here.
    #
    # ON `show_y` PANELS BY DEFAULT, which is the same rule the tick labels follow:
    # the standalone is 2x2 and the two panels of a row share one metric and one set
    # of limits, so the left column carries both.
    #
    # ⚠️ **`show_ylabel` UNBINDS THE NAME FROM THE SCALE**, because they answer
    # different questions and only one of them is answered by the panel to the left.
    # `plot_chili_combined.py` puts its whole 1x4 row on ONE scale (`ylim`), so the
    # tick labels are column 0's alone — but the row still spans TWO METRICS, R² in
    # columns 0-1 and weighted F1 in 2-3, and a reader scanning it must not have to
    # remember which pair they are in. So that caller labels every panel and ticks
    # one, which is exactly what `plot_chili_four_arms.panel` does in the row above
    # it (user, 2026-08-26). A shared scale is a reason to drop TICKS. It is never a
    # reason to drop the name of the quantity.
    if show_y if show_ylabel is None else show_ylabel:
        ax.set_ylabel("wF1" if is_clf else "R²")
    if show_title:
        ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)


def load_seed(seed):
    """`(sources, docs, scratch_fp, finetune_fp)` for ONE seed, row-match asserted.

    Split out of `main` on 2026-08-24 so `tools/plot_chili_combined.py` reads the same
    rungs under the same checks — the asserts are what make the three schemes one
    experiment at each n, and a second caller must not get a weaker set of them. Kept
    as its own function when the figure moved to 3-seed means (2026-08-27) so those
    checks run once PER SEED rather than once on an already-averaged number, where a
    mismatched rung would have been silently averaged away.
    """
    probe, probe_docs = load_probe(seed)
    scratch, scratch_docs, scratch_fp = load_gradient(seed, pretrained=False)
    finetune, finetune_docs, finetune_fp = load_gradient(seed, pretrained=True)
    sources = {"probe": probe, "scratch": scratch, "finetune": finetune}
    docs = {"probe": probe_docs, "scratch": scratch_docs, "finetune": finetune_docs}

    # The whole figure rests on the three arms being the same experiment at the same
    # n. Assert it from the JSONs rather than trusting the paths.
    for frac in FRACS:
        for name, doc in ((n, d[frac]) for n, d in docs.items()):
            assert "chili" in doc["registry"], f"{name}@{frac}: registry {doc['registry']!r}"
            assert doc["n_test"] == probe_docs[1.0]["n_test"], f"{name}@{frac}: test split moved"
        for target, _title, _clf in PANELS:
            n = {name: sources[name][(target, frac)][1] for name in sources}
            assert len(set(n.values())) == 1, (
                f"row match broken: {target} at {frac:g} fits {n}")

    assert (scratch_docs[1.0]["finetune_cfg"]["batch_size"]
            == finetune_docs[1.0]["finetune_cfg"]["batch_size"]), \
        "batch floor differs between the gradient schemes — the single rule lies"
    return sources, docs, scratch_fp, finetune_fp


def load_sources():
    """The figure's data: `(sources, docs, scratch_fp, finetune_fp, per_seed)`.

    `sources` holds the MEAN over `SEEDS` at every point and is the only thing the
    drawing reads, so `panel` and `tools/plot_chili_combined.py` are unchanged by the
    move off a single seed. `docs` and the two fingerprints come from the FIRST seed:
    they carry the batch size, the modal floor, the labelled counts and the learning
    rates, all of which are asserted equal across seeds below, so any seed's copy is
    the same object for those purposes.

    `per_seed` is `{seed: sources}`, kept for `--table` and for nothing else — the
    appendix reports the spread this figure no longer draws, and it has to come from
    the same load as the means or the two can disagree.
    """
    per_seed, docs, scratch_fp, finetune_fp = {}, None, None, None
    for seed in SEEDS:
        src, d, sfp, ffp = load_seed(seed)
        per_seed[seed] = src
        if docs is None:
            docs, scratch_fp, finetune_fp = d, sfp, ffp
        else:
            # The fingerprint pins encoder, latent_dim and every GRADIENT_INVARIANT,
            # so this is what makes the three seeds one arm rather than three arms.
            assert (sfp, ffp) == (scratch_fp, finetune_fp), (
                f"seed {seed} is not the same arm as seed {SEEDS[0]}:\n"
                f"  scratch  {sfp} vs {scratch_fp}\n  finetune {ffp} vs {finetune_fp}")

    sources = {}
    for source in per_seed[SEEDS[0]]:
        sources[source] = {}
        for key, (_score, n, *rest) in per_seed[SEEDS[0]][source].items():
            # n_train must match across seeds too, not only across schemes: averaging
            # cells fitted on different numbers of rows would put the mean at an x
            # position none of its seeds occupies.
            ns = {per_seed[s][source][key][1] for s in SEEDS}
            assert len(ns) == 1, f"{source} {key}: n_train differs across seeds: {ns}"
            scores = [per_seed[s][source][key][0] for s in SEEDS]
            # `rest` is the gradient arms' epoch-ceiling flag; a rung counts as
            # ceiling-hit if ANY seed hit it, which is the conservative reading —
            # it marks the point as a lower bound wherever one exists.
            hit = [any(per_seed[s][source][key][2] for s in SEEDS)] if rest else []
            sources[source][key] = (statistics.fmean(scores), n, *hit)
    return sources, docs, scratch_fp, finetune_fp, per_seed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paper", action="store_true",
                    help="paper-ready: drop the provenance block above the axes and "
                         "print a LaTeX caption carrying it instead")
    ap.add_argument("--table", action="store_true",
                    help="print the per-seed values and standard deviations as a LaTeX "
                         "booktabs table and exit, drawing nothing. This is the appendix's "
                         "half of the 2026-08-27 decision to plot means and report spread "
                         "in text rather than as whiskers")
    ap.add_argument("--out", default=None, help="figure stem (default depends on --paper)")
    args = ap.parse_args()

    sources, docs, scratch_fp, finetune_fp, per_seed = load_sources()
    if args.table:
        print(seed_table(per_seed, sources))
        print()
        print(seed_table(per_seed, sources, stat="sd"))
        return
    probe, scratch, finetune = sources["probe"], sources["scratch"], sources["finetune"]
    probe_docs, scratch_docs = docs["probe"], docs["scratch"]
    finetune_docs = docs["finetune"]

    print(f"invariants (encoder, latent_dim, {', '.join(GRADIENT_INVARIANTS)}) — "
          f"identical across all {len(FRACS)} rungs of each arm:\n"
          f"  supervised {scratch_fp}\n  finetuned  {finetune_fp}")
    for target, title, _clf in PANELS:
        print(f"\n{title} ({target}, {metric_key(target)}, {PROTOCOL})")
        for label, source, _c, _m in ARMS:
            fr = PROBE_FRACS if source == "probe" else FRACS
            print(f"    {label:24s} " + "  ".join(
                f"{f*100:g}%={sources[source][(target, f)][0]:.3f}" for f in fr))
        print(f"    {'n_train (matched @ shared)':24s} "
              + "  ".join(f"{f*100:g}%={probe[(target, f)][1]}" for f in PROBE_FRACS))
    for label, source, _c, _m in ARMS:
        if source == "probe":
            continue  # closed-form: no epoch budget to hit
        src = sources[source]
        hits = [(t, f) for (t, f), (_s, _n, hit) in src.items() if hit]
        print(f"\n{len(hits)}/{len(src)} {label} cells never early-stopped"
              + (f" (lower bounds): {sorted(hits)}" if hits else ""))

    paperstyle.use()
    # WIDTH is fixed in every mode — the type sizes are calibrated to that printed
    # width, and rescaling in \includegraphics breaks the match to body text.
    # Paper height compressed to the collision limit (2026-08-21, from 3.6): below
    # ~3.0 the "batch floor" label and the y-tick labels start crowding the panels.
    fig, axes = plt.subplots(2, 2, figsize=(paperstyle.WIDTH, 3.0 if args.paper else 3.9))
    # One xlim for all four panels so the round XTICKS land identically; padded the
    # same way the old per-panel limits were.
    ns_all = [probe[(t, f)][1] for t, _, _ in PANELS for f in PROBE_FRACS]
    xlim = (min(ns_all) * 0.84, max(ns_all) * 1.19)
    for i, (ax, (target, title, is_clf)) in enumerate(zip(axes.flat, PANELS)):
        panel(ax, sources, target, title, is_clf, scratch_docs,
              show_xlabel=i >= 2, show_y=i % 2 == 0, xlim=xlim)

    # Handles carry the marker — that is what discharges the low-contrast slots'
    # stated label obligation (paperstyle: never colour alone) — but on a SHORT
    # handle, so the line-marker-line sequence reads as a marker rather than as a
    # dash pattern these solid lines do not use.
    handles = [plt.Line2D([], [], color=c, marker=m, markeredgecolor="white",
                          markeredgewidth=0.8) for _l, _s, c, m in ARMS]
    # NO `labelcolor`: the entries name the three schemes, which are data, so they
    # take `text.color` like every other data-naming string in the figure. It was
    # `labelcolor=MUTED` until 2026-08-25 — the same grey the metric label carried,
    # and greyed out for the same reason (it reads as furniture), which is exactly
    # the reading the review rejected. This was the only paper figure setting it; the
    # other three never did, so they were already black and Figure 3 was the outlier.
    fig.legend(handles, [lab for lab, _s, _c, _m in ARMS], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.4,
               handlelength=1.1, handletextpad=0.55)

    if args.paper:
        # `pad=0.3`, not the 1.08 default: the default leaves ~0.13 in of blank on
        # every side, and since `paperstyle.save` then CROPS to the ink, that blank is
        # not whitespace in the output — it is 0.26 in of canvas the axes never got to
        # use, which came off the saved width instead and had the figure rescaling 5%
        # in \includegraphics. Tightening it here spends the same space on the panels.
        # `rect` bottom 0.075 is MEASURED against the legend, not guessed: it is the
        # band the axes give up to `fig.legend`, and the legend does not shrink with
        # it, so shrinking the band pulls the x label DOWN toward a fixed legend top.
        # At the old 0.09 the clearance between the x label and the legend was
        # 0.063 in of dead space; 0.075 leaves 0.018 in, and 0.06 overlaps them by
        # 0.027 in. Re-measure before changing it — the figure does not error when
        # they collide, it just prints them on top of each other.
        paperstyle.layout(fig, rect=(0, 0.075, 1, 1))
    else:
        ft = finetune_docs[1.0]["finetune_cfg"]
        fig.suptitle(
            # BOTH LINES UNDER ~80 CHARS: at WIDTH and 8 pt a longer string runs off
            # the canvas rather than wrapping, and `save` crops to the ink, so the
            # overrun lands in the saved WIDTH (measured: 5.841 in before this trim).
            "Fit AUGMENTED → eval AUGMENTED · cnn_vicreg_mpfull_final · "
            f"mean of seeds {'/'.join(str(x) for x in SEEDS)}\n"
            f"probe: ridge · baseline: lr {scratch_docs[1.0]['finetune_cfg']['lr_enc']:g} · "
            f"fine-tuned: lr {ft['lr_enc']:g}/{ft['lr_head']:g} · 300 ep",
            x=0.012, ha="left", fontsize=8)
        paperstyle.layout(fig, rect=(0, 0.075, 1, 0.90))

    # PDF only (2026-08-21): this figure ships to the paper, and the companion PNG
    # `paperstyle.save` also writes was going stale beside the regenerated PDF.
    #
    # ⚠️ VIA `pdf_only`, NOT VIA A LOCAL `fig.savefig` (2026-08-25). Hand-rolling the
    # write is how this became the ONE paper figure without a tight crop: the crop is
    # applied in `paperstyle.save`, so a figure that does not call it keeps the blank
    # band around the axes. It saved at exactly 5.5 x 3.0 in while every other figure
    # was trimmed to its ink, which is the whitespace the review saw above this one.
    stem = args.out or f"probe_datafrac{'_paper' if args.paper else ''}"
    paperstyle.save(fig, stem, pdf_only=True)
    if args.paper:
        print("\n" + caption(probe_docs, scratch_docs, finetune_docs))


if __name__ == "__main__":
    main()
