"""tools/plot_chili_four_arms.py — the CHILI arm comparison for `cnn_vicreg_mpfull_final`.

    python tools/plot_chili_four_arms.py --finetune untuned --paper   # THE DELIVERABLE
    python tools/plot_chili_four_arms.py                    # untuned finetune (default)
    python tools/plot_chili_four_arms.py --finetune lpft    # the LP-FT arm
    python tools/plot_chili_four_arms.py --finetune tuned   # the val-selected LR arm

**THE PAPER FIGURE IS `chili_arms_untuned_paper.pdf`** (promoted 2026-08-21), i.e.
the `untuned` arm under `--paper` — the headline policy in FINETUNE_SOURCES, rendered
without the provenance header. The plain stems keep that header deliberately: it is
the only thing that names a saved variant's LR and warm-start state, so it is the
diagnostic view and never the one that ships.

Three ways of getting a number out of CHILI — each drawn TWICE, once fitted on the
FIXED spectra and once on VARIED ones, evaluated on the SAME 360-row varied
test split in every cell:

    supervised end-to-end              runs/pdf/supervised/cnn_lr0.001_seed42
    frozen pretrained + linear probe   <run>/probe_chili.json
    finetuned pretrained (unfrozen)    see FINETUNE_SOURCES below

    fixed -> protocol `clean_aug` (solid)  ·  varied -> `aug_aug` (hatched)

    (The protocol KEYS keep the older clean/aug spelling because that is what is on
    disk in every JSON; only the reader-facing words changed. See CHANNELS.)

The first uses no pretrained weights; rows 2-3 start from
`cnn_vicreg_mpfull_final` (full-MP VICReg, one of the two headline models —
`RESULTS.md` -> *The headline models*).

**EVERY ARM CARRIES ITS OWN VARIED-FIT CEILING (2026-08-21), DRAWN AS A HATCHED
BAR NESTED AT THE SAME x.** This supersedes two earlier decisions on the same
question, and all three are recorded because the argument recurs. 2026-08-14 dropped
the supervised+augmented cell entirely: *"it is the standing Q1 bar and it beats all
three of these on all four targets — but it answers a different question, because it
is the only arm that gets to see the shift during fitting. Removing it makes this
figure ask one question with one controlled variable."* Earlier the same day as this
entry it came back as a FOURTH BAR, on the argument that a reader who cannot see the
ceiling cannot tell whether the probe's 0.938 on `np_size` is close to solved or
merely the best of three poor options.

The present layout keeps both readings and trades neither away. The varied-fit cell is
no longer a fourth ARM competing with the other three; it is a SECOND FIT CHANNEL of
each arm, in that arm's own hue, at that arm's own x. So the figure asks two questions,
each with one controlled variable, separated by geometry rather than by omission:
ACROSS ARMS at a fixed fill, does pretraining pay and should the encoder be frozen;
WITHIN an arm, what does that arm gain from being allowed to see the shift while
fitting. **The caption must keep saying that the hatched bars are the ones fitted on
varied spectra**; without that clause this figure does become the thing 2026-08-14
refused to ship.

⚠️ **THE THREE OVERHANGS ARE THE POINT, and they are not the same size.** On R² the
varied fit wins all twelve cells, so the hatched bar always rises past the solid one
and the visible overhang IS the gap: baseline +0.22 to +0.56, fine-tuned (LP-FT) +0.12 to
+0.24, probe +0.03 to +0.10. The probe's overhang being the SHORTEST — the frozen
pretrained encoder is the arm that least needs shifted labels — is a comparison the
four-bar layout could not make, because there the ceiling belonged to the supervised
arm alone. It holds on all four targets under `--finetune lpft`; under `untuned` the
fine-tuned arm is narrower on M–O distance (+0.058) and under `tuned` on nanoparticle
size (+0.023), so the claim is the LP-FT figure's, which is the headline one.

⚠️ **UNDER `--metric mae` THE NESTING INVERTS**, because there lower is better: the
varied fit is the SHORTER bar and the hatch sits INSIDE the solid fixed bar rather
than above it. Nothing is hidden either way — the draw order is computed from the two
values per arm, never assumed — but the reader's cue flips, which is why the metric
is named on the y-axis label in both modes. Under MAE the value label also moves from
inside the bar to just above it, since there the fixed bar is the outer one.

⚠️ **THE SOLID AND HATCHED BARS OF AN ARM ARE THE SAME RUN AND THE SAME JSON**,
differing in FIT CHANNEL and in nothing else: identical architecture, seed, LR, splits
and `target_cn` mask by construction, not by assertion. That is the strongest matching
guarantee in the figure, it is free, and it now holds for all three rows rather than
only the supervised one.

⚠️ **EVERY LR IN THE FIGURE IS CLEAN-SELECTED, and there is no aug-channel search to
fix that.** `RESULTS.md` -> Q3 records 1e-3 as argmin val loss on `target_np_size`,
**clean channel**; the other cells of that grid (3e-5 ... 3e-3) ran `clean_clean` on
`np_size` only. The LP-FT LR was selected the same way — `scripts/chili_lpft.slurm`
STAGE=lr runs `--aug-signal none`. So no `aug_aug` re-selection exists on disk for any
arm and every hatched bar inherits an LR chosen for a channel it is not fitted on. The
bias runs AGAINST the ceilings, which is the harmless direction here, but the caption
says it.

⚠️ **THIS FIGURE'S NUMBERS ARE 3-SEED MEANS AND `RESULTS.md` QUOTES 3-SEED MEDIANS**,
so the two will differ in the third decimal on the same cells. Both are correct and
they are different statistics of the same three runs; a number lifted from here into
that file must be re-derived rather than copied. (Before 2026-08-27 the mismatch was
larger and different in kind — this figure was seed 42 alone.)

WHY THESE FOUR TARGETS. `target_mo_bond` is r = 0.999 with `target_mean_bond`
(`PROGRESS.md` -> s20; 0.9985 on the test split) so plotting both double-counts one
task, and `target_metal` separates nothing (every arm ~0.00-0.03). **The bond panel
is `mo_bond`, switched from `mean_bond` 2026-08-17 for CROSS-FIGURE CONSISTENCY**:
`plot_chem_probe_arms.py` settled on `mo_bond` on validity grounds (it measures the
same quantity in both registries; `mean_bond` does not), and after the switch all
three paper figures show one bond target under one axis label. Within CHILI-3K the
two are interchangeable for any FIXED model — at these score levels the same
predictor's R² can move ≤ 0.06 between them (triangle inequality on the 0.010 Å
residual vs 0.18 Å target sd) and the deterministic probe moves 0.005. The GRADIENT
arms move more (scratch seed 42: 0.477 → 0.367) because each target is a separate
end-to-end run (`analysis/finetune.py` trains per target) and clean→aug transfer is
checkpoint-unstable; the sign of that gap flips across scratch seeds 43/44, so it is
training noise, not a target property. Regression reads R², classification weighted
F1 with the modal-class floor drawn as a reference line (never a hue —
`analysis/paperstyle.py`).

MATCHING, VERIFIED FROM THE JSONs (2026-08-14), not assumed:
- **Same architecture.** All three sources record `encoder=cnn`, `latent_dim=256`,
  and `CNNEncoder`'s geometry depends on NOTHING ELSE — so the scratch arm is the
  same network as the pretrained one, unlike the transformer case that
  `analysis/grid2x2.py` excludes for arch mismatch.
- **Same data.** One registry, `signal=xpdf` / `aug_signal=xpdf_aug`,
  `signal_len=5000`, `r_range=[0, 49.99]`, splits 2530/290/360, and an identical
  `target_cn` mask (2310/265/340) in every arm.
- **No train-size loophole.** The probe fits on the same 2530 rows; it simply does
  not consume the 290 val rows, having no hyperparameter to select.
- **No test leakage.** Every LR was selected on val loss with test never read
  (`RESULTS.md` -> Q3), and the gradient arms early-stop on the CLEAN val split.

⚠️ **TWO ASYMMETRIES THAT SURVIVE, both stated in the figure rather than fixed:**
1. **THE BARS ARE 3-SEED MEANS EXCEPT THE PROBE'S, WHICH IS n = 1** (2026-08-27,
   superseding the seed-42-only decision of 2026-08-14). Baseline and, under
   `untuned`, fine-tuned average seeds 42/43/44; the probe cannot, because at full
   data the ridge fit on a fixed encoder over fixed rows is deterministic and its only
   variance source is the pretraining seed. So this figure mixes two statistics and
   **the caption must say which bar is which** — averaging also narrows the two
   gradient arms, so the probe bar is the noisy one despite looking identical.
   What the change cost, measured: the baseline rises on three targets (seed 42 was
   its low seed), so **the probe's lead on the fixed channel narrows by 0.07-0.08** —
   `cn` +0.426 → +0.342, `oxidation` +0.492 → +0.416, `mo_bond` +0.296 → +0.230. Still
   far outside the noise floor. ⚠️ On `mo_bond` the baseline's fixed cell is
   0.434 ± 0.213 (0.367 / 0.672 / 0.262), so ONE OF ITS SEEDS (0.672) BEATS THE
   PROBE'S 0.663 — the appendix says so rather than leaving it to be found.
   The earlier note that seed 42 was "the scratch arm's lowest on three of four" is
   why the means moved the way they did, and is kept in the appendix table.
   `--finetune tuned` and `lpft` have seed 42 only, so under those the third bar is a
   mean of one.
2. **LR.** Scratch is val-selected (1e-3). Which pretrained arm you get depends on
   `--finetune`; see FINETUNE_SOURCES.

`probe_chili.json`, NOT `probe.json`: the default name was clobbered by the s32
RRUFF sweep (the overwrite `downstream_eval.py`'s own --out-name help warns about),
so the CHILI probe was re-run 2026-08-14 under a registry-qualified name:

    python -m analysis.downstream_eval --runs runs/pdf/sweep \
        --filter '*cnn_vicreg_mpfull_final*' --out-name probe_chili.json

COLOR. Arm colours come from `analysis.paperstyle.ARM_COLORS`, re-searched 2026-08-25
and validated there under the all-pairs gate, which that set is the first arm palette
here to pass outright. Two consequences for this figure specifically:
  - every slot holds a WHITE value label (8.4:1 / 4.6:1 / 8.3:1), so the luminance
    switch this module used to carry (`on_bar_ink`) is gone and every number in the
    figure is the same ink. Under the old ramp `fine-tuned` needed dark ink while the
    other two took white, which read as emphasis on one arm that was never intended;
  - `baseline` is now the lone WARM hue and `probe`/`fine-tuned` a COOL pair, so the
    colours group the way the experiment groups — the two pretrained schemes share an
    encoder and the baseline shares nothing with either.
Arms are keyed in the figure legend, not on the axis, which is what discharges the
palette's never-colour-alone obligation.

⚠️ **ROTATED TO VERTICAL BARS ON 2026-08-25** (review feedback on the workshop
submission), together with: the fill key moved out of the dead band above the bars to a
figure-level legend under the row; "fit clean"/"fit aug." renamed to the "fixed"/
"varied" the paper text and Figure 4 already use; the modal-floor label moved off the
line and into that legend; and the shared style changes in `paperstyle.use` — black
spines and ticks, no grid, and a tight save crop. `panel` and `fill_key` carry the
reasoning for each; `analysis/paperstyle.py` carries the palette's.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from analysis import paperstyle
from analysis.paperstyle import ARM_COLORS, INK, MODAL, MUTED

REPO = Path(__file__).resolve().parent.parent

RUN_STEM = "2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372"
PRETRAIN_RUN = REPO / "runs/pdf/sweep" / RUN_STEM

#: EVERY BAR IS THE MEAN OVER THESE SEEDS (2026-08-27, user) — where they exist. The
#: spread is reported in the appendix (`--table`) rather than drawn: the value label
#: is pinned to the bar's top edge and an error bar is centred on that same edge, so
#: the two occupy one place by construction, and at a quarter-WIDTH panel the label is
#: 0.21 in against a 0.25 in bar, leaving nowhere for the whisker to go.
#:
#: ⚠️ **THE PROBE IS n = 1 AND CANNOT BE OTHERWISE, so this figure mixes statistics
#: and the caption says so.** Every cell here is full data, where
#: `analysis.finetune.subsample_train_mask` returns the mask untouched, and the ridge
#: fit on a fixed encoder over fixed rows is deterministic — the seed has nothing to
#: act on. Its only variance source is the PRETRAINING seed, of which there is one
#: (PROGRESS.md item 12). Averaging three draws also narrows the other two arms, so
#: the probe bar is noisier than its neighbours in a way the figure cannot show.
#:
#: ⚠️ `--finetune tuned` and `--finetune lpft` HAVE SEED 42 ONLY, so under those the
#: third bar is a mean of one. `load_sources` averages whatever seeds exist per arm
#: and reports the count, rather than refusing — the two diagnostic variants would
#: otherwise stop working to buy uniformity in a figure that does not ship.
SEEDS = (42, 43, 44)
SCRATCH_SEEDED = {s: REPO / f"runs/pdf/supervised/cnn_lr0.001_seed{s}/finetune.json"
                  for s in SEEDS}
#: The untuned fine-tune at three seeds. Its seed-42 cell is also in
#: `runs/pdf/sweep/<stem>/finetune.json`, which has no seed in its path — all three
#: were re-run into this tree (job 5503644) so the arm is one batch rather than a
#: 2026-07 point spliced against two new ones. The re-run reproduced the old cell to
#: within 0.0011 on every panel target.
UNTUNED_SEEDED = {s: REPO / f"runs/pdf/datafrac_full/seed{s}/{RUN_STEM}_enc1e-05_head0.0001/finetune.json"
                  for s in SEEDS}

#: Which finetune the third bar reads, and what each one is.
#:
#: `untuned` — lr_enc=1e-5 / lr_head=1e-4, random head init. The repo's headline
#:     policy (`RESULTS.md` -> *the UNTUNED arm is the headline*): the LR came from
#:     prior work rather than selection on this data, which keeps it non-circular,
#:     and the asymmetry against the val-selected scratch arm biases AGAINST it.
#:     But it predates the s32 LP-FT recipe, so it is not the strongest finetune.
#: `tuned` — lr 1e-3, val-selected, LR-matched to the scratch arm. Still a random
#:     head init. Its LR was selected on `target_np_size` clean val loss alone, and
#:     `RESULTS.md` Q3 measures that criterion costing shift robustness
#:     (rho = -0.634): it wins its own selection target and loses `oxidation` by 0.18.
#: `lpft` — `--warm-start-head`, the recipe `RESULTS.md` Q5 calls "done properly",
#:     at its own val-selected LR. Written by `scripts/chili_lpft.slurm`; the LR is
#:     not known until stage 1 selects it, so the path is globbed.
FINETUNE_SOURCES = {
    "untuned": PRETRAIN_RUN / "finetune.json",
    "tuned": REPO / "runs/pdf/ssl_lr" / f"{RUN_STEM}_enc0.001_head0.001" / "finetune.json",
    "lpft": REPO / "runs/pdf/lpft",  # globbed — see resolve_finetune
}

#: (target, panel title, is_classification). One row, regression then
#: classification (2x2 -> 1x4 on 2026-08-19, to reclaim vertical page space).
#: Under `--metric mae` the two classification panels drop out — MAE is undefined
#: for a class label, and a "distance" between oxidation states would be fiction.
PANELS = (
    ("target_np_size", "Nanoparticle size", False),
    ("target_mo_bond", "M–O distance", False),
    ("target_cn", "Coordination number", True),
    ("target_oxidation", "Oxidation state", True),
)

#: Physical unit of each regression target, for the MAE axis. Both are angstroms —
#: `data/builders/chili.py`: np_size is the particle diameter, mo_bond the minimum
#: metal-oxygen distance. The panel title matches `plot_chem_probe_arms.py`'s.
UNITS = {"target_np_size": "Å", "target_mo_bond": "Å"}

#: One ROW each, drawn top-to-bottom. (label, source key, colour)
#:
#: THE LABELS ARE THE UNIFIED PROBE-FIGURE NAMES (renamed 2026-08-19): every figure
#: that draws these arms — this one, `plot_probe_datafrac.py`,
#: `plot_chem_probe_arms.py` — uses exactly "baseline" / "probe" / "fine-tuned", so
#: a reader carries one vocabulary across all of them. The pedantic objection to
#: short names is correct and worth recording: ALL THREE ARMS ARE SUPERVISED on the
#: same labels, so "baseline" vs "probe" does not name the axis that actually
#: varies. What varies is (a) whether the encoder starts random or pretrained, and
#: (b) whether it is updated. The precise labels would be "random init, encoder
#: trained" / "pretrained, encoder frozen" / "pretrained, encoder finetuned".
#:
#: The short ones are used anyway because they fit the legend and read instantly.
#: **That makes the caption load-bearing**: it must say that all three are supervised
#: on identical labels, that arms 2-3 share one pretrained encoder, and that the
#: hatch is the fit channel. `--paper` prints that caption; do not ship the figure
#: with a caption that omits it.
#:
#: SINCE THE 2026-08-25 ROTATION THESE NAMES ARE LEGEND ENTRIES, NOT AXIS TICKS.
#: They were y tick labels while the bars ran horizontally; vertical bars would need
#: them as x tick labels, and "baseline" + "probe" + "fine-tuned" side by side need
#: ~1.5 in against the ~1.0 in a quarter-WIDTH panel has. The colour is now the only
#: in-panel cue for the arm, which is why `fill_key` is mandatory rather than nice to
#: have.
ARMS = (
    ("baseline", "scratch", ARM_COLORS["supervised"]),
    ("probe", "probe", ARM_COLORS["probe"]),
    ("fine-tuned", "finetune", ARM_COLORS["finetuned"]),
)

#: (protocol, hatch, legend label). BOTH are drawn on EVERY row — this is the pair
#: that used to be a fourth arm; the module docstring's entry on the nested layout is
#: the decision record and should be read before changing anything here.
#:
#: `///` AND NOT `//` (2026-08-25, review): at bar width the sparser pattern put two
#: or three strokes on a bar and one in the legend swatch, which reads as a smudge
#: rather than as a fill. Density lives HERE and nowhere else, so the bars and the key
#: cannot drift apart — a key hatched differently from the thing it keys is worse than
#: no key. `plot_chem_probe_arms` carries the same value for the same reason.
#:
#: HATCH, NOT A HUE, FOR THE FIT CHANNEL. It is the convention
#: `plot_chem_probe_arms.py` already settled on 2026-08-21 for exactly this contrast
#: ("within a hue, solid vs hatched differ in the fit CHANNEL alone"), so the two
#: figures teach one reading. Colour follows the ARM; a fourth hue for a channel would
#: need a fresh palette validation and would break the ramp's meaning. The white hatch
#: rides on the existing white edgecolor.
#:
#: The order is the DRAW order only in the sense that `sorted` reverses it when the
#: metric inverts — see the draw loop. The legend key follows this tuple's order in
#: both modes, so it always reads fixed-then-varied.
#:
#: **THE LABELS ARE "fixed"/"varied", NOT "fit clean"/"fit aug." (2026-08-25).** The
#: protocol KEYS stay `clean_aug`/`aug_aug` — they name what is on disk and renaming
#: them would orphan every JSON — but the reader-facing words now match what the paper
#: text and Figure 4 already say. Review feedback on the submission: Figure 2 said
#: "clean"/"aug" for the same distinction Figure 4 called "fixed"/"varied", so the two
#: figures taught two vocabularies for one axis. `fixed` = CHILI's own single
#: instrument realization, `varied` = re-simulated at drawn instrument parameters.
CHANNELS = (
    ("clean_aug", None, "fixed"),
    ("aug_aug", "///", "varied"),
)


#: The fill key's neutral swatch. LIGHT with a dark hatch, because the Set2 arms are
#: light and a key that does not look like the thing it keys is worse than none.
#: ColorBrewer's own Set2 grey, because a hand-picked grey beside eight designed ones
#: is the kind of drift `paperstyle` exists to stop.
KEY_NEUTRAL = "#b3b3b3"

#: Height of a 7.5 pt value label, in y-DATA units, used only to decide whether the
#: modal floor falls inside that label's box. A constant and not a measurement because
#: `panel` runs BEFORE `tight_layout`, so `ax.transData` at that point does not yet
#: describe the axes the label will be drawn in.
#:
#: **ANY VALUE IN [0.066, 0.29] GIVES IDENTICAL OUTPUT**, which is why a constant is
#: safe here rather than merely convenient. The test fires for a label iff the floor
#: sits within LABEL_H below its top edge, and the four targets leave a wide gap
#: between the two cells that must fire and the four that must not:
#:     must fire   cn baseline    top 0.441 vs floor 0.418  -> needs >= 0.023
#:                 oxid baseline  top 0.316 vs floor 0.250  -> needs >= 0.066
#:     must not    cn finetuned   top 0.713 vs floor 0.418  -> fires at >= 0.295
#:                 oxid finetuned top 0.718 vs floor 0.250  -> fires at >= 0.468
#: 0.11 is the true height at the current 1.0 in axes (7.5 pt = 0.104 in over a span
#: of 1.04); 0.075 was the true height at the 1.45 in axes this figure had before it
#: was shortened. Both sit inside the window, so the shortening needed no re-tune —
#: and a geometry that moved LABEL_H outside it would have to be extreme.
LABEL_H = 0.11

#: THE MODAL-FLOOR LINE, AND WHY IT IS NOT PLAIN `MODAL` GREY (2026-08-25).
#: This is the one reference line in the paper that has to cross FILLED BARS — the
#: floor sits at 0.42 (cn) and 0.25 (oxidation) and every bar in both panels is taller
#: than that — so `paperstyle`'s #5f5e5a, which is tuned to RECEDE against white page,
#: recedes into the bars too and the review could not see it. Near-black at 1.1 pt
#: reads on both backgrounds. It stays a neutral, so `paperstyle`'s "a reference line
#: never takes a hue" rule is intact.
#:
#: ⚠️ IT NEEDS A WHITE HALO THE MOMENT THE ARMS GO DARK. The Set2 arms are pastels, so
#: the dark dash clears 22-27 dE against every bar on its own. A brief Tol-dark pass
#: put the bars at L 0.28-0.50, where a near-black dash on #222255 is close to
#: invisible, and the line needed `path_effects=[withStroke(2.8, "white")]` to survive.
#: If the palette ever darkens again, that halo comes back with it.
MODAL_LINE = {
    "color": INK,
    "linewidth": 1.1,
    "linestyle": (0, (3, 2)),
    "zorder": 4,          # above the bars (3), BELOW the value labels (5)
}


def resolve_finetune(which):
    """The finetune.json for `which`. `lpft`'s LR is only known after selection.

    `runs/pdf/lpft/` holds BOTH stages: the 6 stage-1 LR cells and, at the winning
    LR, the stage-2 run that overwrote that cell in place. So the directory name
    cannot identify the arm — only the CONTENTS can. Stage 1 ran one target on the
    clean channel only (`--aug-signal none`); stage 2 ran five targets across three
    protocols. Selecting on "has clean_aug for every target this figure draws"
    therefore picks the final run and cannot pick a search cell, whatever LR won.
    """
    path = FINETUNE_SOURCES[which]
    if which != "lpft":
        return path
    wanted = {t for t, _, _ in PANELS}
    hits = [p for p in sorted(path.glob(f"{RUN_STEM}_enc*/finetune.json"))
            if wanted <= set(json.loads(p.read_text())["tasks"])
            and all("clean_aug" in json.loads(p.read_text())["tasks"][t] for t in wanted)]
    if len(hits) != 1:
        raise SystemExit(
            f"expected exactly 1 completed LP-FT run under {path}, found {len(hits)}"
            f" (stage-1 LR cells are excluded by content, not by name).\n"
            "Run scripts/chili_lpft.slurm STAGE=lr, then analysis.ssl_lr_select, then "
            "STAGE=final, and rsync the tree back."
        )
    return hits[0]


def load(path):
    with open(path) as fh:
        return json.load(fh)


#: Column heads for `--table`, in the layout the thesis already uses: one row per
#: (scheme, fit channel), one column per target, the two metrics grouped.
TABLE_COLS = (("target_np_size", "NP size"), ("target_mo_bond", "M--O dist."),
              ("target_cn", "Coord.\\ no."), ("target_oxidation", "Oxid.\\ state"))


def seed_table(per_seed, metric, stat="mean"):
    """The appendix table for this figure, in the thesis's own layout.

    The other half of the 2026-08-27 decision — bars are means, spread is reported
    here. Generated from the same `load_sources` call that draws the figure, so a
    number here cannot disagree with the bar above it.

    ⚠️ A CELL WITH NO PARENTHESIS IS n = 1, NOT sd = 0, and the caption says so: the
    probe fits on all the labels, so there is no subsample to reseed and a closed-form
    fit on a fixed encoder over fixed rows is deterministic.
    """
    def cell(source, target, protocol, is_clf):
        # ONE NUMBER PER CELL, never `value (sd)`. The inline form was tried and
        # withdrawn (2026-08-27, user): it triples every column and overruns the text
        # block in both this table and the shift table. Two tables of one shape cost a
        # float and keep the layout the thesis already uses.
        k = "mae" if metric == "mae" else ("f1_weighted" if is_clf else "r2")
        docs = per_seed[source]
        ys = [docs[sd]["tasks"][target][protocol][k] for sd in sorted(docs)]
        if stat == "sd":
            return f"{statistics.stdev(ys):.3f}" if len(ys) > 1 else "---"
        return f"{statistics.fmean(ys):.3f}"

    cols = [(t, h) for t, h in TABLE_COLS
            if metric != "mae" or not dict((x[0], x[2]) for x in PANELS)[t]]
    clf = dict((t, c) for t, _title, c in PANELS)
    body = []
    for i, (label, source, _c) in enumerate(ARMS):
        if i:
            body.append(r"        \addlinespace")
        for j, (protocol, _hatch, chname) in enumerate(CHANNELS):
            name = label if j == 0 else ""
            body.append(f"        {name:<10s} & {chname:<6s} & " + " & ".join(
                cell(source, t, protocol, clf[t]) for t, _h in cols) + r" \\")
    heads = " & ".join(h for _t, h in cols)
    n_seeds = len(SEEDS)
    if stat == "sd":
        return rf"""% --- generated by tools/plot_chili_four_arms.py --finetune untuned --table ---
\begin{{table}}[h]
    \centering
    \caption{{Standard deviations over the {n_seeds} seeds behind
    Table~\ref{{tab:fig2}}. A dash marks $n{{=}}1$: the probe cannot be seeded here,
    because every cell fits on all the labels, so there is no subsample to draw and a
    closed-form fit on a fixed encoder over fixed rows is deterministic --- its only
    variance source is the pretraining run, of which there is one. Entries are the
    sample standard deviation at $n{{=}}3$ and are correspondingly poorly determined;
    differences below the $0.0143$ fixed-seed re-run noise floor should not be read in
    either direction. The largest entry is the baseline's on fixed spectra for M--O
    distance, where the three seeds run 0.367, 0.672 and 0.262 --- one of them above
    the probe's 0.663.}}
    \label{{tab:fig2_sd}}
    \begin{{tabular}}{{ll{"c" * len(cols)}}}
        \toprule
        & & \multicolumn{{2}}{{c}}{{$R^2$}} & \multicolumn{{2}}{{c}}{{weighted F1}} \\
        \cmidrule(lr){{3-4}} \cmidrule(lr){{5-6}}
        Scheme & Training labels & {heads} \\
        \midrule
{chr(10).join(body)}
        \bottomrule
    \end{{tabular}}
\end{{table}}"""
    return rf"""% --- generated by tools/plot_chili_four_arms.py --finetune untuned --table ---
\begin{{table}}[h]
    \centering
    \caption{{Exact numbers behind Figure \ref{{fig:instrument}}. Test performance of
    each scheme trained on fixed and on varied spectra. Each entry is the mean over
    {n_seeds} seeds, which vary the head initialisation and the batch order; standard
    deviations are in Table~\ref{{tab:fig2_sd}}. The probe's rows are single runs and
    cannot be otherwise --- every cell fits on all the labels, so there is no subsample
    to draw, and a closed-form fit on a fixed encoder over fixed rows is deterministic;
    its only variance source is the pretraining run, of which there is one. Averaging
    narrows the two end-to-end schemes but not the probe, so the probe's entries carry
    more uncertainty than their neighbours despite being written the same way.}}
    \label{{tab:fig2}}
    \begin{{tabular}}{{ll{"c" * len(cols)}}}
        \toprule
        & & \multicolumn{{2}}{{c}}{{$R^2$}} & \multicolumn{{2}}{{c}}{{weighted F1}} \\
        \cmidrule(lr){{3-4}} \cmidrule(lr){{5-6}}
        Scheme & Training labels & {heads} \\
        \midrule
{chr(10).join(body)}
        \bottomrule
    \end{{tabular}}
\end{{table}}"""


def caption(sources, ft_cfg, metric, seed_list=None):
    """LaTeX caption for `--paper`, built from the JSONs so it cannot drift.

    Everything the shortened bar labels stop saying has to be here: that all three
    arms are SUPERVISED, that rows 2-3 share one pretrained encoder, that the HATCHED
    bar of each row is the same run fitted on AUGMENTED labels, and the asymmetries
    (single seed; that seed being the worst of the scratch arm's three; every LR being
    clean-selected). A caption that drops those is not a shorter caption, it is a
    wrong one — see the ARMS/CHANNELS comments and the module docstring's entry on the
    nested layout.
    """
    probe, ft = sources["probe"], sources["finetune"]
    scale = ("Bars are R\\textsuperscript{2} for the two regression targets and "
             "weighted F1 for the two classification targets, whose modal-class "
             "floor is the dashed line."
             if metric == "r2" else
             "Bars are mean absolute error in \\AA{}. Each panel carries its own "
             "axis: the two targets differ by ${\\sim}80\\times$ in spread, so bar "
             "lengths are not comparable between panels. Lower is better.")
    seed_list = seed_list or ", ".join(str(x) for x in SEEDS)
    return rf"""% --- generated by tools/plot_chili_four_arms.py --paper ---
\caption{{%
  \textbf{{Pretraining substitutes for shifted labels on CHILI.}}
  Three training schemes, each fitted twice on the same {ft['n_train']} CHILI
  examples and evaluated on the same {ft['n_test']} augmented test examples
  ({probe['n_labelled']['target_cn']['test']} for coordination number, where
  mixed-coordination structures are masked).
  \textbf{{Solid bars are fitted on the \emph{{fixed}} spectra and hatched bars on
  \emph{{varied}} ones}}: the two bars of a scheme are the same run differing in fit
  channel alone --- identical architecture, seed, learning rate and splits --- so
  each hatched overhang is what that scheme gains from seeing the instrument shift
  while fitting, and marks the ceiling its fixed-fitted bar is reaching for.
  At a fixed fill, the three schemes differ only in how the encoder is
  initialised and whether it is updated.
  \emph{{Baseline}} trains a randomly initialised CNN end to end
  (learning rate $10^{{-3}}$, selected on validation loss).
  \emph{{Probe}} and \emph{{fine-tuned}} share one encoder pretrained with
  VICReg on simulated PDFs of the full Materials Project
  (\texttt{{cnn\_vicreg\_mpfull\_final}}), disjoint from CHILI;
  the probe holds it frozen and fits a linear head in closed form, while
  \emph{{fine-tuned}} updates it end to end from a probe-initialised head
  (LP-FT, learning rate ${ft_cfg['lr_enc']:g}$, also validation-selected).
  Value labels are the fixed fit; schemes are keyed by color in the legend.
  Every learning rate here was selected on the \emph{{fixed}} channel, so the
  varied bars inherit a rate chosen for a channel they are not fitted on, which
  biases against them.
  {scale}
  \textbf{{The baseline and fine-tuned bars are means of three seeds
  ({seed_list}); the probe bar is a single run.}} The probe cannot be otherwise:
  every cell here fits on all the labels, so there is no subsample to reseed, and a
  closed-form fit on a fixed encoder over fixed rows is deterministic --- its only
  variance source is the pretraining seed, of which there is one. Averaging narrows
  the two gradient schemes, so the probe bar carries more uncertainty than its
  neighbours despite being drawn the same way, and the pretraining run behind both
  pretrained schemes is $n{{=}}1$ throughout. Per-seed values and standard deviations
  are tabulated in the appendix; the largest is the baseline's on the fixed fit of
  M--O distance, where one of its three seeds exceeds the probe.
}}"""


#: Metric keys a bar can be drawn from. `f1_baseline` is deliberately NOT here: the
#: modal floor is a property of the test split, not of a run, and averaging it would
#: hide a split that had moved instead of failing on it (`mean_doc` asserts it fixed).
MEAN_KEYS = ("r2", "mae", "f1_weighted")


def mean_doc(docs):
    """One doc with every drawn metric replaced by its mean over `docs`' seeds.

    Returning a DOC rather than a table of numbers is what keeps `panel` and
    `tools/plot_chili_combined.py` unchanged by the 2026-08-27 move off a single seed:
    they still read `sources[arm]["tasks"][target][protocol][key]`, and whether that
    number came from one run or three is not their concern. The template is the first
    seed's doc, so every non-metric field — protocols, n_labelled, finetune_cfg, the
    modal floor — is that seed's, which the asserts below make safe.
    """
    seeds = sorted(docs)
    base = copy.deepcopy(docs[seeds[0]])
    if len(seeds) == 1:
        return base
    for target, cells in base["tasks"].items():
        for protocol, cell in cells.items():
            for key in MEAN_KEYS:
                if key not in cell:
                    continue
                cell[key] = statistics.fmean(docs[s]["tasks"][target][protocol][key]
                                             for s in seeds)
            if "f1_baseline" in cell:
                floors = {docs[s]["tasks"][target][protocol]["f1_baseline"] for s in seeds}
                assert len(floors) == 1, \
                    f"{target}/{protocol}: modal floor differs across seeds {floors}"
    for s in seeds:
        assert docs[s]["finetune_cfg"]["seed"] == s, \
            f"cell filed under seed {s} reports seed {docs[s]['finetune_cfg']['seed']}"
    return base


def load_sources(which):
    """The three arms' JSONs for `--finetune which`, every cross-arm invariant asserted.

    Split out of `main` on 2026-08-24 so `tools/plot_chili_combined.py` reads the same
    three files under the same checks. The asserts are what make the three bars one
    experiment; a second caller must not get a weaker set of them.
    """
    ft_path = resolve_finetune(which)
    # `{arm: {seed: doc}}`. The probe has one entry because a closed-form fit at full
    # data is deterministic; `tuned`/`lpft` have one because only seed 42 was run.
    # `untuned` reads UNTUNED_SEEDED rather than the run dir, so its three cells are
    # one batch — see that constant.
    per_seed = {
        "scratch": {s: load(p) for s, p in SCRATCH_SEEDED.items()},
        "probe": {SEEDS[0]: load(PRETRAIN_RUN / "probe_chili.json")},
        "finetune": ({s: load(p) for s, p in UNTUNED_SEEDED.items()} if which == "untuned"
                     else {SEEDS[0]: load(ft_path)}),
    }
    sources = {arm: mean_doc(docs) for arm, docs in per_seed.items()}

    # The whole figure rests on these three being the same experiment. Assert it here
    # rather than trusting the paths: a mis-globbed LP-FT dir or a re-run probe against
    # another registry would otherwise plot silently.
    ref = sources["scratch"]
    for name, doc in sources.items():
        assert "chili" in doc["registry"], f"{name}: wrong registry {doc['registry']!r}"
        for key in ("encoder", "latent_dim", "signal", "aug_signal", "signal_len", "n_train", "n_test"):
            assert doc[key] == ref[key], f"{name}: {key} {doc[key]!r} != scratch {ref[key]!r}"
        # `train`/`test` only: the probe has no `val` entry at all, because a
        # closed-form head has no hyperparameter to select on one. That is the
        # documented asymmetry, not a mismatch — but the rows it FITS and SCORES on
        # must match the gradient arms exactly, per target, mask included.
        for tgt, counts in doc["n_labelled"].items():
            for split in ("train", "test"):
                assert counts[split] == ref["n_labelled"][tgt][split], \
                    f"{name}: {tgt} {split} rows {counts[split]} != scratch {ref['n_labelled'][tgt][split]}"
    assert sources["finetune"]["finetune_cfg"].get("warm_start_head", False) == (which == "lpft"), \
        f"--finetune {which} does not match the run's warm_start_head flag"
    # Every arm now needs BOTH channels. A run missing one would otherwise KeyError
    # deep inside the draw loop with no indication of which arm asked for what — the
    # stage-1 LP-FT cells, which ran `clean_clean` only, are exactly that case.
    for label, source, _ in ARMS:
        for protocol, _, _ in CHANNELS:
            assert protocol in sources[source]["protocols"], \
                f"{label}: {source} has no {protocol} channel (has {sources[source]['protocols']})"
    return sources, per_seed


#: The value label inside each bar. **IT IS SET BY THE BAR WIDTH, and the binding
#: panel is `tools/plot_chili_combined.py`'s, not this module's.** Both figures are
#: 1x4 at `paperstyle.WIDTH`, but the combined one spends width on a spacer column and
#: on a second y axis mid-row, so its bars are 0.220 in against this module's 0.250.
#: "0.94" is 0.031 in per point of type, so 7.5 pt is 0.233 in — WIDER THAN THE
#: COMBINED FIGURE'S BAR, which is what it measured at on 2026-08-26, overhanging both
#: edges by 0.006 in (user). 6.8 pt is 0.211 in: it clears the narrow bar, and on the
#: standalone's wider one it restores exactly the 0.020 in per side that 7.5 pt used
#: to buy there. ONE SIZE FOR BOTH FIGURES rather than a per-caller argument — they
#: print the same numbers from the same JSONs and a reader comparing them should not
#: meet two type sizes for one quantity. `paperstyle`'s band for in-panel annotations
#: is 6.5-7.5, so this stays inside it; below 6.5 the fix would have to be a wider
#: bar or a wider panel instead.
VALUE_FONTSIZE = 6.8


def panel(ax, sources, target, title, is_clf, metric, show_y):
    """One target's three arms x two fit channels, drawn into `ax` as VERTICAL bars.

    `show_y` draws the y tick labels — true for the leftmost panel of the row only,
    since all four share one 0-1 scale. The metric NAME is a y-axis label on every
    panel regardless, because two panels are R² and two are weighted F1 and a reader
    scanning the row must not have to remember which pair they are in.

    ⚠️ **ROTATED FROM HORIZONTAL BARS ON 2026-08-25**, on review feedback: the four
    panels fit on one line either way, and turning them puts the metric on the y axis,
    where a 0-1 scale reads as a height. What the rotation COSTS is the arm names,
    which were y tick labels and cannot be x tick labels here — at a quarter of WIDTH
    a panel is ~1.0 in and "baseline"/"probe"/"fine-tuned" need ~1.5 in side by side.
    They move to the figure legend, which is where `plot_probe_datafrac.py` already
    keeps them, so the two figures now key their schemes the same way.

    THE NESTING IS UNCHANGED and still the point: both bars of an arm start at 0 and
    share an x position, so the TALLER is drawn first and the shorter sits in front of
    it. Which channel is taller flips with the metric — varied wins every R² cell
    (hatch above the solid bar) and every MAE cell (hatch inside it) — so the order is
    computed per arm from the two values and never assumed.
    """
    values, outer = [], []
    for x, (_, source, colour) in enumerate(ARMS):
        cells = sources[source]["tasks"][target]
        paired = {}
        for protocol, _, _ in CHANNELS:
            cell = cells[protocol]
            paired[protocol] = cell["mae"] if metric == "mae" else (
                cell["f1_weighted"] if is_clf else cell["r2"])
        for protocol, hatch, _ in sorted(CHANNELS, key=lambda c: -paired[c[0]]):
            ax.bar(x, paired[protocol], width=0.78, color=colour,
                   edgecolor=INK, linewidth=0.6, hatch=hatch, zorder=3)
        # The value text is the FIXED fit's, on every bar; the varied one is read as
        # an overhang, not as a number. Two numbers per bar is what a quarter-width
        # panel cannot take — at 7.5 pt they collide wherever the pair is close,
        # which on the probe arm is every panel.
        values.append(paired["clean_aug"])
        outer.append(max(paired.values()))

    # THE BAR WIDTH IS SET BY THE VALUE LABEL, not by taste. A quarter-WIDTH panel is
    # ~1.0 in across and holds 3.16 x-units, so a 0.78-wide bar is 0.25 in — and
    # "0.94" at 7.5 pt is 0.21 in. At the 0.62 the first draft used, every label
    # overhung both edges of its own bar.
    ax.set_xlim(-0.58, len(ARMS) - 0.42)
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)

    if metric == "mae":
        # ONE AXIS PER PANEL, in the target's own unit. Both targets are in
        # angstroms but differ ~80x in spread (test sd 14.67 vs 0.180), so the
        # shared axis that makes the R² grid comparable at a glance would flatten
        # mo_bond to nothing. Same trade as tools/plot_au_four_arms.py --metric
        # mae: physical units, and NO cross-panel comparison of bar length.
        top = max(outer)
        fmt = "%.3f" if top < 0.5 else "%.2f"
        for x, v in enumerate(values):
            # ABOVE the bar, not in it: under MAE the fixed bar is the OUTER one, so
            # its top is clear of everything drawn on that x.
            ax.text(x, v + top * 0.03, fmt % v, ha="center", va="bottom",
                    fontsize=VALUE_FONTSIZE, color=INK, zorder=5)
        ax.set_ylim(0, top * 1.30)
        # Capped at 4: the auto locator puts 6 ticks on the mo_bond panel, whose
        # labels are 5 characters each ("0.025"), and they collide at this width.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_ylabel(f"MAE ({UNITS[target]}), lower better")
    else:
        # The floor is needed BEFORE the labels are placed, because a label that
        # would sit on the line moves rather than masking it.
        #
        # NO IN-PANEL LABEL FOR THIS LINE — it is keyed in the figure legend. There is
        # nowhere left to put the word once the figure is rotated: the floor sits at
        # 0.42 (cn) and 0.25 (oxidation) and EVERY bar in both panels crosses it, so
        # any text above the line lands on a bar and any text below it lands inside
        # one. Outside the right spine costs ~0.25 in of crop width on the last panel,
        # which `paperstyle.save` explains is the one thing the tight bbox cannot
        # absorb.
        floor = (sources["probe"]["tasks"][target]["clean_aug"]["f1_baseline"]
                 if is_clf else None)
        if is_clf:
            ax.axhline(floor, **MODAL_LINE)
        for x, v in enumerate(values):
            # INSIDE the bar, hugging its top edge — NOT above it, where under R²
            # the label would land squarely on the varied overhang, and NOT in a
            # white bbox, which would ERASE that overhang. `va="top"` pins the text
            # box's top edge a fixed 0.03 below the bar's top, so the gap is the
            # same in every panel; the earlier `va="center"` left the digits sitting
            # optically low in the bar, which is what the review picked up.
            top = v - 0.03
            # ⚠️ **THE LABEL MOVES, THE LINE NEVER BREAKS.** In exactly two cells the
            # floor falls inside a label's box: `cn` baseline (0.471, box down to
            # ~0.396) against a floor of 0.418, and `oxidation` baseline (0.346, box
            # to ~0.271) against 0.250. Something has to give in those two, and an
            # earlier draft gave the label a bbox in the bar's own colour, masking
            # the dashes behind the digits — i.e. it cut a hole in a REFERENCE LINE
            # to protect a value that is also printed in Table 5. That is backwards.
            # The floor is what a reader scans across all three bars to compare
            # against, so it stays continuous and the label drops below it; the label
            # is still inside its own bar and still centred on it, so nothing about
            # its attribution changes.
            if floor is not None and top - LABEL_H <= floor <= top + 0.012:
                top = floor - 0.022
            ax.text(x, top, f"{v:.2f}", ha="center", va="top",
                    fontsize=VALUE_FONTSIZE, color=INK, zorder=5)
        ax.set_ylim(0, 1.04)
        # NOT `sharey`: it propagates a blanked tick-label list to every axis it is
        # shared with, so hiding the labels on panels 2-4 hid them on panel 1 as well.
        # All four carry the same explicit limits anyway, so sharing bought nothing.
        # EVERY DRAWN TICK IS LABELLED. The blanks that used to sit at 0.25 and 0.75
        # left tick MARKS with no numbers beside them, which reads as a missing label
        # rather than as a deliberate minor tick — if a gridline-free axis is going to
        # put a mark somewhere, the mark should say what it is. Blank the whole list
        # on the non-leading panels instead, where the scale is already stated.
        ax.set_yticks((0, 0.25, 0.5, 0.75, 1.0),
                      ("0", "0.25", "0.5", "0.75", "1") if show_y else [""] * 5)
        ax.set_ylabel("wF1" if is_clf else "R²")
    # ONE LINE, CENTRED — and the centring is what makes one line possible. These
    # titles are wider than a quarter-WIDTH panel ("Coordination number" is 1.26 in
    # against 0.89-1.01), so they were wrapped to two lines until 2026-08-26 (user
    # asked for one). A one-line title HAS to overrun its panel; centred, it overruns
    # by half as much and spends it on the GUTTERS, which are empty at title height in
    # both callers. `loc="left"` cannot: it puts the whole 0.37 in of overrun on the
    # right, straight into the next panel's title. Measured clearance between adjacent
    # titles is 0.11 in at the tightest seam (combined figure, coordination number vs
    # oxidation state) and neither figure's crop widens, since the outer two titles
    # stay inside the margins the y labels and x ticks already claim. A LONGER TARGET
    # NAME WOULD BREAK THIS — check both figures, not just one.
    ax.set_title(title, loc="center")


def fill_key(fig, arm_handles=None, y=-0.012):
    """THE FIGURE-LEVEL KEY, AND IT IS NOT OPTIONAL. It carries three things that the
    panels no longer say for themselves: the arm each hue means (the rotation took the
    arm names off the y axis), the fit channel each fill means (the hatch is the only
    thing separating them), and the modal-class floor (the dashed line lost its
    in-panel label to the same rotation).

    ONE ROW UNDER THE PANELS, which is where `plot_probe_datafrac.py` puts its scheme
    key, so a reader meets the same three names in the same place in both figures. It
    replaces the in-axes key that used to sit in a top margin above the first panel —
    the dead band between the title and the bars that the review asked to reclaim.

    `arm_handles` REPLACES the three solid patches, and exists for exactly one caller:
    `tools/plot_chili_combined.py` draws these arms as bars in one row and as marked
    curves in another, so its key has to be the line+marker handle or the two rows get
    two different keys for one set of schemes. It must stay ORDER-MATCHED to `ARMS`;
    the labels are always this module's, so the two figures cannot drift apart on what
    the schemes are called. `y` moves the anchor for a caller whose bottom band is not
    this figure's.

    Neutral grey handles for the two fill patches, never an arm hue: the key states
    what the FILL means, and borrowing one arm's colour would read as if the channel
    belonged to that arm. Swatch and hatch follow the bars — light fill, dark hatch —
    and they flip with the palette's lightness; a key that does not look like the
    thing it keys is worse than none.
    """
    if arm_handles is None:
        arm_handles = [Patch(facecolor=colour, edgecolor=INK, linewidth=0.6)
                       for _, _, colour in ARMS]
    assert len(arm_handles) == len(ARMS), "arm_handles must be order-matched to ARMS"
    handles = list(arm_handles)
    labels = [label for label, _, _ in ARMS]
    handles += [Patch(facecolor=KEY_NEUTRAL, edgecolor=INK, linewidth=0.6, hatch=hatch)
                for _, hatch, _ in CHANNELS]
    labels += [label for _, _, label in CHANNELS]
    handles += [Line2D([], [], **{k: v for k, v in MODAL_LINE.items() if k != "zorder"})]
    labels += ["modal floor"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               # THE SWATCH IS SIZED BY THE HATCH, not by the text. At handlelength
               # 1.7 / handleheight 1.1 the "varied" patch is 0.165 x 0.107 in and
               # fits about two hatch strokes, which reads as a smudge rather than as
               # the pattern on the bars. 2.4 x 1.5 fits enough of them to be
               # recognisably the same fill. The DENSITY is deliberately unchanged —
               # a denser hatch in the key than on the bars would be a key for
               # something the figure does not draw.
               fontsize=7, handlelength=2.4, handleheight=1.5, handletextpad=0.4,
               columnspacing=1.1, borderaxespad=0.0, bbox_to_anchor=(0.5, y))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--finetune", choices=tuple(FINETUNE_SOURCES), default="untuned",
                    help="which finetune the third bar reads; see FINETUNE_SOURCES")
    ap.add_argument("--metric", choices=("r2", "mae"), default="r2",
                    help="r2: all four targets, one shared 0-1 axis. mae: the TWO "
                         "regression targets in angstroms, one axis PER PANEL")
    ap.add_argument("--paper", action="store_true",
                    help="paper-ready: drop the provenance block above the axes and "
                         "print a LaTeX caption carrying it instead. WITHOUT this the "
                         "block stays, because the three --finetune variants are "
                         "otherwise indistinguishable once saved")
    ap.add_argument("--table", action="store_true",
                    help="print the per-seed values and standard deviations as a LaTeX "
                         "booktabs table and exit, drawing nothing — the appendix's half "
                         "of the 2026-08-27 means-in-the-figure decision")
    ap.add_argument("--out", default=None, help="figure stem (default depends on the flags)")
    args = ap.parse_args()


    sources, per_seed = load_sources(args.finetune)
    if args.table:
        print(seed_table(per_seed, args.metric))
        print()
        print(seed_table(per_seed, args.metric, stat="sd"))
        return
    ft_cfg = sources["finetune"]["finetune_cfg"]
    paperstyle.use()

    # HEIGHT ONLY: the width is paperstyle.WIDTH in every mode, because the type
    # sizes are calibrated to that printed width and rescaling in \includegraphics
    # would break the match to 10 pt body text (paperstyle's docstring). Paper mode
    # reclaims the ~0.35 in the provenance block occupied. Both metrics draw ONE
    # ROW (r2 was 2x2 until 2026-08-19): four quarter-width panels cost bar length
    # but halve the figure's height on the page.
    #
    # RAISED ~0.3 in ON 2026-08-21 for the fill KEY, not for a fourth row: the y axis
    # spans len(ARMS) + 2.15 units because the top margin now has to hold a two-line
    # legend as well as the modal-class label pinned at y = -0.52. These are not free
    # parameters to trim back — at the old 1.7 in the legend lands on the first bar.
    if args.metric == "mae":
        panels = [p for p in PANELS if not p[2]]
        fig, axes = plt.subplots(1, len(panels),
                                 figsize=(paperstyle.WIDTH, 1.55 if args.paper else 1.85))
    else:
        panels = list(PANELS)
        fig, axes = plt.subplots(1, len(panels),
                                 figsize=(paperstyle.WIDTH, 1.55 if args.paper else 1.85))
    for i, (ax, (target, title, is_clf)) in enumerate(zip(axes.flat, panels)):
        panel(ax, sources, target, title, is_clf, args.metric, show_y=i == 0)


    if args.paper:
        # `rect` reserves the legend band at the bottom. `fig.legend` is invisible to
        # the layout solver, so without it the key lands on the panels' y tick labels.
        paperstyle.layout(fig, rect=(0, 0.07, 1, 1))
        fill_key(fig)
    else:
        # Two lines, both kept under ~80 characters: at WIDTH inches and 8 pt, longer
        # strings run off the canvas rather than wrapping.
        fig.suptitle(
            "Fit FIXED (solid) / VARIED (hatched) → eval VARIED CHILI test\n"
            f"finetune {args.finetune}: lr {ft_cfg['lr_enc']:g}, warm start "
            f"{'ON' if ft_cfg.get('warm_start_head') else 'off'} · both supervised: lr 1e-3",
            x=0.012, ha="left", fontsize=8)
        # `rect` reserves the legend band only; this matplotlib's tight_layout
        # accounts for the suptitle itself, and reserving headroom on top of that
        # leaves a blank band under the title.
        paperstyle.layout(fig, rect=(0, 0.07, 1, 1))
        fill_key(fig)

    paperstyle.save(fig, args.out or (
        f"chili_arms_{args.finetune}"
        f"{'_mae' if args.metric == 'mae' else ''}"
        f"{'_paper' if args.paper else ''}"))
    if args.paper:
        print("\n" + caption(sources, ft_cfg, args.metric))


if __name__ == "__main__":
    main()
