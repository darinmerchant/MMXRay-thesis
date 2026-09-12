"""tools/plot_chem_shift_interaction.py — SHIFT CONDITION x TRAINING SCHEME for `cnn_vicreg_mpfull_final`.

    python tools/plot_chem_shift_interaction.py            # provenance block above the axes
    python tools/plot_chem_shift_interaction.py --paper    # paper-ready + LaTeX caption

The question this figure answers — and `chem_probe_arms` does not (its axes encode
the FIT channel, so "no shift" is not drawn at all and "materials only" appears
nowhere): **does the progressive shift — none, then a change of materials, then
additionally a change of instrument distribution — affect the three training
schemes differently?** An interaction plot is that claim drawn literally —
non-parallel lines ARE the finding.

    x axis   three conditions: none -> materials -> both
    lines    the three schemes (baseline / probe / fine-tuned), unified vocabulary
    y axis   R2, shared across panels, reaching below 0 for the baseline collapse

**THE INSTRUMENT-ONLY CELL IS NOT DRAWN (final form by decision 2026-08-21,
overriding the same-day one-panel verdict; the full trail is in PROGRESS.md).**
The x axis is a PROGRESSION — each step adds one shift — and the instrument-only
cell belongs to the instrument-axis figure (`plot_chili_four_arms` owns clean-3K
vs aug-3K). ⚠️ THE OMISSION CREATES AN ATTRIBUTION HAZARD: without that column,
the baseline's failure at "both" reads as an effect of the combination, when the
instrument shift ALONE already costs it most of the drop (0.985 -> 0.367 on
`mo_bond`, in the cells this module still loads). The caption's attribution
sentence states this and is mandatory; do not remove it to shorten the caption.

⚠️ **QUOTE `np_size` FROM THIS FIGURE AS "FLAT" AND NOTHING FINER.** Its spread
across every (scheme, condition) cell is 0.87 to 1.00, i.e. 0.13 — inside the
~0.15 same-cell rerun jitter measured on the doubly-shifted channel (RESULTS.md
Q7 §3e). The panel is drawn by explicit decision (it shows the envelope target
carries across the progression for every scheme); differences WITHIN it are not
resolvable, and the shared axis is what keeps them from being dramatized.

FORM HISTORY (2026-08-21, all built, reviewed and deleted the same day): a
four-condition two-panel lines form; grouped vertical bars; `chili_arms`-style
nested bars; a condensed 3-condition dumbbell with ordinal marker fills; a
one-panel `mo_bond` four-condition form. This three-condition two-panel lines
form is the settled figure.

EVERY CELL FITS ON CLEAN NATIVE SPECTRA, and that is what makes the conditions
composable: the materials shift enters through the FIT REGISTRY (CHILI-3K ->
CHILI-100K binary oxides, compositions disjoint from the test set) and the
instrument shift through the EVAL CHANNEL (clean -> `PDF_RANGES` re-simulation).
The aug-FIT arms of `chem_probe_arms` are a different experiment (matching the fit
channel to the eval distribution) and are deliberately not in this figure.

    condition    fit set          protocol       source (per scheme, see below)
    none         clean CHILI-3K   clean_clean    3K doc
    materials    clean CHILI-100K cross_clean    100K doc
    instrument   clean CHILI-3K   clean_aug      3K doc
    both         clean CHILI-100K cross_aug      100K doc

**`--ii` — DESIGN II (2026-08-24), the paper body's framing.** Evaluation is pinned
to the varied channel everywhere (§3.1's "evaluation always uses the varied test
spectra" then holds with no exception) and the shifts enter through TRAINING: first
the fit registry (3K -> 100K, fit channel still varied), then the fit channel
(varied -> the fixed native spectra, a single instrument realization). The final
cell is design I's `both` unchanged; the first two swap to the aug-fit docs.

    condition    fit set          protocol       source
    none         aug CHILI-3K     aug_aug        3K doc
    materials    aug CHILI-100K   cross_aug      100K AUG doc (fit_signal xpdf_aug)
    both         clean CHILI-100K cross_aug      100K doc (as design I)

    scheme      100K AUG doc (materials under --ii)
    baseline    runs/pdf/chem_lr_scratch_aug/ @ per-target val-selected LR
                (`mo_bond` selects 3e-3 at the GRID EDGE within 10%% of the
                runner-up — chem_lr_select flags it a tie; RESULTS.md Q7)
    probe       <run>/probe_chem100k_aug.json
    fine-tuned  runs/pdf/chem_ft_untuned_aug/ (untuned 1e-5/1e-4)

Under design II the materials step is fit-channel-matched, so the baseline carries
it well (RESULTS.md Q7's aug-fit baseline finding) and the collapse attributes
cleanly to the instrument restriction in the final step.

ALL 12 CELLS ARE READ OFF EXISTING TREES AT ZERO NEW COMPUTE, n_train = 2530 in
every one (CHILI-3K's full train; the 100K fits are budget-matched to it by the
0.6286 draw). Seed 42 throughout, n = 1 by the standing decision.

    scheme      3K doc (none / instrument)                     100K doc (materials / both)
    baseline    runs/pdf/supervised/cnn_lr0.001_seed42/        runs/pdf/chem_lr_scratch/ @ per-target
                (val-selected on 3K, s25 stage 1)              val-selected LR (chem_lr_select)
    probe       <run>/probe_chili.json                         <run>/probe_chem100k.json
    fine-tuned  <run>/finetune.json (untuned 1e-5/1e-4)        runs/pdf/chem_ft_untuned/ (same LRs)

⚠️ **THE LR POLICIES DIFFER BY SCHEME, AND THE CAPTION SAYS SO** (same stance as
`chem_probe_arms`): the baseline is val-selected on each fit set it fits (1e-3 on
CHILI-3K from the s25 supervised sweep; per-target via `analysis.chem_lr_select` on
CHILI-100K), the fine-tuned scheme is fixed at the untuned defaults everywhere (the
`chili_arms_untuned` policy — RESULTS.md Q7 §3c's val-misalignment finding is the
license), and the probe has no LR. Selection details live where each tree is
documented; this figure only refuses to hide the asymmetry.

⚠️ **NOT EXPERIMENTAL VALIDATION.** Both registries come out of the same
DebyeCalculator forward model, so every systematic simulator error is common-mode.
This is the chemistry/instrument SIMULATION axis; RESULTS.md Q5/Q6 are the
measured-data questions.

⚠️ Same-cell rerun jitter is ~0.15 R2 on the doubly-shifted channel (RESULTS.md Q7
§3e, measured against `chem_transfer_val20`'s identical-config cell), so orderings
are quotable and small gaps are not — the caption's claims are ordering claims.

WHY `mo_bond` AND NOT `mean_bond`: identical to `chem_probe_arms` (see
`DEFAULT_TARGETS` there) — `mean_bond`'s definition moves with the registry, so it
does not measure one quantity across the materials shift. `--targets` redraws with
any subset, same flag as the sibling.

COLOR & MARKS. `analysis.paperstyle.ARM_COLORS`, the three-slot subset this figure
uses (`supervised`+`probe`+`finetuned`) is validated all-pairs in the paperstyle
docstring (CVD dE 14.0, normal 16.4). `finetuned` carries a 2.64:1 contrast WARN
whose mandatory relief here is: a distinct MARKER per scheme (identity never
color-alone), series names direct-labeled at the line ends in ink, and the value
labels on the terminal cells. Lines 2 pt, markers >= 6 pt, grid recessive.
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
import numpy as np

from analysis import paperstyle
from analysis.chem_lr_select import read as read_lr_cells, select as select_lr
from analysis.paperstyle import ARM_COLORS, INK, MUTED

REPO = Path(__file__).resolve().parent.parent

RUN_STEM = "2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372"
PRETRAIN_RUN = REPO / "runs/pdf/sweep" / RUN_STEM

PROBE_3K = PRETRAIN_RUN / "probe_chili.json"
PROBE_100K = PRETRAIN_RUN / "probe_chem100k.json"
PROBE_100K_AUG = PRETRAIN_RUN / "probe_chem100k_aug.json"
FT_3K = PRETRAIN_RUN / "finetune.json"
FT_100K_ROOT = REPO / "runs/pdf/chem_ft_untuned"
FT_100K_AUG_ROOT = REPO / "runs/pdf/chem_ft_untuned_aug"

#: EVERY POINT IS THE MEAN OVER THESE SEEDS (2026-08-27, user), spread reported in the
#: appendix via `--table` rather than drawn. See CHEM_SEEDS for where the 100K cells
#: come from and MEAN_DOC_KEYS for what gets averaged.
SEEDS = (42, 43, 44)

#: The seeded CHILI-100K tree (job 5503996, 2026-08-27). It exists rather than seeds
#: being added to `chem_lr_scratch{,_aug}`/`chem_ft_untuned{,_aug}` because `one_run`
#: and `scratch_100k` both glob without a seed — a seed-43 run at the selected LR
#: would have made them match three paths and raise. SEED 42 WAS RE-RUN INTO IT so all
#: three cells of every arm come from one batch; the old trees are untouched and the
#: re-run doubles as a reproduction check.
#:
#: ⚠️ **THAT CHECK FAILED FOR THE BASELINE AND PASSED FOR THE OTHER TWO, which is a
#: fact about the schemes and belongs in the caption.** Re-running seed 42 reproduced
#: the probe cells exactly and the fine-tuned cells to within 0.0035, but the
#: baseline's moved by up to 0.061 — three of its four deltas clear the 0.0143 noise
#: floor. From-scratch training on this fit set is NOT reproducible at fixed seed, so
#: the baseline's appendix sd is an upper bound containing run-to-run nondeterminism
#: as well as seed effect, while the probe's is pure subsample-draw variance.
CHEM_SEEDS = REPO / "runs/pdf/chem_seeds"

#: The baseline's per-target LR, val-selected by `analysis.chem_lr_select` at seed 42
#: and CARRIED to 43/44 unchanged rather than re-selected per seed — re-selecting
#: would put a learning-rate search inside the spread the appendix reports, and the
#: aug channel's `mo_bond` selection is a grid-EDGE tie the selector flags both ways.
#: The cost, which the caption owes: the selection itself stays n = 1.
#: Read from `scratch_100k` at draw time, not hardcoded; this dict only says which
#: run-dir name to look for once the LR is known.
MEAN_DOC_KEYS = ("r2", "mae", "f1_weighted")
#: The s25 supervised baseline at its stage-1 val-selected LR (1e-3 for the CNN).
BASE_3K = REPO / "runs/pdf/supervised/cnn_lr0.001_seed42/finetune.json"
BASE_100K_ROOT = REPO / "runs/pdf/chem_lr_scratch"
BASE_100K_AUG_ROOT = REPO / "runs/pdf/chem_lr_scratch_aug"
#: See plot_chem_probe_arms.SCRATCH_STEM — the scratch cells were launched off the
#: infonce config, which supervised mode reads for architecture only.
SCRATCH_STEM = "cnn_infonce_mpfull_final"

#: (condition label, which doc, protocol key). The order is the x axis — a
#: progression, each step adding one shift. The instrument-only cell
#: (`3k`/`clean_aug`) is deliberately not a row: module docstring, attribution
#: hazard.
#: Tick labels are the user-facing names (2026-08-21); the last wraps to two lines
#: because "Material + Instrument shift" overruns a third of the panel on one.
CONDITIONS = (("No shift", "3k", "clean_clean"),
              ("Material shift", "100k", "cross_clean"),
              ("Material +\nInstrument shift", "100k", "cross_aug"))

#: Design II (`--ii`): eval pinned to the varied channel, shifts enter through
#: training. Module docstring has the full cell map and rationale.
#: Tick labels carry the CONDITION NAME the prose uses and, on a second line, the
#: TRAINING SET that defines it (2026-08-24). Evaluation is constant under design II
#: so it need not appear. Two lines each is the height design I's last tick already
#: took, so the axis costs no extra page space while the figure stops depending on
#: the body text to say what the conditions are.
CONDITIONS_II = (("No shift\n3K varied", "3k", "aug_aug"),
                 ("Material shift\n100K varied", "100k_aug", "cross_aug"),
                 ("+ Instrument\n100K fixed", "100k", "cross_aug"))

ALL_PANELS = (("target_np_size", "Nanoparticle size"),
              ("target_mo_bond", "M–O distance"),
              ("target_mean_bond", "Mean bond length"))
#: `mean_bond` is excluded on the sibling figure's validity grounds; `np_size` is
#: drawn by decision with a resolution warning (module docstring: quote it as
#: "flat" and nothing finer).
DEFAULT_TARGETS = ("target_np_size", "target_mo_bond")

#: Drawn in this order; marker is the WARN-relief secondary encoding.
SCHEMES = (("baseline", ARM_COLORS["supervised"], "o"),
           ("probe", ARM_COLORS["probe"], "s"),
           ("fine-tuned", ARM_COLORS["finetuned"], "^"))


def load(path):
    with open(path) as fh:
        return json.load(fh)


def mean_doc(docs):
    """One doc with every drawn metric replaced by its mean over `docs`' seeds.

    Returning a DOC keeps the whole cell-assembly and assertion block below unchanged
    by the 2026-08-27 move off a single seed: it still reads
    `doc["tasks"][target][protocol]["r2"]`, and whether that came from one run or
    three is not its concern. The template is the first seed's, so protocols,
    n_labelled and finetune_cfg are that seed's — which the asserts downstream check.
    """
    seeds = sorted(docs)
    base = copy.deepcopy(docs[seeds[0]])
    if len(seeds) == 1:
        return base
    for target, cells in base["tasks"].items():
        for protocol, cell in cells.items():
            for key in MEAN_DOC_KEYS:
                if key in cell:
                    cell[key] = statistics.fmean(
                        docs[s]["tasks"][target][protocol][key] for s in seeds)
    return base


def seeded(root, pattern):
    """`{seed: doc}` for a `chem_seeds` arm — one run per seed, asserted."""
    out = {}
    for s in SEEDS:
        hits = sorted(root.glob(pattern.format(seed=s) + "/finetune.json"))
        if len(hits) != 1:
            raise SystemExit(f"expected 1 run matching {pattern.format(seed=s)} "
                             f"under {root}, found {len(hits)}")
        d = load(hits[0])
        assert d["finetune_cfg"]["seed"] == s, \
            f"{hits[0]}: cfg seed {d['finetune_cfg']['seed']} != {s}"
        out[s] = d
    return out


def one_run(root, pattern):
    hits = sorted(root.glob(f"{pattern}/finetune.json"))
    if len(hits) != 1:
        raise SystemExit(f"expected 1 run matching {pattern} under {root}, found {len(hits)}")
    return load(hits[0])


def scratch_100k(target, root=BASE_100K_ROOT, per_seed=None):
    """The baseline's 100K doc at that target's val-selected LR (chem_lr_select's rule).

    THE LR IS STILL SELECTED FROM THE SEED-42 GRID (`root`), and the three seeded cells
    at that LR are then read from `CHEM_SEEDS` and averaged — see CHEM_SEEDS for why
    the selection is carried rather than re-run per seed, and why the seeded cells do
    not live in `root` alongside the grid they were selected from.
    """
    lr, _, note = select_lr(read_lr_cells(root), SCRATCH_STEM, target)
    if lr is None:
        raise SystemExit(f"no scratch cells for {target} under {root}")
    print(f"baseline {root.name} LR {target:18s} {lr:>8.0e}  {note or 'interior, no tie'}")
    seeded_root = CHEM_SEEDS / ("scratch_aug" if root is BASE_100K_AUG_ROOT else "scratch")
    docs = seeded(seeded_root, f"{SCRATCH_STEM}_lr{lr:g}_seed{{seed}}"
                               f"_enc{lr:g}_head{lr:g}_*_seed{{seed}}")
    if per_seed is not None:
        per_seed[f"baseline/{seeded_root.name}/{target}"] = docs
    return lr, mean_doc(docs)


def caption(cells, panels, base_lr_100k, base_lr_100k_aug=None, ii=False):
    """LaTeX caption for `--paper`, built from the loaded cells so it cannot drift.

    The `np_size`-omission sentence is mandatory: a reader of the paper's other
    figures will notice the missing envelope target, and "its spread is inside the
    rerun noise" is a caveat they cannot infer from the axes.
    """
    def _mid(title):
        plain = len(title) > 1 and title[1].islower()
        return ((title[0].lower() + title[1:]) if plain else title).replace("–", "--")

    def _lrs(lr_map):
        sel = {lr_map[t] for t, _ in panels}
        return (rf"${sel.pop():g}$" if len(sel) == 1 else
                ", ".join(rf"${lr_map[t]:g}$ for {_mid(title)}" for t, title in panels))

    if ii:
        lr_note = (rf"on CHILI-3K $0.001$; on varied-spectra CHILI-100K {_lrs(base_lr_100k_aug)}; "
                   rf"on fixed-spectra CHILI-100K {_lrs(base_lr_100k)}")
    else:
        lr_note = rf"on CHILI-3K $0.001$, on CHILI-100K {_lrs(base_lr_100k)}"
    if ii:
        headline = r"""\textbf{Trained on varied spectra, every scheme carries the materials shift;
  restricting training to a single instrument realization separates them —
  end-to-end training from random initialisation fails outright while both
  pretrained schemes degrade gracefully.}"""
        # The tick glosses come FIRST: the second line of each tick is the only
        # place the conditions are defined outside the body text, and a reader who
        # cannot decode it cannot read the axis at all.
        conditions_sentence = r"""The second line of each tick names the training set and the spectra it
  is fitted on. Evaluation uses the varied CHILI-3K test spectra throughout,
  so the horizontal axis moves the training data only. The first step changes
  the materials to CHILI-100K COD binary metal oxides, with compositions
  disjoint from the test set. The second restricts training to the fixed
  native spectra, a single instrument realization, which alone already
  collapses the baseline. The nanoparticle-size panel is flat, its spread
  across all schemes and conditions being within the run-to-run variation of
  a single cell."""
        fit_sentence = "Every cell fits"
    else:
        headline = r"""\textbf{A change of materials alone costs all three training schemes roughly
  equally and modestly; adding the instrument shift on top separates them —
  end-to-end training from random initialisation fails outright while both
  pretrained schemes degrade gracefully.}"""
        conditions_sentence = r"""Each step of the horizontal axis adds one shift between training and
  evaluation: first the materials (training labels from CHILI-100K COD binary
  metal oxides, compositions disjoint from the CHILI-3K test set), then
  additionally the instrument distribution (evaluation spectra re-simulated at
  instrument parameters drawn from the pretraining augmentation ranges). The
  instrument shift alone, not shown, already accounts for most of the failure
  at the final step. The nanoparticle-size panel is flat: its spread across
  all schemes and conditions is within the run-to-run variation of a single
  cell."""
        fit_sentence = "All training uses clean spectra and"
    return rf"""% --- generated by tools/plot_chem_shift_interaction.py --paper ---
\caption{{%
  {headline}
  {conditions_sentence}
  {fit_sentence} {cells['n_fit']} labeled particles. Every point is the mean of
  three seeds; the probe's no-shift point is a single run, because that cell fits
  on all the labels and a closed-form fit on fixed rows is deterministic. Per-seed
  values and standard deviations are tabulated in the appendix. The baseline's
  spread is the widest and is an upper bound: re-running one seed moved it by up
  to 0.061 where the two pretrained schemes reproduced to 0.004, so from-scratch
  training on this fit set is not reproducible at fixed seed.
  The baseline trains the CNN end to end from random initialisation at a
  learning rate selected on validation per training set ({lr_note}); the
  probe freezes a VICReg encoder pretrained on
  simulated PDFs of the Materials Project and fits a head in closed form; the
  fine-tuned scheme trains that same pretrained encoder end to end at fixed
  default learning rates (encoder $10^{{-5}}$, head $10^{{-4}}$).
  The bond target is the minimum metal--oxygen distance, defined identically
  in both datasets.
  Both datasets come from one forward model, so the materials axis is not a
  test on measured data.
}}"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paper", action="store_true",
                    help="paper-ready: drop the provenance block and print a LaTeX caption")
    ap.add_argument("--ii", action="store_true",
                    help="design II: eval pinned to the varied channel, shifts enter "
                         "through training (module docstring); output stem gains _II")
    ap.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS),
                    choices=[t for t, _ in ALL_PANELS],
                    help="which panels to draw; target_mean_bond is excluded by "
                         "default on the sibling figure's validity grounds, and "
                         "np_size carries the module docstring's resolution warning")
    ap.add_argument("--out", default=None, help="figure stem")
    ap.add_argument("--table", action="store_true",
                    help="print the figure's exact cells as a LaTeX booktabs table "
                         "and draw nothing — same loads, same assertions, so the "
                         "table cannot disagree with the figure")
    args = ap.parse_args()

    panels = [p for p in ALL_PANELS if p[0] in args.targets]
    conditions = CONDITIONS_II if args.ii else CONDITIONS

    # `per_seed` mirrors every doc below as `{arm_key: {seed: doc}}`, for `--table`.
    # The drawing reads only the means.
    per_seed = {}

    def _mean(key, docs):
        per_seed[key] = docs
        return mean_doc(docs)

    # ⚠️ THE 3K PROBE IS n = 1 AND CANNOT BE OTHERWISE. It is the full-data fit, where
    # `subsample_train_mask` returns the mask untouched and the closed-form solve is
    # deterministic — the seed has nothing to act on. The two 100K probes DO have a
    # seed: they carry `n_train_frac 0.6286, subsample_seed <s>`, the draw that
    # row-matches the fit set to CHILI-3K's 2530. So the probe line carries a mean of
    # three at both shifted conditions and a single run at "no shift".
    probe_3k = _mean("probe/3k", {SEEDS[0]: load(PROBE_3K)})
    probe_100k = _mean("probe/100k",
                       {sd: load(PRETRAIN_RUN / f"probe_chem100k_seed{sd}.json") for sd in SEEDS})
    ft_3k = _mean("fine-tuned/3k",
                  {sd: load(REPO / f"runs/pdf/datafrac_full/seed{sd}/{RUN_STEM}"
                                   "_enc1e-05_head0.0001/finetune.json") for sd in SEEDS})
    base_3k = _mean("baseline/3k",
                    {sd: load(REPO / f"runs/pdf/supervised/cnn_lr0.001_seed{sd}/finetune.json")
                     for sd in SEEDS})
    ft_100k = _mean("fine-tuned/100k",
                    seeded(CHEM_SEEDS / "ft", "*_cnn_vicreg_mpfull_final_*_seed{seed}"))
    base_100k = {t: scratch_100k(t, per_seed=per_seed) for t, _ in panels}
    docs_100k_aug, base_100k_aug = {}, {}
    if args.ii:
        docs_100k_aug = {
            "probe": _mean("probe/100k_aug",
                           {sd: load(PRETRAIN_RUN / f"probe_chem100k_aug_seed{sd}.json")
                            for sd in SEEDS}),
            "fine-tuned": _mean("fine-tuned/100k_aug",
                                seeded(CHEM_SEEDS / "ft_aug",
                                       "*_cnn_vicreg_mpfull_final_*_seed{seed}")),
        }
        base_100k_aug = {t: scratch_100k(t, BASE_100K_AUG_ROOT, per_seed) for t, _ in panels}

    # The two fixed-policy gradient schemes must be at the LRs the docstring claims.
    fts = [("3K", ft_3k), ("100K", ft_100k)] + \
          ([("100K aug", docs_100k_aug["fine-tuned"])] if args.ii else [])
    for name, ft in fts:
        c = ft["finetune_cfg"]
        assert (c["lr_enc"], c["lr_head"]) == (1e-5, 1e-4), \
            f"fine-tuned {name} LRs {c['lr_enc']!r}/{c['lr_head']!r} are not the untuned defaults"
        assert ft["pretrained"], f"fine-tuned {name}: init is not what its name claims"
    assert not base_3k["pretrained"], "baseline 3K: init is not what its name claims"
    bcfg = base_3k["finetune_cfg"]
    assert (bcfg["lr_enc"], bcfg["lr_head"]) == (1e-3, 1e-3), \
        f"baseline 3K LRs {bcfg['lr_enc']!r}/{bcfg['lr_head']!r} != the s25 val-selected 1e-3"

    # One experiment: same encoder geometry, same eval registry and test rows, and the
    # materials docs really fit the other registry (on the channel their name claims)
    # while the 3K docs fit none.
    docs_3k = {"probe": probe_3k, "fine-tuned": ft_3k, "baseline": base_3k}
    docs_100k = {"probe": probe_100k, "fine-tuned": ft_100k}
    cross_clean = list(docs_100k.items()) + [(f"baseline/{t}", d) for t, (_, d) in base_100k.items()]
    cross_aug = list(docs_100k_aug.items()) + \
                [(f"baseline/{t}", d) for t, (_, d) in base_100k_aug.items()]
    for name, doc in list(docs_3k.items()) + cross_clean + cross_aug:
        assert "chili_registry" in doc["registry"], f"{name}: eval registry {doc['registry']!r}"
        for key in ("encoder", "latent_dim", "signal", "aug_signal", "n_test"):
            assert doc[key] == probe_3k[key], f"{name}: {key} {doc[key]!r} != {probe_3k[key]!r}"
    for name, doc in cross_clean + cross_aug:
        assert doc["fit_registry"] == probe_100k["fit_registry"], \
            f"{name}: fit registry {doc['fit_registry']!r}"
    for name, doc in cross_clean:
        assert doc.get("fit_signal") in ("xpdf", None), f"{name}: fit channel {doc.get('fit_signal')!r}"
    for name, doc in cross_aug:
        assert doc.get("fit_signal") == "xpdf_aug", f"{name}: fit channel {doc.get('fit_signal')!r}"
    for name, doc in docs_3k.items():
        assert doc.get("fit_registry") is None, f"{name}: unexpectedly a cross fit"

    # Label budget matched in all cells.
    n_fit = probe_3k["n_labelled"]["target_np_size"]["train"]
    for name, doc in list(docs_3k.items()) + cross_clean + cross_aug:
        for t, _ in panels:
            n = doc["n_labelled"][t]["train"]
            assert n == n_fit, f"{name}: {n} fit rows != {n_fit}"

    def doc_for(scheme, which, target):
        if scheme == "baseline" and which != "3k":
            return (base_100k if which == "100k" else base_100k_aug)[target][1]
        if which == "100k_aug":
            return docs_100k_aug[scheme]
        return (docs_3k if which == "3k" else docs_100k)[scheme]

    cells = {t: {scheme: [doc_for(scheme, which, t)["tasks"][t][proto]["r2"]
                          for _, which, proto in conditions]
                 for scheme, _, _ in SCHEMES} for t, _ in panels}
    cells["n_fit"] = n_fit

    # `(scheme, which, target) -> {seed: doc}`, so the table can recover the spread
    # behind each drawn mean. The baseline's key is per-target because its LR is.
    sd_docs = {}
    for (scheme, which, target), key in (
            (("probe", "3k", None), "probe/3k"),
            (("probe", "100k", None), "probe/100k"),
            (("probe", "100k_aug", None), "probe/100k_aug"),
            (("fine-tuned", "3k", None), "fine-tuned/3k"),
            (("fine-tuned", "100k", None), "fine-tuned/100k"),
            (("fine-tuned", "100k_aug", None), "fine-tuned/100k_aug"),
            (("baseline", "3k", None), "baseline/3k")):
        if key in per_seed:
            for t, _ in panels:
                sd_docs[(scheme, which, t)] = per_seed[key]
    for t, _ in panels:
        for which, root in (("100k", "scratch"), ("100k_aug", "scratch_aug")):
            k = f"baseline/{root}/{t}"
            if k in per_seed:
                sd_docs[("baseline", which, t)] = per_seed[k]

    if args.table:
        # 3 decimals — the JSONs' quotable precision in RESULTS.md; the figure's
        # 2-decimal labels are a rendering choice, not the record. Column heads are
        # shortened from the figure's tick names (repeated per target they overrun
        # a column width); "$+$ Instrument" keeps the progression reading.
        # Column heads are the condition NAME only: the tick labels' second line
        # (the training set) is a figure affordance, and repeated per target here it
        # would overrun the column. The caption carries the definitions.
        short = {"No shift": "No shift", "Material shift": "Material",
                 "Material +\nInstrument shift": "$+$ Instrument",
                 "No shift\n3K varied": "No shift",
                 "Material shift\n100K varied": "Material",
                 "+ Instrument\n100K fixed": "$+$ Instrument"}
        conds = " & ".join(short[c] for c, _, _ in conditions)
        ncond = len(conditions)
        heads = " & ".join(
            rf"\multicolumn{{{ncond}}}{{c}}{{{title.replace('–', '--')} ($R^2$)}}"
            for _, title in panels)
        rules = " ".join(rf"\cmidrule(lr){{{2 + k * ncond}-{1 + (k + 1) * ncond}}}"
                         for k in range(len(panels)))
        # ONE NUMBER PER CELL — the inline `value (sd)` form was tried and withdrawn
        # (2026-08-27, user): it triples every column and overruns the text block. The
        # spread gets a companion table of the same shape; a dash there is n = 1.
        def _cell(t, scheme, j, stat):
            docs = sd_docs.get((scheme, conditions[j][1], t))
            if stat == "sd":
                if not docs or len(docs) < 2:
                    return "---"
                ys = [d["tasks"][t][conditions[j][2]]["r2"] for d in docs.values()]
                return rf"${statistics.stdev(ys):.3f}$"
            return rf"${cells[t][scheme][j]:.3f}$"
        def _rows(stat):
            return "\n".join(
                "        " + scheme + " & " + " & ".join(
                    _cell(t, scheme, j, stat) for t, _ in panels for j in range(ncond))
                + r" \\"
                for scheme, _, _ in SCHEMES)
        rows, sd_rows = _rows("mean"), _rows("sd")
        def lr_str(bmap):
            sel = {bmap[t][0] for t, _ in panels}
            return (rf"${sel.pop():g}$" if len(sel) == 1 else
                    ", ".join(rf"${bmap[t][0]:g}$" for t, _ in panels))
        lr_note = ((rf"on CHILI-3K $0.001$; on varied-spectra CHILI-100K {lr_str(base_100k_aug)}; "
                    rf"on fixed-spectra CHILI-100K {lr_str(base_100k)}") if args.ii else
                   rf"on CHILI-3K $0.001$, on CHILI-100K {lr_str(base_100k)}")
        conditions_sentence = (r"""Test $R^2$ of the three training schemes as shifts are added through
    training; evaluation always uses the varied test spectra. The first step
    changes the materials (training labels from CHILI-100K COD binary metal
    oxides, compositions disjoint from the CHILI-3K test set) while training
    spectra stay varied; the second keeps CHILI-100K and restricts training
    to its fixed native spectra, a single instrument realization, which alone
    already collapses the baseline. Every cell fits""" if args.ii else
                               r"""Test $R^2$ of the three training schemes as shifts are added between
    training and evaluation: none, a change of materials (training labels from
    CHILI-100K COD binary metal oxides, compositions disjoint from the
    CHILI-3K test set), then additionally a change of instrument distribution
    (evaluation spectra re-simulated at instrument parameters drawn from the
    pretraining augmentation ranges). The instrument shift alone, not shown,
    already accounts for most of the failure in the last column. All training
    uses clean spectra and""")
        print(rf"""% --- generated by tools/plot_chem_shift_interaction.py{' --ii' if args.ii else ''} --table ---
\begin{{table}}[h]
    \centering
    \caption{{Exact numbers behind Figure \ref{{fig:matshift}}. Test performance under
    the three shift conditions. The no-shift condition trains on CHILI-3K, the other
    two on CHILI-100K, and evaluation always uses the varied test spectra of CHILI-3K.
    Each entry is the mean over three seeds; standard deviations are in
    Table~\ref{{tab:matshift_sd}}. The baseline's learning rate is selected on
    validation per training set ({lr_note}) at one seed and carried to the other two;
    the fine-tuned scheme uses the fixed default learning rates (encoder $10^{{-5}}$,
    head $10^{{-4}}$). Both datasets come from one forward model, so the materials
    axis is not a test on measured data.}}
    \label{{tab:matshift}}
    \begin{{tabular}}{{l{" c" * (ncond * len(panels))}}}
        \toprule
         & {heads} \\
        {rules}
        Training scheme & {conds} & {conds} \\
        \midrule
{rows}
        \bottomrule
    \end{{tabular}}
\end{{table}}

\begin{{table}}[h]
    \centering
    \caption{{Standard deviations over the three seeds behind
    Table~\ref{{tab:matshift}}. A dash marks $n{{=}}1$: the probe's no-shift cell fits
    on all the labels, so there is no subsample to draw and the closed-form fit is
    deterministic. Entries are the sample standard deviation at $n{{=}}3$ and are
    correspondingly poorly determined; differences below the $0.0143$ fixed-seed
    re-run noise floor should not be read in either direction. The baseline's entries
    are upper bounds rather than seed effect alone: re-running one seed unchanged
    moved its cells by up to $0.061$ where both pretrained schemes reproduced to
    $0.004$, so from-scratch training on this fit set is not reproducible at fixed
    seed; the probe's entries are pure subsample-draw variance.}}
    \label{{tab:matshift_sd}}
    \begin{{tabular}}{{l{" c" * (ncond * len(panels))}}}
        \toprule
         & {heads} \\
        {rules}
        Training scheme & {conds} & {conds} \\
        \midrule
{sd_rows}
        \bottomrule
    \end{{tabular}}
\end{{table}}""")
        return

    paperstyle.use()
    lo = min(0.0, min(min(vals) for t, _ in panels for vals in cells[t].values()))
    x = np.arange(len(conditions))

    # A single panel takes ~half the column, not the full WIDTH: the type sizes stay
    # true at any canvas size as long as the PDF is included at natural size — the
    # paperstyle calibration warns against RESCALING, not against smaller canvases.
    width = paperstyle.WIDTH if len(panels) > 1 else paperstyle.WIDTH * 0.55
    fig, axes = plt.subplots(1, len(panels), sharey=True,
                             # 1.5 in, down from 1.9 (2026-08-25). At 1.9 the axes
                             # were 1.378 in against Figure 2's 0.97 in and Figure
                             # 3's ~1.0 in per row, so this figure read as taller
                             # than its neighbours for the same amount of data —
                             # three x positions and three lines. The decorations
                             # here are fixed in inches (0.20 of title, 0.33 of
                             # two-line tick labels), so the cut comes entirely off
                             # the axes and lands them at ~0.98 in, matching.
                             figsize=(width, 1.5 if args.paper else 2.0))
    axes = np.atleast_1d(axes)
    for i, (ax, (target, title)) in enumerate(zip(axes.flat, panels)):
        for scheme, colour, marker in SCHEMES:
            v = cells[target][scheme]
            ax.plot(x, v, color=colour, lw=2.0, marker=marker, ms=6,
                    mec="white", mew=0.8, zorder=3, clip_on=False)
        # Terminal values only — the "both" cell is where the schemes separate
        # furthest and is the cell the caption quotes; labelling all 12 points per
        # panel collides in the top cluster. The full table is RESULTS.md Q7 §3e.
        # The label y's are de-collided top-down (probe and fine-tuned end 0.006
        # apart on both panels), the leader stays honest because the value is
        # printed.
        style = {s: (c, m) for s, c, m in SCHEMES}
        ends = sorted(((cells[target][s][-1], s) for s, _, _ in SCHEMES), reverse=True)
        min_gap = 0.14
        ys = []
        for v, _ in ends:
            ys.append(v if not ys else min(v, ys[-1] - min_gap))
        # Markers are ALL-OR-NONE PER PANEL (2026-08-21, after a mixed draft): a
        # panel where displaced labels carry DISTINCT values (np_size: 0.93/0.92/
        # 0.87 off three converged lines) is ambiguous, and every label gets its
        # scheme's marker — marking only the displaced ones leaves the panel
        # half-keyed. A panel whose displacements change nothing the reader could
        # misattribute (mo_bond: the displaced 0.55 reads identically to its
        # neighbour, and −0.35 sits at its own line) stays bare — there a marker
        # beside a dot reads as a duplicated data point. The text stays in ink.
        txts = [f"{v:.2f}" for v, _ in ends]
        moved = [abs(y - v) > 0.01 for (v, _), y in zip(ends, ys)]
        keyed = any(m and txts[k] != txts[k - 1] for k, m in enumerate(moved) if k)
        for (v, s), y, txt in zip(ends, ys, txts):
            colour, marker = style[s]
            if keyed:
                ax.plot(x[-1] + 0.13, y, marker, ms=3.5, mfc=colour, mec=colour,
                        clip_on=False, zorder=5)
            ax.text(x[-1] + (0.21 if keyed else 0.13), y, txt,
                    va="center", ha="left", fontsize=7, color=INK, zorder=5)
        # NO ZERO LINE (2026-08-25, review). A grey rule at R² = 0 used to be drawn
        # wherever a panel dipped below it, to mark "worse than predicting the test
        # mean". The y ticks already carry 0 and the one point that goes under it is
        # labelled -0.35 in ink, so the rule was restating what the axis and the
        # annotation both say, and it was the only horizontal furniture in a figure
        # whose whole content is three descending lines.
        # 7 pt, not 7.5: at the half-column single-panel width "materials" and
        # "instrument" are adjacent long words and touch at 7.5.
        ax.set_xticks(x, [c for c, _, _ in conditions])
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlim(-0.25, len(conditions) - 0.45)
        ax.set_title(title, loc="left")
        # THE METRIC IS A Y-AXIS LABEL (2026-08-25, review). It was a right-aligned
        # second `set_title` in MUTED at 8 pt, inherited from `plot_probe_datafrac`
        # — and when that figure's version was called out for putting the name of the
        # quantity as far from its own axis as the panel allows, and greying it out
        # on top, this copy had to move with it or the two figures would teach two
        # conventions for one thing. Leading panel only: `sharey=True` here, so the
        # panels are one scale and the left edge is where its tick labels already are.
        if i == 0:
            ax.set_ylabel("R²")
        ax.set_ylim(lo - 0.08, 1.04)
        # Explicit half-unit ticks: at 1.9 in the auto locator's 0.25 steps crowd.
        ax.set_yticks([t for t in (-0.5, 0, 0.5, 1.0) if t >= lo - 0.08])
        ax.grid(axis="x", visible=False)
    # Legend in the first panel's lower-left band, between the zero line and the
    # data: on the default mo_bond panel every line stays >= 0.53 over the legend's
    # x extent (the collapse happens on the right half). Marker + hue together
    # carry identity.
    axes.flat[0].legend(handles=[plt.Line2D([], [], color=c, marker=m, ms=5.5, lw=2,
                                            mec="white", mew=0.8, label=s)
                                 for s, c, m in SCHEMES],
                        loc="lower left", bbox_to_anchor=(0.0, 0.30),
                        fontsize=6.5, frameon=False,
                        handletextpad=0.6, labelspacing=0.4)

    if args.paper:
        paperstyle.layout(fig)
    else:
        def lr_hdr(bmap):
            sel = {bmap[t][0] for t, _ in panels}
            return (f"{sel.pop():g}" if len(sel) == 1 else
                    "/".join(f"{bmap[t][0]:g}" for t, _ in panels))
        lrs = lr_hdr(base_100k) + (f" · aug {lr_hdr(base_100k_aug)}" if args.ii else "")
        fit_note = "eval aug pinned, fit aug→clean" if args.ii else "fit clean spectra"
        # Line lengths sized to the canvas: the single-panel default is ~half the
        # full WIDTH, so the provenance block wraps to four short lines there.
        header = (
            f"Shift condition × training scheme · cnn_vicreg_mpfull_final · {fit_note}\n"
            f"n_train {n_fit} · seed 42 · baseline lr 0.001 (3K) / {lrs} (100K) · ft lr 1e-5/1e-4 fixed"
        ) if len(panels) > 1 else (
            "Shift condition × training scheme\n"
            f"cnn_vicreg_mpfull_final · {fit_note}\n"
            f"n_train {n_fit} · seed 42 · ft lr 1e-5/1e-4 fixed\n"
            f"baseline lr 0.001 (3K) / {lrs} (100K)"
        )
        fig.suptitle(header, x=0.012, ha="left", fontsize=8)
        paperstyle.layout(fig, rect=(0, 0, 1, 0.87 if len(panels) > 1 else 0.76))
    stem = f"chem_shift_interaction{'_paper' if args.paper else ''}{'_II' if args.ii else ''}"
    paperstyle.save(fig, args.out or stem)
    if args.paper:
        print("\n" + caption(cells, panels, {t: base_100k[t][0] for t, _ in panels},
                             {t: base_100k_aug[t][0] for t, _ in panels} if args.ii else None,
                             ii=args.ii))


if __name__ == "__main__":
    main()
