"""tools/plot_appendix_chem_shift.py — appendix A.3.8: the structural breadth of the two section 3.4 registries.

    /opt/anaconda3/envs/mmxray/bin/python -m tools.plot_appendix_chem_shift

Section 3.4 fits on CHILI-100K and evaluates on CHILI-3K test, and calls the gap a
CHEMISTRY shift. The reader's first objection to that is a STRUCTURAL one — that the
fit set is simply a narrow neighbourhood the eval set sits outside of — and this
figure removes it. Five panels, two series each, one series per registry:

    row 1  SYMMETRY         crystal system (share), distinct space groups (count)
    row 2  UNIT CELL PARAMS a, b, c (share)

**THE ANSWER IS THAT THE FIT SET IS BROADER ON EVERY AXIS HERE** — 7 crystal systems
against 4, 58 space groups against 9, and unit cell parameters running to 24.5 A
against 8.9. So
section 3.4 is not narrow -> broad generalization, and structural coverage is not the
explanation for what it measures.

THE CELL ANGLES WERE DRAWN AND THEN CUT, and the measurement is recorded here so the
ground is not re-covered: every angle in both registries is at least 84% concentrated
on exactly 90 or 120 degrees, which makes a histogram one spike with an invisible
tail and forces a {90, 120, other} bar chart instead. Binned that way the fit rows are
1.3% / 8.1% / 1.3% off-lattice on alpha / beta / gamma and the eval rows are 0.0% on
all three — real, and consistent with everything else here, but three panels spending
a third of the figure to say that CHILI-3K's idealized prototypes have right angles.
The one number worth a sentence in the caption is that eval `cell_alpha` and
`cell_beta` are 90.000 degrees on all 360 rows, exactly.

What CHILI-100K does NOT cover is ELEMENT IDENTITY: 10 of the 24 evaluation metals
never appear in it (`RESULTS.md` section 3f), which is 150 of the 360 test rows and
is where `mean_bond` collapses to R2 0.004. **This figure deliberately does not draw
that.** It is the mechanism, it is a different claim, and it belongs beside the
section 3f table. This figure's job is to eliminate the competing explanation so the
element one is the only one left standing.

⚠️ **THE FIT ROWS ARE THE SUBSAMPLE, NOT THE REGISTRY.** Section 3.4 fits 2530 of
CHILI-100K's 4025 train rows, drawn at `--n-train-frac 0.6286 --subsample-seed 42`
to row-match CHILI-3K's own train split so the chemistry effect is not confounded
with a data-volume one. Plotting all 4025 would describe a population NEITHER arm
saw. The draw is taken by importing `analysis.finetune.subsample_train_mask` rather
than reimplementing it: a figure that characterizes the fit set has to be reading the
same rows the fit read, and calling the same function is the only durable guarantee.
This is why the loader reads the target columns even though NO TARGET IS DRAWN — the
draw is over the per-target LABELLED mask, so the targets define which rows exist to
be drawn from. The three are fully labelled here, so their masks are identical and
the draw is one draw; that is asserted, because a future target arriving with NaNs
would silently give every panel its own 2530 rows.

⚠️ **`overlap_3k` IS NOT DRAWN, and its absence is deliberate.** 765 of the 4025 fit
rows are tier `none` and the other 3260 match a CHILI-3K row at formula, cell or
formula+spacegroup. Shown without its tier definitions that reads as a direct
contradiction of section 3.4's disjointness claim, when in fact
`data/builders/chili100k.py`'s leak rule drops any row whose reduced formula appears
in a CHILI-3K TEST row — so overlap with test is exactly 0 and every one of those
matches is against train/val. That census is A.3.6's table, where the tiers can be
defined. A bar chart cannot carry the caveat that makes it true.

EVERY PANEL PLOTS SHARE, following A.3.3: the two series differ 7x in size (2530 vs
360), so a count axis would compare the registries' SIZES rather than their SHAPES.
Histogram panels are share PER BIN and therefore scale with 1/`BINS`, which is why
the bin count is a named constant and belongs in the caption.

BIN EDGES ARE COMPUTED OVER BOTH SERIES AT ONCE and handed to both `hist` calls.
Two `ax.hist` calls with default binning pick edges from their own data, so the two
outlines would be sampled on different grids and the overlap a reader reads off the
panel would be an artifact of the two ranges. This is the one place in this figure
where the obvious call is the wrong one.

THE UNIT CELL ROW SHARES ITS SCALE, both axes. `a`, `b` and `c` are ONE quantity
measured along three directions, and "is this cell isotropic" is the first thing a
reader asks of the three panels — unanswerable if each silently rescales to its own
maximum. `b` stopping at 16.4 A where `a` runs to 24.5 A is CONTENT, and on per-panel
axes it is invisible. The symmetry row is deliberately NOT shared: its two panels are
not even in the same units.

⚠️ **THE SPACE-GROUP PANEL IS A COUNT, NOT A SHARE**, and it is the only panel here
that is. That is why this figure has no single `supylabel`: three of the five panels
are in share, one is in distinct space groups, and one figure-level label would be
wrong for whichever it did not mean. Each panel that needs a y label carries its own,
and the unit cell row needs exactly one because it shares its axis. The count is also
annotated on the bar — 58 against 9 is the entire content of the panel, and a reader
should not have to read it off a tick.

STEP OUTLINES, NEVER FILLED, on the histogram panels (`ps.STEP_HIST`) — two filled
distributions on one axis occlude each other and the occlusion depends on draw order.
The categorical panels are grouped bars for the same reason: side by side, never
stacked, since a stacked pair reads as a total that means nothing here.

Reads the two registries only — no checkpoint, no run directory, nothing trained.

IMPORT ORDER IS LOAD-BEARING: `import torch` precedes pandas and every `analysis.*`
import. This module reaches `analysis.finetune`, which is a torch module, and
touching pandas first segfaults the process with no traceback (`docs/TRAPS.md` ->
the import-order bug). Unlike `tools/plot_appendix_labels.py`, this guard is NOT
optional here.
"""
from __future__ import annotations

import torch  # noqa: F401 — MUST precede pandas/analysis.*; see docs/TRAPS.md

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis import paperstyle as ps
from analysis.finetune import subsample_train_mask

REPO = Path(__file__).resolve().parents[1]
FIT_REGISTRY = REPO / "data" / "downstream" / "chili100k" / "chili100k_registry_val20.parquet"
EVAL_REGISTRY = REPO / "data" / "downstream" / "chili" / "chili_registry.parquet"

#: Section 3.4's own draw. Both numbers are the flags the runs were launched with;
#: changing either here makes this figure describe rows no cell was ever fit on.
N_TRAIN_FRAC = 0.6286
SUBSAMPLE_SEED = 42

#: Not drawn — they define the LABELLED mask the row draw indexes into. See the
#: docstring; this is why a figure with no target panel still reads target columns.
MASK_TARGETS = ["target_np_size", "target_mo_bond", "target_mean_bond"]

COLUMNS = ["split", "crystal_system", "space_group_number", "cell_a", "cell_b", "cell_c"]

#: Bins on every histogram panel. Load-bearing: a bar's height is its bin's share of
#: the series, so it scales with 1/BINS and is only comparable to a categorical bar
#: at a stated bin count.
BINS = 30

#: Increasing symmetry, which is the crystallographic order and the only one that is
#: not arbitrary. Alphabetical would put Cubic first and Triclinic fourth, which reads
#: as a ranking of nothing.
#:
#: WRITTEN OUT IN FULL, after a first version truncated them to four characters. The
#: truncation could not be fixed by shortening further, which is the reason it went:
#: `Triclinic` and `Trigonal` share their first three letters, so any abbreviation
#: that separates them is already four characters, and `Tric`/`Trig` differ in one
#: glyph at 8 pt — two categories the reader has to spell out to tell apart, sitting
#: three positions from each other on the axis. The full names cost ~0.2 in of panel
#: height at 45 degrees and are unambiguous.
SYSTEMS = ["Triclinic", "Monoclinic", "Orthorhombic",
           "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]

#: `subplot_mosaic` over a 6-column grid: two half-width panels on the symmetry row,
#: three third-width panels on the unit cell row. A plain 2x3 grid cannot do this — it
#: would shrink the two symmetry panels to third-width and leave a hole, and those two
#: are the ones carrying the longest tick labels and the annotated counts.
MOSAIC = [["sys", "sys", "sys", "sg", "sg", "sg"],
          ["a", "a", "b", "b", "c", "c"]]


def _fit_rows(registry: Path) -> pd.DataFrame:
    """The 2530 CHILI-100K rows section 3.4 actually fits, in registry order.

    The mask is built over the FULL registry (not the train slice) because that is the
    frame `subsample_train_mask` indexes into inside `finetune.py`; slicing first would
    renumber the rows and draw a different 2530 from the same seed.
    """
    df = pd.read_parquet(registry, columns=COLUMNS + MASK_TARGETS)

    masks = {t: torch.from_numpy(
        ((df.split == "train") & df[t].notna()).to_numpy().copy()) for t in MASK_TARGETS}
    ref = masks[MASK_TARGETS[0]]
    for t, m in masks.items():
        assert torch.equal(m, ref), f"{t} is labeled on different rows — see the docstring"

    kept = subsample_train_mask(ref, N_TRAIN_FRAC, SUBSAMPLE_SEED)
    print(f"  fit  CHILI-100K: {int(ref.sum())} train rows -> "
          f"{int(kept.sum())} drawn at frac={N_TRAIN_FRAC} seed={SUBSAMPLE_SEED}", flush=True)
    return df[kept.numpy()]


def _hist_panel(ax, fit: np.ndarray, ev: np.ndarray, title: str, xlabel: str,
                frame: tuple[float, float] | None = None) -> None:
    """Two share-per-bin outlines on ONE set of edges. `frame` fixes those edges to a
    meaningful interval instead of the observed range — see the space-group note."""
    edges = (np.linspace(*frame, BINS + 1) if frame
             else np.histogram_bin_edges(np.concatenate([fit, ev]), bins=BINS))
    # `density=False` overrides `ps.STEP_HIST`'s default, and the override is the
    # point: `density` renormalizes by bin WIDTH, which would put these panels in
    # 1/Å while the categorical panels stay in share and break the one shared y
    # label. The 1/len weights do the same job `density` was there for — making two
    # series that differ 7x in size comparable — in the units the figure uses.
    style = {**ps.STEP_HIST, "density": False}
    for v, key in ((fit, "fit"), (ev, "eval")):
        ax.hist(v, bins=edges, color=ps.REGISTRY_COLORS[key],
                weights=np.full(len(v), 1 / len(v)), **style)
    ax.set_title(title, loc="left", pad=3)
    ax.set_xlabel(xlabel)
    if frame:
        ax.set_xlim(*frame)
    ax.set_ylim(bottom=0)


def _bar_panel(ax, labels: list[str], fit: np.ndarray, ev: np.ndarray,
               title: str, ylabel: str, rotate: bool = False) -> None:
    """Grouped bars, side by side. Never stacked: a stacked pair reads as a total.

    No x label on any caller: the tick labels ARE the categories, so a label under
    them would only restate the panel title in other words (the same rule A.3.3's
    categorical row follows).
    """
    x = np.arange(len(labels))
    for off, share, key in ((-0.2, fit, "fit"), (0.2, ev, "eval")):
        ax.bar(x + off, share, 0.38, color=ps.REGISTRY_COLORS[key], zorder=3)
    ax.set_xticks(x, labels, rotation=45 if rotate else 0,
                  ha="right" if rotate else "center")
    ax.set_title(title, loc="left", pad=3)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)


def build(fit_registry: Path, eval_registry: Path, out_stem: str, sub: str | None) -> None:
    ps.use()

    fit = _fit_rows(fit_registry)
    ev = pd.read_parquet(eval_registry, columns=COLUMNS)
    ev = ev[ev.split == "test"]
    print(f"  eval CHILI-3K test: {len(ev)} rows", flush=True)

    # 4.0, not 3.7: `tight_layout` splits the canvas by mosaic ROW, so the seven
    # written-out crystal systems rotated 45 degrees come off the symmetry row's PLOT
    # height rather than off the figure's. At 3.7 that row's axes were a third shorter
    # than the edge row's for no reason a reader could see.
    fig, ax = plt.subplot_mosaic(MOSAIC, figsize=(ps.WIDTH, 4.0))

    # ---- row 1: symmetry -------------------------------------------------------
    sys_fit = np.array([float((fit.crystal_system == s).mean()) for s in SYSTEMS])
    sys_ev = np.array([float((ev.crystal_system == s).mean()) for s in SYSTEMS])
    _bar_panel(ax["sys"], SYSTEMS, sys_fit, sys_ev,
               "Crystal system", "share of rows", rotate=True)

    # DISTINCT space groups PRESENT, not a distribution over them. One number per
    # registry is the whole panel, so the number is annotated on its bar rather than
    # left to be read off a tick — and the x ticks stay empty, because naming the two
    # registries under the bars would only repeat the figure legend.
    n_sg = np.array([fit.space_group_number.nunique(), ev.space_group_number.nunique()])
    for x, (n, key) in enumerate(zip(n_sg, ("fit", "eval"))):
        ax["sg"].bar(x, n, 0.55, color=ps.REGISTRY_COLORS[key], zorder=3)
        ax["sg"].annotate(str(n), (x, n), textcoords="offset points", xytext=(0, 2),
                          ha="center", va="bottom", fontsize=7.5, color=ps.INK)
    ax["sg"].set_xticks([])
    ax["sg"].set_xlim(-0.6, 1.6)
    ax["sg"].set_ylim(0, n_sg.max() * 1.18)  # headroom the annotation sits in
    ax["sg"].set_title("Space groups present", loc="left", pad=3)
    ax["sg"].set_ylabel("distinct space groups")

    print(f"  crystal systems  fit {int((sys_fit > 0).sum())}  eval {int((sys_ev > 0).sum())}"
          f"   |   space groups  fit {n_sg[0]}  eval {n_sg[1]}", flush=True)

    # ---- row 2: unit cell parameters --------------------------------------------
    # ONE frame across all six series, so the three panels are on one ruler — see the
    # shared-scale note. Computed from the data rather than fixed, because a parameter
    # longer than 24.5 A in a rebuilt registry must widen the axis, never clip.
    # NOT named `edges` — that is what `_hist_panel` calls its bin boundaries, and one
    # module using the word for two unrelated things is how a later reader misreads it.
    params = {k: (fit[c].to_numpy(), ev[c].to_numpy())
              for k, c in (("a", "cell_a"), ("b", "cell_b"), ("c", "cell_c"))}
    flat = np.concatenate([v for pair in params.values() for v in pair])
    frame = (float(flat.min()), float(flat.max()))
    for key, (f, e) in params.items():
        _hist_panel(ax[key], f, e, f"Unit cell param. ${key}$", f"${key}$ (Å)", frame=frame)
        print(f"  cell_{key}       fit {f.mean():6.3f}±{f.std():5.3f} [{f.min():.2f},{f.max():.2f}]"
              f"   eval {e.mean():6.3f}±{e.std():5.3f} [{e.min():.2f},{e.max():.2f}]", flush=True)
    top = max(ax[k].get_ylim()[1] for k in params)
    for k in params:
        ax[k].set_ylim(0, top)
    ax["a"].set_ylabel("share of rows")  # the row shares one axis, so one label

    # Colour is the ONLY thing separating the two series, so the key is not optional.
    # Figure-level and above the panels, so nothing hangs outside an axis and widens
    # the tight crop (`ps.save`'s note on what a stray decoration costs).
    # THE LEGEND SAYS train/test WHILE THE CODE SAYS fit/eval, and the mismatch is
    # deliberate. `ps.REGISTRY_COLORS`' keys name the ROLE each registry plays in
    # section 3.4 — one is fitted on, one is evaluated on — and that is the vocabulary
    # `RESULTS.md` uses throughout. The reader is shown the SPLIT each set of rows
    # actually is, which is the more checkable statement: these are CHILI-100K's train
    # split and CHILI-3K's test split. Renaming the keys is not the fix — `ps.SPLIT_COLORS`
    # already owns train/val/test meaning one split of one registry, and two `train`
    # keys in one module would be worse than this comment.
    handles = [plt.Line2D([], [], color=ps.REGISTRY_COLORS[k], lw=1.4, label=lab)
               for k, lab in (("fit", "CHILI-100K (train)"), ("eval", "CHILI-3K (test)"))]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))

    ps.layout(fig, rect=(0, 0, 1, 0.95))
    ps.save(fig, out_stem, sub=sub)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-registry", default=str(FIT_REGISTRY))
    ap.add_argument("--eval-registry", default=str(EVAL_REGISTRY))
    ap.add_argument("--out", default="appendix_chem_shift")
    ap.add_argument("--sub", default="appendix")
    args = ap.parse_args()
    build(Path(args.fit_registry), Path(args.eval_registry), args.out, args.sub)


if __name__ == "__main__":
    main()
