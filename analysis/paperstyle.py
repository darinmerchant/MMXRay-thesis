"""One look for every workshop-paper figure — palette, type sizes, and the save path.

WHY THIS EXISTS. The paper's figures were written independently and drifted: one
carried a #fafafa canvas with hand-picked hexes, another used matplotlib defaults,
and the same arm appeared in two different colours across two figures the reader is
meant to compare. A reader who has to re-learn the colour key per figure is being
made to do work the figures should have done.

THE ARM COLOURS ARE **COLORBREWER SET2**, slots 1-3 (2026-08-25, by request).
`probe` #66c2a5, `supervised` #fc8d62, `finetuned` #8da0cb. Measured with the dataviz
skill's validator (ported to Python in scratchpad — no node on this Mac; the port
reproduces the reference palette's documented worst adjacent CVD dE 9.1 and
normal-vision 19.6 exactly), under the ALL-PAIRS gate, which is the honest one here
because all three arms appear together in every panel:

    Lightness band       PASS   all 3 inside L 0.43-0.77
    Chroma floor         FAIL   probe C=0.099, finetuned C=0.067 — both read grey-ish
    CVD separation       PASS   worst all-pairs 9.0 (deutan, probe<->finetuned)
    Normal-vision floor  FAIL   worst all-pairs 13.1 (probe<->finetuned), floor 15
    Contrast vs surface  WARN   2.14 / 2.30 / 2.62 : 1, all below the 3:1 mark gate

⚠️ **SET2 FAILS TWO CHECKS AND WARNS ON A THIRD. THIS IS A KNOWN, ACCEPTED TRADE, NOT
AN OVERSIGHT — but do not quote these figures as validated.** Set2 is a pastel family:
it is designed for filled areas on a light page, read at size, and it buys familiarity
and print-friendliness at the cost of separation. The searched palette it replaced
(rust #843600 / cyan-blue #007ea8 / indigo #1248a8) passed every check outright at
all-pairs CVD 14.1 and normal 15.3, so the switch costs **-5.1 dE of colourblind
separation and -2.2 dE of normal-vision separation**, concentrated in the
`probe`<->`finetuned` pair. If a reviewer or a printer cannot separate those two, that
pair is why, and the searched palette is the drop-in fix — its rationale is preserved
in git history at this commit's parent.

**THE WARM SLOT IS `supervised`, AND THE ASSIGNMENT IS NOT SET2's SLOT ORDER.** Set2's
own order would give slot 1 to `supervised`, which puts the orange on `probe` and makes
the HEADLINE arm the odd one out. The three-arm experiment groups as (no pretrained
encoder) versus (pretrained, frozen) and (pretrained, updated), so the lone warm hue
belongs on `supervised` and the two cool slots on the pretrained pair. Reassigning
WITHIN a chosen slot set is safe — the all-pairs gate does not depend on which arm
holds which hex, only on the set — but the SET must stay slots 1..N, which is the
CVD-safety mechanism and is not reassignable.

TWO ALTERNATIVES WERE MEASURED AND REJECTED, both while chasing white labels. Recorded
so the same ground is not re-covered:

    palette              white text on it       all-pairs CVD / normal   gate
    Set2 (in use)        2.14 / 2.30 / 2.62     9.0 / 13.1               fails 2, warns 1
    ColorBrewer Dark2    3.39 / 3.76 / 4.44     11.6 / 19.9              PASSES ALL
    Tol dark (y/c/b)     5.98 / 8.40 / 14.71    10.0 / 12.0              fails 3
    searched (retired)   8.37 / 4.62 / 8.34     14.1 / 15.3              PASSES ALL

  - **Dark2 is the same hue order as Set2, just darker, so it is a true drop-in** and
    it is the best-scoring recognised palette here. It does NOT solve white labels: at
    mid-lightness neither ink clears 4.5:1 on any slot (#7570b3 is 4.44 white against
    4.43 dark, essentially the crossover), so it only trades Set2's comfortable dark
    labels for two uncomfortable options.
  - **Tol's dark scheme solves white text outright** (5.98-14.71:1) but it is a
    TEXT/BACKGROUND scheme, not a data scheme — Tol pairs it with his "pale" set for
    marking text — and its low chroma shows: all three slots miss the chroma floor,
    two miss the lightness band, and the triple is fragile. Only yellow/cyan/blue is
    usable; the red/cyan/blue a warm baseline would want collapses to **CVD dE 5.4,
    below the floor of 6**, i.e. dark red and dark cyan merge under deutan. Do not
    reach for the red slot here.

Set2 was kept over both, deliberately, after seeing all three rendered.

⚠️ **NO SET2 SLOT HOLDS WHITE TEXT, so the value labels are DARK.** Contrast on white
is 2.14 / 2.30 / 2.62 : 1, nowhere near the 4.5:1 a 7.5 pt label needs; against INK it
is 9.21 / 8.55 / 7.52 : 1. The review that prompted the palette work asked for slots
that all take ONE ink, so that a bar's number never reads as emphasis nobody intended,
and dark-on-pastel satisfies that as squarely as white-on-dark did — it is the same
requirement with the polarity flipped. `tools/plot_chili_four_arms.py` prints every
number in INK; there is still no `on_bar_ink` luminance switch and there must not be.

⚠️ **A WHITE HATCH IS INVISIBLE ON A PASTEL, and the hatch is load-bearing.** Two paper
figures use hatch-within-a-hue for the fit channel, so their bars carry `edgecolor=INK`
rather than the `"white"` they used while the arms were dark. A bar figure that adopts
these hues and keeps a white edge silently deletes its own fit-channel distinction.
Fill keys follow: light neutral swatch, dark hatch, matching the bars.

⚠️ **`raw` IS NOT A SET2 SLOT, AND IT WAS RE-STEPPED #440154 → #805bfe ON 2026-09-09
BECAUSE A DARK BAR CANNOT CARRY A DARK LABEL.** `plot_crystal_ladder` moved its value
labels INSIDE the bars to match `plot_chili_four_arms`, and viridis's #440154 measures
1.3:1 against INK — the number on the `raw` bar was simply unreadable. The one fix this
module forbids is the other one: there is no `on_bar_ink` luminance switch and there
must not be (see the label paragraph above), so the BAR moved, not the ink.

The replacement was SEARCHED, not picked, against four gates at once — the two the
validator applies to any series (CVD dE >= 8 and normal-vision dE >= 15 against all
three arms), plus INK >= 4.5:1 so it can hold a 6.8 pt label, plus >= 3:1 against the
white page so it survives as a LINE. That last gate is what rules out the obvious
in-family answer: Set2's own remaining slots either collide with an arm (slot 4 pink
is CVD 1.5 against `finetuned`, slot 5 green 6.3 against `supervised`, slot 7 tan 5.7
against `probe`, slot 8 grey 2.1 against `probe`) or, in slot 6 yellow's case, clear
every separation gate and then measure 1.38:1 against the page, which is invisible as
a 1.8 pt line. A PASTEL violet is unavailable for the same reason at the other end:
`finetuned` already occupies that region, and #b39ddb measures CVD 2.4 against it.

#805bfe is the best-separated point in the window: CVD dE 18.6 and normal 20.0 against
its nearest arm (`finetuned`), INK 4.5:1, page 4.33:1, L 0.601, C 0.23. It is NEVER the
worst pair in any set it is drawn in — every remaining failure in those sets is the
Set2 arms' own `probe`<->`finetuned` collision documented above. Measured all-pairs on
white: with all four arms, CVD 9.0 / normal 13.1 (both that same pair); with
`supervised` + `finetuned`, 15.0 / 20.0; with `supervised` + `probe`
(`plot_xrd_crystal_ladder`'s triple), 10.0 / 22.3.

Its old lightness-band failure is GONE — #440154 sat at L 0.285, #805bfe at 0.601 — so
`raw` now passes a check the rest of this palette does not. THREE FIGURES DRAW IT and
all three were checked before the re-step, because two of them do not draw it as a
filled bar with an ink edge: `plot_crystal_ladder` (bars, ink edge),
`plot_sim2real` (bars, NO edge) and `plot_xrd_crystal_ladder` (a LINE plus a 0.15-alpha
band). The page-contrast gate above exists for those two. `plot_pdf_invariance`, which
this note used to name as the only caller, has not drawn `raw` since it moved to its
own POS/NEG tokens.

`supervised_aug` KEEPS #440154 AND DID NOT FOLLOW (2026-08-21, re-stated 2026-09-09).
It was written as "the same hex as `raw`", which it no longer is — that is the
"two names, one hex, never co-drawn" case working exactly as intended: two keys because
two meanings, and a re-step of one must not drag the other. Read it now as its own
colour, not as a reference to `raw`'s. It still has NO caller, so nothing was measured
for it; a figure that adopts it needs its own validation, and inherits viridis's dark
lightness along with the dark-label problem that just cost `raw` its hex.

⚠️ **SET2 SLOTS 1-3 ARE NOW ALSO NEAR THE APPENDIX SETS' TERRITORY.** `SLOTS` below is
a DIFFERENT documented palette (blue/orange/aqua) and the appendix figures draw from
it. The two never appear in one figure — arms are arms and splits are splits — but the
old guarantee that "the arm ramp is reserved and no appendix colour can be mistaken
for an arm" was a property of viridis, and Set2's orange against `SLOTS`' orange
#eb6834 is closer than that. Worth knowing before a new figure mixes them.

BLACK AXES AND NO GRID (2026-08-25), from the same review. `use()` had painted the
spines and ticks in GRID/MUTED and turned `axes.grid` on, which put a horizontal rule
at every y tick. The feedback was that single-direction guiding lines "give some Excel
vibes" and that the bar figures do not need them at all, and that the axes should be
matplotlib's default black. Both are now the default for every figure in the repo; a
plot that genuinely needs a grid turns it on locally and says why.

MODAL was darkened #898781 → #5f5e5a at the same time. Against the new `probe` the old
grey measured CVD 5.5 — below the floor of 6 — so the modal-class reference line and
the headline series became hard to separate under deutan. (The old palette was no
better here: it measured 5.1. The collision predates viridis; it is simply cheap to
fix while the tokens are being touched.) #5f5e5a clears dE 8.6 against the four viridis arms, and was re-checked against Set2:
CVD dE 22.4 / 26.5 / 23.1 on supervised / probe / finetuned. It separates easily from
pastels — but `plot_chili_four_arms` still overrides it to near-black for the one
reference line that crosses filled bars; see that module's MODAL_LINE.

THE TYPE SCALE, and the one rule it has to obey. Titles and axis labels 8.5, tick
labels and legends 8, in-panel annotations 6.5-7.5 by density. **NO LABEL MAY OUTRANK
THE TITLE OF THE PANEL IT SITS IN.** Dropping titles 9.5 -> 8.5 without touching
`axes.labelsize` broke exactly that for one commit: axis labels at 9 became the largest
type in every figure, which is why "labeled training samples" started reading as
oversized in Figure 3 — it was uniform in POINTS with every other axis label in the
paper, and simply bigger than the titles above it. Both are 8.5 now. If these ever move
again, move them together.

PANEL TITLES ARE 8.5 pt, DOWN FROM 9.5 (2026-08-25, review). At quarter-WIDTH a panel
title is the largest type in the figure and was competing with the 10 pt body text it
sits under; 8.5 keeps it clearly a label. It does NOT buy one-line titles anywhere that
matters — "Coordination number" still renders 1.26 in at 8.5 pt against a 0.78 in panel
in the arms figure — so two-line wraps stay a constraint, but it does return ~0.04 in of
height per wrapped title row.

TYPE SIZES assume the figure is placed at WIDTH inches and NOT rescaled in LaTeX.
Scaling a figure in \\includegraphics rescales the type with it and breaks the match
to body text, so set the width here rather than in the document.

PDF is the deliverable (vector, and `fonttype 42` keeps fonts out of Type 3, which
several venue checkers reject); the PNG twin is for quick viewing only. `save` crops
to the drawn extent (`bbox_inches="tight"`, 0.01 in of pad) so no figure ships a band
of blank page around it — see `save`'s own note on what that costs.

THE APPENDIX VOCABULARY (added 2026-08-20). `ARM_COLORS` answers "which arm"; the
appendix figures ask four other questions and none of them is an arm. Those sets are
kept clear of the arm hues, so a reader who has learned "teal-green = probe" never
meets that colour meaning "validation split". They come
from the dataviz skill's documented categorical palette, assigned in ITS fixed slot
order, which is the CVD-safety mechanism and is not reassignable.

Measured against THIS module's surface — WHITE, since `use()` sets a white canvas,
not the skill's #fcfcfb default. The validator was re-ported to Python for this Mac
(still no node); the port reproduces the skill's own documented reference numbers
exactly (worst adjacent CVD dE 9.1, normal-vision 19.6) AND the retired viridis ramp's
numbers exactly (all-pairs CVD 14.0, normal 16.4, `finetuned` 2.64:1), which is what
licenses both the arm measurements above and everything below:

    VIEW_COLORS    slots 1-2   all-pairs CVD dE 24.7   normal 33.6   both >= 3:1
    SPACE_COLORS   slots 1-2   the same two slots — see below
    SPLIT_COLORS   slots 1-3   all-pairs CVD dE  9.2   normal 24.0   see the WARN
    VICREG_COLORS  slots 1-3   the same three slots — see below

ALL-PAIRS, NOT ADJACENT, for all three. Overlaid step histograms and overlaid G(r)
traces put every pair on screen at once, so the adjacent pairlist — which assumes only
neighbours touch — would hide a collapse. That choice is what caps these sets at
THREE: the documented palette clears all-pairs with its first three slots and no
ordering of more can (the all-pairs pairlist does not depend on order). A fourth
split or a fourth view is a facet or a fold-to-other, never a fourth hue.

`test` (#1baf7a) measures 2.82:1 on white, below the 3:1 mark gate. It carries a
non-dismissable obligation: every figure using it ships a legend or direct labels,
never colour alone. (The arm slots no longer need that relief — all three clear the
gate — but neither do the Set2 arms, so the obligation is now universal here.)

SPACE_COLORS REUSES VIEW_COLORS' TWO HEXES, for the same reason SPLIT_COLORS and
VICREG_COLORS share three. A.1.5 asks "which view", A.6.1 asks "which space"; the two
figures never co-appear and neither reads a value off the other's key. Two names, two
meanings, one measured pair — and a later re-step of one must not drag the other.

SPLIT_COLORS AND VICREG_COLORS ARE THE SAME THREE HEXES ON PURPOSE. Slots are assigned
1..N per figure — that is the skill's rule, not an accident — and the two sets never
appear in one figure, so blue reads as "train" in A.3.3 and as the invariance term in
A.2.3 with no collision any reader can see. They are two names because they are two
figures, and a later re-step of one must not silently drag the other with it.

A.1.5 HAS NO CORPUS COLOUR, deliberately, and that is a statement about A.1.5 —
NOT a ban on the corpus ever taking a hue. That figure is three panels (MP, CHILI-3K,
CHILI-100K) each holding one view pair. Colour there encodes VIEW; the corpus is the
panel title. Giving the corpus a hue as well would spend the identity channel on what
the layout already says, and would collide with the view pair drawn inside every panel.

`REGISTRY_COLORS` IS THE CASE WHERE THE CORPUS *IS* THE SERIES (added 2026-08-27, the
A.3.8 chemistry-shift figure). There the two registries are overlaid INSIDE one panel
and the panel title names the quantity, not the corpus — the exact inverse of A.1.5's
layout — so the identity channel has nothing else to spend itself on. Same two hexes
as VIEW_COLORS and SPACE_COLORS, for the same documented reason those two share: the
three figures never co-appear, no reader reads a value off another's key, and three
names exist so a later re-step of one cannot silently drag the others.

SEQUENTIAL is for confusion-matrix cells: continuous magnitude, one hue, light->dark,
off the skill's documented blue ramp. Being SEQUENTIAL, it is checked for lightness
monotonicity (PASS), step spacing (all dL >= 0.06, PASS) and single hue (4 deg spread,
PASS) — and NOT for the ordinal light-end floor, which it misses at 1.32:1 BY DESIGN,
because the lightest step means "near zero" and is supposed to recede into the
surface. Do not "fix" that number; the skill says outright it fails by construction.

THERE IS NO WIDTH_FULL. `WIDTH` already IS the full text width of this venue's
single-column layout, so a "wider" appendix figure would overrun the text block and
then get scaled in LaTeX — the one thing the type sizes above cannot survive. What
varies between figures is HEIGHT, and it stays an explicit per-figure number at the
call site (existing figures use 2.5-3.0 for one row, ~1.9/row for two) rather than a
constant here, because the right height depends on how many tick labels a panel
carries, which this module cannot know.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "figures"

#: Column width in inches. NeurIPS-style single column, full text width.
WIDTH = 5.5

#: One colour per ARM, reused across every figure. Keyed by the arm's name in the
#: paper, so a plotter never spells a hex. Viridis at t = 0, 0.217, 0.433, 0.65 —
#: the ORDER IS THE RAMP and carries meaning (increasing use of pretraining), so
#: arms are not reassignable between slots without re-reading the docstring.
#: ColorBrewer **Set2**, slots 1-3. The WARM slot is `supervised` on purpose — see the
#: docstring; the two pretrained arms take the two cool slots.
ARM_COLORS = {
    "raw": "#805bfe",         # violet — no training at all; NOT a Set2 slot, and searched
    "supervised": "#fc8d62",  # Set2 2, orange — the arm with no pretrained encoder
    "probe": "#66c2a5",       # Set2 1, teal-green — the headline arm
    "finetuned": "#8da0cb",   # Set2 3, blue-purple — the other pretrained arm
    "supervised_aug": "#440154",  # SAME hex as `raw`; see below
}

#: Reference-line ink: anything on an axis that is NOT a series — the modal-class
#: floor, a parity plot's identity line, a marked true value. It never takes a hue.
#: Darkened from #898781 (2026-08-17): CVD 5.5 against `probe` was below the floor.
REFLINE = "#5f5e5a"

#: The original name for the same ink. Kept because plotters import it directly
#: (`from analysis.paperstyle import ARM_COLORS, MODAL, MUTED`), and because on an
#: arm figure the reference line IS the modal floor. New call sites want REFLINE.
MODAL = REFLINE

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

#: The dataviz skill's documented categorical palette, in ITS fixed order. ONLY the
#: first three are exposed, and that is the series cap, not laziness: slot 4 puts
#: yellow beside orange at normal-vision dE 13.7, and no re-ordering rescues it.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua

#: Two views of one structure overlaid on one axis (A.1.5). The two are exchangeable
#: random draws of p_inst, so the pair is NOMINAL — neither view is "first" in any
#: sense the reader should see, and a ramp here would invent an order that isn't real.
VIEW_COLORS = {"view_1": SLOTS[0], "view_2": SLOTS[1]}

#: The two spaces one measurement is taken in (A.6.1 invariance decay): the encoder's
#: h, and the min-max-normalized G(r) it was computed from. Nominal — input space is
#: the BASELINE, not a lesser rung of the same ladder, so a ramp would invent an order.
#: Same two hexes as VIEW_COLORS; see the docstring.
SPACE_COLORS = {"latent": SLOTS[0], "input": SLOTS[1]}

#: The two REGISTRIES of the chemistry-shift experiment (A.3.8), overlaid in one panel:
#: the CHILI-100K rows section 3.4 fits on, and the CHILI-3K test rows it evaluates on.
#: Nominal — neither is a rung of the other, and the shift between them is the subject.
#: Same two hexes as VIEW_COLORS/SPACE_COLORS; see the docstring.
REGISTRY_COLORS = {"fit": SLOTS[0], "eval": SLOTS[1]}

#: The two PAIR TYPES of the three invariance-histogram figures (2026-09-09 redesign):
#: positive = two views of one structure, negative = two different structures. ONE
#: color pair across all three dataset figures — the earlier per-representation hues
#: were redundant with the panel titles and made the three figures read as three keys.
#: Same two hexes as VIEW_COLORS/SPACE_COLORS/REGISTRY_COLORS, same documented reason:
#: the figures never co-appear with those, and separate names keep a later re-step of
#: one from dragging the others. TINTS are the second trace of an example pair — the
#: same hue lightened, so the top row never spends a third hue on "which trace".
PAIR_COLORS = {"pos": SLOTS[0], "neg": SLOTS[1]}
PAIR_TINTS = {"pos": "#8ab5e8", "neg": "#f4ac8f"}

#: Dataset splits (A.3.3 label distributions, A.1.5 realized-draw marginals). Nominal:
#: train/val/test are not ordered, and density-normalizing is what makes them
#: comparable, not the colour.
SPLIT_COLORS = {"train": SLOTS[0], "val": SLOTS[1], "test": SLOTS[2]}

#: The three VICReg terms (A.2.3), drawn as WEIGHTED contributions — never the total,
#: which scales with lambda_cov by construction and therefore cannot rank runs.
VICREG_COLORS = {"inv": SLOTS[0], "var": SLOTS[1], "cov": SLOTS[2]}

#: Confusion-matrix cells (A.3.7): continuous magnitude, one hue, light->dark.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "mmx_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

#: SCATTER MARKS coloured by a continuous value (A.3.8, A.3.9). NOT `SEQUENTIAL`, and
#: the reason is a property of the mark: that ramp's light end is specified to recede
#: into the white surface, which is right for a filled cell with neighbours and wrong
#: for a 1 pt dot with nothing around it, which simply vanishes (its palest point is
#: 1.32:1). This is `plasma` truncated where contrast first falls below 1.8:1
#: (t = 0.83, cap #fdb22f): palest point 1.81:1, perceptually uniform, CVD-safe, and a
#: blue -> magenta -> orange hue sweep carrying discriminability that a single-hue ramp
#: must get from lightness alone.
#:
#: `viridis` is deliberately avoided even though it scores similarly: ARM_COLORS
#: reserves teal for arms, so a reader who has learned teal = "probe" must never meet
#: teal meaning "particle size 30 A".
#:
#: Promoted here from tools/plot_appendix_embedding.py when a second figure
#: (tools/plot_appendix_aug_pca.py) needed the same ramp — one figure kept it local,
#: two make it a convention, and two copies would drift.
MARKS = LinearSegmentedColormap.from_list(
    "mmx_plasma_marks", colormaps["plasma"](numpy.linspace(0.0, 0.83, 256)))

#: Where a label drawn ON a SEQUENTIAL cell must flip from INK to white. This is the
#: MAXIMIN point, computed over the ramp rather than guessed: below it INK is the more
#: readable ink, above it white is, and they cross at 4.44:1. That 4.44 is therefore
#: the worst label contrast anywhere on the ramp — it clears the 3:1 large-text gate
#: and misses the 4.5:1 normal-text gate by 0.06, at the single value v ~ 0.58 and
#: nowhere else. Print counts beside the matrix instead if that matters for a venue.
CELL_INK_SWITCH = 0.583

#: Overlaid distributions are STEP outlines, never filled bars: three filled
#: histograms on one axis occlude each other and the occlusion depends on draw order,
#: so the figure would change meaning if the splits were plotted in another sequence.
#: `density=True` because the splits differ ~9x in size (train 2530 vs val 290).
STEP_HIST = {"histtype": "step", "density": True, "linewidth": 1.4}


def use():
    """Apply the shared rcParams. Call once, before creating any figure."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 9,
        "axes.labelsize": 8.5,
        # `supxlabel`/`supylabel` default to "large" and to black-by-theme, so a
        # figure-level axis label came out a different size AND a different colour
        # from a panel-level one. They are the same kind of object; give them the
        # same tokens rather than letting each call site pick.
        "figure.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": INK,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        # Hatch strokes at the matplotlib default of 1.0 are hairlines, and the two
        # figures that use hatch use it to carry MEANING — solid vs hatched is the fit
        # channel, the only thing separating a scheme's two bars. 1.3 keeps that
        # readable at legend-swatch size without the fill reading as a texture.
        "hatch.linewidth": 1.3,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def cell_ink(value):
    """Ink for a label drawn ON a SEQUENTIAL cell, given that cell's normalized value.

    `value` is the cell's position on the ramp in [0, 1] — the same number handed to
    `SEQUENTIAL`, NOT the raw count. Passing a raw count silently returns INK for
    everything, because every count >= 1 lands above the switch only by accident.
    """
    return "#ffffff" if value >= CELL_INK_SWITCH else INK


#: `tight_layout`'s pad, in font-size units, for every figure that ships. NOT the
#: matplotlib default of 1.08, which at 9 pt reserves 0.135 in of blank on each side.
#: That blank is invisible in the output — `save` crops to the ink — so it does not
#: read as whitespace; it is canvas the AXES never got to use, and it comes off the
#: saved WIDTH instead. A figure declared at WIDTH then saves at 5.22 in and
#: `\includegraphics[width=\textwidth]` scales it, and every type size with it, by
#: 5%. At 0.1 the same figure saves at ~5.47 in, i.e. within half a percent of the
#: width its type is calibrated for.
LAYOUT_PAD = 0.1


def layout(fig, rect=None):
    """`tight_layout` at the shared pad, so figures fill WIDTH instead of being cropped
    back from it. Call this instead of `fig.tight_layout()`.

    `rect` is passed straight through, for figures that must reserve a band the layout
    solver cannot see — a `fig.legend`, a `suptitle`, a shared x label. Everything
    else about the call is deliberately not configurable: the pad is the whole point.

    ⚠️ **TWO PASSES, AND THE SECOND ONE IS NOT SUPERSTITION.** `tight_layout` sizes the
    margins around the tick labels that exist WHEN IT RUNS, and then changes the axes
    size — which lets the locator re-pick the ticks, sometimes wider than what was
    measured. `plot_appendix_training` is the case that found this: its learning-rate
    panel went from 3-character labels to `0.00`/`0.25`/`0.50` after the first pass, so
    its column was sized for the narrow ones and the labels hung 0.059 in off the LEFT
    of the canvas. `save` crops to the ink, so that overhang became WIDTH, and the
    figure saved at 5.55 in — over the text block, which at natural size is an overfull
    hbox rather than a silent rescale. The draw between the passes is what settles the
    locator; laying out twice without it changes nothing.
    """
    kw = {"pad": LAYOUT_PAD, **({"rect": rect} if rect else {})}
    fig.tight_layout(**kw)
    fig.canvas.draw()
    fig.tight_layout(**kw)


def save(fig, stem, sub=None, pdf_only=False):
    """Write `figures/[<sub>/]<stem>.pdf` (the deliverable) and `.png` (quick viewing).

    `pdf_only` skips the PNG twin. It exists because `plot_probe_datafrac` wants the
    PDF alone, and for a while it got that by calling `fig.savefig` itself — which is
    how it became **the one paper figure that never got the tight crop**, since the
    crop lives here. It saved at exactly its declared 5.5 x 3.0 in while every other
    figure was trimmed, i.e. it kept the blank band the crop exists to remove. A
    plotter that needs a variation of this function should take a parameter, never
    its own `savefig`.

    `sub` puts the pair in a subdirectory — `sub="appendix"` for the appendix set.
    It is created on demand, `parents=True`, which the old `mkdir(exist_ok=True)`
    could not do: a nested stem used to fail at `savefig` with a bare
    FileNotFoundError naming a path the caller never typed.

    Note what a subdirectory does NOT buy: `/figures/` is gitignored (DESIGN.md, "one
    gitignored `figures/` at the repo root"), so this organizes output on THIS machine
    and records nothing in the repo. The regeneration command in each plotter's
    docstring remains the only durable record that a figure exists.

    ⚠️ **`bbox_inches="tight"` MEANS THE SAVED FILE IS NOT `figsize` WIDE.** It crops to
    what was actually drawn, so a figure declared at WIDTH = 5.5 in saves at whatever
    the outermost ink plus 0.01 in of pad comes to. That is the point — it is what
    removes the blank band around every figure — but it interacts with the type-size
    rule in this module's docstring: the sizes assume the figure is placed at WIDTH and
    NOT rescaled, and `\\includegraphics[width=\\textwidth]` on a 5.36 in crop scales the
    type up by 2.6%. Two consequences, neither of them optional:
      - keep `tight_layout` in the plotter, so the crop only ever removes true blank
        margin (hundredths of an inch) rather than compensating for a sloppy layout;
      - do not let a decoration hang far outside the axes. One label 0.25 in past the
        right spine widens the crop by 0.25 in, and THEN the rescale is 5%, which is
        visible against body text. This is why `plot_chili_four_arms` keys its modal
        floor in the figure legend rather than annotating it beside the line.
    """
    out = FIGURES / sub if sub else FIGURES
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf",) if pdf_only else ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.01)
        paths.append(p)
    # THE SAVED WIDTH IS REPORTED, NOT ASSUMED. These figures are included at their
    # NATURAL size (`\includegraphics{...}`, no `width` key) so the type sizes land
    # exactly as calibrated — which means the saved width is the printed width, and a
    # figure wider than WIDTH does not get quietly scaled down any more, it runs into
    # the margin as an Overfull \hbox. The crop is driven by the outermost ink, so
    # that happens whenever a decoration hangs past the canvas edge, which no amount
    # of care in `figsize` prevents on its own. One line of arithmetic here turns a
    # LaTeX warning nobody reads into a message at the point of authorship.
    w_in = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * 0.01
    over = " ⚠️ WIDER THAN WIDTH — will overfull \hbox at natural size" if w_in > WIDTH else ""
    print("wrote " + " + ".join(str(p.relative_to(REPO)) for p in paths)
          + f"  [{w_in:.3f} in of {WIDTH}]{over}")
    return paths
