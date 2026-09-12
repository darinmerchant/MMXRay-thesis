"""core/transforms.py — seedable simulate-transform: PDF only, for now.

The piece `core/simulate.py` deliberately left out: PARAM SAMPLING. A
`Transform` takes a record + an integer seed and returns one simulated view,
reproducibly (same seed -> same params -> same signal). Two-view pretraining
is just calling it twice with two seeds.

Minimal contract on purpose — one method, no `Compose`, no post-op stage. XRD
will need a second stage (noise/background/texture/shift layered on ONE
simulated view, so the simulator only reruns for physics-changing params; see
DESIGN.md -> Throughput). That stage's shape is deferred until real XRD ops are
in front of us (see PROGRESS.md) — `PDFSimulate` is written against a contract
narrow enough that it won't need rework when `Compose` lands.

This module also hosts `minmax_normalize`, the encoder-input convention applied
at the dataset read path (see DESIGN.md -> Signal normalization). It lives here,
not in `PDFSimulate`, so the simulator and the bank stay raw physical G(r).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.registry import MaterialRecord
from core.simulate import simulate
from modalities.xrd.grid import CROP_HI, CROP_LO

__all__ = [
    "TransformResult", "Transform", "PDFSimulate", "minmax_normalize", "max_normalize",
    "draw_pdf_params", "PDF_RANGES", "PDF_BOUNDS", "PDF_OOB_RANGES", "ranges_at",
    "XRDSimulate", "draw_xrd_params", "XRD_RANGES", "PREF_DIRS",
    "BKG_ORDER", "BKG_COEF_DECAY",
]


def minmax_normalize(y, eps: float = 1e-8):
    """Per-sample min-max scale a signal into [0, 1] — the encoder-input convention.

    Each signal is individually rescaled so its absolute amplitude is dropped,
    leaving only peak positions and relative peak structure. Absolute G(r)
    amplitude does NOT transfer from simulated PDFs to experimental / other-source
    PDFs (CHILI xPDF, real xPDF), so an encoder allowed to use it would bind to a
    feature that vanishes under the sim->real domain shift (see DESIGN.md ->
    Signal normalization).

    Applied at the dataset READ path only (`core/dataset_base.py`) — NOT in the
    simulator (`modalities/pdf/simulate.py`) or the bank (`tools/bank_pdf.py`),
    which stay raw. Keeping the bank raw makes a standardization ablation (z-score
    / L2) a one-line read-path change instead of a full re-bank.

    Duck-typed: works on a numpy array or a torch tensor — both provide
    `.min()`/`.max()` and elementwise arithmetic. A constant signal maps to all
    zeros (the denominator is `eps`).

    ONE signal at a time. `min()`/`max()` here take no axis, so a stacked `(N, L)`
    batch would be normalized against the batch's global extremes rather than each
    sample's own — silently wrong, and not the distribution encoders are trained
    on. That mistake is rejected rather than tolerated (it bit `analysis/
    downstream_eval.py` once); map over the batch instead.
    """
    if getattr(y, "ndim", 1) != 1:
        raise ValueError(
            f"minmax_normalize takes ONE signal, got shape {tuple(y.shape)}. "
            f"min()/max() take no axis, so this would normalize globally across the "
            f"batch instead of per-sample. Map over the batch instead."
        )
    lo = y.min()
    return (y - lo) / (y.max() - lo + eps)

def max_normalize(y, eps: float = 1e-8):
    """Per-sample MAX scale — the encoder-input convention for XRD.

    Deliberately NOT `minmax_normalize`. An XRD channel has a meaningful zero:
    CHILI's resampled pattern is zero-filled outside its angular coverage (5.8%
    of the [10, 80] deg window is structurally zero on every row), so its min is
    0, while a simulated pattern carrying a Chebyshev background has min > 0.
    Min-max would subtract a different floor from each and destroy exactly the
    baseline difference the domain-shift experiment is measuring — the two would
    stop being comparable in the way DESIGN.md -> Signal normalization requires.
    Scaling by the max alone leaves the zero where it is.

    `clip(y, 0, None)` is explicit rather than assumed: intensity is
    non-negative physically, but a simulated pattern with noise added can dip
    below zero, and a negative excursion left in place would shift the whole
    pattern once divided through.

    Duck-typed over numpy arrays and torch tensors, ONE signal at a time, for
    the same reasons as `minmax_normalize` — see its docstring.
    """
    if getattr(y, "ndim", 1) != 1:
        raise ValueError(
            f"max_normalize takes ONE signal, got shape {tuple(y.shape)}. "
            f"max() takes no axis, so this would normalize globally across the "
            f"batch instead of per-sample. Map over the batch instead."
        )
    y = y.clip(0, None)
    return y / (y.max() + eps)


# Uniform draw ranges, ported as-is from the reference's AUG_RANGES
# (PDF/Experiments/03_ssl_augmented/generate_mpfull.py). Draw order (uiso, qmax,
# qbroad, qmin) matches `simulate_one()` there.
UISO_RANGE = (1e-3, 0.03)
QMAX_RANGE = (15.0, 30.0)
QBROAD_RANGE = (0.01, 0.06)
QMIN_RANGE = (0.5, 2.0)
PDF_RANGES = {
    "uiso": UISO_RANGE, "qmax": QMAX_RANGE, "qbroad": QBROAD_RANGE, "qmin": QMIN_RANGE,
}

# Physically admissible bounds — the widest each param can go before the simulated
# PDF stops carrying structure. NOT the training ranges (that is PDF_RANGES above,
# inherited as-is from the reference repo); these are the ENDPOINTS of the
# augmentation-strength sweep, and exist because "a multiple of an arbitrary range"
# is still anchored to that arbitrary range. Rationale per param:
#   uiso    floor >0 (thermal motion is never zero); 0.10 A^2 is RMS displacement
#           ~0.32 A, past which peaks merge into a featureless envelope.
#   qmax    floor 5 A^-1 — below it, termination ripples dominate and almost no
#           structural detail survives. 40 A^-1 exceeds typical synchrotron reach
#           but sits far under the grid's Nyquist limit (pi/rstep ~ 314 A^-1).
#   qbroad  floor 0 (negative gives an imaginary peak width); 0.15 washes out
#           high-r oscillation by ~20 A.
#   qmin    floor 0 (negative Q is meaningless); 4 A^-1 removes enough low-Q that
#           G(r) is dominated by baseline distortion.
# Chosen so `qmin` can never be drawn above `qmax` in any single-param sweep arm:
# the qmax arm floors at 5.0 while qmin stays at its default max 2.0, and the qmin
# arm ceilings at 4.0 while qmax stays at its default min 15.0.
PDF_BOUNDS = {
    "uiso": (1e-4, 0.10),
    "qmax": (5.0, 40.0),
    "qbroad": (0.0, 0.15),
    "qmin": (0.0, 4.0),
}

# Out-of-band draw ranges: per param, the one-sided segment between the
# PDF_RANGES edge and the PDF_BOUNDS endpoint, on the side real instruments
# degrade toward (qmax down toward lab reach, uiso/qbroad/qmin up). This is the
# region the invariance ladder measured as OUTSIDE the trained range, so a
# channel drawn from it shifts the eval domain to instrument settings the SSL
# augmentations never practiced — the counterpart to PDF_RANGES, which is both
# the pretraining augmentation and (via tools/augment_chili.py) the in-band
# eval shift. qmin can never cross qmax here: qmax floors at 5.0 while qmin
# ceilings at 4.0, the same non-overlap PDF_BOUNDS was chosen for.
PDF_OOB_RANGES = {
    "uiso": (0.03, 0.10),
    "qmax": (5.0, 15.0),
    "qbroad": (0.06, 0.15),
    "qmin": (2.0, 4.0),
}


def draw_pdf_params(seed: int, ranges: dict = PDF_RANGES) -> dict[str, float]:
    """One seeded uniform draw of the four PDF instrument params.

    Shared by `PDFSimulate` (pretraining views, diffpy on a periodic cell) and
    `tools/augment_chili.py` (downstream views, Debye sum on a nanoparticle) so
    the two cannot drift apart in draw ORDER — the order fixes which param gets
    which number out of the stream, so changing it silently re-labels every
    view ever generated at a given seed.

    A ZERO-WIDTH range `(c, c)` is the way to switch a param off (see
    `ranges_at`): `uniform(c, c)` returns `c` and still consumes one draw, so the
    stream stays aligned with every other range setting. Skipping the draw
    instead would desynchronize it, and sweep points would no longer be paired
    on the same underlying uniforms.
    """
    rng = np.random.default_rng(seed)
    params = {name: float(rng.uniform(*lo_hi)) for name, lo_hi in ranges.items()}
    # Defensive: PDF_BOUNDS is chosen so no sweep arm can violate this, but an
    # empty/inverted Q window is silent garbage rather than a simulator error.
    if "qmin" in params and "qmax" in params and params["qmin"] >= params["qmax"]:
        raise ValueError(
            f"drew qmin={params['qmin']:.4g} >= qmax={params['qmax']:.4g} — the Q window "
            f"is empty or inverted. Ranges were qmin={tuple(ranges['qmin'])}, "
            f"qmax={tuple(ranges['qmax'])}."
        )
    return params


def ranges_at(
    swept: str, t: float, *, background: str = "default"
) -> dict[str, tuple[float, float]]:
    """The four aug ranges for one point of the augmentation-strength sweep.

    `swept` is opened out from its center `c` toward `PDF_BOUNDS` by strength
    `t`: at `t=0` it collapses to the single value `c` (no augmentation on that
    axis), at `t=1` it spans the full physically admissible range.

    `c` is the midpoint of the SCAN — `PDF_BOUNDS`, not `PDF_RANGES` — so the
    widening is symmetric and `t` means the same fraction of the physical span in
    both directions. Pivoting on the inherited default instead would anchor the
    whole sweep to a range that was ported unchanged from the reference repo and
    is lopsided against the physical bounds (uiso's default reaches 94% of the way
    to its floor but only 17% toward its ceiling). The trade is that `t=0`, and any
    param switched off, sits at the middle of the physical range rather than at a
    typical instrument value — e.g. uiso 0.05 A^2 is RMS displacement ~0.22 A
    against a typical 0.07-0.14 A. That degraded baseline is shared by every point
    in the sweep, so it costs absolute downstream performance but not comparability.

    `background` decides what the other three params do:
      "default" — their inherited `PDF_RANGES` width. This is the InfoMin protocol
                  (Tian et al. 2020, Figs 13/14): vary ONE augmentation's magnitude
                  inside an otherwise-normal pipeline, so every point stays a
                  realistic model.
      "fixed"   — collapsed to their own centers, making `swept` the only source of
                  view-to-view variation. Used by the pilot's solo blocks to measure
                  each param's standalone contribution to I_NCE, which is
                  path-independent in a way cumulative stacking is not.

    The inherited default is NOT a point on this line — it is neither centered on
    `c` nor scaled from it — so the production config is a separately banked
    anchor, not some value of `t`. Same for the pilot's solo blocks, which put one
    param at its inherited default width: pass those ranges explicitly.

    Key order matches `PDF_RANGES`, which `draw_pdf_params` relies on: the order
    fixes which param takes which number out of the seeded stream.
    """
    if swept not in PDF_RANGES:
        raise KeyError(f"unknown param {swept!r}; expected one of {list(PDF_RANGES)}")
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be in [0, 1], got {t}")
    if background not in ("default", "fixed"):
        raise ValueError(f"background must be 'default' or 'fixed', got {background!r}")

    out: dict[str, tuple[float, float]] = {}
    for name, (lo, hi) in PDF_RANGES.items():  # iterate PDF_RANGES: it fixes the draw order
        floor, ceil = PDF_BOUNDS[name]
        c, half = 0.5 * (floor + ceil), 0.5 * (ceil - floor)
        if name == swept:
            out[name] = (c - t * half, c + t * half)
        elif background == "fixed":
            out[name] = (c, c)
        else:
            out[name] = (lo, hi)
    return out


@dataclass(frozen=True)
class TransformResult:
    x: np.ndarray
    y: np.ndarray
    params: dict[str, float]  # the drawn params, for logging / the invariance eval


class Transform:
    """The whole contract, for now: a seedable record -> TransformResult step."""

    def __call__(self, record: MaterialRecord, seed: int) -> TransformResult:
        raise NotImplementedError


class PDFSimulate(Transform):
    """Sample (Uiso, Qmax, Qbroad, Qmin) and simulate one PDF view.

    All four are physics-changing (PDF has no cheap post-ops yet), so every
    call re-runs the diffpy backend. `rmin`/`rmax`/`rstep` are fixed grid
    settings, not sampled — pass to override `modalities.pdf.simulate`'s
    defaults; leave unset to use them.
    """

    def __init__(
        self,
        *,
        rmin: float | None = None,
        rmax: float | None = None,
        rstep: float | None = None,
        uiso_range: tuple[float, float] = UISO_RANGE,
        qmax_range: tuple[float, float] = QMAX_RANGE,
        qbroad_range: tuple[float, float] = QBROAD_RANGE,
        qmin_range: tuple[float, float] = QMIN_RANGE,
    ) -> None:
        self._grid = {k: v for k, v in dict(rmin=rmin, rmax=rmax, rstep=rstep).items() if v is not None}
        self._ranges = {
            "uiso": uiso_range, "qmax": qmax_range, "qbroad": qbroad_range, "qmin": qmin_range,
        }

    def __call__(self, record: MaterialRecord, seed: int) -> TransformResult:
        params = draw_pdf_params(seed, self._ranges)
        x, y = simulate(record, "pdf", **self._grid, **params)
        return TransformResult(x=x, y=y, params=params)


# ---------------------------------------------------------------------------
# XRD
# ---------------------------------------------------------------------------

# The XRD augmentation ranges, from DESIGN.md -> XRD arm, inherited from the
# reference repo VERBATIM. `tau_nm`'s [3, 50] nm is DISJOINT from CHILI's real
# [0.74, 5.67] nm — known, deliberate, and biasing AGAINST SSL, which is the safe
# direction. It is kept and reported as a measured property, not tuned away.
TAU_NM_RANGE = (3.0, 50.0)       # nm    crystallite domain size (Scherrer)
U_ISO_RANGE = (0.005, 0.05)      # A^2   isotropic ADP (Debye-Waller)
TEXTURE_RANGE = (0.0, 0.5)       #       preferred-orientation strength
SHIFT_DEG_RANGE = (-0.3, 0.3)    # deg   2theta zero-point offset
AMP_FRAC_RANGE = (0.0, 0.05)     #       diffuse background amplitude, fraction of max
NOISE_C_RANGE = (0.001, 0.005)   #       measurement-noise std, fraction of max
XRD_RANGES = {
    "tau_nm": TAU_NM_RANGE,
    "u_iso": U_ISO_RANGE,
    "max_texture": TEXTURE_RANGE,
    "shift_deg": SHIFT_DEG_RANGE,
    "amp_frac": AMP_FRAC_RANGE,
    "noise_c": NOISE_C_RANGE,
}

# There is deliberately no `XRD_BOUNDS` / `ranges_at` counterpart. Those exist for
# the PDF augmentation-strength sweep; the XRD one is deferred until the 2x2 comes
# back ambiguous, and inventing physical endpoints nobody sweeps over would be six
# guesses with no measurement behind them.

# Background SHAPE controls. They live here, not in `modalities/xrd/simulate.py`,
# because they are draw-time properties: the forward model evaluates whatever
# coefficients it is handed, so no parameter is silently ignored there. Orders
# above BKG_ORDER are zeroed and coefficient k is damped by DECAY**k, which is
# what keeps the baseline stiffer than any Bragg peak — measured at 8x the
# broadest in-window peak in `tests/test_xrd_simulate.py`, not assumed.
BKG_ORDER = 2
BKG_COEF_DECAY = 0.3

# The 7 non-zero directions in {0,1}^3 — the reference's `pref` support, ENUMERATED
# rather than rejection-sampled. Order is part of the contract: it is what the
# single indexed draw in `draw_xrd_params` indexes into.
PREF_DIRS = ((0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))


def draw_xrd_params(seed: int, ranges: dict = XRD_RANGES) -> dict[str, float]:
    """One seeded draw of the 13 XRD forward-model params.

    ELEVEN draws produce thirteen params: six ranged uniforms, ONE for the
    preferred direction, four for the Chebyshev coefficients. The count is FIXED
    — it does not depend on the values drawn — which is what lets two sweep
    points at the same seed stay paired on the same underlying uniforms. Two
    things in the reference break that and are deliberately not ported:

      - `_sample_pref_dir` is a REJECTION LOOP (draw {0,1}^3 until non-zero),
        which consumes a variable number of values: measured 12.45% of draws
        consume more than one, up to 7. Here it is a single `random()` indexed
        into `PREF_DIRS`. `integers(0, 7)` would read better, but numpy's bounded
        integer generator is itself rejection-based, so its consumption is only
        PROBABILISTICALLY fixed; `random()` makes it structurally fixed, which is
        what a bankability contract should rest on.
      - the reference's docstring states the draw order as (tau_nm, u_iso,
        max_texture, pref, shift_deg, amp_frac, cheb, noise_c). Its CODE draws
        pref FIRST and cheb SECOND, because dict-literal values evaluate in
        source order. The docstring is wrong about its own function; nothing here
        was derived from it.

    KEY ORDER IS THE DRAW ORDER, exactly as in `draw_pdf_params`. The order fixes
    which param takes which number out of the stream, so changing it silently
    re-labels every view ever generated at a given seed.

    A ZERO-WIDTH range `(c, c)` switches a param off and STILL consumes its draw,
    so the stream stays aligned with every other range setting — same convention
    as `draw_pdf_params`, and the reason `pref` and `cheb` are drawn last: nothing
    after them can be knocked out of alignment by a range change.

    `seed` is NOT one of the returned keys. It is passed separately to the
    backend by `XRDSimulate` (the XRD forward model needs it for the noise), and
    `tools/verify_bank.py` reconstructs it from `seed0 + gidx*n_views + view`
    exactly as it already does for PDF.
    """
    rng = np.random.default_rng(seed)
    params: dict[str, float] = {
        name: float(rng.uniform(*lo_hi)) for name, lo_hi in ranges.items()
    }
    pref = PREF_DIRS[int(rng.random() * len(PREF_DIRS))]
    params["pref_h"], params["pref_k"], params["pref_l"] = (int(v) for v in pref)
    cheb = rng.uniform(-1.0, 1.0, size=4)
    for k in range(4):
        params[f"cheb_{k}"] = float(cheb[k] * BKG_COEF_DECAY ** k) if k <= BKG_ORDER else 0.0
    return params


class XRDSimulate(Transform):
    """Sample the 13 XRD params and simulate one view on the encoder window.

    Mirrors `PDFSimulate`'s constructor shape — one `<name>_range` keyword per
    ranged param — so `tools/bank_pdf.py`'s
    `**{f"{name}_range": lo_hi for ...}` line works for either modality without
    growing a branch. There is no grid keyword: the XRD grid is fixed by
    `modalities/xrd/grid.py` and is not a per-block setting the way `rmin`/`rmax`
    are for PDF.

    THE SAME SEED FEEDS BOTH STAGES. `draw_xrd_params(seed)` and the backend's
    noise generator each build their own `default_rng(seed)`, so they are two
    different uses of one stream rather than two independent streams. Harmless —
    the noise is additive and independent of the pattern it lands on — and
    deliberately not dressed up with a `SeedSequence.spawn`, because any
    derivation would become part of the bankability contract that
    `verify_bank`'s bit-exact re-simulation has to reproduce, and that formula
    already lives in two files.

    THE CROP HAPPENS HERE, not at the read path. The backend returns the full
    [0, 180] deg grid because an out-of-window reflection leaks a fifth of its
    height into the window once broadened, so the physics has to be computed
    wide; but what gets banked is the 4999-point encoder window, which is what
    DESIGN.md's 28.9 GB block figure is. Normalization stays a read-path step —
    the bank holds raw intensity (DESIGN.md -> Signal normalization).
    """

    #: No per-block grid settings — the XRD grid is fixed by modalities/xrd/grid.py,
    #: unlike PDF's rmin/rmax/rstep. `tools/bank_pdf._config_hash` reads `tf._grid`
    #: from either transform, and that function's source is pinned (T0.1's AST
    #: check), so the accommodation lives here rather than there. Never mutated,
    #: so a shared class-level dict is safe.
    _grid: dict = {}

    def __init__(
        self,
        *,
        tau_nm_range: tuple[float, float] = TAU_NM_RANGE,
        u_iso_range: tuple[float, float] = U_ISO_RANGE,
        max_texture_range: tuple[float, float] = TEXTURE_RANGE,
        shift_deg_range: tuple[float, float] = SHIFT_DEG_RANGE,
        amp_frac_range: tuple[float, float] = AMP_FRAC_RANGE,
        noise_c_range: tuple[float, float] = NOISE_C_RANGE,
    ) -> None:
        self._ranges = {
            "tau_nm": tau_nm_range, "u_iso": u_iso_range,
            "max_texture": max_texture_range, "shift_deg": shift_deg_range,
            "amp_frac": amp_frac_range, "noise_c": noise_c_range,
        }

    def __call__(self, record: MaterialRecord, seed: int) -> TransformResult:
        params = draw_xrd_params(seed, self._ranges)
        x, y = simulate(record, "xrd", seed=seed, **params)
        return TransformResult(x=x[CROP_LO:CROP_HI], y=y[CROP_LO:CROP_HI], params=params)
