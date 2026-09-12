"""modalities/pdf/from_xrd.py — measured XRD pattern -> *virtual* PDF, by sine FT.

The other way to get a G(r). `simulate.py` goes CIF -> diffpy -> G(r); this goes
measured I(2theta) -> G(r) with no structure involved, so it runs on experimental
data where no CIF is trusted. Ported from Szymanski et al., npj Comput. Mater. 10,
45 (2024), Eqs. 4-5 — the "virtual PDF" of `XRD-AutoAnalyzer`.

    2theta -> Q      Q = 4*pi*sin(theta)/lambda
    S(Q)   = Ic(Q) / <Ic(Q)>                          (their Eq. 4, + see below)
    G(r)   = (2/pi) * INT Q[S(Q) - 1] sin(Qr) dQ      (their Eq. 5)

Eq. 4 is the paper's own simplification of the proper total-scattering S(Q)
(their Eq. 3): the <b>^2 normalization needs the composition, and the whole point
of running this on unknown experimental samples is that the composition is not
known. So the form factors are dropped and the intensity is used raw. The output
is therefore NOT a physically calibrated G(r) — absolute amplitude is
meaningless, and only peak POSITIONS and relative structure survive. That is
exactly the part the encoder is allowed to use anyway, since the read path
min-max normalizes every PDF (`core/transforms.minmax_normalize`).

**The `/ <Ic(Q)>` is not in the paper, and omitting it destroys the channel.**
Eq. 5 subtracts 1 because a structure function OSCILLATES ABOUT 1 — that is the
Q -> infinity limit of S(Q), and the subtraction is what removes the DC term so
the transform sees only the structural modulation. An intensity array does not
obey that convention on its own. RRUFF patterns arrive max-normed to [0, 1] with
a near-zero baseline, so taking S = Ic literally puts `S - 1` at a MEAN OF -0.972
instead of ~0: F(Q) = Q[S-1] becomes a large negative ramp, essentially the same
ramp for every sample, and its transform buries the structure under a common
curve. Measured on all 148 RRUFF minerals (2026-08-07):

    S = Ic literally      mean pairwise corr between the 148 G(r) = 0.9994
                          (99.88% of variance is ONE shared curve)
                          corr vs diffpy at matched qmax        = -0.014
    S = Ic / <Ic>         mean pairwise corr between the 148    = 0.078
                          corr vs diffpy at matched qmax        = +0.863

So the division is what makes the channel carry the material at all. The scale
factor itself is arbitrary — read-path min-max normalization discards it — and
the mean is a stand-in for the inaccessible high-Q limit (Cu K-alpha stops at
5.24 A^-1, nowhere near where S(Q) actually flattens). Its ONLY job is to put the
DC term at zero, and any divisor that does that would serve equally well.

A consequence worth knowing: because S is rescaled to mean 1, `virtual_pdf` is
INVARIANT to the overall scale of its input. Feeding raw counts or a max-normed
pattern gives the identical G(r), so the caller never has to pre-normalize.

**No background subtraction.** The paper applies a rolling-ball subtraction to
remove incoherent scattering before Eq. 4. It is deliberately omitted here: the
one dataset this module feeds (RRUFF, `data/builders/rruff.py`) arrives with its
baseline already at ~0.1% of max, so a rolling ball would have nothing to remove
and would only erode the feet of real peaks. Re-add it here, behind a flag, if a
source with a real background ever needs it — not before.

**Read the Qmax caveat before interpreting any score from this.** Cu K-alpha over
2theta in [10, 80] reaches only Q = 5.24 A^-1. The PDF encoders in this repo were
pretrained on diffpy G(r) at qmax 15-30 A^-1 (`simulate.py` -> `QMAX_RANGE`), so a
virtual PDF sits a factor of ~3-6 below the bottom of the pretraining range. Two
consequences, both real and neither fixable by tuning:

* **Resolution.** Real-space resolution is ~pi/Qmax ~ 0.6 A. Nearest-neighbour
  distances (1.5-3 A) are barely separated; fine structure the encoder learned to
  read at qmax 22.5 is simply not present.
* **Termination ripple.** Truncating the integral at 5.24 A^-1 rings at period
  2*pi/Qmax ~ 1.2 A across the whole r range. **This does NOT make high r useless**,
  which is the intuitive but wrong conclusion — a diffpy reference truncated at the
  same Qmax rings at the same period, so against a matched reference the ripple is
  shared signal. Measured over the 7 minerals of `tools/plot_rruff_pdf_views.py`,
  agreement with matched-qmax diffpy IMPROVES with r: 0.863 on [1, 20] A, 0.874 on
  [0, 50], 0.891 on [20, 50] alone. The worst-agreeing region is low r, where the
  Qmin = 0.71 cutoff distorts the baseline. Judge a virtual PDF against a reference
  at ITS OWN Qmax; against a sharp one, ripple reads as error that is not there.

Both are properties of the method, not bugs, and the paper lives with them by
training its PDF network on virtual PDFs rather than on simulated ones. This repo
does the opposite — it takes an encoder pretrained on diffpy PDFs and feeds it
virtual ones — so the domain gap is the thing under test and should be reported as
the headline caveat, not buried.
"""

from __future__ import annotations

import numpy as np

from modalities.xrd.grid import WAVELENGTH_A

__all__ = ["two_theta_to_q", "virtual_pdf"]


def two_theta_to_q(tt_deg, wavelength: float = WAVELENGTH_A):
    """2theta (deg) -> Q (A^-1) via ``Q = 4*pi*sin(theta)/lambda``.

    The forward direction of `modalities.xrd.grid.q_to_2theta_deg`, and total on
    [0, 180] — every angle has a Q, so unlike the inverse there is no NaN branch
    and no boundary case to get wrong.
    """
    tt_deg = np.asarray(tt_deg, dtype=np.float64)
    return 4.0 * np.pi * np.sin(np.radians(tt_deg) / 2.0) / wavelength


def virtual_pdf(tt_deg, intensity, r, *, wavelength: float = WAVELENGTH_A):
    """Sine-transform a measured pattern to G(r). Szymanski Eqs. 4-5.

    `intensity` is ``(..., n_2theta)`` — a single pattern or a whole stack, since
    the sin kernel depends only on (Q, r) and is built once and shared. Returns
    ``(..., n_r)`` float64.

    `tt_deg` must be sorted ascending and is NOT resampled: the caller decides
    what window the transform sees, because that window sets Qmax and therefore
    the resolution of everything downstream (see the module docstring). Integration
    is trapezoidal on whatever spacing `tt_deg` implies — note the Q grid is NOT
    uniform even when the 2theta grid is, which is why the weights are computed
    from the Q values rather than assumed constant.
    """
    tt_deg = np.asarray(tt_deg, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    if tt_deg.ndim != 1:
        raise ValueError(f"tt_deg must be 1-D, got shape {tt_deg.shape}")
    if intensity.shape[-1] != tt_deg.size:
        raise ValueError(
            f"intensity last axis is {intensity.shape[-1]}, expected {tt_deg.size} to match tt_deg"
        )
    if tt_deg.size < 2:
        raise ValueError("need at least 2 points to integrate")
    if np.any(np.diff(tt_deg) <= 0):
        raise ValueError("tt_deg must be strictly increasing")

    q = two_theta_to_q(tt_deg, wavelength)

    # S(Q) = Ic(Q) / <Ic(Q)>, so S oscillates about 1 and [S - 1] has zero mean.
    # Without this the DC term dominates and every sample transforms to the same
    # curve — see the module docstring for the measured numbers.
    mean = intensity.mean(axis=-1, keepdims=True)
    if np.any(mean <= 0):
        raise ValueError("intensity must have a positive mean to normalize S(Q) to 1")
    f_q = q * (intensity / mean - 1.0)

    # Trapezoid weights from the Q values themselves (the Q grid is non-uniform).
    w = np.empty_like(q)
    w[0] = (q[1] - q[0]) / 2.0
    w[-1] = (q[-1] - q[-2]) / 2.0
    w[1:-1] = (q[2:] - q[:-2]) / 2.0

    # (n_q, n_r) kernel, built once and reused across every row of `intensity`.
    kernel = np.sin(np.outer(q, r))
    return (2.0 / np.pi) * ((w * f_q) @ kernel)
