"""modalities/xrd/debye.py — one nanoparticle's XRD pattern, by Debye sum.

The downstream counterpart of `modalities/xrd/simulate.py`, and the sibling of
`modalities/pdf/debye.py`. Where `simulate.py` takes a periodic CIF and gets a
reflection list from pymatgen, this takes a particle's ACTUAL atoms and sums over
pairs, because that is the only way the finite-size envelope `target_np_size` is
read from survives.

    elements + coords  ->  DebyeCalculator.iq  ->  I(Q)
                       ->  Q -> 2theta at Cu K-alpha
                       ->  2theta zero shift        [shift_deg]
                       ->  resample onto TT_GRID
                       ->  rescale so max = 100
                       ->  Chebyshev background     [amp_frac, cheb_0..3]
                       ->  Gaussian noise, clipped  [noise_c, seed]

**NO SCHERRER BROADENING, AT ALL.** This is the load-bearing difference from the
pretraining forward model and it is settled in `DESIGN.md` -> *XRD arm*: the atoms
already carry the particle size, so the width is physics the Debye sum produces
rather than a parameter. Adding Scherrer at `tau = np_size` on top inflates every
peak by a factor of sqrt(2) while leaving `target_np_size` monotone — R^2 stays
high, nothing looks broken, and `clean_aug` would then be measuring a size bias
instead of the domain shift it exists to measure. `tests/test_chili_xrd.py` T6.2
is built to catch exactly that: a double-count lands at sqrt(2) and fails.

`tau_nm` is therefore not a parameter here, and neither are the four texture
parameters — a Debye sum over real atoms has no reflection list for
`_apply_texture` to scale. This function takes ONLY the eight params it actually
applies. The caller (`tools/augment_chili_xrd.py`) records twelve, four of them
inert; keeping that asymmetry at one documented seam in the tool is deliberate,
so that nothing in the physics layer accepts a parameter it silently ignores.

The background and noise terms are IMPORTED from `modalities/xrd/simulate.py`
rather than reimplemented, so the two channels cannot drift in what `amp_frac`
and `noise_c` mean.

Needs `debyecalculator`. Imported inside the function, as the PDF sibling does.
"""

from __future__ import annotations

import math

import numpy as np

from modalities.xrd.grid import TT_GRID, WAVELENGTH_A, q_to_2theta_deg
from modalities.xrd.simulate import _add_background

__all__ = ["simulate_xrd_debye", "QSTEP", "Q_LO", "Q_HI", "TT_MARGIN_DEG"]

#: Q sampling for the Debye sum. MEASURED trade, 2026-08-07: cost is
#: O(n_atoms^2 * n_q) and CHILI's tail runs to 14793 atoms, so the whole 3180-particle
#: job is ~5 h at this step against ~31 h at 0.00075 (which would match the encoder
#: grid step everywhere). Nothing physical is lost: CHILI's particles are
#: 0.74-5.67 nm, so the NARROWEST peak that can exist is ~1.45 deg FWHM, and this
#: samples it at 0.077-0.092 deg — already ~16x oversampled. The finer grid would
#: resolve structure no CHILI particle can produce.
QSTEP = 0.005

#: Simulate a little wider than the encoder window so the `shift_deg` draw, which
#: spans +/-0.3 deg, never has to be extrapolated at either end.
TT_MARGIN_DEG = 0.5


def _q_at(two_theta_deg: float) -> float:
    """2theta (deg) -> Q (A^-1) at this wavelength. The inverse of `q_to_2theta_deg`."""
    return 4.0 * math.pi * math.sin(math.radians(two_theta_deg / 2.0)) / WAVELENGTH_A


Q_LO = _q_at(float(TT_GRID[0]) - TT_MARGIN_DEG)
Q_HI = _q_at(float(TT_GRID[-1]) + TT_MARGIN_DEG)


def simulate_xrd_debye(
    elements: list[str],
    coords: np.ndarray,
    *,
    u_iso: float,
    shift_deg: float,
    amp_frac: float,
    cheb_0: float,
    cheb_1: float,
    cheb_2: float,
    cheb_3: float,
    noise_c: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(2theta, intensity) for one nanoparticle on `TT_GRID`, at the given params.

    `u_iso` reaches `DebyeCalculator` as `Biso = 8*pi^2*u_iso`, the same
    conversion `modalities/pdf/debye.py` makes and the same convention
    `modalities/xrd/simulate.py` hands to pymatgen — so one drawn `u_iso` means
    the same displacement in all three places.

    Returns float32 intensity of length `len(TT_GRID)` (4999) on exactly
    [10, 80] deg. Unlike the pretraining backend this returns the WINDOW, not the
    full [0, 180] grid: there is no crop step downstream of it, and a Debye sum
    over a finite particle has no out-of-window reflections to leak in — the
    simulate-wide-then-crop argument is about a periodic lattice's reflection
    list, which this does not have.
    """
    from debyecalculator import DebyeCalculator

    dc = DebyeCalculator(
        qmin=Q_LO, qmax=Q_HI, qstep=QSTEP,
        biso=8.0 * math.pi ** 2 * u_iso,
        device="cpu", radiation_type="xray",
    )
    q, iq = dc.iq((elements, np.asarray(coords, dtype=float)))
    q = np.asarray(q, dtype=np.float64)
    iq = np.asarray(iq, dtype=np.float64)

    # Q -> 2theta, then the goniometer zero shift, on the axis rather than on the
    # finished profile — the same ordering `simulate_xrd` uses.
    tt = q_to_2theta_deg(q) + shift_deg
    good = ~np.isnan(tt)
    tt, iq = tt[good], iq[good]
    order = np.argsort(tt)  # np.interp needs an increasing x
    y = np.interp(TT_GRID, tt[order], iq[order])

    m = y.max()
    if m > 0:
        y = 100.0 * y / m
    y = _add_background(y, amp_frac, (cheb_0, cheb_1, cheb_2, cheb_3), TT_GRID)

    sigma_n = noise_c * y.max()
    if sigma_n > 0:
        y = y + np.random.default_rng(seed).normal(0.0, sigma_n, size=y.shape)
    return TT_GRID.copy(), np.clip(y, 0.0, None).astype(np.float32)
