"""modalities/xrd/grid.py — the XRD 2theta grid, and CHILI's Q -> 2theta map.

Split out from `simulate.py` (not yet built) because it is the only part the
DOWNSTREAM channel needs: `analysis/downstream_eval.py` has to know what grid an
XRD signal is supposed to live on, and the step-0 gate has to put CHILI's native
Q-grid channel onto it. The simulator will import these constants rather than
redeclare them.

Ported from `PDF_XRD_Fusion/XRD/src/xrd_preprocess.py`, which calls itself the
"single source of truth" for CHILI XRD domain matching and is imported by that
repo's linear probe, finetune and supervised baseline alike. TWO deliberate
departures, both settled in `DESIGN.md` -> *XRD arm*:

1. **The grid.** The reference uses `linspace(0, 180, 5000)` cropped to [10, 80]
   -> 1944 points, whose endpoints are NOT exactly 10 and 80. Here the native
   grid is `linspace(0, 180, 12853)` cropped `[714:5713]` -> **4999 points with
   endpoints exactly 10.0 and 80.0**. Exactly-5000-on-exactly-[10,80] is
   arithmetically impossible (see `docs/MODULES.md`), and `CNNEncoder` is
   length-agnostic, so the round number was the cheaper thing to give up.
2. **Normalization is NOT done here.** The reference max-norms inside
   `preprocess_xrd`. In this repo normalization is a READ-PATH step
   (`DESIGN.md` -> *Signal normalization*): registries and banks store raw
   physical intensity so a normalization change is a read-path edit, not a
   regeneration. `resample_chili_xrd` therefore returns RAW intensity.

The zero-fill outside CHILI's angular coverage is kept exactly as the reference
has it, and is load-bearing: it is why the XRD channel must be max-normed rather
than min-max normed (a zero-filled row has min = 0 while a simulated pattern with
a background has min > 0, so min-max would treat the two differently).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "WAVELENGTH_A", "N_FULL", "TT_FULL", "CROP_LO", "CROP_HI", "TT_GRID", "GRID_LEN",
    "Q_MAX", "q_to_2theta_deg", "resample_chili_xrd",
]

#: Cu K-alpha. The one constant coupling pretraining to the CHILI Q -> 2theta map;
#: a Kalpha1 (1.54056) vs Kalpha (1.5406) slip moves 2theta by ~7e-3 deg at 26.6 deg.
WAVELENGTH_A = 1.5406

#: Simulate WIDE, then crop. Out-of-window reflections leak into [10, 80] once
#: broadened (20.2% of an 8 deg peak's height at tau = 3 nm), so simulating
#: directly on the window is wrong. See DESIGN.md -> XRD arm.
N_FULL = 12853
TT_FULL = np.linspace(0.0, 180.0, N_FULL)

#: `int(...)` so these stay plain ints: they index, and a numpy scalar here would
#: propagate into shapes and JSON.
CROP_LO = int(np.searchsorted(TT_FULL, 10.0, side="left"))    # 714
CROP_HI = int(np.searchsorted(TT_FULL, 80.0, side="right"))   # 5713

#: The encoder input grid: 4999 points, endpoints EXACTLY 10.0 and 80.0 deg.
TT_GRID = TT_FULL[CROP_LO:CROP_HI]
GRID_LEN = len(TT_GRID)

#: Above this there is no real scattering angle at this wavelength: Q = 4pi/lambda
#: maps to exactly 2theta = 180 deg. 75.2% of CHILI's stored Q grid sits above it.
Q_MAX = 4.0 * np.pi / WAVELENGTH_A


def q_to_2theta_deg(q):
    """Q (A^-1) -> 2theta (deg), NaN where no real angle exists.

    Inverts `Q = 4*pi*sin(theta)/lambda`. `arg <= 1.0` rather than `< 1.0`: at
    `Q = Q_MAX` exactly, `arg == 1` and the point maps to exactly 180 deg, which
    is a real angle. One character the other way silently drops it.

    DIVIDES BY `Q_MAX` rather than computing `q * lambda / (4*pi)` as the
    reference does. The two are algebraically identical and differ by one ulp,
    but that ulp lands exactly on the boundary: the reference's form round-trips
    `Q_MAX` to **1.0000000000000002**, so its own `arg <= 1.0` test fails and the
    180 deg point becomes NaN. `q / Q_MAX` is exactly 1.0 there by construction.
    Measured while porting (2026-08-06); the reference has the boundary bug.
    """
    q = np.asarray(q, dtype=np.float64)
    arg = q / Q_MAX
    arg = np.where(arg <= 1.0, arg, np.nan)
    return 2.0 * np.degrees(np.arcsin(arg))


def resample_chili_xrd(q, intensity):
    """CHILI's (Q, intensity) -> RAW intensity on `TT_GRID`, zero-filled.

    Drops the unphysical high-Q tail, sorts into increasing 2theta (`np.interp`
    requires it and the Q -> 2theta map is monotone only over the physical part),
    then interpolates onto `TT_GRID` with **zero outside CHILI's coverage** —
    `left=0.0, right=0.0`, exactly as the reference does.

    This is an UPSAMPLE and says so: CHILI keeps ~85 real points inside [10, 80]
    at 0.759 deg spacing, so the result is an 85-node polyline stretched over
    4999 samples, not 4999 measurements (`docs/TRAPS.md`). Callers must not read
    it as resolution it does not have.

    Returns float64 raw intensity; normalization is the read path's job.
    """
    q = np.asarray(q, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if q.shape != intensity.shape:
        raise ValueError(f"q {q.shape} and intensity {intensity.shape} must have the same shape")

    tt = q_to_2theta_deg(q)
    valid = ~np.isnan(tt)
    if not valid.any():
        raise ValueError(f"no physical scattering angle in this row: all {q.size} Q values exceed {Q_MAX:.6f}")
    tt, iv = tt[valid], intensity[valid]
    order = np.argsort(tt)
    return np.interp(TT_GRID, tt[order], iv[order], left=0.0, right=0.0)
