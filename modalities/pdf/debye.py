"""modalities/pdf/debye.py — finite-NANOPARTICLE PDF via the Debye scattering equation.

The second PDF forward model. `simulate.py` in this package is diffpy
``PDFCalculator`` on a PERIODIC cell (an infinite crystal); this one is the
``debyecalculator`` package's Debye sum over the actual atom positions of a
finite particle. They are not interchangeable:

    simulate.py   CIF -> G(r) of the infinite parent crystal
    debye.py      (elements, coords) -> G(r) of one nanoparticle

Only the Debye route carries the particle-size envelope — the low-frequency
decay of G(r) that encodes how big the particle is — so it is the only one that
can produce an augmented view of a CHILI sample without destroying
``target_np_size``. It is also the engine CHILI-3K's own native xPDF was
generated with, which is why an augmented view stays on-distribution with the
clean one.

Deliberately NOT registered with `core.simulate`: that seam's contract is
``(cif, **params) -> (x, y)`` and a nanoparticle is not a CIF. The one caller
(`tools/augment_chili.py`) imports this function directly. If a second caller
ever appears, generalizing the seam is the change to make — not adding a fake
CIF round-trip here.

``debyecalculator`` is a heavy optional dependency (and drags torch in with it),
so it is imported inside the function; importing this module is free.

QBROAD IS NOT AN ENVELOPE — the reference repo got this wrong, and so did the
first version of this module. It multiplied G(r) by ``exp(-0.5*(r*qbroad)**2)``,
which is measurably diffpy's **Qdamp**, not its Qbroad: diffpy's
``g(qdamp=X)/g(qdamp=0)`` equals that expression to 1.1e-16, while its
``g(qbroad=X)/g(qbroad=0)`` is not an envelope at all (it changes sign).

That mattered here rather than being a naming quibble. A decaying r-envelope is
exactly how a finite particle's size shows up in G(r), so drawing one at random
was overwriting ``target_np_size``: at qbroad=0.06 the envelope halves G(r) by
r = 19.6 A, and 80% of CHILI's particles (7.4-56.7 A) are larger than that, i.e.
the augmentation truncated the PDF before the particle itself did.

Qbroad is r-dependent PEAK BROADENING (diffpy's Jeong width model with
delta1=delta2=0): ``sigma(r) = sigma0 * sqrt(1 + (qbroad*r)**2)``, so the width
to add in quadrature on top of an already-broadened G(r) is
``sigma_add(r) = sigma0 * qbroad * r``. Measured against diffpy s19: the
best-fit sigma0 lands on ``sqrt(2*uiso)`` exactly at every (uiso, qbroad) tried
(0.002/0.005/0.02 x 0.02/0.05), correlation >= 0.9977. `sqrt(2*uiso)` is the
pair-distance width for two uncorrelated atoms of equal Uiso, which is the same
convention `biso` feeds to DebyeCalculator.
"""

from __future__ import annotations

import math

import numpy as np

# xPDF grid + fixed instrument settings, from the reference's augment.ipynb
# (which took them from CHILI-3K's own generation, Appendix A.4 Table 7).
#
# XPDF_-prefixed because simulate.py in this package exports RMIN/RMAX/RSTEP too,
# on a DIFFERENT grid (0-50 A, the pretraining one). Same origin and step, longer
# rmax; tools/augment_chili.py asserts that relationship, since downstream_eval
# truncates this grid to that one rather than resampling.
XPDF_RMIN, XPDF_RMAX, XPDF_RSTEP = 0.0, 60.0, 0.01  # -> 6000 pts, CHILI's native xPDF
QSTEP = 0.05
QDAMP = 0.04  # DebyeCalculator's default, used to generate CHILI-3K


def simulate_pdf_debye(
    elements: list[str],
    coords: np.ndarray,
    *,
    uiso: float,
    qmax: float,
    qbroad: float,
    qmin: float,
    rmin: float = XPDF_RMIN,
    rmax: float = XPDF_RMAX,
    rstep: float = XPDF_RSTEP,
) -> tuple[np.ndarray, np.ndarray]:
    """(r, G(r)) for one nanoparticle, at the given instrument params.

    `elements` are chemical symbols and `coords` their (n, 3) ABSOLUTE positions
    in Angstrom — a finite cluster, not a unit cell.

    Two conversions the caller must not have to know about:
      * DebyeCalculator takes Biso, not Uiso: ``Biso = 8*pi**2 * Uiso``.
      * DebyeCalculator has no qbroad parameter, so the r-dependent peak
        broadening is applied afterwards by `_qbroad_blur` — NOT as an r-envelope,
        which is Qdamp and would corrupt the particle size (see module docstring).

    Qdamp itself stays fixed at CHILI's own generation value; it is instrument
    resolution, not something this augmentation varies.

    Deterministic in (elements, coords, params) — the drawing of those params is
    `core.transforms.draw_pdf_params`, not this function.
    """
    from debyecalculator import DebyeCalculator

    dc = DebyeCalculator(
        qmin=qmin, qmax=qmax, qstep=QSTEP, biso=8 * math.pi**2 * uiso,
        rmin=rmin, rmax=rmax, rstep=rstep, qdamp=QDAMP, device="cpu",
    )
    r, g = dc.gr((elements, np.asarray(coords, dtype=float)))
    r = np.asarray(r, dtype=np.float32)
    g = _qbroad_blur(r, np.asarray(g, dtype=np.float64), uiso=uiso, qbroad=qbroad)
    return r, g.astype(np.float32)


def _qbroad_blur(r: np.ndarray, g: np.ndarray, *, uiso: float, qbroad: float) -> np.ndarray:
    """Add diffpy's Qbroad peak broadening to an already-computed G(r).

    Each point is smeared into a Gaussian of width ``sqrt(2*uiso) * qbroad * r``
    evaluated at ITS OWN r, so peaks at larger r broaden more — that r-dependence
    is the whole content of Qbroad, and is what an r-envelope cannot express.
    Widths add in quadrature under convolution, which is why this composes
    correctly with the thermal width already in `g`.

    The kernel is normalized to unit sum, so the blur conserves area and cannot
    fake the amplitude decay that encodes particle size.
    """
    if qbroad <= 0:
        return g
    dr = float(r[1] - r[0])
    sigma = math.sqrt(2 * uiso) * qbroad * r
    out = np.zeros_like(g)
    # Plain loop over source points: the kernel width changes with r, so this is a
    # non-stationary convolution and np.convolve does not apply. ~6000 iterations
    # against a Debye sum measured in seconds-to-minutes — not the bottleneck.
    for j in np.nonzero(g)[0]:
        s = sigma[j]
        if s < dr / 10:  # narrower than the grid: nothing to spread
            out[j] += g[j]
            continue
        half = int(6 * s / dr) + 1
        lo, hi = max(0, j - half), min(len(g), j + half + 1)
        k = np.exp(-0.5 * ((r[lo:hi] - r[j]) / s) ** 2)
        out[lo:hi] += g[j] * k / k.sum()
    return out
