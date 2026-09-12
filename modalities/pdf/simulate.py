"""modalities/pdf/simulate.py — CIF -> pair distribution function G(r) via diffpy.

The PDF forward model ported from the reference repo (PDF's generate_mpfull.py /
load.ipynb): CIF -> diffpy Structure -> set an isotropic ADP (Uiso) on every
atom -> diffpy.srreal ``PDFCalculator`` on a fixed r-grid with instrument params
(qmax, qbroad, qmin) -> G(r).

It is a PURE deterministic function of (cif, params): no random draws here. In
the old repo these same four params (uiso, qmax, qbroad, qmin) were *sampled*
per view to make two augmented views; that sampling belongs in
core/transforms.py, not in the forward model. This module just evaluates the
model at whatever params it is handed.

diffpy (``diffpy.structure`` + ``diffpy.srreal``) is a heavy, env-specific
dependency, so it is imported inside the function; importing this module only
registers the backend with core.simulate.

Defaults reproduce the reference's fixed r-grid (0..50 A, step 0.01 -> 5000 pts);
the instrument defaults are nominal mid-range values (the old repo only ever
sampled them, so there is no single "pristine" upstream value to copy).
"""

from __future__ import annotations

from io import StringIO

import numpy as np

from core.simulate import register

# Fixed r-grid from generate_mpfull.py (RMIN/RMAX/DR). This is THE pretraining
# grid — every encoder checkpoint expects 5000 points on it. debye.py's parallel
# XPDF_* constants are CHILI's 0-60 A xPDF grid; don't cross the two.
RMIN, RMAX, RSTEP = 0.0, 50.0, 0.01
# Nominal instrument params: mid-range of the reference's augmentation ranges
# (UisoJitter 1e-3..0.03, QmaxJitter 15..30, QbroadJitter 0.01..0.06, QminJitter 0.5..2).
UISO, QMAX, QBROAD, QMIN = 0.005, 22.5, 0.03, 1.0


def simulate_pdf(
    cif: str,
    *,
    rmin: float = RMIN,
    rmax: float = RMAX,
    rstep: float = RSTEP,
    uiso: float = UISO,
    qmax: float = QMAX,
    qbroad: float = QBROAD,
    qmin: float = QMIN,
) -> tuple[np.ndarray, np.ndarray]:
    """(r, G(r)) for one CIF on the r-grid, at the given instrument params.

    Deterministic: the params are the forward model, not jitter. Raises whatever
    diffpy raises on an unparseable/unsimulable structure — callers that batch
    over a registry decide whether to skip (as the builders do).
    """
    from diffpy.srreal.pdfcalculator import PDFCalculator
    from diffpy.structure import loadStructure

    struc = loadStructure(StringIO(cif), fmt="cif")
    for atom in struc:
        atom.Uisoequiv = uiso
    pc = PDFCalculator()
    pc.rmin, pc.rmax, pc.rstep = rmin, rmax, rstep
    pc.qmax, pc.qbroad, pc.qmin = qmax, qbroad, qmin
    r, g = pc(struc)
    return np.asarray(r, dtype=np.float32), np.asarray(g, dtype=np.float32)


register("pdf", simulate_pdf)
