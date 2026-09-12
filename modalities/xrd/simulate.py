"""modalities/xrd/simulate.py — CIF -> powder XRD pattern I(2theta).

The XRD forward model, ported from `PDF_XRD_Fusion/XRD/src/xrd_simulate.py`.
Physical order, each term a separate function below:

    CIF
    -> Debye-Waller-damped sticks                [u_iso]
    -> texture scaling                           [max_texture, pref_h/k/l]
    -> 2theta zero-point shift                   [shift_deg]
    -> Scherrer broadening onto the grid         [tau_nm]
    -> rescale so max = 100
    -> Chebyshev background                      [amp_frac, cheb_0..3]
    -> additive Gaussian noise, clipped >= 0     [noise_c, seed]

Like `modalities/pdf/simulate.py` this is a PURE deterministic function of
(cif, params) — the `seed` is a param, not ambient state (see below). Param
*sampling* is not here; it lands in `core/transforms.py` (build step 3).

The grid comes from `modalities/xrd/grid.py` and is NOT redeclared: the output
lives on `TT_FULL`, the full `[0, 180]` deg grid, and cropping to the encoder
window `[10, 80]` is the read path's job. That split is the whole point of
"simulate wide, then crop" (DESIGN.md -> XRD arm): an out-of-window reflection
leaks a fifth of its height into the window once broadened, so a simulator that
worked on `[10, 80]` directly would be missing intensity that is really there.
`tests/test_xrd_simulate.py` measures the leak rather than asserting the comment.

FOUR DEPARTURES from the reference, all deliberate:

1. **The grid is a parameter, not a default argument.** The reference's
   `_broaden(..., steps=STEPS_2T, step_size=STEP_SIZE)` binds both at `def`
   time, so it cannot be repointed and a benchmark that patched the module
   constant measured nothing at all (`docs/TRAPS.md`). Here every helper takes
   `tt` explicitly and derives its step from `tt` itself.
2. **The step is derived from the grid.** The reference computes
   `STEP_SIZE = (MAX_2T - MIN_2T) / N_STEPS`, which is off by one — the real
   step of `linspace(0, 180, N)` is `180/(N-1)`. It feeds
   `gaussian_filter1d(row, sigma / step_size)`, so every peak in the reference
   corpus is ~0.02% too wide. Measured 2026-08-06 while porting.
3. **`_broaden` evaluates the Gaussian analytically over a +/-4 sigma window**
   instead of scattering deltas into an `(n_reflections, n_grid)` array and
   convolving each row. Same physics, three consequences: peak centres are exact
   rather than snapped to the nearest grid point, nothing allocates
   `n_reflections x 12853`, and it is **130x faster on a 19-reflection cell and
   167x on a 4889-reflection one** (28 ms -> 0.2 ms, 9.0 s -> 54 ms; measured
   2026-08-06 against the reference repointed at this grid). The cost of a
   pattern is now pymatgen's `get_pattern` — 908 ms of the 954 ms a
   4889-reflection structure takes — so the broadening is no longer worth
   optimizing and the banking budget should be re-estimated from `get_pattern`,
   not from the reference's 377 ms figure. The 4 sigma truncation drops 6.3e-5
   of each peak's area; `tests/test_xrd_simulate.py` T4 holds the two
   implementations to 0.1%.
4. **The noise RNG is local and seeded.** The reference draws from global
   `np.random`, which makes it unbankable under `tools/bank_pdf.py`'s fork pool
   — a worker's realization would depend on how many prior tasks it happened to
   pull (`docs/TRAPS.md`). `seed` is a required keyword here so a caller cannot
   silently get correlated views.

Every other keyword defaults to its own physical no-op (`u_iso=0` is B=0,
`max_texture=0` is the identity, `amp_frac=0` is no baseline, `noise_c=0` is
noiseless), so a test can switch on one term at a time and `simulate_xrd(cif,
tau_nm=..., seed=...)` is the clean Bragg-plus-Scherrer pattern. `tau_nm` has no
no-op — an unbroadened pattern is a comb of deltas, not a physical limit — so it
is required.

pymatgen is imported inside `_sticks`, matching the PDF backend: importing this
module only registers the backend with `core.simulate`.
"""

from __future__ import annotations

import numpy as np

from core.simulate import register
from modalities.xrd.grid import N_FULL, TT_FULL, WAVELENGTH_A

__all__ = ["simulate_xrd", "MAX_SIGMA_DEG"]

#: Scherrer broadening carries a 1/cos(theta) factor that diverges at
#: backscattering: a reflection at 2theta -> 180 deg gets sigma -> inf. In the
#: reference that was a crash — `gaussian_filter1d` sizes its kernel at ~4 sigma
#: and tried to allocate a multi-exabyte `np.arange`. The windowed evaluation
#: here cannot crash on it (the window is clipped to the grid), but an uncapped
#: sigma still smears one reflection into a flat pedestal across the whole
#: pattern, so the cap is kept and is now about physics rather than memory. At
#: the low end of AUG_RANGES (tau = 3 nm) sigma is 6.5 deg at 2theta = 160 and
#: 1.5 deg at the 80 deg window edge — MEASURED, the reference's docstring says
#: 7.7 deg — so 10 deg leaves every physical peak untouched and only tames the
#: singularity.
MAX_SIGMA_DEG = 10.0


def _sticks(cif: str, u_iso: float):
    """CIF -> (angles_deg, intensities, hkls) with isotropic Debye-Waller damping.

    `B = 8*pi^2*u_iso` is handed to `XRDCalculator` per element. pymatgen applies
    it to the scattering AMPLITUDE as `exp(-B*s^2)` with `s = sin(theta)/lambda`,
    and intensity is the modulus square, so the intensity damping is
    `exp(-2*B*s^2)`. That factor of two is what T2.3 pins.

    Computed over the full `[0, 180]` deg range — the same range `TT_FULL`
    spans, and wider than the encoder window on purpose (see the module
    docstring). Intensities are pymatgen's `scaled=True` values (max = 100), so
    they carry no absolute scale; only ratios within one pattern mean anything.

    `hkls` takes ONE representative per reflection — pymatgen groups the whole
    multiplicity family under a single 2theta and `_apply_texture` needs a
    direction. See its docstring for what that approximates away.
    """
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.core import Structure

    struc = Structure.from_str(cif, fmt="cif")
    B = 8.0 * np.pi ** 2 * u_iso
    calc = XRDCalculator(
        wavelength=WAVELENGTH_A,
        debye_waller_factors={el.symbol: B for el in struc.composition.elements},
    )
    pat = calc.get_pattern(struc, two_theta_range=(float(TT_FULL[0]), float(TT_FULL[-1])))
    hkls = [v[0]["hkl"] for v in pat.hkls]
    return np.asarray(pat.x, dtype=np.float64), np.asarray(pat.y, dtype=np.float64), hkls


def _scherrer_sigma_deg(two_theta_deg: float, tau_nm: float) -> float:
    """Gaussian sigma (deg 2theta) for a crystallite of size `tau_nm`.

    Scherrer: `beta = K*lambda/(tau*cos(theta))` in radians, with `beta` the
    integral FWHM and K = 0.9 (the usual spherical-ish shape factor). A Gaussian
    of FWHM w has `sigma = w / (2*sqrt(2*ln2))` — dropping that conversion is a
    factor of 2.355, and doing the arithmetic in Angstrom instead of nm is a
    factor of 10. T2.2 sees both.

    `two_theta_deg` is clipped into [0, 180]: the 2theta zero shift can push a
    near-backscattering reflection past 180 deg, where `cos(theta)` turns
    negative and would hand back a NEGATIVE sigma. There is no scattering angle
    out there, so the width is evaluated at the boundary instead.
    """
    K, wl_nm = 0.9, WAVELENGTH_A / 10.0
    theta = np.radians(min(max(two_theta_deg, 0.0), 180.0) / 2.0)
    fwhm_deg = np.degrees(K * wl_nm / (np.cos(theta) * tau_nm))
    return float(min(fwhm_deg / (2.0 * np.sqrt(2.0 * np.log(2.0))), MAX_SIGMA_DEG))


def _broaden(angles_deg, intensities, tau_nm: float, tt):
    """Reflection list -> a profile on `tt`, each peak a Scherrer-width Gaussian.

    Each reflection is written as `I * N(ang, sigma)` sampled on `tt` over a
    +/-4 sigma window, so its SAMPLE SUM is its stick intensity `I`: integrated
    intensity is the conserved quantity, and peak height falls as 1/sigma.

    That is the same convention the reference gets by convolving a delta with
    `gaussian_filter1d` (whose kernel is normalized to sum 1), but evaluated in
    closed form instead. Three differences, all in this direction:

      - the reference snaps each reflection to the nearest grid point before
        convolving, quantizing peak centres to 0.014 deg — a fifth of the
        `shift_deg` range it is supposed to be able to express. Here the centre
        is exact.
      - it allocates `len(angles) x len(tt)` floats and filters every row over
        the whole grid; this touches only the ~8 sigma each peak occupies.
      - the 4 sigma truncation drops 6.3e-5 of each peak's area, which the
        reference's normalized kernel does not. T4 holds the two to 0.1%.

    `tt` and its step are arguments, never defaults — a grid captured in
    `__defaults__` cannot be repointed, and the benchmark that tried measured
    nothing at all (`docs/TRAPS.md`).
    """
    step = float(tt[1] - tt[0])
    y = np.zeros(len(tt), dtype=np.float64)
    for ang, inten in zip(angles_deg, intensities):
        sigma = _scherrer_sigma_deg(float(ang), tau_nm)
        lo = int(np.searchsorted(tt, ang - 4.0 * sigma, side="left"))
        hi = int(np.searchsorted(tt, ang + 4.0 * sigma, side="right"))
        if hi <= lo:
            continue  # the whole peak fell off the grid
        w = tt[lo:hi]
        amp = inten * step / (sigma * np.sqrt(2.0 * np.pi))
        y[lo:hi] += amp * np.exp(-0.5 * ((w - ang) / sigma) ** 2)
    return y


def _miller(hkl) -> tuple[int, int, int]:
    """pymatgen's hkl -> a 3-index Miller direction.

    For a HEXAGONAL lattice pymatgen returns four Miller-Bravais indices
    `(h, k, i, l)` with the redundant `i = -h-k`; everything else returns three.
    The reference takes `hkl[:3]`, which for a hexagonal cell keeps the
    redundant index and DROPS `l` — so every basal `(0 0 0 l)` reflection
    becomes the zero vector and gets the maximum texture suppression no matter
    what `pref` is, while every other family silently loses its c component.
    Measured in `tests/test_xrd_simulate.py` on a hexagonal Mg cell: 6
    reflections zero out and texture scales move by up to 0.50 once fixed. It
    was 3 of 28 on the two-atom hcp cell it was first found on — the count is a
    property of the cell, the defect is not. That is a fifth defect in the reference,
    found 2026-08-06 by porting it; it is structure-class-dependent, so it never
    shows up on the cubic cells a spot-check would use.
    """
    return (int(hkl[0]), int(hkl[1]), int(hkl[-1]))


def _apply_texture(intensities, hkls, pref_dir, max_texture: float):
    """Scale each reflection by how well its plane normal aligns with `pref_dir`.

    `f = |cos(angle between hkl and pref)|` in [0, 1] maps linearly onto
    `[1 - max_texture, 1]`, so `max_texture = 0` is exactly the identity and a
    reflection perpendicular to the preferred direction is suppressed most.

    TWO APPROXIMATIONS, both inherited and both deliberate:

      - the cosine is taken between raw index triples, i.e. in a CARTESIAN
        metric. Miller indices only span a Euclidean space for a cubic lattice;
        anywhere else the true angle needs the reciprocal metric tensor. This is
        a plausibility-shaped augmentation, not a measurement of real preferred
        orientation, so the cheap version is kept.
      - `hkls` carries ONE arbitrary representative of each multiplicity family
        (pymatgen collapses e.g. all six {100} into a single 2theta), so which
        member gets read decides the scale factor. That makes an exact expected
        value a statement about pymatgen's sort order rather than about physics
        — which is why T2.4 asserts bounds and identities only.
    """
    pref = np.asarray(pref_dir, dtype=float)
    bound = 1.0 - max_texture
    scales = np.empty(len(hkls), dtype=np.float64)
    for i, hkl in enumerate(hkls):
        v = np.asarray(_miller(hkl), dtype=float)
        norm = np.linalg.norm(v) * np.linalg.norm(pref)
        f = abs(float(np.dot(v, pref)) / norm) if norm > 0 else 0.0
        scales[i] = bound + (1.0 - bound) * f
    return intensities * scales


def _add_background(y, amp_frac: float, cheb_coeffs, tt):
    """Add a smooth Chebyshev baseline, rescaled to sit in `[0, amp_frac*y.max()]`.

    The grid is mapped onto Chebyshev's natural `[-1, 1]`, the polynomial is
    evaluated, its minimum is subtracted (so the baseline never removes
    intensity) and it is rescaled to a fixed fraction of the pattern maximum.
    Because of the min-shift and the rescale, `cheb_0` — the constant term —
    washes out entirely; the shape is set by the higher coefficients alone.

    The coefficients are whatever the caller passes. Their ORDER MASK and
    geometric damping, which are what keep this baseline stiffer than any Bragg
    peak, are draw-time properties and belong with the sampler (build step 3,
    `core/transforms.py`), not here. `tests/test_xrd_simulate.py` T2.5 measures
    the resulting stiffness rather than asserting it from this docstring.
    """
    x = 2.0 * (tt - tt[0]) / (tt[-1] - tt[0]) - 1.0
    B = np.polynomial.chebyshev.chebval(x, list(cheb_coeffs))
    B = B - B.min()
    if B.max() > 0:
        B = B / B.max() * (amp_frac * y.max())
    return y + B


def simulate_xrd(
    cif: str,
    *,
    tau_nm: float,
    seed: int,
    u_iso: float = 0.0,
    max_texture: float = 0.0,
    pref_h: int = 1,
    pref_k: int = 1,
    pref_l: int = 1,
    shift_deg: float = 0.0,
    amp_frac: float = 0.0,
    cheb_0: float = 0.0,
    cheb_1: float = 0.0,
    cheb_2: float = 0.0,
    cheb_3: float = 0.0,
    noise_c: float = 0.0,
):
    """(2theta, intensity) for one CIF on `TT_FULL`, at the given params.

    Returns float32 intensity of length `N_FULL` on the full [0, 180] deg grid;
    the caller crops to the encoder window and normalizes.
    """
    angles, intensities, hkls = _sticks(cif, u_iso)
    if len(angles) == 0:
        # No reflection in [0, 180] at this wavelength. Return the FULL grid, not
        # the cropped one — a caller that crops must get the same length either way.
        return TT_FULL.copy(), np.zeros(N_FULL, dtype=np.float32)

    intensities = _apply_texture(intensities, hkls, (pref_h, pref_k, pref_l), max_texture)
    # The 2theta zero-point error is a goniometer misalignment: it moves where the
    # reflection SITS, and the Scherrer width is then evaluated at the angle it
    # actually lands on. Shifting the finished profile instead would be a rigid
    # translation quantized to the grid step, which is a different (and smaller)
    # augmentation — this ordering is why texture and shift cannot be read-time
    # post-ops on a banked pattern (DESIGN.md -> XRD arm, reversing the earlier plan).
    angles = angles + shift_deg
    y = _broaden(angles, intensities, tau_nm, TT_FULL)

    # Rescale to max 100, as the reference does. Purely cosmetic for the encoder
    # (normalization is a read-path step, DESIGN.md -> Signal normalization) and
    # the intensities were never absolute anyway — get_pattern(scaled=True)
    # already dropped that. It is kept because it makes the later `amp_frac` and
    # `noise_c`, both defined as fractions of the running maximum, read as plain
    # percentages of a banked row.
    m = y.max()
    if m > 0:
        y = 100.0 * y / m

    y = _add_background(y, amp_frac, (cheb_0, cheb_1, cheb_2, cheb_3), TT_FULL)

    # Measurement noise, as a fraction of the pattern maximum. The generator is
    # LOCAL and seeded from a param: a backend that drew from global np.random
    # would hand `tools/bank_pdf.py`'s fork pool a realization depending on how
    # many prior tasks each worker happened to pull, making the block
    # irreproducible across a re-run or a different --workers (docs/TRAPS.md).
    # Skipped entirely at noise_c=0 so the clean leg does not pay for a draw it
    # would multiply by zero; the result is identical either way.
    sigma_n = noise_c * y.max()
    if sigma_n > 0:
        y = y + np.random.default_rng(seed).normal(0.0, sigma_n, size=y.shape)

    # Explicit, and unconditional: intensity is non-negative, and saying so here
    # is what lets the read path's max-norm leave the zero where it is.
    return TT_FULL.copy(), np.clip(y, 0.0, None).astype(np.float32)


register("xrd", simulate_xrd)
