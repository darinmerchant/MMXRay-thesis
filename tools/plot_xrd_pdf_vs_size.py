"""tools/plot_xrd_pdf_vs_size.py — one crystal at five carved diameters, XRD beside PDF.

    /opt/anaconda3/envs/mmxray/bin/python -m tools.plot_xrd_pdf_vs_size

The size-breakdown figure: the SAME TiO2 crystal carved at the five diameters
CHILI-100K ships (53.6 down to 12.3 A), each particle pushed through both Debye
forward models. Down the XRD column the Bragg peaks broaden into each other until
the smallest particle is a pair of diffuse humps; down the PDF column the low-r
peaks stay sharp at every size and only the envelope contracts to the diameter.
That asymmetry is the figure's one point — the reader should be able to say which
diameter each PDF came from after the XRD column has stopped telling them.

ONE MATERIAL, CHILI-100K's OWN CARVINGS. COD 1511015 (TiO2, 71-6,861 atoms) —
a compound a reader knows, with well-separated reflections at the large end. The
five particles come straight from the raw .h5 (`DiscreteParticleGraphs`, the same
groups `tools/augment_chili.py` reads), so every size is the identical structure
cut at a different radius — nothing about chemistry, lattice, or carving protocol
varies down a column. The material is deliberately NOT named on the figure; the
caption may name it.

BOTH TRACES ARE THE CLEAN CHANNEL AT NOMINAL INSTRUMENT SETTINGS. PDF at the
pretraining simulator's own nominals (uiso 0.005, qmax 22.5, qbroad 0.03,
qmin 1.0); XRD at the same u_iso with zero shift, zero background, zero noise.
Randomized instrument draws would add trace-to-trace variation that reads as part
of the size story; here size must be the only thing that changes.

NO SCHERRER TERM ANYWHERE, and that is the point of using the Debye route: the
atoms already carry the particle size, so the XRD broadening shown IS the physics
of a finite particle, not a width parameter chosen to illustrate it (DESIGN.md ->
XRD arm; `modalities/xrd/debye.py`).

EACH TRACE IS SCALED TO ITS OWN MAXIMUM, XRD by max I (its floor is the diffuse
background, part of the story), G(r) by max |G| so the zero baseline survives.
Absolute amplitude grows with atom count in both channels and would compress the
small-particle rows to invisibility — the repo drops it at the encoder input for
the same reason (`core.transforms.minmax_normalize`). The caption must say
"individually scaled". On top of that per-trace scale the G(r) column takes ONE
global compression, set by the deepest trough of any row, so a trace never
crosses into the row below and the zero line sits at the same height in every
row; a per-row compression instead would silently re-vary the amplitudes the
per-trace scale just equalized.

COLOR IS VIRIDIS over the full ramp, purple for the largest particle down to
yellow for the smallest. Diameter is also the y tick label, so color is
reinforcement, never the only carrier.

THE DEBYE SUMS ARE CACHED (`figures/xrd_pdf_vs_size_sim.npz`, ~35 s to build the
largest particle's XRD): delete the file to re-simulate. The cache keys include
nothing drawn at random, so a rebuild is byte-stable.

`import torch` first — importing pandas or `core.*` ahead of it segfaults with no
traceback (docs/ENVIRONMENT.md).
"""

from __future__ import annotations

import torch  # noqa: F401  MUST precede core.* / pandas — see module docstring

import h5py
import numpy as np
from ase.data import chemical_symbols

from analysis import paperstyle as ps
from modalities.pdf.debye import simulate_pdf_debye
from modalities.pdf.simulate import QBROAD, QMAX, QMIN, UISO
from modalities.xrd.debye import simulate_xrd_debye

COD = "1511015"  # TiO2
H5 = ps.REPO / "data" / "raw" / "chili100k" / "cod_output_080124" / f"{COD}.h5"
CACHE = ps.FIGURES / "xrd_pdf_vs_size_sim.npz"

#: The XRD panel starts here, not at TT_GRID's 10 deg. A finite particle scatters
#: strongly toward the beam (the small-angle regime), and the tail of that signal
#: still rises steeply over 10-14 deg — at 12 A it is the pattern's global maximum,
#: so normalizing the full window scales the Bragg humps down by ~3x and the reader's
#: eye lands on a spike the figure is not about. TiO2's first reflection is at
#: 27.4 deg; nothing structural lives below 15. Both the crop and the per-trace
#: maximum are taken over this window.
TT_MIN_DEG = 15.0


def _simulate() -> dict[str, np.ndarray]:
    """All ten Debye sums for the five carved sizes, smallest first."""
    from data.builders.chili100k import _sizes

    out: dict[str, np.ndarray] = {}
    with h5py.File(H5, "r") as h5:
        sizes = _sizes(h5)
        out["diameters"] = np.array([float(s.rstrip("Å")) for s in sizes])
        for i, s in enumerate(sizes):
            g = h5[f"DiscreteParticleGraphs/{s}"]
            z = np.asarray(g["NodeFeatures"][()])[:, 0].round().astype(int)
            elements = [chemical_symbols[int(v)] for v in z]
            coords = np.asarray(g["AbsoluteCoordinates"][()], dtype=float)
            print(f"simulating {s} ({len(elements)} atoms)", flush=True)
            out["tt"], out[f"xrd_{i}"] = simulate_xrd_debye(
                elements, coords, u_iso=UISO, shift_deg=0.0, amp_frac=0.0,
                cheb_0=0.0, cheb_1=0.0, cheb_2=0.0, cheb_3=0.0, noise_c=0.0, seed=0)
            out["r"], out[f"pdf_{i}"] = simulate_pdf_debye(
                elements, coords, uiso=UISO, qmax=QMAX, qbroad=QBROAD, qmin=QMIN)
    return out


def main() -> None:
    if CACHE.exists():
        d = dict(np.load(CACHE))
    else:
        d = _simulate()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, **d)
        print(f"cached {CACHE.relative_to(ps.REPO)}")

    diameters = d["diameters"]
    n = len(diameters)
    order = np.argsort(diameters)[::-1]  # largest first = top row

    ps.use()
    import matplotlib.pyplot as plt

    fig, (ax_x, ax_p) = plt.subplots(1, 2, figsize=(ps.WIDTH, 3.4))

    # ONE row spacing for both panels, so row i sits at the same height left and
    # right. XRD traces span [0, 1] of a band. G(r) swings negative, so every row
    # gets one global compression k, fixed by the deepest trough anywhere: the
    # zero line then sits at the same height in every band, no trace leaves its
    # band, and peak tops land exactly where the XRD tops do.
    keep = d["tt"] >= TT_MIN_DEG
    tt = d["tt"][keep]
    xrd = [d[f"xrd_{i}"][keep] / d[f"xrd_{i}"][keep].max() for i in order]
    pdf = [d[f"pdf_{i}"] / np.abs(d[f"pdf_{i}"]).max() for i in order]
    s = 1.12
    gmin = min(g.min() for g in pdf)
    k = 1.0 / (1.0 - gmin)
    z0 = -k * gmin  # height of G = 0 above a band's floor

    # Viridis, purple for the largest particle down to yellow for the smallest.
    colors = [plt.cm.viridis(t) for t in np.linspace(0.0, 1.0, n)]

    for row, (y, g, c) in enumerate(zip(xrd, pdf, colors)):
        base = (n - 1 - row) * s
        ax_x.plot(tt, y + base, color=c, linewidth=1.0)
        ax_p.plot(d["r"], k * g + z0 + base, color=c, linewidth=0.9)

    labels = [f"{diameters[i]:.0f} Å" for i in order]
    ax_x.set_yticks([(n - 1 - row) * s + 0.45 for row in range(n)], labels)
    ax_x.tick_params(axis="y", length=0)
    ax_p.set_yticks([])
    ax_x.set_ylabel("nanoparticle diameter")

    ax_x.set_title("(a) X-ray diffraction")
    ax_x.set_xlabel(r"2$\theta$ (deg)")
    ax_x.set_xlim(float(tt[0]), float(tt[-1]))
    ax_p.set_title("(b) Pair distribution function")
    ax_p.set_xlabel("r (Å)")
    ax_p.set_xlim(float(d["r"][0]), float(d["r"][-1]))
    for ax in (ax_x, ax_p):
        ax.set_ylim(-0.03, (n - 1) * s + 1.06)
    ax_p.spines["left"].set_visible(False)

    ps.layout(fig)
    ps.save(fig, "xrd_pdf_vs_size")


if __name__ == "__main__":
    main()
