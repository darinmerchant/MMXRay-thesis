"""tools/plot_pdf_aug_effect.py — one figure per PDF augmentation, for the paper.

Four standalone PDFs, one per augmentation param (uiso, qmax, qbroad, qmin).
Each sweeps its own param across the PRETRAINING range (`PDF_RANGES`) while the
other three sit at the forward model's nominal values, so a panel shows the
effect of exactly one axis of the augmentation pipeline as it is actually run.

DETERMINISTIC, not seeded. `draw_pdf_params` is bypassed on purpose: a seeded
draw moves all four params at once, which is the right thing for training and
the wrong thing for a figure whose whole claim is "this is what qmax does".

Each panel is plotted over its OWN r-window (`R_WINDOW`), because the four
parameters do not act in the same place: qbroad's damping only becomes obvious
at high r, qmin's baseline distortion only exists at low r.

Vector PDF with `pdf.fonttype=42`, so every label stays editable text in
Illustrator rather than being flattened to outlines. No titles — captions and
panel letters get added at assembly time.

Requires a working diffpy backend:
    /opt/anaconda3/envs/mmxray/bin/python -m tools.plot_pdf_aug_effect
    ... --material-id mp-81 --n-curves 9 --outdir figures/aug_effect
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from core.registry import MaterialRegistry
from core.simulate import simulate
from core.transforms import PDF_RANGES
from modalities.pdf.simulate import UISO, QMAX, QBROAD, QMIN

DEFAULT_REGISTRY = REPO_ROOT / "data" / "registry" / "material_registry.parquet"
DEFAULT_OUT = REPO_ROOT / "figures" / "aug_effect"
# fcc Au, one atom in the cell. Chosen because a sparse PDF is the only kind that
# stays readable with nine curves overlaid at 0-20 A: on TiO2 rutile (mp-2657) the
# peaks are dense enough that the qmax panel reads as a solid mass. Au is also the
# repo's existing downstream arm (data/builders/au_selfdriving.py).
DEFAULT_MATERIAL = "mp-81"

# The values the three unswept params are held at: the forward model's own
# nominals (modalities/pdf/simulate.py), which are mid-range of PDF_RANGES for
# qmax/qbroad and near it for qmin. uiso's 0.005 A^2 is SHARPER than its range
# midpoint 0.0155, and deliberately so — held at the midpoint, thermal broadening
# swamps Q-space termination and the qmax panel shows almost no visible effect.
# The sharper hold costs some legibility in exchange for each panel actually
# demonstrating its own parameter.
NOMINAL = {"uiso": UISO, "qmax": QMAX, "qbroad": QBROAD, "qmin": QMIN}

# Viridis truncated at 0.95 — its lightest yellow is too faint to read on white.
# Built once so the curves and the colorbar cannot map values differently.
CMAP = LinearSegmentedColormap.from_list(
    "viridis95", plt.get_cmap("viridis")(np.linspace(0.0, 0.95, 256))
)

# The r-window each panel is plotted over, PER PARAM — these are not cosmetic
# crops, they are where each parameter's effect actually lives:
#   qbroad  damping grows with r, so [20, 40] shows the effect at full strength
#           where [0, 20] shows its weakest end.
#   qmin    distorts the low-r baseline only; [0, 10] resolves it instead of
#           compressing it into the leftmost eighth of the axis.
#   qmax    acts on peak SHAPE (width, termination ripple), not on an envelope,
#           so it needs a narrow window where a linewidth is resolvable — at
#           [0, 20] the difference is sub-linewidth and the curves superimpose.
# The curves are SLICED to the window, not just xlim'd to it — matplotlib
# autoscales y over everything plotted, so leaving the low-r peaks in the data
# would squash qbroad's window against a y-range set by peaks it never shows.
R_WINDOW = {
    "uiso": (0.0, 20.0),
    "qmax": (0.0, 10.0),
    "qbroad": (20.0, 40.0),
    "qmin": (0.0, 10.0),
}

# Axis label per param. Colorbar labels only — the panels carry no title.
LABELS = {
    "uiso": r"$U_{\mathrm{iso}}$ ($\mathrm{\AA}^2$)",
    "qmax": r"$Q_{\mathrm{max}}$ ($\mathrm{\AA}^{-1}$)",
    "qbroad": r"$Q_{\mathrm{broad}}$ ($\mathrm{\AA}^{-1}$)",
    "qmin": r"$Q_{\mathrm{min}}$ ($\mathrm{\AA}^{-1}$)",
}


def _style() -> None:
    """Large type, editable text in the PDF, no chartjunk."""
    plt.rcParams.update({
        "pdf.fonttype": 42,          # TrueType, not outlines — stays editable in Illustrator
        "font.size": 20,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    })


def sweep(record, param: str, n: int):
    """(values, r, G) for `param` stepped across its PDF_RANGES span.

    Simulated from r=0 and then sliced to `R_WINDOW[param]` rather than simulated
    from the window's floor: G(r) is evaluated pointwise, so the two agree, and
    starting at 0 keeps every panel on the one pretraining grid.
    """
    lo, hi = PDF_RANGES[param]
    r_lo, r_hi = R_WINDOW[param]
    values = np.linspace(lo, hi, n)
    curves = []
    for v in values:
        params = dict(NOMINAL, **{param: float(v)})
        r, g = simulate(record, "pdf", rmax=r_hi, **params)
        curves.append(g)
    keep = r >= r_lo
    return values, r[keep], np.asarray(curves)[:, keep]


def plot_panel(param: str, values, r, curves, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    norm = Normalize(vmin=values[0], vmax=values[-1])
    for v, g in zip(values, curves):
        ax.plot(r, g, color=CMAP(norm(v)), lw=1.6, solid_joinstyle="round")

    ax.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$G(r)$ ($\mathrm{\AA}^{-2}$)")
    ax.set_xlim(*R_WINDOW[param])  # not r[-1]: the grid stops one rstep short, hiding the last tick
    ax.spines[["top", "right"]].set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=CMAP)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(LABELS[param], size=24)
    cbar.ax.tick_params(labelsize=20, width=1.2, size=6)
    cbar.outline.set_linewidth(1.2)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--material-id", default=DEFAULT_MATERIAL)
    # 5, not 9: at nine the ramp reads as a continuous band rather than as
    # distinguishable settings, which defeats the point of a colorbar.
    ap.add_argument("--n-curves", type=int, default=5)
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    _style()
    record = MaterialRegistry.load(Path(args.registry)).get(args.material_id)
    outdir = Path(args.outdir)
    for param in PDF_RANGES:
        values, r, curves = sweep(record, param, args.n_curves)
        out = outdir / f"aug_{param}.pdf"
        plot_panel(param, values, r, curves, out)
        print(
            f"saved {out}  ({param}: {values[0]:.4g} -> {values[-1]:.4g}, "
            f"{args.n_curves} curves, r={R_WINDOW[param]})"
        )


if __name__ == "__main__":
    main()
