"""tools/plot_appendix_aug_effect.py — appendix A.1: one instrument parameter varied alone.

    /opt/anaconda3/envs/mmxray/bin/python -m tools.plot_appendix_aug_effect

A 2x2 grid, one panel per augmentation parameter in the draw order the paper
states, (Uiso, Qmax, Qbroad, Qmin). Each panel sweeps its own parameter across
the PRETRAINING range while the other three sit at the forward model's nominal
values, on the r-window where that parameter's effect actually lives.

WHY THIS EXISTS. These four panels were the insets of the paper's Figure 1. The
AI4Mat version drops the insets from Figure 1 to fit the page limit, so they move
to the appendix as one figure, in paperstyle rather than the 24-pt Illustrator
style of `tools/plot_pdf_aug_effect.py`.

WHAT IS REUSED. The sweep itself — `sweep`, `R_WINDOW`, `LABELS`, `NOMINAL`,
`CMAP`, `DEFAULT_MATERIAL` — is imported from `tools/plot_pdf_aug_effect.py`, so
the appendix figure and the Figure 1 insets can never show different physics.
This module only lays out panels. Everything that module's docstring says about
the material choice (fcc Au, mp-81), the r-windows and the sharper-than-midpoint
Uiso hold applies here unchanged.

IMPORT ORDER IS LOAD-BEARING: `import torch` precedes every `core.*` import
(`docs/TRAPS.md` -> the import-order bug).
"""
from __future__ import annotations

import torch  # noqa: F401 — MUST precede pandas/core.*; see docs/TRAPS.md

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from analysis import paperstyle as ps
from core.registry import MaterialRegistry
from tools.plot_pdf_aug_effect import (
    CMAP, DEFAULT_MATERIAL, DEFAULT_REGISTRY, LABELS, R_WINDOW, sweep,
)

#: The paper's draw order (Appendix A.1), read left-to-right, top-to-bottom.
PARAM_ORDER = ("uiso", "qmax", "qbroad", "qmin")


def build(record, n_curves: int, out_stem: str, sub: str | None) -> None:
    ps.use()
    fig, axes = plt.subplots(2, 2, figsize=(ps.WIDTH, 3.6))
    for ax, param in zip(axes.ravel(), PARAM_ORDER):
        values, r, curves = sweep(record, param, n_curves)
        norm = Normalize(vmin=values[0], vmax=values[-1])
        for v, g in zip(values, curves):
            ax.plot(r, g, color=CMAP(norm(v)), lw=1.0, solid_joinstyle="round")
        ax.set_xlim(*R_WINDOW[param])
        ax.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), ax=ax, pad=0.02)
        cbar.set_label(LABELS[param])
        cbar.outline.set_linewidth(0.8)
        print(f"{param}: {values[0]:.4g} -> {values[-1]:.4g}, {n_curves} curves, r={R_WINDOW[param]}")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$G(r)$ ($\mathrm{\AA}^{-2}$)")
    ps.layout(fig)
    ps.save(fig, out_stem, sub=sub)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--material-id", default=DEFAULT_MATERIAL)
    ap.add_argument("--n-curves", type=int, default=5)
    ap.add_argument("--out", default="appendix_aug_effect")
    ap.add_argument("--sub", default="appendix")
    args = ap.parse_args()
    record = MaterialRegistry.load(Path(args.registry)).get(args.material_id)
    build(record, args.n_curves, args.out, args.sub)


if __name__ == "__main__":
    main()
