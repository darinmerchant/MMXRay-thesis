"""tools/plot_appendix_aug_pca.py — is each PDF augmentation still there, and where?

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      /opt/anaconda3/envs/mmxray/bin/python -m tools.plot_appendix_aug_pca

One 4 x 3 grid on ONE point set: 3,000 Materials Project structures, one banked
augmented view each (`tools/carve_mp_aug3k.py`). ROWS are the four parameters the
PDF augmentation draws — Qmax, Qmin, Qbroad, Uiso; COLUMNS are three representation
states of the same rows:

    1. the raw min-max normalized G(r) — the INPUT-SPACE baseline
    2. the frozen pretrained encoder's `h` — what SSL pretraining kept
    3. an encoder FINETUNED to recover that row's own parameter — what a downstream
       gradient can put back

Read a row left to right and it answers, for one augmentation: is it in the signal
at all, did pretraining discard it, and can finetuning recover it. Read column 2
down and it is the invariance claim itself.

WHY THIS IS NOT tools/plot_appendix_embedding.py. That figure asks the same question
on DOWNSTREAM rows (3,180 augmented CHILI-3K particles) and carries structural
targets beside the instrument ones. This one asks it on the PRETRAINING corpus, at
MP's own crystal-system distribution, with one column per augmentation-recovering
encoder — four encoders that were each asked for exactly the axis their panel is
coloured by, which the CHILI figure's single mo_bond-trained column is not. The two
share an estimator and a ramp, deliberately, so their stamps sit on one footing.

THE 3,000 ROWS ARE STRATIFIED TO MP'S CRYSTAL-SYSTEM DISTRIBUTION, so a colour
gradient in any panel cannot be dismissed as a sampling artefact of an over-cubic
subset (the 45k `material_registry` bank holds 4.2% triclinic against full-MP's
11.3%; this set is carved from the mpfull bank the figure's encoder was pretrained
on). Every row was seen in pretraining — the same "training corpus only" limit
RESULTS.md -> Q4 -> *The theta-probe* states, and the reason no held-out claim is
made here.

ONE JOINT DRAW PER ROW. All four parameters were drawn independently for the same
view (`core.transforms.draw_pdf_params`), so the four rows are literally the same
3,000 points recoloured, and no row can be read against a different point set. The
cost is that this measures each parameter AS DRAWN ALONGSIDE the other three, not a
one-at-a-time sweep: where a panel looks unstructured, `Qmax` especially, the honest
reading is "not linearly readable in this design", never "the axis does nothing".

WHY PCA. The claim is LINEAR readability — every stamp here and every probe number
in RESULTS.md is a linear fit — so the projection matches the claim. Deterministic,
no hyperparameter to defend; centre-only, no whitening, which would re-weight
directions by inverse variance and destroy the geometry the figure is about. UMAP
would test a different claim this figure does not make.

ONE PCA PER PANEL, fit on all 3,000 rows. Columns 1 and 2 have no per-row encoder so
their positions repeat down the column; column 3's encoder changes per row, so its
positions do too.

WHY THE R^2 STAMPS. A scatter shows the leading 2-D subspace of a 256-d space, so
"maybe it lives in PC7" is a fair objection the picture cannot answer. Each panel
carries the FULL-SPACE ridge R^2 for its own (features, parameter): fit on
`split == "train"` (2,100 rows), scored on `split == "test"` (600), lambda selected
by grouped 5-fold CV inside train, `analysis.theta_probe._svd_ridge` imported rather
than re-derived so these numbers and the theta-probe's stay comparable. The picture
is decorative relative to the stamps.

GROUPING IS A NO-OP HERE, AND THAT IS THE POINT. `grouped_folds` is kept because the
protocol is shared with the CHILI figure, but this registry holds ONE view of each of
3,000 DISTINCT materials, so every group is a singleton and grouped CV degenerates to
plain 5-fold. The near-duplicate leakage that makes GCV under-regularize on CHILI
(docs/TRAPS.md -> *GCV under grouped rows*) cannot arise; the folds are honest by
construction rather than by repair.

COLOUR SPANS THE TRAINING RANGE, not the realized min/max: limits come from
`core.transforms.PDF_RANGES`, the uniform bounds the draws were taken from. A
percentile clip (which the CHILI figure needs for its skewed structural targets) would
be wrong for a uniform draw — it would throw away 4% of the range for nothing.

READ ORDER IS LOAD-BEARING: `import torch` precedes pandas and every `core.*` import,
or the process segfaults with no traceback (docs/TRAPS.md).

Runtime ~4 min on the Mac, dominated by the raw column's 2100 x 5000 SVDs.
"""
from __future__ import annotations

import torch  # noqa: F401 — MUST precede pandas/core.*; see docs/TRAPS.md

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis import paperstyle as ps
from analysis.downstream_eval import _single_threaded, embed, load_encoder, load_probe_data
from analysis.probe_cv import grouped_folds
from analysis.theta_probe import _LAMBDAS, _svd_ridge
from core.config import TrainConfig
from core.registry import MATERIAL_ID, SPLIT
from core.train import build_model, pick_device
from core.transforms import PDF_RANGES

REPO = Path(__file__).resolve().parents[1]

REGISTRY = REPO / "data" / "downstream" / "mp_aug3k" / "mp_aug3k.parquet"
#: The carved MP set's one channel — every banked view IS augmented; there is no clean twin.
SIGNAL = "mpaug"
#: The pretraining grid, which this registry is written on exactly (no slice needed).
SIGNAL_LEN = 5000
N_ROWS_EXPECTED = 3000

#: Column 2, and the encoder column 3 finetunes from. The Q1 headline CNN, pretrained
#: on the same mpfull bank these views were carved from.
PRETRAINED_CKPT = (REPO / "runs" / "pdf" / "sweep"
                   / "2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372" / "ckpt_best.pt")
#: `scripts/aug_pca_encoders.slurm`'s output: one finetune per augmentation.
AUG_PCA = REPO / "runs" / "pdf" / "aug_pca"
FT_RUNDIR = "2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372_enc1e-05_head0.0001"

#: (column key, header, is-per-row). Per-row columns get one encoder per augmentation.
COLUMNS = [
    ("raw", "Raw input ν", False),
    ("pretrained", "Pretrained,\nfrozen h", False),
    ("finetuned", "Fine-tuned to\nrecover it", True),
]

#: (registry column, row label, PDF_RANGES key). Order follows the augmentation as it
#: is discussed in the thesis, not `draw_pdf_params`' draw order — the draws are
#: independent, so no ordering claim is made either way.
ROWS = [
    ("aug_qmax", r"$Q_{\max}$ (Å$^{-1}$)", "qmax"),
    ("aug_qmin", r"$Q_{\min}$ (Å$^{-1}$)", "qmin"),
    ("aug_qbroad", r"$Q_{broad}$ (Å$^{-1}$)", "qbroad"),
    ("aug_uiso", r"$U_{iso}$ (Å$^2$)", "uiso"),
]

#: Five is the repo's outer-fold default. See the module docstring on why grouping is
#: a no-op on this registry and is kept anyway.
N_LAMBDA_FOLDS = 5


def load_points():
    """`(X, meta)` — all 3,000 rows in one order, with their realized draws.

    `X` is `(3000, 5000)`, min-max normalized per sample by `load_probe_data`
    exactly as the pretraining read path does. `meta` is the same rows as a
    DataFrame, read separately because `load_probe_data` projects columns; the two
    orders are ASSERTED equal element-for-element rather than assumed, which is the
    whole of the row-alignment argument.
    """
    cols = [c for c, _, _ in ROWS]
    splits = ("train", "val", "test")
    data, _ = load_probe_data(REGISTRY, SIGNAL, cols, SIGNAL_LEN, splits=splits)

    df = pd.read_parquet(REGISTRY, columns=[MATERIAL_ID, SPLIT, "crystal_system", *cols])
    meta = pd.concat([df[df[SPLIT] == s] for s in splits], ignore_index=True)

    X = torch.cat([data[s][0] for s in splits])
    ids = [i for s in splits for i in data[s][2]]
    assert ids == meta[MATERIAL_ID].tolist(), "load_probe_data and the meta read disagree on row order"
    assert len(meta) == N_ROWS_EXPECTED, f"{len(meta)} rows, expected {N_ROWS_EXPECTED}"
    assert meta[MATERIAL_ID].nunique() == len(meta), "a material appears twice — grouping is not a no-op"
    assert torch.isfinite(X).all(), "non-finite value in the normalized signal"
    for col, _, _ in ROWS:
        assert meta[col].notna().all(), f"{col} has nulls; every row carries all four draws"
    return X, meta


def load_finetuned(target: str, device):
    """The encoder half of one `--save-weights` file, frozen.

    A near-twin of `plot_appendix_embedding.load_saved`, kept local for its error
    message alone: that one points at `scripts/embed_fig_encoders.slurm`, and a
    reader who follows it here would rerun the wrong job. The head is discarded —
    the figure is about `h`, and every stamp fits its own fresh readout.
    """
    path = AUG_PCA / target / FT_RUNDIR / f"finetune_{target}_clean.pt"
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO)} — run `sbatch --array=0-3 "
            f"scripts/aug_pca_encoders.slurm` on SuperCloud and rsync runs/pdf/aug_pca/ back"
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    names = {f.name for f in fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items() if k in names})
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def features(X, device, batch_size):
    """`{(column key, target or None): (N, d) float64}` — all six feature spaces.

    EVERY torch forward pass happens here, before any linear algebra below: a
    numpy/LAPACK call followed by a torch BatchNorm forward deadlocks this env's
    three OpenMP runtimes at 0% CPU with no message (docs/TRAPS.md), and encoding up
    front means that interleaving never arises.
    """
    out = {("raw", None): X.double()}
    pretrained, _, _ = load_encoder(PRETRAINED_CKPT, device)
    out[("pretrained", None)] = embed(pretrained, X, device, batch_size).double()
    print("  embedded pretrained (shared)", flush=True)
    for target, _, _ in ROWS:
        out[("finetuned", target)] = embed(
            load_finetuned(target, device), X, device, batch_size).double()
        print(f"  embedded the {target} finetune", flush=True)
    return out


def pca2(H):
    """`(scores (N, 2), (var1, var2))` — centre, SVD, keep two components."""
    with _single_threaded():
        U, S, _ = torch.linalg.svd(H - H.mean(0, keepdim=True), full_matrices=False)
    var = (S**2) / (S**2).sum()
    return (U[:, :2] * S[:2]).numpy(), (float(var[0]), float(var[1]))


def ridge_r2(H, meta, target):
    """Full-space test R², ridge, λ by grouped 5-fold CV inside train — never GCV.

    One SVD per fold serves the whole λ grid; the grid is scaled once by the full
    train design's mean squared singular value so it stays dimensionless. Identical
    arithmetic to `plot_appendix_embedding.ridge_r2`, on `material_id` groups.
    """
    train = (meta[SPLIT] == "train").to_numpy()
    test = (meta[SPLIT] == "test").to_numpy()
    Htr, Hte = H[train], H[test]
    groups = meta.loc[train, MATERIAL_ID].to_numpy()
    y = torch.from_numpy(meta[target].to_numpy(dtype=np.float64).copy())

    with _single_threaded():
        predict_test, S = _svd_ridge(Htr, Hte)
        scale = float((S**2).mean())
        sse = np.zeros(len(_LAMBDAS))
        for held in grouped_folds(groups, N_LAMBDA_FOLDS, seed=0):
            keep = np.setdiff1d(np.arange(len(Htr)), held)
            predict, _ = _svd_ridge(Htr[keep], Htr[held])
            yt = y[train]
            for j, lam in enumerate(_LAMBDAS):
                sse[j] += float(((yt[held] - predict(yt[keep], float(lam) * scale)) ** 2).sum())
        j = int(np.argmin(sse))
        # An argmin at either end means the CV wanted a λ outside the grid, so the
        # "selection" is a boundary rather than an optimum. A high edge is the right
        # answer for a variable with no signal (shrink to the intercept); a low edge
        # is not, and is worth looking at. Either way it is printed, never silent.
        if j in (0, len(_LAMBDAS) - 1):
            print(f"    note: {target} λ at grid {'low' if j == 0 else 'high'} edge "
                  f"({float(_LAMBDAS[j]):.3g}× mean(s²))", flush=True)
        pred = predict_test(y[train], float(_LAMBDAS[j]) * scale)
        truth = y[test]
        ss_res = float(((truth - pred) ** 2).sum())
        ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def build(meta, feats, out_stem, sub):
    ps.use()

    panels = {}
    for target, _, _ in ROWS:
        for key, _, per_row in COLUMNS:
            H = feats[(key, target if per_row else None)]
            panels[(target, key)] = (pca2(H), ridge_r2(H, meta, target))
        print(f"  projected + stamped row {target}", flush=True)

    # Margins on the gridspec, NOT tight_layout, which recomputes hspace/wspace and
    # would undo the tight packing this grid needs.
    fig = plt.figure(figsize=(ps.WIDTH, 6.2))
    gs = fig.add_gridspec(len(ROWS), len(COLUMNS) + 1,
                          width_ratios=[1] * len(COLUMNS) + [0.05],
                          # Same colourbar-label rule as plot_appendix_embedding.
                          left=0.075, right=0.945, top=0.855, bottom=0.025,
                          hspace=0.11, wspace=0.07)

    for i, (target, row_label, range_key) in enumerate(ROWS):
        values = meta[target].to_numpy(dtype=float)
        lo, hi = PDF_RANGES[range_key]
        norm = plt.Normalize(lo, hi)
        for j, (key, header, _) in enumerate(COLUMNS):
            ax = fig.add_subplot(gs[i, j])
            (scores, (v1, v2)), r2 = panels[(target, key)]
            sc = ax.scatter(scores[:, 0], scores[:, 1], c=values, cmap=ps.MARKS, norm=norm,
                            s=1.0, linewidths=0, alpha=0.85, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(True)
                side.set_color(ps.GRID)
            ax.text(0.05, 0.95, f"$R^2$ {r2:.2f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=5.5, color=ps.INK,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                              boxstyle="square,pad=0.12"))
            # Variance explained sits INSIDE the panel: in a per-row column it is a
            # per-panel property, not a column header.
            ax.text(0.05, 0.04, f"{100 * v1:.0f}/{100 * v2:.0f}%", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=4.8, color=ps.MUTED)
            if i == 0:
                ax.set_title(header, pad=3, fontsize=7)
            if j == 0:
                ax.set_ylabel(row_label, fontsize=7, color=ps.INK)
        cax = fig.add_subplot(gs[i, -1])
        cb = fig.colorbar(sc, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=5, length=2)
        cb.set_ticks([lo, hi])
        cb.ax.set_yticklabels([f"{lo:g}", f"{hi:g}"])

    fig.suptitle("PC1 vs PC2 of the same 3,000 Materials Project views — three "
                 "representation states,\nfour augmentation parameters", fontsize=8, y=0.988)
    # y sits BELOW the two-line suptitle and ABOVE the two-line column headers, which
    # the gridspec's own `top` fixes at ~0.895. Both are two lines; neither is optional.
    fig.text(0.115, 0.918,
             "column 3 is a different encoder in every row: the one finetuned to "
             "recover that row's own parameter",
             fontsize=5.5, color=ps.MUTED, ha="left")
    ps.save(fig, out_stem, sub=sub)
    plt.close(fig)
    return panels


CAPTION = """\
PC1 vs PC2 of three representation states of the SAME 3,000 Materials Project
structures — one simulated PDF view each, drawn from the 32-view mpfull bank the
encoder was pretrained on, sampled to match MP's own crystal-system distribution —
recoloured by the four parameters the augmentation draws.

Column 1 is the raw min-max normalized G(r) the encoder reads; column 2 is the frozen
pretrained CNN's 256-d h; column 3 is that same encoder after a standard finetune
(enc 1e-5 / head 1e-4, 300 epochs) trained to REGRESS that row's own parameter, so it
is a different encoder in every row and its point positions change accordingly.
Columns 1 and 2 have no target and repeat down the figure. One PCA per panel,
center-only; the small percentages are PC1/PC2 variance explained for that panel.

The corner number is NOT a property of the projection — it is the FULL-SPACE ridge
R^2 for that panel's (features, parameter), fit on the 2,100 train rows and scored on
the 600 test rows, with lambda selected by grouped 5-fold CV inside train. Color
limits are the uniform draw bounds the augmentation samples from, not the realized
extremes.

All four parameters were drawn jointly for one view per structure, so each panel
measures a parameter as it appears alongside the other three; an unstructured panel
means "not linearly readable in this design", not "this axis does nothing to the
signal". Every row was part of pretraining, so column 2 is a training-corpus
measurement, and column 3's stamps are what a standard downstream gradient recovers,
not an upper bound on recoverability.\
"""


def report(panels):
    print(f"\nFull-space ridge R², fit on train (2,100), scored on test (600), λ by "
          f"grouped {N_LAMBDA_FOLDS}-fold CV inside train.\n")
    head = " ".join(f"{h.replace(chr(10), ' '):>22}" for _, h, _ in COLUMNS)
    print(f"{'parameter':<12} {head}")
    for target, _, _ in ROWS:
        cells = " ".join(f"{panels[(target, k)][1]:>22.3f}" for k, _, _ in COLUMNS)
        print(f"{target:<12} {cells}")
    print("\nPC1/PC2 variance explained")
    for target, _, _ in ROWS:
        cells = " ".join(f"{100 * panels[(target, k)][0][1][0]:>8.1f}/"
                         f"{100 * panels[(target, k)][0][1][1]:<5.1f}" for k, _, _ in COLUMNS)
        print(f"  {target:<12} {cells}")
    print("\ncaption\n-------\n" + CAPTION)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help='"auto" | "cpu" | "cuda"')
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="appendix_aug_pca")
    ap.add_argument("--sub", default=None, help="subdirectory under figures/")
    args = ap.parse_args()

    device = pick_device(args.device)
    X, meta = load_points()
    print(f"{len(meta)} rows of signal_{SIGNAL} on {device}; "
          f"splits {meta[SPLIT].value_counts().to_dict()}, "
          f"{meta['crystal_system'].nunique()} crystal systems", flush=True)
    feats = features(X, device, args.batch_size)
    report(build(meta, feats, args.out, args.sub))


if __name__ == "__main__":
    main()
