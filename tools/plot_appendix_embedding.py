"""tools/plot_appendix_embedding.py — thesis appendix: what each representation state organizes.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      /opt/anaconda3/envs/mmxray/bin/python -m tools.plot_appendix_embedding

A 4 x 6 grid over ONE point set: all 3,180 rows of the augmented CHILI-3K channel
(`signal_xpdf_aug_*`), which is the evaluation channel every domain-shift number in
this repo is scored on. ROWS are the four downstream targets; COLUMNS are six
representation states of those same rows:

    1  raw input nu                    no encoder
    2  supervised from scratch, fit on the CLEAN channel
    3  supervised from scratch, fit on the AUGMENTED channel
    4  pretrained encoder, FROZEN      no target, no fit
    5  fine-tuned, fit on the CLEAN channel
    6  fine-tuned, fit on the AUGMENTED channel

**COLUMNS 2, 3, 5 AND 6 ARE PER-TARGET**: each of those 16 panels shows an encoder
trained on its OWN row's target, not one encoder reused down the column
(`scripts/embed_fig_encoders.slurm`, 8 array cells x 2 fit channels). That is the
whole point of the layout — it asks what a representation trained FOR a target
organizes, against what pretraining organizes without being told any target.

WHAT REPEATS AND WHAT DOES NOT. Columns 1 and 4 have no target, so their projection
is identical down all four rows and only the colouring changes. In the per-target
columns the ENCODER changes with the row, so the point positions change too: the
3,180 *samples* are the same everywhere, their *coordinates* are not. Do not read a
shape as persisting across a per-target column.

ONE PCA PER PANEL for the per-target columns, one per column for the shared two.
Centre-only, no per-dimension scaling — whitening would re-weight directions by
inverse variance, which is the geometry being looked at. PC1/PC2 variance explained
is printed inside each panel rather than in the column header, precisely because it
is no longer a column-level property.

WHY PCA AND NOT UMAP. The claim is *linear* readability — every probe number in this
repo is a linear readout — so the projection has to be linear too. PCA is
deterministic and has no hyperparameter to defend. UMAP would test a different,
nonlinear claim that nothing here makes.

THE STAMPS, AND WHY THEY ARE TWO DIFFERENT METRICS. Each panel carries the
FULL-SPACE score for its own (features, target) — the scatter shows a 2-D shadow, so
"maybe it lives in PC7" is a fair objection the picture cannot answer, and the stamp
is what answers it. The metric follows the target's kind, matching how the rest of
the repo scores these four:

  np_size, mo_bond   REGRESSION  -> R^2, ridge, lambda by `group_key`-grouped 5-fold
                                   CV inside train (never GCV: docs/TRAPS.md ->
                                   *GCV under grouped rows*). Ridge imported from
                                   `analysis.theta_probe`, not re-derived.
  cn, oxidation      CLASSIFICATION -> weighted F1, via `downstream_eval`'s OWN
                                   `fit_classifier`/`score_classifier`, so the number
                                   is directly comparable to `probe.json` and Q1.

Both are fit on `split == "train"` and scored on `split == "test"`; `val` is never
touched. **The two metrics are not comparable to each other** — read down a column
within a row's own metric, never across the R^2/F1 boundary. Note also that the
logistic head's `C` is NOT tuned (sklearn's default, as everywhere else here), while
the ridge lambda IS selected; that asymmetry is inherited from the repo's protocols,
not introduced here.

⚠️ **THE COLUMNS ARE NOT AN ARM COMPARISON.** The supervised arm runs at
lr_enc = lr_head = 1e-3 and the fine-tuned arm at 1e-5 / 1e-4 — each at the LR its
own RESULTS.md-quoted counterpart uses, which is what makes each column comparable to
the literature beside it and precisely NOT to the other column.
`docs/TRAPS.md` -> *A "from-scratch control" that does not share the backbone LR is
not a control* is about exactly this. Every panel is read against **column 4**, the
frozen encoder, which has no LR and no target at all. Additionally the supervised
1e-3 was val-selected on `np_size` alone and is applied unre-selected to the other
three, which biases that arm down by an unknown amount on those rows.

`target_cn` IS NaN ON CHILI'S 265 SPINEL ROWS (the metal sites disagree). Those rows
are drawn in grey and excluded from that row's fit and score — so its two encoders
saw 2,915 rows, while all 3,180 are still embedded and plotted. Every other row is
fully labelled.

WHY AUGMENTED CHILI-3K. It is the one dataset carrying the realized instrument draw
and the structural labels on the SAME rows, and it is the channel the shift results
are scored on. The augmentation parameters were drawn independently of structure.

READ ORDER IS LOAD-BEARING, twice. `load_probe_data` slices to the pretraining grid
and THEN min-max normalizes; the other order would rescale by extrema in the 50-60 A
tail the encoder never reads. And `import torch` precedes pandas and every `core.*`
import, or the process segfaults with no traceback (docs/TRAPS.md).

FIGURE SIZE IS A DELIBERATE DEPARTURE from `paperstyle.WIDTH`. Six columns at 5.5 in
leaves 0.72 in per panel, which cannot show structure in 3,180 points. This is sized
for a LANDSCAPE full-page thesis float at 7.2 in and must be placed at that width —
scaling it in \\includegraphics rescales the type and breaks paperstyle's match to
body text, which is the one thing its type sizes cannot survive.

`--columns` DRAWS A SUBSET OF THE SIX. `--columns raw,pretrained,ft_clean` is the
thesis's three-state cut: the input, what pretraining organizes with no target, and
what fine-tuning on the clean channel does to it. In subset mode the corner
full-space stamp, the two title lines and the stamps JSON are all dropped — the 24
stamps already live in `analysis/out/appendix_embedding_pca.json` and RESULTS.md's
rule is that a measured number lives in exactly one place, so the subset must not
mint a second copy of twelve of them. Dropping the stamp is also what makes it
cheap: no ridge CV, no logistic fits, seconds off the feature cache rather than
~45 min. The reading context the dropped subtitle carried moves into `CAPTION_SUBSET`.

Runtime ~45 min on the Mac: 17 encoder passes over 3,180 signals, then the raw
column's 2530 x 5000 SVDs.
"""
from __future__ import annotations

import torch  # noqa: F401 — MUST precede pandas/core.*; see docs/TRAPS.md

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

from analysis import paperstyle as ps
from analysis.downstream_eval import (
    _single_threaded, embed, fit_classifier, load_encoder, load_probe_data,
    score_classifier,
)
from analysis.probe_cv import grouped_folds
from analysis.theta_probe import _LAMBDAS, _svd_ridge
from core.config import TrainConfig
from core.registry import MATERIAL_ID, SPLIT
from core.train import build_model, pick_device

REPO = Path(__file__).resolve().parents[1]

REGISTRY = REPO / "data" / "downstream" / "chili" / "chili_registry.parquet"
SIGNAL = "xpdf_aug"
SIGNAL_LEN = 5000
N_ROWS_EXPECTED = 3180

PRETRAINED_CKPT = (REPO / "runs" / "pdf" / "sweep"
                   / "2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372" / "ckpt_best.pt")
EMBED_FIG = REPO / "runs" / "pdf" / "embed_fig"
#: The two run-dir stems `scripts/embed_fig_encoders.slurm` produces, per target.
ARM_DIR = {
    "sup": "cnn_infonce_mpfull_final_lr0.001_seed42_enc0.001_head0.001",
    "ft": "2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372_enc1e-05_head0.0001",
}

#: (column key, header, source). `source` None means the column has no target and is
#: therefore SHARED down every row; otherwise `(arm, fit channel)` and the panel's
#: encoder is the one trained on that row's own target.
COLUMNS = [
    ("raw", "Raw input ν", None),
    ("sup_clean", "Supervised\nfit clean", ("sup", "clean")),
    ("sup_aug", "Supervised\nfit aug.", ("sup", "aug")),
    ("pretrained", "Pretrained,\nfrozen h", None),
    ("ft_clean", "Fine-tuned\nfit clean", ("ft", "clean")),
    ("ft_aug", "Fine-tuned\nfit aug.", ("ft", "aug")),
]

#: Column headers the SUBSET cut overrides. Two DROP a qualifier the six-column grid
#: needs and the cut does not: with `ft_aug` not drawn beside it there is nothing for
#: "fit clean" to disambiguate, and with no other pretrained column "frozen h" is
#: distinguishing the column from nothing. Both facts move into `CAPTION_SUBSET`.
#: `raw` goes the other way and ADDS its argument, spelling ν(G(r)) the way
#: `tools/plot_appendix_views.py` labels the same quantity — the wider panels have
#: room for it, and it names what the encoder actually reads rather than a bare ν.
#: The six-column grid keeps the full headers, where the distinctions are the point.
SUBSET_HEADER = {"raw": "Raw input ν(G(r))", "pretrained": "Pretrained",
                 "ft_clean": "Fine-tuned"}

#: (registry column, row label, kind). `kind` picks the stamp metric AND the colour
#: treatment: "reg" is a continuous ramp + R^2, "cls" a discrete one + weighted F1.
TARGETS = [
    ("target_np_size", "particle size (Å)", "reg"),
    ("target_mo_bond", "M–O bond (Å)", "reg"),
    ("target_cn", "coordination no.", "cls"),
    ("target_oxidation", "oxidation state", "cls"),
]

#: The scatter-mark ramp, and why it is not `paperstyle.SEQUENTIAL`, now live in
#: `paperstyle.MARKS` — promoted there when tools/plot_appendix_aug_pca.py needed
#: the same ramp. Aliased rather than renamed at ~20 use sites below.
POINT_CMAP = ps.MARKS

#: Rows with no label for this row's target (only `target_cn`, 265 Spinel rows). Drawn
#: rather than dropped, so the point set stays literally the same in every panel, and
#: in a neutral grey that is not on the ramp so it cannot be misread as a value.
UNLABELLED = "#c9c8c3"

#: Grouped-CV folds for the ridge lambda. Five is the repo's outer-fold default; with
#: 265 groups over 2530 train rows each fold holds ~53 groups.
N_LAMBDA_FOLDS = 5
#: Robust colour limits for the continuous rows — a few outliers on `mo_bond` (max
#: 3.32 Å against a 2.29 Å upper quartile) would compress everything else into one end.
CLIP_PCT = (2.0, 98.0)


def load_points():
    """`(X, meta)` — all 3,180 augmented rows in one order, with their labels.

    `X` is `(3180, 5000)`, sliced to the pretraining grid and THEN min-max
    normalized, straight out of `analysis/downstream_eval.load_probe_data`.

    `meta` is the same rows as a DataFrame, read separately because
    `load_probe_data` projects columns and does not carry `group_key`. The two orders
    are not assumed to agree: this rebuilds the same `split`-block concatenation and
    ASSERTS the material ids match element-for-element, which is the whole of the
    row-alignment argument.
    """
    cols = [c for c, _, _ in TARGETS]
    splits = ("train", "val", "test")
    data, _ = load_probe_data(REGISTRY, SIGNAL, cols, SIGNAL_LEN, splits=splits)

    df = pd.read_parquet(REGISTRY, columns=[MATERIAL_ID, SPLIT, "group_key", *cols])
    meta = pd.concat([df[df[SPLIT] == s] for s in splits], ignore_index=True)

    X = torch.cat([data[s][0] for s in splits])
    ids = [i for s in splits for i in data[s][2]]
    assert ids == meta[MATERIAL_ID].tolist(), "load_probe_data and the meta read disagree on row order"
    assert len(meta) == N_ROWS_EXPECTED, f"{len(meta)} rows, expected {N_ROWS_EXPECTED}"
    assert torch.isfinite(X).all(), "non-finite value in the normalized signal"
    # `target_cn` is legitimately NaN on 265 rows; every other target must be complete.
    for col, _, _ in TARGETS:
        n_nan = int(meta[col].isna().sum())
        if col == "target_cn":
            assert n_nan == 265, f"target_cn has {n_nan} nulls, expected CHILI's 265 Spinel rows"
        else:
            assert n_nan == 0, f"{col} has {n_nan} nulls, expected none"
    return X, meta


def load_saved(path: Path, device):
    """The encoder half of a `--save-weights` file, as a frozen model.

    `analysis/finetune.py` writes `model_state` (the full pretraining model, whose
    `.encode()` is `h`) beside a separate `head_state`. The head is discarded: the
    figure is about `h`, and every stamp fits its own fresh readout anyway.
    """
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO)} — run scripts/embed_fig_encoders.slurm "
            f"and rsync runs/pdf/embed_fig/ back"
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


def weight_path(target: str, arm: str, channel: str) -> Path:
    return EMBED_FIG / target / ARM_DIR[arm] / f"finetune_{target}_{channel}.pt"


def cache_key(col: str, target) -> str:
    """`"<column>|<target or empty>"` — npz keys must be strings, tuples are not."""
    return f"{col}|{target or ''}"


def features(X, device, batch_size, cache: Path | None = None):
    """`{(column key, target or None): (N, d) float64}` — every feature space.

    Shared columns are keyed on `None` and computed once; per-target columns are
    keyed on their target. 17 encoder passes in all.

    EVERY torch forward pass happens in this function, before any linear algebra
    below. A numpy/LAPACK call followed by a torch BatchNorm forward deadlocks this
    env's three OpenMP runtimes at 0% CPU with no message (docs/TRAPS.md); doing all
    the encoding up front means that interleaving never arises.
    """
    if cache is not None and cache.exists():
        # The cache exists because UMAP CANNOT RUN IN THIS INTERPRETER: umap-learn
        # needs sklearn >= 1.6 and this env is pinned to 1.5.2 for the numpy-1 ABI
        # (docs/ENVIRONMENT.md), so the projection step lives in a second env and the
        # features have to cross on disk. Re-deriving them costs ~40 min of encoder
        # passes; that is the whole reason this is not recomputed per projection.
        # float32 is LOSSLESS here — `embed` returns float32 and the raw channel is
        # float32; the .double() below is an upcast for the linear algebra only.
        z = np.load(cache)
        out = {}
        for k in z.files:
            col, target = k.split("|", 1)
            out[(col, target or None)] = torch.from_numpy(z[k]).double()
        print(f"  loaded {len(out)} feature spaces from {cache.name}", flush=True)
        return out

    out = {("raw", None): X.double()}
    pretrained, _, _ = load_encoder(PRETRAINED_CKPT, device)
    out[("pretrained", None)] = embed(pretrained, X, device, batch_size).double()
    print("  embedded pretrained (shared)", flush=True)
    for target, _, _ in TARGETS:
        for key, _, src in COLUMNS:
            if src is None:
                continue
            model = load_saved(weight_path(target, *src), device)
            out[(key, target)] = embed(model, X, device, batch_size).double()
        print(f"  embedded 4 per-target encoders for {target}", flush=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{cache_key(c, t): v.float().numpy() for (c, t), v in out.items()})
        print(f"  wrote {cache}", flush=True)
    return out


def pca2(H):
    """`(scores (N, 2), (var1, var2))` — centre, SVD, keep two components."""
    with _single_threaded():
        U, S, _ = torch.linalg.svd(H - H.mean(0, keepdim=True), full_matrices=False)
    var = (S**2) / (S**2).sum()
    return (U[:, :2] * S[:2]).numpy(), (float(var[0]), float(var[1]))


def _split_masks(meta, labelled):
    train = ((meta[SPLIT] == "train").to_numpy() & labelled).copy()
    test = ((meta[SPLIT] == "test").to_numpy() & labelled).copy()
    return train, test


def ridge_r2(H, meta, target):
    """Full-space test R², ridge, λ by `group_key`-grouped CV inside train.

    Never GCV: it assumes exchangeable rows, and CHILI's 3,180 rows sit in 265
    groups, so GCV leaks its effective validation set through near-duplicates and
    under-regularizes (docs/TRAPS.md → *GCV under grouped rows*). One SVD per fold
    serves the whole λ grid; the grid is scaled once by the full train design's mean
    squared singular value so it stays dimensionless.
    """
    labelled = meta[target].notna().to_numpy()
    train, test = _split_masks(meta, labelled)
    Htr, Hte = H[train], H[test]
    groups = meta.loc[train, "group_key"].to_numpy()
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
        # An argmin at either end means the CV wanted a λ the grid does not contain,
        # so the "selection" is a boundary rather than an optimum. Silent edge
        # selection is the failure docs/TRAPS.md records for two other searches here.
        if j in (0, len(_LAMBDAS) - 1):
            print(f"    note: {target} λ at grid {'low' if j == 0 else 'high'} edge "
                  f"({float(_LAMBDAS[j]):.3g}× mean(s²))", flush=True)
        pred = predict_test(y[train], float(_LAMBDAS[j]) * scale)
        truth = y[test]
        ss_res = float(((truth - pred) ** 2).sum())
        ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def logistic_f1(H, meta, target):
    """Weighted F1 on test, via `downstream_eval`'s own classification probe.

    Uses `fit_classifier`/`score_classifier` rather than a local logistic fit, so the
    number lands on the same footing as `probe.json` and Q1 — same standardizer, same
    solver, same untuned `C`, same weighted-F1 definition. A reimplementation would
    make the two incomparable in a way nothing in the output would show.

    The vocabulary is the union over all labelled rows, so a class absent from train
    still has a slot rather than vanishing.
    """
    labelled = meta[target].notna().to_numpy()
    train, test = _split_masks(meta, labelled)
    y = meta[target].to_numpy()
    classes = sorted(np.unique(y[labelled]).tolist())
    head = fit_classifier(H[train], y[train], classes)
    return score_classifier(head, H[test], y[test])["f1_weighted"]


def stamp(H, meta, target, kind):
    return ridge_r2(H, meta, target) if kind == "reg" else logistic_f1(H, meta, target)


def _colour(meta, target, kind):
    """`(values, cmap, norm, ticks, ticklabels, labelled mask)` for one row."""
    v = meta[target].to_numpy(dtype=float)
    labelled = ~np.isnan(v)
    if kind == "reg":
        lo, hi = np.percentile(v[labelled], CLIP_PCT)
        return v, POINT_CMAP, plt.Normalize(lo, hi), [lo, hi], [f"{lo:.3g}", f"{hi:.3g}"], labelled
    # Discrete: both class targets are ORDINAL numbers (cn 4/6/8, oxidation 1..6), so
    # they keep the same ramp sampled at N levels rather than taking categorical hues
    # — `paperstyle`'s categorical set caps at three slots and oxidation has five.
    classes = np.unique(v[labelled])
    cmap = LinearSegmentedColormap.from_list(
        f"{target}_levels", POINT_CMAP(np.linspace(0.0, 1.0, len(classes))), N=len(classes))
    bounds = np.arange(len(classes) + 1) - 0.5
    codes = np.full_like(v, np.nan)
    for i, c in enumerate(classes):
        codes[v == c] = i
    ticks = np.arange(len(classes))
    return (codes, cmap, BoundaryNorm(bounds, len(classes)), ticks,
            [f"{c:.3g}" for c in classes], labelled)


def build(meta, feats, out_stem, sub, coords=None, columns=COLUMNS, stamps=True):
    """`coords` None -> project with PCA in-process; otherwise a dict of precomputed
    2-D layouts keyed by `cache_key`, which is how UMAP arrives (it is computed in a
    second env — see `tools/umap_coords.py`).

    `columns` is a subsequence of `COLUMNS`; `stamps` False leaves each panel's
    full-space score unmeasured (`None`) and drops the corner number with it, which
    is the subset figure's mode — see the `--columns` paragraph in the docstring.
    `stamps=False` also skips every ridge CV and logistic fit, so the run is seconds
    rather than minutes."""
    ps.use()

    panels = {}
    for target, _, kind in TARGETS:
        for key, _, src in columns:
            ck = (key, None if src is None else target)
            H = feats[ck]
            if coords is None:
                proj = pca2(H)
            else:
                # UMAP has no variance-explained: it is not a projection onto ranked
                # orthogonal axes, so there is no share-of-variance to report and the
                # per-panel percentage is dropped rather than faked.
                proj = (coords[cache_key(*ck)], None)
            value = stamp(H, meta, target, kind) if stamps else None
            panels[(target, key)] = (proj, value)
        print(f"  projected{' + stamped' if stamps else ''} row {target}", flush=True)

    # Margins on the gridspec, NOT tight_layout, which recomputes hspace/wspace and
    # would undo the tight packing this grid needs.
    # ⚠️ WIDTH, NOT 7.2 in (2026-08-25). `paperstyle` has no WIDTH_FULL on purpose —
    # WIDTH already IS this venue's full text width — so a 7.2 in figure overran the
    # text block and then got scaled DOWN 14% by \includegraphics, taking every type
    # size with it. Height scales with it (5.4 * 5.5/7.2) to hold the panel aspect.
    # Height is per-COLUMN-COUNT, not fixed: the six-column grid is 4.15 in, and the
    # three-column cut keeps the SAME canvas width (WIDTH is the text block, see
    # below) so its panels double in width and would read as letterboxes at 4.15 in.
    # 6.3 in puts three columns at roughly square panels; the top margin also comes
    # back to 0.965 because subset mode draws no title above the headers.
    subset = len(columns) < len(COLUMNS)
    fig = plt.figure(figsize=(ps.WIDTH, 6.3 if subset else 4.15))
    gs = fig.add_gridspec(len(TARGETS), len(columns) + 1,
                          width_ratios=[1] * len(columns) + [0.045],
                          # ⚠️ THESE MARGINS HOLD TEXT, NOT BLANK PAGE, and they had to
                          # widen when the canvas narrowed. This figure was authored at
                          # 7.2 in; at WIDTH the gridspec FRACTIONS mean the same
                          # margins are 24% less room in inches, while the row labels
                          # and colourbar tick labels stayed the same POINT size. They
                          # then drew past the canvas, and since `save` crops to the
                          # ink the overhang became figure width — 5.98 in against a
                          # 5.5 in text block, which at natural size is an overfull
                          # hbox. 0.15/0.87 buys 0.69 in back, against 0.48 in of
                          # overhang. `save` prints the width; check it after changing.
                          # ⚠️ SUBSET MARGINS ARE NOT THE SIX-COLUMN ONES, for the
                          # reason the note above gives in reverse. `save` crops to the
                          # ink, so a margin wider than the text it holds becomes lost
                          # WIDTH: at 0.15/0.87 the three-column cut saved at 4.385 in,
                          # and \includegraphics[width=\textwidth] on a 4.385 in crop
                          # scales every type size up 25%. 0.06/0.95 holds the same
                          # rotated row label and colourbar ticks and lands the crop
                          # within a few hundredths of WIDTH — the printed width in
                          # `save`'s line is the check, after any change here.
                          left=0.06 if subset else 0.15,
                          right=0.95 if subset else 0.87,
                          top=0.965 if subset else 0.875,
                          bottom=0.015 if subset else 0.02,
                          hspace=0.13, wspace=0.08)

    # ⚠️ TYPE SIZES ARE PER-MODE, and the subset's are simply `paperstyle`'s. Every
    # other appendix figure hand-sets nothing and takes `use()`'s rcParams (titles
    # 8.5, ticks 8, INK for text, MUTED only for de-emphasis); the 4.8/5/7 pt
    # overrides below are NOT a house style, they are what 0.65 in panels forced on
    # the six-column grid. At 1.55 in the subset has no such excuse, so it hands back
    # every override — an empty dict means "let the rcParams decide". The variance
    # text also comes off MUTED: in the six-column grid it is the quieter of two
    # annotations, and with the stamp gone there is nothing for it to be quieter than.
    type_kw = {} if subset else {"fontsize": 7}
    var_kw = {"color": ps.INK} if subset else {"fontsize": 4.8, "color": ps.MUTED}
    cb_kw = {"length": 2} if subset else {"labelsize": 5, "length": 2}

    for i, (target, row_label, kind) in enumerate(TARGETS):
        values, cmap, norm, ticks, ticklabels, labelled = _colour(meta, target, kind)
        metric = "$R^2$" if kind == "reg" else "wF1"
        for j, (key, header, src) in enumerate(columns):
            ax = fig.add_subplot(gs[i, j])
            (scores, var), value = panels[(target, key)]
            if (~labelled).any():
                ax.scatter(scores[~labelled, 0], scores[~labelled, 1], c=UNLABELLED,
                           s=0.9, linewidths=0, rasterized=True, zorder=1)
            sc = ax.scatter(scores[labelled, 0], scores[labelled, 1], c=values[labelled],
                            cmap=cmap, norm=norm, s=0.9, linewidths=0, alpha=0.85,
                            rasterized=True, zorder=2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(True)
                side.set_color(ps.GRID)
            if value is not None:
                ax.text(0.05, 0.95, f"{metric} {value:.2f}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=5.5, color=ps.INK,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                                  boxstyle="square,pad=0.12"))
            # Variance explained sits INSIDE the panel, not in the column header,
            # because in a per-target column it is a per-panel property.
            if var is not None:
                # The plate is SUBSET-ONLY, and it is the dropped stamp's. On the six
                # column grid this text sits under a stamp that already carries one,
                # in a corner the eye reaches with something to compare against; with
                # the stamp gone it is the panel's only annotation and it lands on the
                # frozen column's densest corner, where 18/13% read as 18/1_%.
                ax.text(0.05, 0.04, f"{100 * var[0]:.0f}/{100 * var[1]:.0f}%",
                        transform=ax.transAxes, ha="left", va="bottom", **var_kw,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                                  boxstyle="square,pad=0.12") if subset else None)
            if i == 0:
                ax.set_title(SUBSET_HEADER.get(key, header) if subset else header,
                             pad=3, **type_kw)
            if j == 0:
                ax.set_ylabel(row_label, color=ps.INK, **type_kw)
        cax = fig.add_subplot(gs[i, -1])
        cb = fig.colorbar(sc, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(**cb_kw)
        cb.set_ticks(ticks)
        cb.ax.set_yticklabels(ticklabels)

    if not subset:
        axes_name = "PC1 vs PC2" if coords is None else "UMAP"
        fig.suptitle(f"{axes_name} of the same 3,180 augmented CHILI-3K particles — "
                     "six representation states, four targets", fontsize=8, y=0.975)
        fig.text(0.105, 0.925,
                 "columns 2/3/5/6 are trained on that row's own target, so their point "
                 "positions change per row; columns 1 and 4 have no target and repeat",
                 fontsize=5.5, color=ps.MUTED, ha="left")
    ps.save(fig, out_stem, sub=sub)
    plt.close(fig)
    return panels


CAPTION = """\
PC1 vs PC2 of six representation states of the SAME 3,180 augmented CHILI-3K
particles (signal_xpdf_aug, the channel every domain-shift result is scored on),
recoloured by four downstream targets.

Columns 2, 3, 5 and 6 are PER-TARGET: each of those 16 panels shows an encoder
trained on its own row's target, at the fit channel named in the header. Their point
positions therefore change from row to row. Columns 1 (raw input) and 4 (the frozen
pretrained encoder) have no target, so their projection repeats down every row and
only the color changes. One PCA per panel, center-only; the small percentages are
PC1/PC2 variance explained for that panel.

The corner number is NOT a property of the projection — it is the FULL-SPACE score
for that panel's (features, target): ridge R^2 for the two regression rows, with
lambda selected by group_key-grouped 5-fold CV inside train, and weighted F1 from the
standard logistic probe for the two classification rows. Both fit on train and score
on test. R^2 and wF1 are not comparable to each other; read within a row.

The two trained arms run at different learning rates by construction (supervised
1e-3/1e-3, fine-tuned 1e-5/1e-4), each at the rate its own published counterpart
uses, so the figure is NOT an arm-versus-arm comparison; every panel is read against
column 4, the frozen encoder, which has no learning rate and no target. The
supervised rate was selected on particle size alone and applied to the other three
rows unchanged.

Gray points in the coordination-number row are CHILI's 265 Spinel particles, which
carry no CN label; they are plotted but excluded from that row's fit and score, so
its encoders saw 2,915 rows while all 3,180 are shown.

Encoders in columns 2/3/5/6 are trainings made for this figure and are not the
archived Q1 cells; a fixed-seed re-run of the finetuner is documented to move by a
median 9.4% in validation loss, so read orderings, not equalities.\
"""


CAPTION_SUBSET = """\
What each representation state organizes. PC1 vs PC2 of the same 3,180 augmented
CHILI-3K particles (signal_xpdf_aug, the channel on which every domain-shift result
here is scored), shown in three representation states and colored by four structural
targets. RAW INPUT nu(G(r)) is the encoder's input itself, G(r) sliced to the
pretraining grid and then min-max normalized; PRETRAINED is
the VICReg encoder frozen, its h taken with no target and no fitting; FINE-TUNED is
that same encoder fine-tuned on the row's own target on the clean channel (encoder
1e-5, head 1e-4). The first two columns have no target, so their projection repeats
down every row and only the colouring changes; the third is per-target, so its
coordinates change from row to row and a shape should not be read as persisting down
it. One PCA per panel, centered and not whitened; the small percentages are PC1/PC2
variance explained for that panel. Gray points in the coordination-number row are
CHILI's 265 spinel particles, which carry no CN label - plotted so the point set stays
identical everywhere, excluded from that row's fitting. No score is stamped on these
panels; the corresponding full-space probe scores are in
analysis/out/appendix_embedding_pca.json. The fine-tuned encoders were trained for
this figure rather than taken from the archived cells, and a fixed-seed re-run of the
finetuner moves validation loss by a median 9.4%, so read orderings, not equalities.\
"""




#: Durable home for the 24 stamps. They are measured numbers someone will quote, and
#: RESULTS.md's rule is that such a number lives in exactly one place — stdout is not
#: a place. Same shape as `analysis/out/theta_probe.json`.
STAMPS_JSON = REPO / "analysis" / "out" / "appendix_embedding_pca.json"


def write_stamps(panels, stem):
    record = {
        "figure": f"figures/appendix/{stem}.pdf",
        "registry": str(REGISTRY.relative_to(REPO)),
        "signal": SIGNAL,
        "n_rows": N_ROWS_EXPECTED,
        "protocol": {
            "regression": f"ridge, test R2, lambda by group_key-grouped {N_LAMBDA_FOLDS}-fold CV in train",
            "classification": "downstream_eval.fit_classifier/score_classifier, weighted F1 on test",
            "fit_split": "train", "score_split": "test",
        },
        "pretrained_ckpt": str(PRETRAINED_CKPT.relative_to(REPO)),
        "columns": [{"key": k, "header": h.replace(chr(10), " "), "source": src} for k, h, src in COLUMNS],
        "rows": {},
    }
    for target, label, kind in TARGETS:
        record["rows"][target] = {
            "label": label,
            "metric": "r2" if kind == "reg" else "f1_weighted",
            "stamps": {k: panels[(target, k)][1] for k, _, _ in COLUMNS},
            "pc_var_explained": {k: (list(v) if (v := panels[(target, k)][0][1]) else None)
                                 for k, _, _ in COLUMNS},
        }
    out = STAMPS_JSON.with_name(f"{stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out.relative_to(REPO)}")


def report(panels):
    print("\nFull-space stamps — R² (regression rows) / weighted F1 (classification rows),\n"
          "fit on train, scored on test; ridge λ by group_key-grouped "
          f"{N_LAMBDA_FOLDS}-fold CV.\n")
    head = " ".join(f"{h.replace(chr(10), ' '):>20}" for _, h, _ in COLUMNS)
    print(f"{'target':<18} {'metric':>6} {head}")
    for target, _, kind in TARGETS:
        cells = " ".join(f"{panels[(target, k)][1]:>20.3f}" for k, _, _ in COLUMNS)
        print(f"{target:<18} {'R2' if kind == 'reg' else 'wF1':>6} {cells}")
    if panels[(TARGETS[0][0], COLUMNS[0][0])][0][1] is not None:
        print("\nPC1/PC2 variance explained")
        for target, _, _ in TARGETS:
            cells = " ".join(f"{100 * panels[(target, k)][0][1][0]:>8.1f}/"
                             f"{100 * panels[(target, k)][0][1][1]:<5.1f}" for k, _, _ in COLUMNS)
            print(f"  {target:<18} {cells}")
    print("\ncaption\n-------\n" + CAPTION)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help='"auto" | "cpu" | "cuda"')
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--features-cache", type=Path,
                    default=REPO / "analysis" / "out" / "appendix_embedding_features.npz",
                    help="load the 17 feature spaces from here if present, else compute "
                         "and write them. Exists because UMAP runs in a SECOND env "
                         "(tools/umap_coords.py) and the features must cross on disk")
    ap.add_argument("--coords", type=Path, default=None,
                    help="npz of precomputed 2-D layouts from tools/umap_coords.py. "
                         "Given, the figure is drawn with UMAP instead of PCA")
    ap.add_argument("--columns", default=None,
                    help="comma-separated subset of " + ",".join(k for k, _, _ in COLUMNS) +
                         ", in draw order. Given, the corner stamp, both title lines "
                         "and the stamps JSON are dropped (see the module docstring)")
    ap.add_argument("--out", default=None,
                    help="figure stem; defaults to appendix_embedding_{pca,umap}, "
                         "or that plus _subset under --columns")
    ap.add_argument("--sub", default="appendix",
                    help="subdirectory under figures/; the appendix family's home")
    args = ap.parse_args()

    by_key = {col[0]: col for col in COLUMNS}
    if args.columns:
        keys = [k.strip() for k in args.columns.split(",") if k.strip()]
        unknown = [k for k in keys if k not in by_key]
        if unknown:
            raise SystemExit(f"unknown column(s) {unknown}; have {list(by_key)}")
        columns = [by_key[k] for k in keys]
    else:
        columns = COLUMNS
    subset = len(columns) < len(COLUMNS)

    device = pick_device(args.device)
    X, meta = load_points()
    print(f"{len(meta)} rows of signal_{SIGNAL} on {device}; "
          f"splits {meta[SPLIT].value_counts().to_dict()}, "
          f"{meta['group_key'].nunique()} groups", flush=True)
    feats = features(X, device, args.batch_size, args.features_cache)
    coords = None
    if args.coords:
        if not args.coords.exists():
            raise SystemExit(f"missing {args.coords} — run tools/umap_coords.py first "
                             f"(in the mmxray-umap env; see its docstring)")
        z = np.load(args.coords)
        coords = {k: z[k] for k in z.files}
        missing = {cache_key(c, None if src is None else t)
                   for c, _, src in columns for t, _, _ in TARGETS} - set(coords)
        if missing:
            raise SystemExit(f"{args.coords} is missing layouts for {sorted(missing)} — "
                             f"it was built from a different feature set")
    stem = args.out or (f"appendix_embedding_{'umap' if coords else 'pca'}"
                        + ("_subset" if subset else ""))
    panels = build(meta, feats, stem, args.sub, coords, columns, stamps=not subset)
    if subset:
        # No write_stamps and no report table: nothing was measured. The subset's
        # numbers are the full grid's, and they are already written there.
        print("\ncaption\n-------\n" + CAPTION_SUBSET)
    else:
        write_stamps(panels, stem)
        report(panels)


if __name__ == "__main__":
    main()
