"""analysis/downstream_eval.py — frozen-encoder linear probe on a downstream registry.

    python -m analysis.downstream_eval --runs runs/pdf/sweep/ --filter 'transformer_*'

Ranks pretrained SSL checkpoints by how linearly decodable downstream targets are
from their FROZEN embeddings. For each `<run>/ckpt_best.pt` it rebuilds the
encoder from the config stored in the checkpoint, embeds a downstream registry's
signal ONCE, then fits a LINEAR head per target on the `train` split and scores it
on `test`. The encoder is frozen throughout and never receives a gradient; the
regression head is closed-form least squares, while the classification head added in
session 20 is an iterative logistic fit (see below).

Results are written to `<run>/probe.json` (`--out-name` to change it — the name
carries no registry, so a second registry's sweep overwrites the first's), nested
per task then per protocol (runs
stay self-contained), and one ranked table per task is printed at the end — not a
mean across tasks, which would invent a composite metric. There is deliberately no
aggregated results file: it would be a derived artifact that silently goes stale.

Defaults probe CHILI's native measured xPDF for six targets — three regression:
  target_np_size    nanoparticle size (A), a property of the nanoparticle
  target_mo_bond    min metal-oxygen distance (A), a property of the parent cell
  target_mean_bond  mean nearest-neighbour M-O distance (A), CrystalNN mean-of-means
and three classification (`CLASSIFICATION_TARGETS`):
  target_oxidation  formal metal charge from stoichiometry — 5 classes
  target_cn         metal coordination number — 3 classes (4, 6, 8)
  target_metal      the metal element, as atomic number — 53 classes
np_size and mo_bond are near-uncorrelated (r=0.029 across CHILI), so they probe
different things. The first five port the ground-truth targets of Na Narong et al.,
npj Comput. Mater. 11, 98 (2025); see `data/builders/chili.py` for how each is
derived, and for why `target_oxidation`/`target_cn` are much easier here than in
that paper — on CHILI both are deterministic functions of `crystal_type`, so a high
F1 means the prototype was identified, not that a local environment was resolved.
`target_metal` (session 21, not from the paper) is the opposite case: it is
independent of `crystal_type` (53 metals x 12 prototypes are fully crossed), so it
probes something the other two classification targets structurally cannot.

CLASSIFICATION HEAD. `sklearn.linear_model.LogisticRegression` on standardized
embeddings, with the scaler fit on the FIT split only. Reported per protocol:
weighted-mean F1, per-class F1, and the F1 of a trivial classifier that always
predicts the fit split's modal class — the paper's metric set. The baseline is the
part that makes the number readable: CHILI's oxidation classes run 8/42/9/33/8 %, so
a weighted F1 of 0.6 may be no better than guessing. This head IS iteratively fit,
unlike the OLS one, but only on the frozen embeddings — no gradient reaches the
encoder, so what is being ranked is unchanged.

PER-TARGET MASKING. Targets may be null on some rows (`target_cn` is NaN for CHILI's
265 Spinel rows, where the metal sites disagree). Each task is fit and scored on its
own finite subset of the SAME single embedding pass; nothing is dropped registry-wide
to satisfy one target. `probe.json` records the per-target labelled counts.

`probe.json` metric keys differ by task kind and are self-describing: `r2`/`mae` for
regression, `f1_weighted`/`f1_per_class`/`f1_baseline`/`classes` for classification.
The nesting (task -> protocol -> metrics) is unchanged from session 19.

PROTOCOLS. If the registry also carries an augmented channel (`--aug-signal`,
written by `tools/augment_chili.py`: the same nanoparticles re-simulated at
randomly drawn instrument params), each task is scored three ways:

  clean_clean  fit on clean train, score on clean test — the baseline
  clean_aug    fit on clean train, score on AUGMENTED test — measures how much
               of the probe survives an instrument-parameter shift it never saw
  aug_aug      fit on augmented train, score on augmented test — separates "the
               encoder cannot represent shifted data at all" from "the clean
               probe simply does not transfer to it"

`clean_aug` minus `clean_clean` is the domain-shift cost; a large drop there with
`aug_aug` staying high means the information is still in the embedding, just
moved. With no augmented channel present only `clean_clean` is computed, and the
output shape is the same.

CHEMISTRY AXIS (`--fit-registry`, 2026-08-14). Fits the head on ONE registry's
clean train rows and scores it on ANOTHER's test rows, giving the frozen-probe
counterpart of `analysis/finetune.py --fit-registry`:

  cross_clean  fit on the FIT registry's clean train -> score on eval clean test
  cross_aug    the SAME head                         -> score on eval AUG test

so the row effect (in-registry vs cross) is the chemistry shift while the column
effect stays exactly the instrument shift. Only the fit registry's `train` split is
read — a closed-form head has no hyperparameter, so its `val` rows are never
touched, and the test rows always come from `--registry`, which is what pins
evaluation while the fit set moves.

REGRESSION ONLY, refused rather than approximated for classification: the class
vocabulary does not survive a change of registry, so a head fitted on one label set
would be scored against units it never trained on. Same guard, same reason, as
`finetune.py`.

⚠️ Both registries must sit on the SAME pretraining grid — checked, not assumed,
because a mismatch would feed the encoder two different r axes. And a cross-registry
R² is scored against the EVAL split's own mean, so it is only meaningful when the
two label distributions overlap; verify that before quoting one.

LABELED-DATA-SIZE SWEEP (`--n-train-frac`, 2026-08-14). Fits every head on a seeded
fraction of the train split's labelled rows, drawn by the SAME function the finetune
sweep uses (`analysis.finetune.subsample_train_mask`, which depends only on
`(mask, frac, seed)`), through the same `load_probe_data` row order — so probe cells
at a given `(frac, seed)` fit the identical rows the `runs/pdf/datafrac/` cells
trained on. That row match is what makes a probe-vs-scratch curve comparable point
for point. Test rows are never subsampled. Use `--ridge` at small fractions: below
n_train = latent_dim, plain OLS interpolates silently (see `fit_probe`).

Note what this measures: the encoders were pretrained on *simulated* diffpy PDFs,
so this is a sim->real transfer test, and a low score confounds a weak
representation with domain shift. The `val` split is untouched — plain least
squares has no hyperparameter to tune, so val stays clean for a later ridge or
finetune.

`import torch` comes first on purpose: importing pandas or `core.*` ahead of it
segfaults with no traceback (exit 139) — see docs/ENVIRONMENT.md.
"""

from __future__ import annotations

import torch  # MUST precede pandas / core.* — see module docstring

import argparse
import fnmatch
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from core.config import REPO_ROOT, TrainConfig
from core.registry import MATERIAL_ID, SPLIT
from core.train import build_model, pick_device
from core.transforms import max_normalize, minmax_normalize
from modalities.pdf.simulate import RMIN, RSTEP  # the grid the encoders were pretrained on
from modalities.xrd.grid import TT_GRID  # the 2theta grid the XRD encoders expect

__all__ = ["load_probe_data", "load_encoder", "embed", "fit_probe", "score",
           "fit_classifier", "score_classifier"]

DEFAULT_REGISTRY = "data/downstream/chili/chili_registry.parquet"
DEFAULT_SIGNAL = "xpdf"
DEFAULT_AUG_SIGNAL = "xpdf_aug"
DEFAULT_TARGETS = ["target_np_size", "target_mo_bond", "target_mean_bond",
                   "target_oxidation", "target_cn", "target_metal"]
#: Targets scored as classification rather than regression. An explicit list, not a
#: guess from dtype or cardinality: `target_cn` is a float column holding 4.0/6.0/8.0
#: and `target_oxidation` holds 8/3, so nothing about the values says "class".
CLASSIFICATION_TARGETS = frozenset({"target_oxidation", "target_cn", "target_metal",
                                    "target_crystal_system"})
#: Fit split -> score split. `clean_clean` always runs; the other two need an
#: augmented channel in the registry.
PROTOCOLS = {"clean_clean": ("clean", "clean"), "clean_aug": ("clean", "aug"),
             "aug_aug": ("aug", "aug")}
#: The CHEMISTRY axis (`analysis/finetune.py --fit-registry`). Same `<fit>_<eval>`
#: naming, but the `cross` fit channel comes from a DIFFERENT registry — the eval
#: registry still supplies both test channels, so the column effect (clean vs aug)
#: stays exactly the instrument shift while the row effect becomes the chemistry
#: shift. Kept as a separate table rather than merged into `PROTOCOLS`: every
#: existing caller iterates that dict, and a fourth entry needing a second registry
#: would break all of them.
CROSS_PROTOCOLS = {"cross_clean": ("cross", "clean"), "cross_aug": ("cross", "aug")}


def channel_contract(signal, signal_len):
    """`(expected grid, read-time normalizer)` for a signal channel, by MODALITY.

    Both are properties of the modality, not of this function, and they move
    together: an encoder is pretrained on one grid with one normalization, and a
    downstream channel has to arrive on both or the comparison is meaningless.

    Keyed on the signal NAME because that is what the registry column carries —
    `xpdf`, `xpdf_aug` -> the simulator's r-grid + min-max; `xrd`, `xrd2t` ->
    the 2theta grid + max-norm (DESIGN.md -> XRD arm; a zero-filled XRD row has
    min = 0 while a simulated one with a background has min > 0, so min-max would
    treat the two differently).

    The returned grid is SLICED to `signal_len`, never resampled — see
    `load_probe_data`, which is what enforces that.
    """
    if signal.startswith("xrd"):
        return TT_GRID[:signal_len], max_normalize
    return RMIN + np.arange(signal_len) * RSTEP, minmax_normalize


def load_probe_data(registry_path, signal, targets, signal_len, splits=("train", "test")):
    """Read a downstream registry into `(out, r_range)`, ready for the encoder.

    `out` maps split name -> `(X, Y, ids)` with `X` of shape `(N, signal_len)`
    and `Y` a dict of target name -> `(N,)`. All targets share one `X`: embedding
    is the expensive step and the heads are nearly free, so every task is scored
    from a single pass over the encoder.

    `splits` names which splits to read. The default is the probe's: `val` is
    deliberately never loaded, because closed-form least squares has no
    hyperparameter to tune against it. `analysis/finetune.py` passes all three —
    a gradient finetune early-stops on val, so it needs the split the probe
    exists to keep clean.

    Nulls in a target are PRESERVED as NaN rather than dropped: rows are shared
    across targets, so dropping a row unlabelled for one target would silently
    shrink every other task. The caller masks per target.

    `X` is of shape `(N, signal_len)`,
    min-max normalized per sample exactly as the pretraining read path does
    (`core.transforms.minmax_normalize`) — the encoder's input distribution has to
    match what it was trained on.

    The registry's signal grid is CHECKED against the simulator's own r-grid
    constants, not assumed to match: CHILI stores 6000 points over 0-59.99 A while
    the encoders were pretrained on 5000 points over 0-49.99 A. Same step, so the
    leading `signal_len` points align exactly and this is a slice, never a
    resample. A registry whose grid does not line up raises rather than silently
    feeding the encoder an off-by-a-step signal.

    Columns are projected at read time: the registry carries inline CIFs (~99 MB
    for CHILI) that a probe never touches.
    """
    x_col, y_col = f"signal_{signal}_x", f"signal_{signal}_y"
    df = pd.read_parquet(registry_path, columns=[MATERIAL_ID, SPLIT, x_col, y_col, *targets])

    grid = np.asarray(df[x_col].iloc[0], dtype=float)
    if len(grid) < signal_len:
        raise ValueError(
            f"{registry_path}: signal_{signal} has {len(grid)} points, need at least "
            f"{signal_len} to match the pretraining grid"
        )
    expected, normalize = channel_contract(signal, signal_len)
    step = float(expected[1] - expected[0])
    if not np.allclose(grid[:signal_len], expected, atol=step / 100):
        raise ValueError(
            f"{registry_path}: signal_{signal} grid does not match the pretraining grid "
            f"(got [{grid[0]:.4f}, {grid[signal_len - 1]:.4f}] over {signal_len} points, "
            f"expected [{expected[0]:.4f}, {expected[-1]:.4f}]) — a resample would be "
            f"needed, which this probe deliberately does not do"
        )

    out = {}
    for name in splits:
        sub = df[df[SPLIT] == name]
        if sub.empty:
            raise ValueError(f"{registry_path}: no rows in split {name!r}")
        # One signal at a time: minmax_normalize takes a bare min()/max() with no
        # axis, so handing it a stacked (N, L) matrix would normalize GLOBALLY
        # across the split rather than per-sample. The pretraining read path calls
        # it on a single signal, and so must this one.
        X = torch.stack([
            # .copy() — the parquet-backed arrays are read-only, which torch warns on
            normalize(torch.from_numpy(np.asarray(v, dtype=np.float32)[:signal_len].copy()))
            for v in sub[y_col]
        ])
        Y = {t: torch.from_numpy(sub[t].to_numpy(dtype=np.float32)) for t in targets}
        out[name] = (X, Y, sub[MATERIAL_ID].tolist())

    return out, (float(expected[0]), float(expected[-1]))


def load_encoder(ckpt_path, device):
    """Rebuild the frozen encoder stored in a `ckpt_best.pt`, plus its TrainConfig.

    The checkpoint's `config` is `asdict(TrainConfig) + git_info()`, so the git
    keys are filtered out before reconstruction. `build_model` returns the full
    pretraining model (encoder + projection head); `embed` reads `.encode()`,
    i.e. the representation `h`, and `embed_z` reads the projected `z`
    (`--space z` on the probes).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    names = {f.name for f in fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items() if k in names})
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, int(ckpt.get("epoch", -1))


@torch.no_grad()
def embed(model, X, device, batch_size=256):
    """`(N, signal_len)` -> `(N, latent_dim)`. Encoders want `(B, 1, L)`."""
    out = [
        model.encode(X[i : i + batch_size].unsqueeze(1).to(device)).cpu()
        for i in range(0, len(X), batch_size)
    ]
    return torch.cat(out)


@torch.no_grad()
def embed_z(model, X, device, batch_size=256):
    """`(N, signal_len)` -> `(N, proj_dim)`: the projector output `z`, eval-mode BN.

    `forward` returns `(z, h)` — z is index 0 (`core/encoders.py`).
    """
    out = [
        model(X[i : i + batch_size].unsqueeze(1).to(device))[0].cpu()
        for i in range(0, len(X), batch_size)
    ]
    return torch.cat(out)


def _design(H):
    """Embeddings plus a bias column, in float64 — least squares is done in double."""
    return torch.cat([H.double(), torch.ones(len(H), 1, dtype=torch.float64)], dim=1)


def fit_probe(H, y, ridge: bool = False):
    """Closed-form linear head. Plain least squares by default; GCV ridge on request.

    **`ridge=False` is the default and is byte-identical to what it always was**, so
    CHILI and every number already measured with it are untouched. n_train (2530 for
    CHILI) far exceeds latent_dim (256), the design matrix is well conditioned, and
    the small-n double-descent peak cannot occur.

    **`ridge=True` exists because the default fails SILENTLY when n_train < latent_dim.**
    RRUFF is that case: 94 training rows against latent_dim 256 makes the design matrix
    94x257, i.e. underdetermined, so `lstsq` returns the minimum-norm solution, which
    interpolates the training set exactly. Measured on RRUFF/xpdf — **train R2 = 1.0000,
    test R2 = -2.07**. Nothing raises, nothing warns; the number just stops meaning
    anything. Lambda is picked by generalized cross-validation, which needs no held-out
    split, so this works on registries (like RRUFF) that have no val rows to spare.
    """
    if not ridge:
        return torch.linalg.lstsq(_design(H), y.double().unsqueeze(1)).solution
    return _fit_ridge_gcv(H.double(), y.double().unsqueeze(1))


#: Log-spaced multipliers on the data's own mean squared singular value, so the grid is
#: dimensionless — an absolute lambda grid would depend on embedding scale and silently
#: mis-regularize a differently-normalized encoder.
_RIDGE_GRID = torch.logspace(-8, 4, 97, dtype=torch.float64)


def _fit_ridge_gcv(H, y):
    """Ridge with lambda by GCV. The bias is NOT penalized, hence the centering.

    Generalized cross-validation is the leave-one-out proxy

        GCV(lam) = n * RSS(lam) / (n - df(lam))^2,   df(lam) = sum_i s_i^2/(s_i^2 + lam)

    evaluated from one SVD of the centered design, so all 97 lambdas cost one
    decomposition rather than 97 fits. Returns `(latent_dim + 1, 1)` with the bias
    LAST, matching `_design`'s column order, so `score()` needs no change.
    """
    n = H.shape[0]
    x_mean, y_mean = H.mean(0, keepdim=True), y.mean()
    Xc, yc = H - x_mean, y - y_mean

    U, s, Vh = torch.linalg.svd(Xc, full_matrices=False)
    uty = U.T @ yc                                     # (k, 1)
    s2 = (s**2).unsqueeze(1)                           # (k, 1)
    lam = _RIDGE_GRID.unsqueeze(0) * s2.mean()         # (1, n_lam), scale-free

    shrink = s2 / (s2 + lam)                           # (k, n_lam)
    # RSS splits into the part inside span(U) and a lambda-independent remainder
    # outside it (U is orthonormal but not complete when n < d).
    rss = ((1.0 - shrink) ** 2 * uty**2).sum(0) + ((yc**2).sum() - (uty**2).sum())
    df = shrink.sum(0)
    gcv = n * rss / (n - df).clamp(min=1e-9) ** 2

    # **GCV DEGENERATES WHEN df CAN REACH n, WHICH IS EXACTLY THE n < d CASE THIS
    # FUNCTION EXISTS FOR.** As lambda -> 0 the fit interpolates: RSS -> 0 and
    # df -> n, and the denominator (n - df)^2 collapses FASTER than the numerator,
    # so GCV -> 0 and the criterion happily elects the interpolating fit it is
    # supposed to rule out. Measured on RRUFF/xpdf/n_atoms — the unrestricted global
    # minimum sits at lambda/mean(s^2) = 1e-8 with GCV = 2.4e-07, df = 93.0 of n = 94,
    # train R2 = 1.0000 and test R2 = -0.57; the genuine INTERIOR minimum is at 1e0
    # with GCV = 34.1, df = 24.1 and test R2 = +0.36.
    #
    # Capping the effective degrees of freedom at n/2 removes the degenerate branch
    # and leaves the interior minimum, which is the one GCV is actually reasoning
    # about. The cap is a guard, not a tuned hyperparameter: it binds only when
    # df > n/2, so it is inert whenever n comfortably exceeds latent_dim (CHILI at
    # n = 2530 tops out at df = 256, well under 1265) and never moves an
    # already-well-posed fit.
    admissible = df <= n / 2.0
    if not bool(admissible.any()):                     # every lambda over-fits: take the largest
        return _weights_at(Vh, U, s, uty, lam[0, -1], x_mean, y_mean)
    gcv = torch.where(admissible, gcv, torch.full_like(gcv, float("inf")))
    best = int(torch.argmin(gcv))

    return _weights_at(Vh, U, s, uty, lam[0, best], x_mean, y_mean)


def _weights_at(Vh, U, s, uty, lam, x_mean, y_mean):
    """Ridge weights at one lambda, plus the unpenalized bias, as `(d + 1, 1)`."""
    w = Vh.T @ ((s / (s**2 + lam)).unsqueeze(1) * uty)
    return torch.cat([w, (y_mean - (x_mean @ w)).reshape(1, 1)], dim=0)


def score(H, y, w):
    """R² and MAE of a fitted probe. R² is variance-explained against `y`'s own mean."""
    pred = (_design(H) @ w).squeeze(1)
    y = y.double()
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"r2": float(1 - ss_res / ss_tot), "mae": float((y - pred).abs().mean()),
            # Per-row output, for the appendix parity panels. Additive: every
            # existing key keeps its meaning, and a reader of the figure can
            # recompute `r2`/`mae` from these two lists alone.
            "y_true": [float(v) for v in y],
            "y_pred": [float(v) for v in pred]}


def _single_threaded():
    """Pin sklearn's native calls to one thread, for the duration of a `with` block.

    NOT a performance knob — it is load-bearing. This env has THREE OpenMP runtimes
    on the library path: conda's `libomp` (pulled in by scikit-learn), conda's
    `libiomp5`, and the copy vendored inside the pip torch wheel
    (`functorch/.dylibs/libomp.dylib`). Importing torch and sklearn together is fine
    in either order, but the first multi-threaded sklearn call then dies — sometimes
    `OMP: Error #179 pthread_mutex_init failed`, sometimes a bare SIGSEGV (exit 139,
    the same no-traceback signature docs/ENVIRONMENT.md records for this stack).

    Measured: `OMP_NUM_THREADS=1` also fixes it and `KMP_DUPLICATE_LIB_OK=TRUE` does
    NOT. This is the scoped form of the former — no global env var for a caller to
    forget, and torch's own threading is left alone outside the block. Both the fit
    and the predict need wrapping; limiting only the fit still segfaults on predict.

    n_train=2530 x latent_dim=256 fits in well under a second single-threaded, so the
    cost of this is not measurable next to the encoder pass.
    """
    from threadpoolctl import threadpool_limits  # ships with scikit-learn

    return threadpool_limits(limits=1)


def _encode(y, classes):
    """Class VALUES -> integer codes into `classes`.

    The target columns hold the classes as floats — `target_cn` is 4.0/6.0/8.0 and
    `target_oxidation` includes 8/3. sklearn inspects the label array and calls any
    non-integral float target "continuous", refusing to fit or to score it. Codes keep
    everything in a discrete domain; `classes` carries the real values back out for
    reporting, so nothing downstream has to know this happened.
    """
    index = {v: i for i, v in enumerate(classes)}
    missing = sorted(set(y.tolist()) - index.keys())
    if missing:
        raise ValueError(f"class values {missing} are absent from the label vocabulary {classes}")
    return np.array([index[v] for v in y.tolist()], dtype=np.int64)


def fit_classifier(H, y, classes):
    """Multinomial logistic regression on standardized embeddings.

    `classes` is the shared label vocabulary — the sorted class values, fixed by the
    caller across protocols AND across the fit/test split, so per-class metrics stay
    positionally comparable and a class the fit split never saw still has a slot
    (scoring 0) rather than silently vanishing from the table.

    Returns `(clf, scaler, modal_code, classes)`. The modal class of the FIT split
    travels with the head because the trivial-classifier baseline it defines must be
    the one the head itself could have fallen back on, not one recomputed from test.

    The scaler is fit here, on the fit split only. Logistic regression is scale
    sensitive and lbfgs will not converge in a sane number of iterations on raw
    encoder embeddings, whose per-dimension scales vary by orders of magnitude.

    This is the one iterative fit in the module. It touches only the frozen
    embeddings, so the ranking still measures pretraining and not this head.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    codes = _encode(y, classes)
    X = H.numpy()
    with _single_threaded():
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=5000).fit(scaler.transform(X), codes)
    return clf, scaler, int(np.bincount(codes, minlength=len(classes)).argmax()), classes


def score_classifier(head, H, y):
    """Weighted-mean F1, per-class F1, accuracy, and the modal-class baseline F1.

    The paper's metric set. The baseline — what a classifier that always predicts the
    fit split's modal class would score — is what makes the F1 readable on an
    imbalanced target, and is reported alongside rather than subtracted.

    PER-CLASS METRICS ARE RESTRICTED TO CLASSES PRESENT IN THIS TEST SET. `classes`
    (the fit vocabulary) is the union of fit- and score-split values, because
    `fit_classifier` needs every FIT-split class to encode `y_tr` — but a class the
    test split never contains has zero true instances, so its "F1" is a 0/0 that
    sklearn reports as 0 regardless of the model. Padding the per-class list with
    those is not a smaller score, it is a meaningless one. Concretely, on CHILI's
    `target_metal`: all 53 metals are in `train`, but composition-level group-disjoint
    splitting leaves only 5 groups/metal (see `data/builders/chili.py`), so 29 of 53
    have ZERO rows in `test`. `f1_weighted`/`f1_baseline` are UNCHANGED by this
    either way — weighted averaging already zero-weights an absent class by
    construction (weight = true support = 0); only the per-class breakdown and the
    `classes` reported alongside it are narrowed, and they stay positionally aligned
    with each other. Since `y` (test truth) doesn't depend on the checkpoint or the
    protocol, the present-class set is identical across every run and both
    non-`clean_clean` protocols for one target — safe for a caller to read off the
    first run and treat as shared, as `analysis/probe_classification.py` does.
    """
    from sklearn.metrics import f1_score

    clf, scaler, modal, classes = head
    truth = _encode(y, classes)
    with _single_threaded():
        pred = clf.predict(scaler.transform(H.numpy()))
    kw_all = dict(labels=list(range(len(classes))), zero_division=0)
    present = sorted(set(truth.tolist()))  # codes actually occurring in y (test truth)
    kw_present = dict(labels=present, zero_division=0)
    return {
        "f1_weighted": float(f1_score(truth, pred, average="weighted", **kw_all)),
        "f1_per_class": [float(v) for v in f1_score(truth, pred, average=None, **kw_present)],
        "f1_baseline": float(f1_score(truth, np.full_like(truth, modal), average="weighted", **kw_all)),
        "accuracy": float((pred == truth).mean()),
        "classes": [float(classes[c]) for c in present],
        "modal_class": float(classes[modal]),
        # Per-row output, for the appendix confusion panels. Stored as class
        # VALUES, not codes: `classes` above is narrowed to the present set,
        # so a code would not index it, and values are self-describing.
        "y_true": [float(classes[c]) for c in truth],
        "y_pred": [float(classes[c]) for c in pred],
    }


def _record_path(path):
    """Repo-relative when the file is inside the tree, absolute otherwise.

    `Path.relative_to` RAISES on a path outside the repo, and this is called while
    building the result dict — i.e. after every embedding pass has already run. A
    registry staged anywhere else (a scratch dir, an external volume) would lose the
    whole computation to a formatting detail. Inside the repo, which is every
    production caller, the recorded string is unchanged.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def find_checkpoints(runs_dir, pattern):
    """`<runs_dir>/*/ckpt_best.pt` whose run-dir name matches `pattern`."""
    runs_dir = Path(runs_dir)
    return sorted(
        p for p in runs_dir.glob("*/ckpt_best.pt") if fnmatch.fnmatch(p.parent.name, pattern)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs/pdf/sweep", help="dir whose */ckpt_best.pt are probed")
    parser.add_argument("--filter", default="*", help="glob over run-dir names, e.g. 'transformer_*'")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help="downstream registry parquet")
    parser.add_argument("--signal", default=DEFAULT_SIGNAL, help="signal_<name>_x/_y to probe")
    parser.add_argument("--aug-signal", default=DEFAULT_AUG_SIGNAL,
                        help="augmented channel for the shift protocols; skipped if absent")
    parser.add_argument("--fit-registry", default=None,
                        help="CHEMISTRY AXIS: fit the head on THIS registry's clean train "
                             "rows and score on --registry's test rows (protocols "
                             "cross_clean/cross_aug). Regression targets only")
    parser.add_argument("--fit-signal", default=None,
                        help="channel to read from --fit-registry (default: --signal)")
    parser.add_argument("--target", nargs="+", default=DEFAULT_TARGETS,
                        help="target_* column(s) to regress; all share one embedding pass")
    parser.add_argument("--device", default="auto", help='"auto" | "cpu" | "cuda"')
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ridge", action="store_true",
                        help="GCV ridge instead of plain OLS for the REGRESSION heads. "
                             "Required when n_train < latent_dim (e.g. RRUFF at 94 vs 256), "
                             "where plain OLS interpolates the train split and reports "
                             "meaningless test scores. No effect on classification heads.")
    parser.add_argument("--n-train-frac", type=float, default=1.0,
                        help="fit every head on a seeded FRACTION of the train split's labeled "
                             "rows. The draw is analysis.finetune.subsample_train_mask, and both "
                             "modules load rows through load_probe_data, so a given (frac, seed) "
                             "fits the IDENTICAL rows the runs/pdf/datafrac/ finetune cells "
                             "trained on — that row match is what makes a probe-vs-scratch curve "
                             "comparable point for point. 1.0 (default) is the untouched path. "
                             "Pass --ridge with small fractions: below n_train=latent_dim plain "
                             "OLS interpolates and the score is meaningless (see fit_probe).")
    parser.add_argument("--subsample-seed", type=int, default=42,
                        help="seed for the --n-train-frac draw; matches finetune.py --seed so "
                             "the fitted rows line up with that seed's datafrac cells")
    parser.add_argument("--space", choices=("h", "z"), default="h",
                        help='"h" (encoder representation, the default) or "z" (projector '
                             "output). z gets its own default --out-name so it can never "
                             "clobber the h-space probe.json readers depend on")
    parser.add_argument("--out-name", default=None,
                        help="filename written next to each checkpoint (default probe.json "
                             "for --space h, probe_z.json for z). The default "
                             "OVERWRITES any previous sweep's results, because the name "
                             "carries no registry: probing a second downstream registry "
                             "destroys the first one's JSONs, and `runs/pdf/sweep/` is gitignored. "
                             "Pass e.g. probe_au.json to keep both. Readers "
                             "(plot_probe_sweep, finetune_curves, train_curves) look for "
                             "the default name, so leave it alone for the main sweep.")
    args = parser.parse_args()
    if args.out_name is None:
        args.out_name = "probe.json" if args.space == "h" else "probe_z.json"
    embed_fn = embed if args.space == "h" else embed_z
    if not (0.0 < args.n_train_frac <= 1.0):
        raise SystemExit(f"--n-train-frac must be in (0, 1], got {args.n_train_frac}")
    if args.fit_registry:
        # Refuse rather than mis-score: a class vocabulary spanning two registries
        # whose label sets disagree leaves the head with units it never trains on.
        # Same guard, same wording, as analysis/finetune.py.
        bad = [t for t in args.target if t in CLASSIFICATION_TARGETS]
        if bad:
            raise SystemExit(
                f"--fit-registry is regression-only; drop {bad} (the class vocabulary "
                f"does not survive a change of registry)"
            )
    if args.n_train_frac < 1.0:
        # Lazy: finetune.py imports this module at load time, so a top-level import
        # here would be circular. The function depends only on (mask, frac, seed).
        from analysis.finetune import subsample_train_mask
    else:
        subsample_train_mask = None

    ckpts = find_checkpoints(args.runs, args.filter)
    if not ckpts:
        raise SystemExit(f"no */ckpt_best.pt under {args.runs} matching {args.filter!r}")

    device = pick_device(args.device)
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = REPO_ROOT / registry_path

    # signal_len is a property of the pretraining grid, so it comes from the first
    # checkpoint's config rather than being a flag; a mismatch across runs raises below.
    first = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    signal_len = int(first["config"]["signal_len"])
    del first

    splits, r_range = load_probe_data(registry_path, args.signal, args.target, signal_len)
    (X_tr, Y_tr, ids_tr), (X_te, Y_te, ids_te) = splits["train"], splits["test"]

    # The augmented channel is optional: a registry built before
    # tools/augment_chili.py ran simply has no such column.
    has_aug = f"signal_{args.aug_signal}_y" in set(pq.read_schema(registry_path).names)
    if has_aug:
        aug, _ = load_probe_data(registry_path, args.aug_signal, args.target, signal_len)
        (A_tr, _, a_ids_tr), (A_te, _, a_ids_te) = aug["train"], aug["test"]
        # Both reads filter the same frame the same way, so the row order must
        # match — the targets loaded with the clean channel are reused for both.
        if a_ids_tr != ids_tr or a_ids_te != ids_te:
            raise ValueError(f"{args.aug_signal} rows are not aligned with {args.signal}")

    # CHEMISTRY AXIS. The fit side is a whole separate registry, so it is deliberately
    # NOT alignment-checked against the eval rows — there is no correspondence between
    # them. Only `train` is read: a closed-form head has no hyperparameter to select,
    # and the test rows always come from the eval registry.
    fit_registry_path = None
    if args.fit_registry:
        fit_registry_path = Path(args.fit_registry)
        if not fit_registry_path.is_absolute():
            fit_registry_path = REPO_ROOT / fit_registry_path
        cross, cross_r_range = load_probe_data(
            fit_registry_path, args.fit_signal or args.signal, args.target, signal_len,
            ("train",))
        # The grids must agree even though the rows do not: `channel_contract` checks
        # each registry against the pretraining grid independently, so a disagreement
        # here means one of them slid off it and the encoder would see two r axes.
        if cross_r_range != r_range:
            raise ValueError(
                f"fit registry grid {cross_r_range} != eval registry grid {r_range}"
            )
        X_fit, Y_fit, _ = cross["train"]
        protocol_table = CROSS_PROTOCOLS
        protocols = [p for p, (_, test) in CROSS_PROTOCOLS.items()
                     if test == "clean" or has_aug]
    else:
        X_fit, Y_fit = X_tr, Y_tr
        protocol_table = PROTOCOLS
        protocols = list(PROTOCOLS) if has_aug else ["clean_clean"]

    print(
        f"device={device} registry={registry_path.name} signal={args.signal}"
        f"{' +' + args.aug_signal if has_aug else ' (no augmented channel)'}\n"
        + (f"fit_registry={fit_registry_path.name} "
           f"fit_signal={args.fit_signal or args.signal}\n" if fit_registry_path else "")
        + f"targets={', '.join(args.target)}  protocols={', '.join(protocols)}\n"
        f"grid={signal_len} pts over [{r_range[0]:.2f}, {r_range[1]:.2f}] A  "
        f"n_train={len(X_fit)} n_test={len(X_te)}\n"
        f"probing {len(ckpts)} checkpoint(s) under {args.runs} matching {args.filter!r}\n"
    )

    rows = []
    for ckpt_path in ckpts:
        run = ckpt_path.parent.name
        print(f"  {run} ...", end=" ", flush=True)
        model, cfg, epoch = load_encoder(ckpt_path, device)
        if cfg.signal_len != signal_len:
            raise ValueError(
                f"{run}: signal_len={cfg.signal_len} but the probe grid is {signal_len} — "
                f"probe one grid at a time (use --filter)"
            )
        # Embed ONCE per channel; every task and protocol reuses these. The encoder
        # pass dominates, the per-task heads are nearly free. Each entry is
        # (fit-side embedding, eval-side embedding); a None means that side is never
        # indexed under the active protocol table.
        if fit_registry_path is None:
            H = {"clean": (embed_fn(model, X_tr, device, args.batch_size),
                           embed_fn(model, X_te, device, args.batch_size))}
            if has_aug:
                H["aug"] = (embed_fn(model, A_tr, device, args.batch_size),
                            embed_fn(model, A_te, device, args.batch_size))
            fit_channels = [c for c in ("clean", "aug") if c in H]
        else:
            # The eval registry's TRAIN rows are never fitted on in cross mode, so
            # they are not embedded — `cross` supplies the only fit side.
            H = {"cross": (embed_fn(model, X_fit, device, args.batch_size), None),
                 "clean": (None, embed_fn(model, X_te, device, args.batch_size))}
            if has_aug:
                H["aug"] = (None, embed_fn(model, A_te, device, args.batch_size))
            fit_channels = ["cross"]
        tasks, labelled = {}, {}
        for t in args.target:
            # Per-target masks, on the FIT rows and the EVAL rows separately. Within one
            # registry these are the same object read twice (the augmented rows are
            # aligned with the clean ones, asserted above, so one mask serves both
            # channels). With --fit-registry the two sides are different registries with
            # no row correspondence, so conflating them would index the eval embeddings
            # with the fit registry's mask.
            m_tr, m_te = torch.isfinite(Y_fit[t]), torch.isfinite(Y_te[t])
            if subsample_train_mask is not None:
                # Shrink the FIT rows only — the test split is never touched, so every
                # fraction is scored on the identical test set.
                m_tr = subsample_train_mask(m_tr, args.n_train_frac, args.subsample_seed)
            y_tr, y_te = Y_fit[t][m_tr], Y_te[t][m_te]
            labelled[t] = {"train": int(m_tr.sum()), "test": int(m_te.sum())}
            # One head per FIT channel, reused by every protocol that fits on it —
            # clean_clean and clean_aug are the same probe scored on two test sets.
            if t in CLASSIFICATION_TARGETS:
                # One vocabulary for the whole task: the union over fit and test, so
                # the per-class lists stay comparable across protocols and no class
                # disappears just because one split lacks it.
                classes = sorted(set(y_tr.tolist()) | set(y_te.tolist()))
                heads = {c: fit_classifier(H[c][0][m_tr], y_tr, classes) for c in fit_channels}
                tasks[t] = {name: score_classifier(heads[fit], H[test][1][m_te], y_te)
                            for name, (fit, test) in protocol_table.items() if name in protocols}
            else:
                heads = {c: fit_probe(H[c][0][m_tr], y_tr, ridge=args.ridge)
                         for c in fit_channels}
                tasks[t] = {name: score(H[test][1][m_te], y_te, heads[fit])
                            for name, (fit, test) in protocol_table.items() if name in protocols}

        result = {
            "run": run,
            "epoch": epoch,
            "encoder": cfg.encoder,
            "loss": cfg.loss,
            "latent_dim": cfg.latent_dim,
            "cov_weight": cfg.cov_weight,
            "temperature": cfg.temperature,
            "registry": _record_path(registry_path),
            "signal": args.signal,
            "aug_signal": args.aug_signal if has_aug else None,
            # Named exactly as analysis/finetune.py writes them, so a reader joining a
            # probe cell to a finetune cell does not have to translate keys.
            "fit_registry": _record_path(fit_registry_path) if fit_registry_path else None,
            "fit_signal": (args.fit_signal or args.signal) if fit_registry_path else None,
            "protocols": protocols,
            "signal_len": signal_len,
            "r_range": list(r_range),
            "n_train": len(X_fit),
            "n_test": len(X_te),
            "space": args.space,
            "ridge": args.ridge,
            "n_train_frac": args.n_train_frac,
            "subsample_seed": args.subsample_seed if args.n_train_frac < 1.0 else None,
            "n_labelled": labelled,
            "classification_targets": [t for t in args.target if t in CLASSIFICATION_TARGETS],
            "tasks": tasks,
        }
        (ckpt_path.parent / args.out_name).write_text(json.dumps(result, indent=2) + "\n")
        rows.append(result)
        # The reference protocol is the first one that ran, not a literal
        # `clean_clean`: under --fit-registry that key does not exist at all.
        ref = protocols[0]
        print("  ".join(
            f"{t}: " + (f"F1={m[ref]['f1_weighted']:.4f}" if t in CLASSIFICATION_TARGETS
                        else f"R2={m[ref]['r2']:.4f} MAE={m[ref]['mae']:.3f}")
            for t, m in tasks.items()))

    # One table per task — averaging R² across targets of different difficulty
    # would invent a composite metric nobody asked for, and the disagreement
    # between the two orderings is itself the interesting part.
    width = max(len(r["run"]) for r in rows)
    for target in args.target:
        clf = target in CLASSIFICATION_TARGETS
        key, metric = ("f1_weighted", "weighted F1") if clf else ("r2", "R²")
        first = rows[0]["tasks"][target][protocols[0]]
        n = rows[0]["n_labelled"][target]
        # The trivial-classifier baseline depends only on the labels, so it is the
        # same for every run — a header line, not a repeated column.
        # classes are rounded for DISPLAY only — probe.json keeps the exact label
        # values, which for a float32 column means 8/3 is 2.6666667461395264.
        # Listed inline for a handful of classes; target_metal's 53 would just be
        # noise, so it gets a count instead. Guarded on `clf`: a regression task's
        # metrics have no `classes` key, and the guard has to be on the whole
        # expression — the `if clf` on `note` alone still evaluates `classes_desc`.
        note = ""
        if clf:
            classes_desc = (f"[{', '.join(f'{c:g}' for c in first['classes'])}]"
                            if len(first["classes"]) <= 10 else f"{len(first['classes'])} classes")
            note = f" — modal-class baseline {first['f1_baseline']:.4f} over {classes_desc}"
        print(f"\n── ranked by {metric} on {target} (test, n={n['train']}/{n['test']})"
              f"{note} " + "─" * 12)
        # Ordered by the clean baseline so the protocol columns stay comparable
        # down the table; re-sorting per protocol would hide which run moved.
        # A regression protocol prints R² AND MAE, in every column. They answer
        # different questions — R² is scale-free and comparable across targets,
        # MAE is in the target's own units (A) and is what a materials reader
        # quotes — so the pair travels together, as it did in the old repo. The
        # earlier layout showed MAE only when there was a single protocol, which
        # meant any registry carrying an augmented channel printed none at all.
        # Classification has no MAE; its table is unchanged.
        print(f"{'run':<{width}} {'encoder':>12} {'loss':>8}"
              + "".join(f"{p:>{13 if clf else 15}}" for p in protocols))
        if not clf:
            print(f"{'':<{width}} {'':>12} {'':>8}" + f"{'R²':>8}{'MAE':>7}" * len(protocols))
        for r in sorted(rows, key=lambda r: r["tasks"][target][protocols[0]][key], reverse=True):
            m = r["tasks"][target]
            cells = "".join(
                f"{m[p][key]:>13.4f}" if clf else f"{m[p]['r2']:>8.4f}{m[p]['mae']:>7.3f}"
                for p in protocols)
            print(f"{r['run']:<{width}} {r['encoder']:>12} {r['loss']:>8}{cells}")


if __name__ == "__main__":
    main()
