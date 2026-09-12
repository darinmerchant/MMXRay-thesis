"""analysis/probe_cv.py — cross-validated frozen probe: pooled out-of-fold R².

`downstream_eval.py` scores one registry-supplied train/test split. That is the
right tool when the test split is large; it is the wrong one on a 92-row registry,
where it was measured to be dominated by split luck rather than by model quality.
This module replaces the single split with repeated group-K-fold over the WHOLE
registry, and reports **pooled out-of-fold R²**.

**Why pooled, and not the mean of per-fold R².** `score()`'s R² is taken against
the *test rows' own* mean, so its denominator is whatever variance the held-out
rows happen to carry. On this registry that denominator is the dominant noise
source, not the fit — measured on the Au self-driving registry over 3 split seeds:

    target   R² per seed                     MAE / fixed full-set sd
    LiCt     -7.587 / +0.525 / -0.487        0.640 / 0.688 / 1.115
    HAuCl4   +0.620 / -1.977 / +0.330        0.504 / 0.926 / 0.877

LiCt seeds 0 and 1 make almost the same error (0.640 vs 0.688) and report R² of
-7.59 versus +0.53, purely because one test draw had 0.20x the training spread and
the other had 1.00x. Averaging per-fold R² would average those explosions in.
Pooling instead — concatenate every fold's held-out prediction, then take ONE R²
over all N rows — puts the FULL dataset variance in the denominator, which is a
constant of the registry rather than a property of the draw. Same honest R², no
blow-up, and the effective test set is N rather than N/n_folds.

**Folds are group-disjoint**, on the same column the registry's own split uses
(`recipe_group` for Au: single-linkage clusters over the synthesis recipe). Rows
in one group never straddle a fold boundary, so the near-duplicate leakage the
registry was built to prevent cannot reappear here.

**Repeats** re-draw the group->fold assignment under a new seed. Each repeat yields
one complete set of out-of-fold predictions and therefore one pooled R²; the spread
across repeats is the error bar. It is a measure of fold-assignment sensitivity
only — it does NOT capture uncertainty from the 92 samples being the 92 they are.

Embeddings are computed ONCE per checkpoint over the whole registry and reused by
every target, repeat and fold — the encoder pass dominates and the ridge refits are
nearly free, which is what makes 15 refits per target affordable.

Unlike `downstream_eval`, this writes **one summary JSON** (`--out`) rather than a
file per run directory. `<run>/probe.json` carries no registry in its name, so two
registries overwrite each other; a single artifact per measurement has no such
collision, and there is no per-run state to keep in sync.

**`--classify` (added s33, for RRUFF)** treats every target as a class label and
reports pooled out-of-fold **weighted F1** instead of R² — same folds, same pooling
logic, `downstream_eval.fit_classifier` as the head. The reason is the same split-luck
argument that created this module: RRUFF's single 30-row test set gives the figure's
probe F1 = 0.371 a bootstrap 95% CI of [0.20, 0.56], wider than every gap quoted from
it. Pooling out-of-fold predictions over all rows makes the effective test n the whole
registry. The modal-class baseline is recomputed per fold from that fold's FIT rows and
pooled the same way — the single-split version (0.031) was an artifact of train's modal
class (trigonal, 24%) drawing only 13% of test. Class vocabulary per target is the union
over ALL rows, so a fold whose fit split misses a rare class still encodes it (RRUFF has
6 hexagonal rows; a fit split missing all 6 is possible and must not crash).

**`--n-train` / `--draw` (added 2026-08-12, for the RRUFF ladder)** fit each fold's
head on only N of its fit rows. This is the labelled-set-size axis: on RRUFF one row
is one measured mineral, so N counts distinct minerals, and the held-out rows are
never touched, so every rung is scored against the same pooled denominator. It
answers "does the pretrained encoder's advantage widen as labels shrink?" without
inventing samples — an augmented copy of a mineral carries a label you already had,
so it would move the row count without moving the label diversity that actually
bounds the from-scratch arm. `subsample_fit` has the seeding contract that keeps the
probe and the gradient arms on identical minerals.

Run:
    python analysis/probe_cv.py --registry data/downstream/au_selfdriving/au_registry.parquet \\
        --signal xpdf --filter '*_cov*' '*_temp*' '*mpfull_final*' \\
        --target target_vol_HAuCl4 ... --out analysis/out/au_probe_cv.json
"""

from __future__ import annotations

import torch  # MUST precede pandas / core.* — see downstream_eval's module docstring

import argparse
import fnmatch
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.downstream_eval import (
    _encode,
    embed,
    fit_classifier,
    fit_probe,
    load_encoder,
    load_probe_data,
)
from core.config import REPO_ROOT
from core.registry import MATERIAL_ID, SPLIT
from core.train import pick_device

#: The registry column whose equal values must stay in one fold. Defaults to the
#: Au registry's recipe clusters; `--group-col material_id` gives plain K-fold.
DEFAULT_GROUP_COL = "recipe_group"


def grouped_folds(groups: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """`n_folds` arrays of ROW indices, disjoint by group.

    Unique groups are shuffled and dealt into `n_folds` near-equal blocks; a row
    follows its group. Fold sizes are therefore only approximately equal — with 84
    groups over 92 rows the imbalance is small, and forcing equal ROW counts would
    require splitting a group, which is the one thing this must not do.
    """
    uniq = np.unique(groups)
    if len(uniq) < n_folds:
        raise ValueError(f"{len(uniq)} groups is fewer than {n_folds} folds")
    order = np.random.default_rng(seed).permutation(uniq)
    fold_of_group = {g: i % n_folds for i, g in enumerate(order)}
    assigned = np.array([fold_of_group[g] for g in groups])
    return [np.flatnonzero(assigned == k) for k in range(n_folds)]


def blocked_groups(values: np.ndarray, groups: np.ndarray, n_blocks: int) -> np.ndarray:
    """Contiguous-block labels over `values`, constant within each `groups` group.

    The blocked robustness arm (see `--block-on`): groups are ordered by their
    median `values` (campaign iteration, for Au) and dealt into `n_blocks`
    contiguous runs of near-equal ROW count. Assignment is per GROUP, never per
    row — a recipe re-proposed a few iterations later would otherwise straddle a
    block boundary and reintroduce exactly the duplicate leakage the groups exist
    to prevent. With `n_blocks == n_folds`, `grouped_folds` then puts one block in
    each fold whatever the shuffle seed, so the fold assignment is deterministic.
    """
    med = {g: float(np.median(values[groups == g])) for g in np.unique(groups)}
    per_block = len(values) / n_blocks
    labels, assigned = {}, 0
    for g in sorted(med, key=med.get):
        labels[g] = min(n_blocks - 1, int(assigned / per_block))
        assigned += int((groups == g).sum())
    return np.array([f"block_{labels[g]}" for g in groups])


def pooled_oof_r2(H: torch.Tensor, y: torch.Tensor, folds: list[np.ndarray],
                  ridge: bool, match_gradient: bool = False,
                  n_train: int | None = None, draw: int = 0,
                  rep: int = 0) -> tuple[float, np.ndarray]:
    """One R² over every row's held-out prediction. Returns `(r2, oof_predictions)`.

    The denominator is the variance of the FULL `y`, not of any one fold — that is
    the whole point of pooling; see the module docstring. `match_gradient` shrinks
    the FIT rows exactly as in `pooled_oof_f1` (see `fit_rows`); held-out rows are
    untouched, so the pooled score stays over all N. Until 2026-08-12 only the F1
    twin honoured the flag and this function silently ignored it (docs/TRAPS.md).

    `n_train`/`draw` shrink the fit rows a second time, to a ladder rung — see
    `subsample_fit`. `rep` is the repeat index and is passed only so that rung
    lands on the same minerals the gradient arms get.
    """
    oof = np.full(len(y), np.nan)
    for k, held in enumerate(folds):
        keep = subsample_fit(fit_rows(folds, k, len(y), match_gradient),
                             n_train, draw, rep, k)
        w = fit_probe(H[keep], y[keep], ridge=ridge)
        design = torch.cat([H[held].double(),
                            torch.ones(len(held), 1, dtype=torch.float64)], dim=1)
        oof[held] = (design @ w).squeeze(1).numpy()
    if not np.isfinite(oof).all():
        raise ValueError("some rows never landed in a fold")
    truth = y.double().numpy()
    ss_res = float(((truth - oof) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot, oof


def fit_rows(folds: list[np.ndarray], k: int, n_rows: int, match_gradient: bool) -> np.ndarray:
    """Row indices the head may FIT on, for fold `k` of `folds`.

    Default: everything except fold `k` — 4/5 of the registry, the natural thing
    for a closed-form head that needs no validation data.

    `match_gradient` additionally drops fold `k+1`, which is what
    `analysis/finetune_cv.run_arm` reserves as its early-stopping val block. The
    probe cannot use those rows for anything, but a bar chart comparing a probe
    fitted on 4/5 against a finetune trained on 3/5 is not comparing methods — it
    is comparing training-set sizes, and at n=148 that difference (~118 vs ~89
    rows) is large enough to swamp the effect under test. Pass this whenever the
    number is going next to a gradient arm; leave it off for a probe-only table,
    where throwing away a fifth of the fit rows would just be a worse measurement.
    """
    drop = folds[k] if not match_gradient else np.concatenate([folds[k], folds[(k + 1) % len(folds)]])
    return np.setdiff1d(np.arange(n_rows), drop)


def subsample_fit(keep: np.ndarray, n_train: int | None, draw: int,
                  rep: int, k: int) -> np.ndarray:
    """`n_train` of the rows in `keep`, drawn without replacement. The LADDER knob.

    Shrinks the FIT side only. Held-out rows are never touched, so the pooled
    out-of-fold score stays over all N and every rung of the ladder is scored
    against the same denominator — the property that makes the rungs comparable
    at all (see `pooled_oof_r2`).

    **THE SEED IS `(draw, rep, k)` AND NOTHING ELSE, AND THAT IS THE WHOLE POINT.**
    `finetune_cv` calls this on its own `train_idx`, which for a `--match-gradient-rows`
    probe is the SAME row set at the same `(rep, k)` — both modules build folds with
    `grouped_folds(groups, n_folds, seed=rep)`. Deriving the RNG from the position in
    the design rather than from anything arm-specific is what makes the probe, the
    finetune and the supervised arm fit on the IDENTICAL minerals at every cell. If
    it depended on the checkpoint, the arm, or a wall-clock seed, the ladder would be
    comparing arms on different subsets and would measure draw luck instead.

    `n_train=None`, or a value at or above `len(keep)`, is the identity — that is the
    top rung, and it must return `keep` untouched so the rung reproduces the
    already-published full-data cell rather than a resampled near-copy of it.

    Rows, not groups. On RRUFF every `material_id` is unique (one measured pattern
    per mineral), so a row count IS a count of distinct minerals and the ladder's
    x-axis reads directly. On a registry with multi-row groups it would not, and
    this function does not pretend otherwise: it takes no group argument, so a
    caller there is asking for rows and gets rows.
    """
    if n_train is None or n_train >= len(keep):
        return keep
    if n_train < 1:
        raise ValueError(f"n_train={n_train} leaves nothing to fit on")
    picked = np.random.default_rng([draw, rep, k]).choice(keep, size=n_train, replace=False)
    return np.sort(picked)


def pooled_oof_f1(H, y: torch.Tensor, folds: list[np.ndarray],
                  match_gradient: bool = False, n_train: int | None = None,
                  draw: int = 0, rep: int = 0) -> tuple[float, float, np.ndarray]:
    """One weighted F1 over every row's held-out prediction:
    `(f1, modal_baseline_f1, oof_class_values)`.

    ⚠️ **THE THIRD RETURN IS CLASS VALUES, NOT THE CODES THE CLASSIFIER PREDICTS**
    (added 2026-09-09, for the appendix confusion matrices). `_encode` narrows to the
    classes PRESENT, so a code indexes `classes` and means nothing without it; a
    consumer holding the code alongside a raw `y` has two encodings of one label and
    no way to see it. Decoding here is what makes the vector self-contained, and it
    follows the rule the 2026-08-25 per-row-predictions commit set for the other two
    scorers ("stores class VALUES rather than codes"). `analysis/finetune_cv.py`
    banks CODES under a same-named key and is not being changed — see docs/TRAPS.md.

    Returning the vector at all is what `pooled_oof_r2` has always done; the
    classification twin computed the same thing and dropped it, which is why every
    probe_cv JSON on disk carries aggregates a confusion matrix cannot be drawn from.

    The classification twin of `pooled_oof_r2`. The baseline is pooled the same way
    the predictions are: each fold's held-out rows get the modal class of that fold's
    FIT rows, and one F1 is taken over the concatenation — so it is exactly what the
    trivial classifier would have scored under this fold assignment, not a number
    recomputed from a split that shifts under it.

    `classes` is the union over ALL rows (see module docstring): a fit split missing
    a rare class entirely still encodes the truth labels; the head simply never
    predicts the class it never saw.

    Every row is still SCORED exactly once under `match_gradient` — that flag changes
    only which rows the head is fitted on, never the held-out set, so the pooled score
    stays over all N rows and remains comparable to a run without it.

    `n_train`/`draw`/`rep` shrink the fit rows to a ladder rung, exactly as in the
    R² twin — see `subsample_fit`.
    """
    from sklearn.metrics import f1_score

    classes = sorted(set(y.tolist()))
    truth = _encode(y, classes)
    oof = np.full(len(y), -1)
    modal = np.full(len(y), -1)
    for k, held in enumerate(folds):
        keep = subsample_fit(fit_rows(folds, k, len(y), match_gradient),
                             n_train, draw, rep, k)
        clf, scaler, modal_code, _ = fit_classifier(H[keep].float(), y[keep], classes)
        oof[held] = clf.predict(scaler.transform(H[held].float().numpy()))
        modal[held] = modal_code
    if (oof < 0).any():
        raise ValueError("some rows never landed in a fold")
    kw = dict(labels=list(range(len(classes))), average="weighted", zero_division=0)
    values = np.asarray(classes, dtype=float)[oof]
    return (float(f1_score(truth, oof, **kw)), float(f1_score(truth, modal, **kw)), values)


def cv_scores(H, Y, targets, groups, n_folds, repeats, ridge, classify,
              match_gradient=False, n_train=None, draw=0) -> dict:
    """`{target: {"r2_mean", "r2_per_repeat"}}` — one pooled R² per repeat.

    Under `classify`, the keys are `f1_mean` / `f1_per_repeat` plus the pooled
    modal baseline (`modal_f1_mean` / `modal_f1_per_repeat`): different names on
    purpose, so nothing can read an F1 as an R² without noticing.

    Nulls are masked PER TARGET, mirroring `downstream_eval`'s per-target masking:
    `load_probe_data` preserves NaN labels so rows stay shared across targets, and
    here the unlabelled rows (with their groups) are dropped before the folds are
    built — RRUFF's `target_cation_cn` is NaN on 21 of 148 rows, and a NaN label
    reaching `fit_probe` would poison the whole ridge solve, not just its row.
    `n_labelled` records how many rows each target's score is actually over.

    ⚠️ **A target with NaN labels drops rows BEFORE building its folds**, so its
    fold assignment — and therefore its `subsample_fit` draw — differs from a fully
    labelled target's and from `finetune_cv`'s. `n_train` is only row-for-row
    comparable against a gradient arm on targets with no nulls; `n_fit_per_fold`
    records what was actually fitted so the mismatch is visible rather than assumed
    away. RRUFF's `cell_b`/`cell_c` have zero nulls, which is why the ladder uses them.
    """
    out = {}
    for t in targets:
        labelled = torch.isfinite(Y[t])
        Ht, yt, gt = H[labelled], Y[t][labelled], groups[labelled.numpy()]
        per_repeat, modal_per_repeat, n_fit, oof_per_repeat = [], [], [], []
        for rep in range(repeats):
            folds = grouped_folds(gt, n_folds, seed=rep)
            n_fit += [len(subsample_fit(fit_rows(folds, k, len(yt), match_gradient),
                                        n_train, draw, rep, k))
                      for k in range(n_folds)]
            if classify:
                f1, modal_f1, oof = pooled_oof_f1(Ht, yt, folds, match_gradient,
                                                  n_train, draw, rep)
                per_repeat.append(f1)
                modal_per_repeat.append(modal_f1)
                oof_per_repeat.append([float(v) for v in oof])
            else:
                r2, _ = pooled_oof_r2(Ht, yt, folds, ridge, match_gradient,
                                      n_train, draw, rep)
                per_repeat.append(r2)
        key = "f1" if classify else "r2"
        out[t] = {f"{key}_mean": float(np.mean(per_repeat)),
                  f"{key}_per_repeat": [float(v) for v in per_repeat],
                  "n_labelled": int(labelled.sum()),
                  "n_fit_per_fold": [int(v) for v in sorted(set(n_fit))]}
        if classify:
            out[t]["modal_f1_mean"] = float(np.mean(modal_per_repeat))
            out[t]["modal_f1_per_repeat"] = [float(v) for v in modal_per_repeat]
            # PER-ROW, so an appendix panel can ask WHICH classes a scheme confuses —
            # a weighted F1 cannot, and the two failure modes that matter (error
            # spread across neighbouring classes, vs collapse onto the modal class)
            # look identical in the aggregate. Banked as CLASS VALUES with the class
            # list beside them, so the file decodes itself; `y_true` is this target's
            # LABELLED rows in registry order, which is the order every
            # `oof_per_repeat` entry is in, since the NaN mask is taken once above.
            out[t]["y_true"] = [float(v) for v in yt.tolist()]
            out[t]["classes"] = [float(v) for v in sorted(set(yt.tolist()))]
            out[t]["oof_per_repeat"] = oof_per_repeat
    return out


def load_all_rows(registry_path, signal, targets, signal_len, group_col):
    """Whole registry as `(X, Y, ids, groups)` — ALL splits concatenated, order fixed.

    `load_probe_data` is reused rather than reimplemented so the grid check and the
    per-sample normalization are bit-identical to `downstream_eval`'s; the registry's
    own split is then discarded, because CV defines its own folds. "All splits"
    includes `val` where one exists (RRUFF): val's only job is finetune early
    stopping, which a closed-form probe has no use for, and leaving 24 of 148 rows
    out would just shrink the measurement this module exists to widen. (Au has no
    val split, so its numbers are unchanged by this.)
    """
    present = pd.read_parquet(registry_path, columns=[SPLIT])[SPLIT].unique()
    extra = set(present) - {"train", "val", "test"}
    if extra:
        raise ValueError(f"unknown split value(s) {sorted(extra)} in {registry_path}")
    order = [s for s in ("train", "val", "test") if s in set(present)]
    splits, r_range = load_probe_data(registry_path, signal, targets, signal_len,
                                      splits=tuple(order))
    X = torch.cat([splits[s][0] for s in order])
    Y = {t: torch.cat([splits[s][1][t] for s in order]) for t in targets}
    ids = [i for s in order for i in splits[s][2]]

    if group_col == MATERIAL_ID:
        # Plain K-fold: every row is its own group. Reading the id column twice
        # would build the lookup from a duplicated projection — just use the ids.
        return X, Y, ids, np.array(ids), r_range
    frame = pd.read_parquet(registry_path, columns=[MATERIAL_ID, group_col])
    lookup = dict(zip(frame[MATERIAL_ID], frame[group_col]))
    missing = [i for i in ids if i not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} id(s) absent from {group_col}, e.g. {missing[:5]}")
    return X, Y, ids, np.array([lookup[i] for i in ids]), r_range


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", required=True)
    ap.add_argument("--signal", default="xpdf")
    ap.add_argument("--target", nargs="+", required=True)
    ap.add_argument("--runs", default="runs/pdf/sweep")
    # nargs="+" unlike downstream_eval's single glob: the canonical 22-model set is
    # three patterns (`*_cov*` `*_temp*` `*mpfull_final*`) and running it as three
    # separate invocations would produce three tables that cannot share a baseline.
    ap.add_argument("--filter", nargs="+", default=["*"])
    ap.add_argument("--group-col", default=DEFAULT_GROUP_COL)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--no-ridge", action="store_true",
                    help="plain OLS heads. Almost always wrong here: n_train < latent_dim "
                         "makes the design matrix underdetermined and lstsq interpolates.")
    ap.add_argument("--match-gradient-rows", action="store_true",
                    help="fit each fold's head on 3/5 of the rows instead of 4/5, "
                         "dropping the block finetune_cv reserves for early stopping. "
                         "Use when the number goes beside a gradient arm; see fit_rows.")
    ap.add_argument("--n-train", type=int, default=None, metavar="N",
                    help="LADDER RUNG: fit each fold's head on only N of its fit rows "
                         "(held-out rows untouched, so rungs stay comparable). Pair with "
                         "--match-gradient-rows so the rung matches a gradient arm's. "
                         "On RRUFF one row is one mineral, so N is a mineral count.")
    ap.add_argument("--draw", type=int, default=0, metavar="D",
                    help="which subsample draw at this rung (0, 1, 2, ...). Seeds the "
                         "row choice as (draw, repeat, fold) — see subsample_fit — so "
                         "every arm at the same (rung, draw) fits identical minerals.")
    ap.add_argument("--classify", action="store_true",
                    help="every target is a class label: pooled out-of-fold weighted F1 "
                         "via fit_classifier, plus the pooled modal baseline. "
                         "--no-ridge is meaningless here (the head is logistic).")
    ap.add_argument("--block-on", default=None, metavar="COLUMN",
                    help="robustness arm: folds become `--folds` CONTIGUOUS blocks of "
                         "this registry column (e.g. `iteration`), assigned per "
                         "--group-col group so near-duplicates never straddle a block. "
                         "Tests extrapolation across campaign phases instead of "
                         "interpolation within them — the difference from the default "
                         "grouping IS the measurement. Deterministic, so requires "
                         "--repeats 1.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.classify and args.no_ridge:
        raise SystemExit("--no-ridge has no effect under --classify; drop one")
    if args.draw and args.n_train is None:
        raise SystemExit("--draw only means something with --n-train; the full rung "
                         "has exactly one draw")
    ridge = not args.no_ridge
    runs_dir = Path(args.runs)
    ckpts = sorted(p for p in runs_dir.glob("*/ckpt_best.pt")
                   if any(fnmatch.fnmatch(p.parent.name, f) for f in args.filter))
    if not ckpts:
        raise SystemExit(f"no */ckpt_best.pt under {runs_dir} matching {args.filter}")

    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = REPO_ROOT / registry_path

    first = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    signal_len = int(first["config"]["signal_len"])
    del first

    X, Y, ids, groups, r_range = load_all_rows(
        registry_path, args.signal, args.target, signal_len, args.group_col)
    if args.block_on:
        if args.repeats != 1:
            raise SystemExit("--block-on makes the fold assignment deterministic; "
                             "pass --repeats 1 (extra repeats would be identical)")
        frame = pd.read_parquet(registry_path, columns=[MATERIAL_ID, args.block_on])
        lookup = dict(zip(frame[MATERIAL_ID], frame[args.block_on]))
        values = np.array([float(lookup[i]) for i in ids])
        groups = blocked_groups(values, groups, args.folds)
        counts = pd.Series(groups).value_counts().sort_index()
        print(f"blocked on {args.block_on}: rows per block "
              f"{counts.to_dict()}\n")
    device = pick_device(args.device)
    n_groups = len(np.unique(groups))
    metric = "f1" if args.classify else "r2"
    print(f"device={device} registry={registry_path.name} signal={args.signal}\n"
          f"grid={signal_len} pts over [{r_range[0]:.2f}, {r_range[1]:.2f}] A  n={len(X)} rows\n"
          f"{args.folds}-fold x {args.repeats} repeats, group-disjoint on "
          f"{args.group_col} ({n_groups} groups)\n"
          f"head={'logistic (weighted F1)' if args.classify else ('ridge GCV' if ridge else 'plain OLS')}"
          f"{', fit rows matched to the gradient arms (3/5)' if args.match_gradient_rows else ''}  "
          f"probing {len(ckpts)} checkpoint(s)\n"
          + (f"LADDER RUNG n_train={args.n_train} draw={args.draw}\n" if args.n_train else ""))

    # The `--match-gradient-rows` incident (docs/TRAPS.md) was a fit-row flag that
    # silently did nothing. This one cannot: a rung wider than the fit set is a typo,
    # not a full-data run, so it stops rather than quietly returning the full rung.
    if args.n_train is not None:
        widest = max(len(fit_rows(grouped_folds(groups, args.folds, seed=rep), k,
                                  len(X), args.match_gradient_rows))
                     for rep in range(args.repeats) for k in range(args.folds))
        if args.n_train > widest:
            raise SystemExit(f"--n-train {args.n_train} exceeds the widest fit set "
                             f"({widest} rows); omit the flag for the full rung")

    # The baseline: the raw signal IS the design matrix, no encoder in the path.
    # Same folds, same fit, same pooling, so it is comparable cell for cell.
    print("raw baseline (no encoder) ...", flush=True)
    results = {"raw_baseline": cv_scores(
        X, Y, args.target, groups, args.folds, args.repeats, ridge, args.classify,
        args.match_gradient_rows, args.n_train, args.draw)}

    models = {}
    for ckpt_path in ckpts:
        run = ckpt_path.parent.name
        print(f"  {run} ...", end=" ", flush=True)
        model, cfg, epoch = load_encoder(ckpt_path, device)
        if cfg.signal_len != signal_len:
            raise ValueError(f"{run}: signal_len={cfg.signal_len} but the probe grid "
                             f"is {signal_len} — probe one grid at a time (use --filter)")
        H = embed(model, X, device, args.batch_size)
        sc = cv_scores(H, Y, args.target, groups, args.folds, args.repeats, ridge,
                       args.classify, args.match_gradient_rows, args.n_train, args.draw)
        models[run] = {"encoder": cfg.encoder, "loss": cfg.loss,
                       "latent_dim": cfg.latent_dim, "epoch": epoch, "tasks": sc}
        print(" ".join(f"{t.replace('target_vol_', '').replace('target_', '')}="
                       f"{sc[t][f'{metric}_mean']:+.3f}" for t in args.target))
    results["models"] = models
    results["config"] = {
        "registry": str(registry_path.relative_to(REPO_ROOT)), "signal": args.signal,
        "targets": list(args.target), "group_col": args.group_col,
        "folds": args.folds, "repeats": args.repeats, "ridge": ridge,
        "classify": args.classify, "match_gradient_rows": args.match_gradient_rows, "block_on": args.block_on,
        "n_train": args.n_train, "draw": args.draw,
        "n_rows": len(X), "n_groups": int(n_groups), "signal_len": signal_len,
    }

    _print_table(results, args.target, metric)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out}")


def _print_table(results, targets, metric) -> None:
    """Raw baseline vs the CNN and transformer means, one row per target."""
    models = results["models"]
    print(f"\n{'target':>16s} | {'raw':>8s} | {'CNN mean':>9s} {'tf mean':>9s} | "
          f"{'best model':>10s}  {'>raw':>5s}")
    print("-" * 74)
    key = f"{metric}_mean"
    for t in targets:
        raw = results["raw_baseline"][t][key]
        by = {e: [m["tasks"][t][key] for m in models.values() if m["encoder"] == e]
              for e in ("cnn", "transformer")}
        best = max((m["tasks"][t][key], r) for r, m in models.items())
        n_beat = sum(m["tasks"][t][key] > raw for m in models.values())
        short = t.replace("target_vol_", "").replace("target_", "")
        print(f"{short:>16s} | {raw:8.3f} | {np.mean(by['cnn']):9.3f} "
              f"{np.mean(by['transformer']):9.3f} | {best[0]:10.3f}  "
              f"{n_beat:2d}/{len(models)}")
        if metric == "f1":
            print(f"{'':>16s} | modal baseline (pooled) "
                  f"{results['raw_baseline'][t]['modal_f1_mean']:.3f}")

    print(f"\nspread across repeats (max-minus-min pooled {metric}, CNN mean):")
    for t in targets:
        rng = [max(m["tasks"][t][f"{metric}_per_repeat"]) - min(m["tasks"][t][f"{metric}_per_repeat"])
               for m in models.values() if m["encoder"] == "cnn"]
        short = t.replace("target_vol_", "").replace("target_", "")
        print(f"  {short:>16s}  mean {np.mean(rng):.3f}  max {np.max(rng):.3f}")


if __name__ == "__main__":
    main()
