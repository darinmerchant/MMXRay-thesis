"""analysis/finetune_cv.py — the gradient arms of the four-way comparison, cross-validated.

`probe_cv.py` gives two of the four bars for free (raw signal, frozen linear probe),
because a closed-form head is cheap enough to refit on every fold. This module adds
the two that need gradient descent, on the SAME folds so all four are comparable:

    finetuned    `finetune_one(cfg, state, ...)` — pretrained encoder, unfrozen
    supervised   `finetune_one(cfg, None,  ...)` — identical loop, random init

Those two differ in initial weights and in nothing else, which is the whole point of
`finetune.py`'s design and is why the supervised baseline is built from the same
function rather than a separate script.

**`--target` takes one column or several, and several means ONE SHARED HEAD.** This
is the only place in the repo where that flag means "joint" rather than "loop" —
`finetune.py` and `probe_cv.py` both iterate — so the arity is guarded and printed
rather than left to be discovered. One target is the original behaviour, byte for
byte. Several trains a single encoder with a `(T, d)` head against all T targets at
once, which is how "can one model read the whole synthesis recipe?" gets asked.

The cost argument used to run the other way: 5 folds x 3 repeats x 2 arms is 30
trainings for ONE target, so all 8 Au targets one at a time is 240, and this module
used to tell you not to. A shared head covers all 8 in **30**, which is what makes
the joint comparison affordable at all.

**Only the gradient arms can be joint, and that is a fact about least squares, not
a scoping decision.** A closed-form head with a per-column bias decomposes exactly
into T independent per-column fits, so a "joint" raw-signal or frozen-probe arm is
the same T solutions the single-target run already produced. `fit_probe` and
`_fit_ridge_gcv` are therefore deliberately single-RHS and untouched.

**A joint run FANS OUT to one JSON per target**, `f"{--out}{short_name(t)}.json"`,
rather than writing one multi-target file. That keeps the schema identical to a
single-target run, which is what lets `tools/plot_au_four_arms.py` read joint files
with a flag rather than a parser, and `tools/plot_au_parity.py` read them with no
change at all — including its hard check that `r2_per_repeat[0]` is recomputable
from `y_true` and `oof_per_repeat[0]` to 1e-9. `config["joint_targets"]` is the
only discriminator; its presence means "this came from a shared head". Note that
`best_epochs` is IDENTICAL across the fanned-out files, because it is one model —
never average them as if they were independent runs.

**The joint arm's defining handicap is that one stopping epoch serves T targets.**
Measured on Au, repeat 0, the same folds: HAuCl4's best epochs were
`[44, 42, 81, 100, 63]` while LiOH's were `[1, 34, 5, 70, 1]`. The aggregate val
loss is dominated by whichever columns are still improving, so a column that wanted
to stop at epoch 1 does not. This is NOT the val-CE/F1 trap in `docs/TRAPS.md` —
mean standardized val MSE *is* monotone in the mean per-target `1 - R²`, so the
selection rule is not inverted — but it does mean a per-target R² drop has two
possible causes. `finetune_one` records `val_loss_per_target` for exactly this
reason, and it is stored per fold at the best epoch.

**Val comes from the fold structure, not from a carved fraction.** `finetune_one`
early-stops on a validation split, and the Au registry was built with `val_frac=0`.
Rather than re-splitting the registry — which would move rows relative to the probe
and break the comparison this module exists to make — fold `k` is the test set and
fold `k+1 (mod n)` is the validation set, leaving the other `n-2` folds to train on.
Val is then group-disjoint from both train and test by construction, and every arm
sees exactly the rows `probe_cv` scored.

The cost is real and must be stated: training sees ~3/5 of the registry (~55 rows at
n=92) where the probe's closed-form head sees ~4/5 (~74). **The gradient arms are
therefore handicapped by ~25% fewer training rows**, which is a property of needing
early stopping, not of the methods. Read a finetune that merely ties the probe with
that in mind.

**Supervised LR is inherited, not re-selected.** `--lr-enc` and `--lr-head` must be
passed EQUAL for the supervised arm (`finetune.py`'s header: there are no pretrained
features to protect, and the 1e-5 finetuning default would leave the backbone barely
moved, measuring random features plus a linear head). The default here is the 3e-4
that `scripts/supervised_baseline.slurm` documents as the stage-2 choice — selected
on CHILI, NOT on this registry. The supervised arm is the most LR-sensitive of the
four, so treat its bar as "the inherited baseline", not "the best from-scratch model
achievable here". **That equal-LR rule predates LP-FT**: with `--warm-start-head` the
head starts at the closed-form probe and a separate (usually lower) head LR is the
measured RRUFF choice — `--sup-lr-head` overrides the coupling for that case.

Pooled out-of-fold R², PER TARGET, same as `probe_cv` — see that module for why pooling rather
than averaging per-fold R² is what keeps the denominator stable. A target in
`CLASSIFICATION_TARGETS` switches the whole module to pooled out-of-fold **weighted
F1** (cross-entropy head inside `finetune_one`, argmax predictions, the modal-class
baseline recomputed from each fold's TRAIN rows and pooled the same way) — detected
from the target name, as `finetune.py` does, so caller and callee cannot disagree.

**RRUFF additions (s34).** `--warm-start-head` passes LP-FT through — without it a
random head's first gradients undo the pretraining being measured (RESULTS.md Q5,
"Finetuning, done properly"). `--arms` selects which arm(s) run: RRUFF's 4 SSL models
share 3 scratch architectures, so coupled arms would either run a duplicate scratch
control or silently re-key one to two different SSL checkpoints; instead the slurm
stage runs `--arms finetuned` per checkpoint and `--arms supervised` per architecture
(the ckpt then supplies ONLY the architecture config — its weights are never loaded).

**`--lr-enc` accepts a grid (s34), same mechanism as `--sup-lr`.** The first RRUFF CV
run inherited both arms' LRs from the single split's 24-row val — selected on
`n_atoms` (a target the label audit later condemned) under a different train size —
and "the finetune ran a mistuned LR" was the strongest alternative explanation for it
losing to its own frozen probe in 3 of 4 models. Giving several values runs the same
stage-1-style search the supervised arm gets: each scored by mean best VAL loss over
repeat 0's folds on THIS module's target and fold geometry, winner used for the real
runs, held-out folds never read. `lr_enc=0` in the grid keeps "leave the encoder
frozen" reachable as an outcome rather than assumed away (stage-1 precedent; the
BatchNorm caveat there applies here too).

**`--n-train` / `--draw` (2026-08-12, the RRUFF ladder).** Shrink each fold's TRAIN
set to N rows and leave val and test alone, so pooled OOF stays over all N and the
rungs are comparable. The seeding contract lives in `probe_cv.subsample_fit`: the
draw is a function of `(draw, repeat, fold)` only, so the frozen probe, the finetuned
arm and the supervised arm at one rung fit the SAME minerals — the comparison is
between methods at a labelled-set size, not between three different subsets.

**The LR is re-selected per rung, and that is not optional.** `select_lr` runs inside
the rung (repeat 0's folds, subsampled), so each rung picks its own LR. Inheriting the
full-data LR down the ladder would starve whichever arm's optimum moves most with n —
the supervised arm, on the evidence — and this repo has already had one conclusion
reversed by exactly that class of bias (RESULTS.md Q5, the 150- vs 600-epoch control).

Run:
    python analysis/finetune_cv.py --registry data/downstream/au_selfdriving/au_registry.parquet \\
        --ckpt runs/pdf/sweep/2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372/ckpt_best.pt \\
        --target target_vol_HAuCl4 --out runs/pdf/sweep/au_finetune_cv_HAuCl4.json

    # the joint head over all 8: --out is a PREFIX, not a file
    python analysis/finetune_cv.py --registry ... --ckpt ... \\
        --target target_vol_HAuCl4 target_vol_LiCt ... --sup-lr 1e-4 \\
        --out runs/pdf/sweep/au_finetune_cv_joint_
"""

from __future__ import annotations

import torch  # MUST precede pandas / core.* — see downstream_eval's module docstring

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.downstream_eval import CLASSIFICATION_TARGETS, _encode
from analysis.finetune import _load_state, finetune_one
from analysis.probe_cv import blocked_groups, grouped_folds, load_all_rows, subsample_fit
from core.config import REPO_ROOT
from core.registry import MATERIAL_ID

ARMS = ("finetuned", "supervised")


def short_name(target: str) -> str:
    """`target_vol_HAuCl4` -> `HAuCl4`. Lives here rather than in the plotters
    because this module is the one that WRITES the filenames they read."""
    return target.replace("target_vol_", "").replace("target_", "")


def pooled_r2(y: np.ndarray, oof: np.ndarray) -> np.ndarray:
    """Pooled out-of-fold R² per column, for `(n, T)` truth and predictions.

    Column by column, not by an axis reduction over the matrix: the two are the
    same arithmetic, but a per-column loop makes it obvious that the number stored
    beside a fanned-out `oof_per_repeat` was computed from that exact column —
    which is the invariant `tools/plot_au_parity.py` re-derives to 1e-9.
    """
    out = np.empty(y.shape[1])
    for j in range(y.shape[1]):
        t, p = y[:, j], oof[:, j]
        out[j] = 1.0 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    return out


def _predict(model, head, X, device, batch_size, classes) -> np.ndarray:
    """`(n, T)` regression predictions in ORIGINAL units (`finetune_one` folds the
    target standardization back into the head, so no stats need restoring here);
    `(n,)` class CODES via argmax when `classes` is set."""
    model.eval()
    head.eval()
    with torch.no_grad():
        out = torch.cat([
            head(model.encode(X[i:i + batch_size].unsqueeze(1).to(device))).cpu()
            for i in range(0, len(X), batch_size)
        ])
    if classes is not None:
        return out.argmax(dim=1).numpy()
    return out.double().numpy()


def run_arm(cfg, state, classes, X, y, folds, *, device, hp, warm_start_head=False,
            n_train=None, draw=0, rep=0):
    """`(pooled OOF scores, OOF predictions, best epochs, criterion, modal, per_target)`.

    `y` is `(n, T)` and the scores come back as a `(T,)` array — length 1 for a
    single target and for classification, so callers have ONE shape to handle
    rather than a shape that depends on the task. `per_target` is the fold-wise
    per-column val loss at the best epoch (`(folds, T)`, joint runs only, else
    None): the audit for a shared stopping epoch, see the module docstring.

    The score is R² for regression, weighted F1 when `classes` is set; `modal` is
    the pooled modal-class baseline F1 (None for regression) — each fold's held-out
    rows scored as if predicted the modal class of that fold's TRAIN rows, exactly
    the trivial classifier this fold assignment permits.

    `state=None` is the supervised baseline. The fourth return value is what LR
    selection keys on — **never the pooled OOF score**, which is computed on the
    held-out folds and is the number being reported.

    **The selection criterion is metric-matched, and this is load-bearing.** For
    regression it is mean best val loss (MSE), which is monotone in the reported R².
    For classification it is `1 - pooled val F1` — NOT val cross-entropy. Measured on
    this registry (s34, `runs/archive/rruff_cv_valce/`): minimizing val CE selected the WORST
    LR on F1 in all four finetuned arms, corr(val CE, F1) running +0.24 to +0.99.
    The mechanism is visible in the epoch counts — a high LR trains 12-54 epochs and
    converges toward the class prior, which is well-calibrated (low CE) but hedged at
    the argmax (low F1), while a low LR stops at epoch 1-5 keeping the pretrained
    features and commits (higher CE, higher F1). CE rewards hedging; weighted F1
    rewards committing. The single-split stages never hit this because they select on
    `n_atoms`, a regression target where loss and metric agree.

    **Val F1 is POOLED across folds, not averaged.** Fold `k`'s val is fold `k+1`, so
    over the 5 folds the val slices tile the whole registry: 148 val predictions
    rather than five ~29-row samples. Same argument as `probe_cv`'s pooling, and it
    stays leakage-free because fold `k`'s val is disjoint from its own test fold.

    **`n_train`/`draw`/`rep` are the ladder rung** (`probe_cv.subsample_fit`): TRAIN
    shrinks to `n_train` rows, val and test do not move. Val staying full-size is the
    point — early stopping must not get noisier as the rung narrows, or the low rungs
    would measure stopping-rule variance instead of labelled-set size. `rep` only
    seeds the draw, and it must be the same `rep` that built `folds`, or this arm
    trains on different minerals than the probe it is plotted against.
    """
    # One shape for both tasks: regression writes T columns, classification writes
    # its codes into column 0 and scores that column.
    n_out = y.shape[1] if classes is None else 1
    oof = np.full((len(y), n_out), np.nan)
    modal = np.full(len(y), -1)
    val_pred = np.full(len(y), np.nan)
    val_seen = np.zeros(len(y), dtype=bool)
    best_epochs, val_losses, per_target = [], [], []
    n = len(folds)
    # Classification labels are class VALUES, not a target matrix: hand `finetune_one`
    # the single column it encodes. Regression hands over all T at once.
    y_fit = y if classes is None else y[:, 0]
    for k, test_idx in enumerate(folds):
        val_idx = folds[(k + 1) % n]
        train_idx = subsample_fit(
            np.setdiff1d(np.arange(len(y)), np.concatenate([test_idx, val_idx])),
            n_train, draw, rep, k)
        # Seed varies per fold so the supervised arm — whose ENCODER init the seed
        # controls — is not 15 replays of one lucky initialization.
        model, head, log, best = finetune_one(
            cfg, state, classes,
            X[train_idx], y_fit[train_idx], X[val_idx], y_fit[val_idx],
            device=device, seed=hp["seed"] + k, warm_start_head=warm_start_head,
            **{k2: hp[k2] for k2 in
            ("epochs", "lr_enc", "lr_head", "patience", "batch_size")})
        pred = _predict(model, head, X[test_idx], device, hp["batch_size"], classes)
        oof[test_idx] = pred if classes is None else pred[:, None]
        if classes is not None:
            modal[test_idx] = np.bincount(
                _encode(y_fit[train_idx], classes), minlength=len(classes)).argmax()
            # This fold's OWN val rows, from this fold's model — the selection signal.
            val_pred[val_idx] = _predict(model, head, X[val_idx], device,
                                         hp["batch_size"], classes)
            val_seen[val_idx] = True
        best_epochs.append(best)
        val_losses.append(float(log[best - 1]["val_loss"]))
        if "val_loss_per_target" in log[best - 1]:
            per_target.append(log[best - 1]["val_loss_per_target"])
        print(f"    fold {k + 1}/{n}: train={len(train_idx)} val={len(val_idx)} "
              f"test={len(test_idx)}  best epoch {best}/{len(log)} "
              f"(val {val_losses[-1]:.4g})", flush=True)
    if not np.isfinite(oof).all():
        raise ValueError("some rows never landed in a fold")
    if classes is not None:
        from sklearn.metrics import f1_score

        truth = _encode(y_fit, classes)
        codes = oof[:, 0].astype(int)
        kw = dict(labels=list(range(len(classes))), average="weighted", zero_division=0)
        # `1 - F1` so the caller's `min(...)` rule is identical for both branches.
        val_f1 = f1_score(truth[val_seen], val_pred[val_seen].astype(int), **kw)
        return (np.array([f1_score(truth, codes, **kw)]), oof, best_epochs,
                float(1.0 - val_f1), float(f1_score(truth, modal, **kw)), None)
    return (pooled_r2(y.double().numpy(), oof), oof, best_epochs,
            float(np.mean(val_losses)), None, per_target or None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", required=True)
    ap.add_argument("--ckpt", required=True, help="the ckpt_best.pt to finetune FROM")
    ap.add_argument("--signal", default="xpdf")
    ap.add_argument("--target", nargs="+", required=True,
                    help="one target_* column, or SEVERAL for a joint head — one "
                         "shared encoder and a (T, d) head trained on all of them "
                         "at once. Unlike finetune.py and probe_cv.py, several "
                         "targets here mean one model, not a loop.")
    ap.add_argument("--group-col", default="recipe_group")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr-enc", type=float, nargs="+", default=[1e-5],
                    help="finetuned arm ENCODER LR(s). One value uses it directly (the "
                         "Au behavior); several run the same val-loss search --sup-lr "
                         "describes. Head LR stays --lr-head throughout, as in stage 1.")
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--sup-lr", type=float, nargs="+", default=[3e-4],
                    help="supervised arm LR(s), used for BOTH encoder and head unless "
                         "--sup-lr-head. Give SEVERAL to run a stage-1 style search: "
                         "each is scored by mean best VAL loss over repeat 0's folds and "
                         "the winner is used for the real runs. Selecting on val, never "
                         "on the held-out fold.")
    ap.add_argument("--patience", type=int, default=15)
    # NOT finetune.py's 64: that default is sized for CHILI (n=2530), and
    # `finetune_one` drops the last partial batch, so 55 training rows at 64 gives
    # ZERO batches and raises. 16 leaves 3 full batches per epoch at this n.
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--warm-start-head", action="store_true",
                    help="LP-FT: start every head at the closed-form probe (both arms), "
                         "as the RRUFF single-split stages do")
    ap.add_argument("--sup-lr-head", type=float, default=None,
                    help="supervised HEAD LR; default couples it to the selected sup LR "
                         "(the pre-LP-FT rule). RRUFF's scratch stage fixed 1e-4 — pass it.")
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS),
                    help="which arm(s) to run; see module docstring for why RRUFF "
                         "decouples them")
    ap.add_argument("--block-on", default=None, metavar="COLUMN",
                    help="robustness arm, mirroring probe_cv's flag: folds become "
                         "`--folds` CONTIGUOUS blocks of this registry column "
                         "(e.g. `iteration`), assigned per --group-col group. "
                         "Deterministic, so requires --repeats 1.")
    ap.add_argument("--n-train", type=int, default=None, metavar="N",
                    help="LADDER RUNG: train each fold on only N of its train rows. "
                         "Val and test are untouched. Must match the --n-train of the "
                         "probe_cv --match-gradient-rows run it is plotted against; "
                         "on RRUFF one row is one mineral. See probe_cv.subsample_fit.")
    ap.add_argument("--draw", type=int, default=0, metavar="D",
                    help="which subsample draw at this rung (0, 1, 2, ...). Must match "
                         "the probe's --draw: together (rung, draw) fix the minerals "
                         "every arm sees.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True,
                    help="one target: the .json to write. SEVERAL targets: a PREFIX, "
                         "since a joint run fans out to one file per target "
                         "(e.g. runs/pdf/sweep/au_finetune_cv_joint_)")
    args = ap.parse_args()

    from core.train import pick_device

    cfg, state, epoch = _load_state(Path(args.ckpt))
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = REPO_ROOT / registry_path

    joint = len(args.target) > 1
    # `--out` carries the arity, so a joint run cannot quietly overwrite a
    # single-target file (or vice versa) by inheriting a stale command line.
    if joint and args.out.endswith(".json"):
        raise SystemExit(f"{len(args.target)} targets means ONE joint head fanning out "
                         f"to one file per target, so --out is a PREFIX, not a .json "
                         f"(e.g. runs/pdf/sweep/au_finetune_cv_joint_)")
    if not joint and not args.out.endswith(".json"):
        raise SystemExit("one target writes one file: --out must end in .json")

    X, Y, ids, groups, r_range = load_all_rows(
        registry_path, args.signal, args.target, cfg.signal_len, args.group_col)
    y = torch.stack([Y[t] for t in args.target], dim=1)      # (n, T)
    bad = [args.target[j] for j in range(y.shape[1]) if not torch.isfinite(y[:, j]).all()]
    if bad:
        raise ValueError(f"{bad} has non-finite values; this module has no mask path")
    if args.block_on:
        if args.repeats != 1:
            raise SystemExit("--block-on makes the fold assignment deterministic; "
                             "pass --repeats 1 (extra repeats would be identical)")
        frame = pd.read_parquet(registry_path, columns=[MATERIAL_ID, args.block_on])
        lookup = dict(zip(frame[MATERIAL_ID], frame[args.block_on]))
        values = np.array([float(lookup[i]) for i in ids])
        groups = blocked_groups(values, groups, args.folds)
        print(f"blocked on {args.block_on}: rows per block "
              f"{pd.Series(groups).value_counts().sort_index().to_dict()}\n")
    if args.draw and args.n_train is None:
        raise SystemExit("--draw only means something with --n-train; the full rung "
                         "has exactly one draw")
    if args.n_train is not None:
        # Same guard as probe_cv's: a rung wider than the train set is a typo, and a
        # fit-row flag that silently no-ops has already cost this repo one result
        # (docs/TRAPS.md, --match-gradient-rows).
        widest = max(len(np.setdiff1d(np.arange(len(y)),
                                      np.concatenate([f[k], f[(k + 1) % args.folds]])))
                     for rep in range(args.repeats)
                     for f in [grouped_folds(groups, args.folds, seed=rep)]
                     for k in range(args.folds))
        if args.n_train > widest:
            raise SystemExit(f"--n-train {args.n_train} exceeds the widest train set "
                             f"({widest} rows); omit the flag for the full rung")
    kinds = [t in CLASSIFICATION_TARGETS for t in args.target]
    if joint and any(kinds):
        raise SystemExit(f"a joint head over classification targets is not built; "
                         f"{[t for t, k in zip(args.target, kinds) if k]} is classification")
    classes = sorted(set(y[:, 0].tolist())) if kinds[0] else None
    metric = "f1" if classes is not None else "r2"
    sup_head = "sup-lr-head" if args.sup_lr_head is not None else "sup lr (coupled)"

    device = pick_device(args.device)
    joint_line = (f"JOINT head over {len(args.target)} targets — ONE model per fold, "
                  f"not {len(args.target)} runs; fanning out to "
                  f"{args.out}<target>.json\n" if joint else "")
    print(f"device={device} registry={registry_path.name} target={args.target}"
          f"{f' ({len(classes)} classes, weighted F1)' if classes else ''}\n"
          f"{joint_line}"
          f"ckpt={Path(args.ckpt).parent.name} (epoch {epoch}, {cfg.encoder}/{cfg.loss})"
          f"{' — config only, weights unused' if args.arms == ['supervised'] else ''}\n"
          f"n={len(X)} rows, {len(np.unique(groups))} groups, "
          f"{args.folds}-fold x {args.repeats} repeats, arms={args.arms}, "
          f"warm_start_head={args.warm_start_head}\n"
          f"finetuned: lr_enc in {[f'{v:g}' for v in args.lr_enc]} lr_head={args.lr_head:g}   "
          f"supervised: lr in {[f'{v:g}' for v in args.sup_lr]}, head={sup_head}\n"
          + (f"LADDER RUNG n_train={args.n_train} draw={args.draw} "
             f"(val and test unchanged)\n" if args.n_train else ""))

    # Stage-1 style LR selection, per arm, by mean best val loss over repeat 0's
    # folds. Originally supervised-only (the untuned finetuned arm biases AGAINST
    # pretraining — the safe direction); the finetuned grid was added for RRUFF,
    # where the inherited LR was itself the leading suspect (module docstring).
    # `1 - pooled val F1` for classification, mean best val loss for regression —
    # both minimized, both computed on val folds only (see run_arm's docstring for
    # why CE is the wrong criterion on a classification target).
    crit_name = "1-val_f1" if classes is not None else "val_loss"

    def select_lr(arm, grid, arm_state, head_lr_of):
        search = {}
        print(f"selecting {arm} LR from {[f'{v:g}' for v in grid]} by {crit_name} "
              f"over repeat 0's folds\n", flush=True)
        folds0 = grouped_folds(groups, args.folds, seed=0)
        for lr in grid:
            hp = dict(epochs=args.epochs, patience=args.patience,
                      batch_size=args.batch_size, seed=args.seed,
                      lr_enc=lr, lr_head=head_lr_of(lr))
            print(f"  {arm} lr_enc={lr:g}", flush=True)
            sc, _, bests, crit, _, _ = run_arm(cfg, arm_state, classes, X, y, folds0,
                                               device=device, hp=hp,
                                               warm_start_head=args.warm_start_head,
                                               n_train=args.n_train, draw=args.draw,
                                               rep=0)
            # The stored score is the per-target list; the printed one is its mean,
            # which for a single target is the same number it always was.
            search[f"{lr:g}"] = {crit_name: crit, metric: [float(v) for v in sc],
                                 "best_epochs": bests}
            print(f"    -> {crit_name} {crit:.4f}  (pooled OOF {metric} "
                  f"{np.mean(sc):+.4f}, NOT the selection criterion)", flush=True)
        best = float(min(search, key=lambda k: search[k][crit_name]))
        print(f"\nselected {arm} lr = {best:g}\n", flush=True)
        return best, search

    ft_lr, ft_lr_search = args.lr_enc[0], {}
    if len(args.lr_enc) > 1 and "finetuned" in args.arms:
        ft_lr, ft_lr_search = select_lr("finetuned", args.lr_enc, state,
                                        lambda lr: args.lr_head)
    sup_lr, lr_search = args.sup_lr[0], {}
    if len(args.sup_lr) > 1 and "supervised" in args.arms:
        sup_lr, lr_search = select_lr(
            "supervised", args.sup_lr, None,
            lambda lr: lr if args.sup_lr_head is None else args.sup_lr_head)

    # Everything below is per (arm, repeat) and holds the FULL (n, T) matrices; the
    # per-target slicing happens once, at write time, so each fanned-out file's R²
    # is computed from the very column stored beside it.
    scores = {arm: [] for arm in args.arms}          # repeat -> (T,) pooled score
    oofs = {arm: [] for arm in args.arms}            # repeat -> (n, T) predictions
    epochs_of = {arm: [] for arm in args.arms}
    val_of = {arm: [] for arm in args.arms}          # repeat -> (folds, T) or None
    modal_per_repeat = []
    for rep in range(args.repeats):
        folds = grouped_folds(groups, args.folds, seed=rep)
        for arm in args.arms:
            lr = ({"lr_enc": ft_lr, "lr_head": args.lr_head} if arm == "finetuned"
                  else {"lr_enc": sup_lr,
                        "lr_head": sup_lr if args.sup_lr_head is None else args.sup_lr_head})
            hp = dict(epochs=args.epochs, patience=args.patience,
                      batch_size=args.batch_size, seed=args.seed + 100 * rep, **lr)
            print(f"  repeat {rep + 1}/{args.repeats}  arm={arm}", flush=True)
            sc, oof, bests, _crit, modal, per_t = run_arm(
                cfg, state if arm == "finetuned" else None, classes,
                X, y, folds, device=device, hp=hp,
                warm_start_head=args.warm_start_head,
                n_train=args.n_train, draw=args.draw, rep=rep)
            scores[arm].append(sc)
            oofs[arm].append(oof)
            epochs_of[arm].append(bests)
            val_of[arm].append(per_t)
            if modal is not None and len(modal_per_repeat) <= rep:
                modal_per_repeat.append(modal)
            print(f"    -> pooled OOF {metric} = "
                  + "  ".join(f"{short_name(t)} {v:+.4f}"
                              for t, v in zip(args.target, sc)), flush=True)

    for arm in args.arms:
        per_target_mean = np.mean(np.vstack(scores[arm]), axis=0)
        for j, t in enumerate(args.target):
            v = [float(s[j]) for s in scores[arm]]
            print(f"\n{arm:>11s} {short_name(t):>12s}: pooled OOF {metric} "
                  f"{per_target_mean[j]:+.4f}  per repeat "
                  f"{' '.join(f'{x:+.3f}' for x in v)}")
    if classes is not None and modal_per_repeat:
        print(f"{'modal':>11s}: pooled OOF f1 {np.mean(modal_per_repeat):+.4f}")

    # One file per target, joint or not, so the schema is the same either way and
    # the plotters need no parser. `oof_per_repeat` keeps every row's held-out
    # prediction (aligned with the top-level `ids`/`y_true`), so parity plots need
    # no re-run of the finetunes.
    for j, target in enumerate(args.target):
        results = {}
        for arm in args.arms:
            results[arm] = {
                f"{metric}_per_repeat": [float(s[j]) for s in scores[arm]],
                "best_epochs": epochs_of[arm],
                "oof_per_repeat": [[float(v) for v in o[:, j]] for o in oofs[arm]],
                f"{metric}_mean": float(np.mean([s[j] for s in scores[arm]])),
            }
            if joint:
                # This column's share of the shared stopping decision, per fold.
                results[arm]["val_loss_per_repeat"] = [
                    [float(f[j]) for f in rep] for rep in val_of[arm]]
        if classes is not None and modal_per_repeat:
            results["modal_f1_per_repeat"] = modal_per_repeat
        results["ids"] = list(ids)
        results["y_true"] = [float(v) for v in y[:, j]]
        results["config"] = {
            "registry": str(registry_path.relative_to(REPO_ROOT)), "signal": args.signal,
            "target": target, "ckpt": str(Path(args.ckpt).parent.name),
            "group_col": args.group_col, "folds": args.folds, "repeats": args.repeats,
            "epochs": args.epochs, "lr_enc": ft_lr, "lr_enc_grid": args.lr_enc,
            "lr_enc_search": ft_lr_search, "lr_head": args.lr_head,
            "sup_lr": sup_lr, "sup_lr_grid": args.sup_lr, "sup_lr_search": lr_search,
            "sup_lr_head": args.sup_lr_head, "arms": list(args.arms),
            "selection_criterion": crit_name,
            "warm_start_head": args.warm_start_head, "block_on": args.block_on,
            "classes": classes,
            "patience": args.patience, "n_rows": len(X), "batch_size": args.batch_size,
            "n_train": args.n_train, "draw": args.draw,
        }
        if joint:
            # The ONLY marker that this came from a shared head. `best_epochs` is
            # identical across these files because it is one model — never average
            # them as if they were independent runs.
            results["config"]["joint_targets"] = list(args.target)
        out = Path(args.out if not joint else f"{args.out}{short_name(target)}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
