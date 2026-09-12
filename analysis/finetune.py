"""analysis/finetune.py — gradient finetuning of pretrained encoders on a downstream registry.

    python -m analysis.finetune --runs runs/pdf/sweep/ --filter 'cnn_*' --target target_np_size

The counterpart to `analysis/downstream_eval.py`. Where the probe FREEZES the
encoder and fits a closed-form linear head, this UNFREEZES it and trains encoder
+ linear head end to end with Adam, early-stopping on the `val` split the probe
deliberately leaves untouched. Same registry, same targets, same protocols, same
metrics — `finetune.json` and `probe.json` are comparable cell for cell, which is
the whole point: the gap between them is what finetuning buys.

Everything about the data is inherited from `downstream_eval` rather than
restated — the target list, which targets are classification, the protocol table,
the registry read path, the class-code encoding, the sklearn thread guard. A
schema change there must reach here, and importing is what guarantees it does.

TASK KIND is one dispatch, not a fork: `classes is None` selects a 1-unit head +
MSE + R²/MAE, otherwise a `len(classes)`-unit head + cross-entropy + the F1 set.
The training loop is byte-identical for both.

PROTOCOLS are `downstream_eval`'s, with one structural difference. A probe fits
two cheap heads on ONE frozen embedding pass, so `clean_clean` and `clean_aug`
cost nothing extra. A finetune MUTATES the encoder, so each fit channel needs its
OWN full finetune from the pretrained weights:

  clean_clean  finetune on clean train  -> score on clean test
  clean_aug    the SAME clean-finetuned model -> scored on AUGMENTED test
  aug_aug      a SECOND, independent finetune on the augmented channel

With no augmented channel in the registry only `clean_clean` runs, and the output
shape is unchanged.

R²/MAE are computed in torch here, reproducing `downstream_eval.score`'s
arithmetic rather than calling `sklearn.metrics.r2_score`: identical footing for
the comparison, and one less multi-threaded sklearn call in a process that also
holds torch (see `_single_threaded`). F1 does need sklearn, and is wrapped.

REGRESSION TARGETS ARE STANDARDIZED for the loss (train-split stats), and the
normalization is folded back into the head so everything reported stays in
original units — see `finetune_one`. This is the one place the finetune must
diverge from the probe: OLS is equivariant to an affine target map and gradient
descent is not, so the target's physical scale silently sets what `lr_head`
means. It is unconditional rather than a flag — there is no result in this repo
that depends on the unstandardized behaviour, so a flag would be a branch with no
caller. Classification is unaffected.

SUPERVISED MODE (`--config <pretrain yaml>`) is the from-scratch baseline: the
same loop, the same data, the same metrics, but the encoder starts at its random
init instead of a checkpoint's weights. The flag IS the mode — a YAML carries no
weights, so a separate `--random_init` would be a second name for one branch. Only
the ARCHITECTURE fields of the config are read (`encoder`, `latent_dim`,
`d_model`, `signal_len`); `loss`/`temperature`/`cov_weight`/`bank` describe a
pretraining run that is not happening here and are recorded as null.

Results go to `runs/pdf/supervised/<config-stem>_lr<lr>_seed<seed>/`, NOT `runs/pdf/sweep/`, and
that is load-bearing rather than tidiness: `finetune_curves.load_runs` derives each
bar's sweep-knob as `temperature if loss == "infonce" else cov_weight` and sorts
on it, so a supervised run sitting in `runs/pdf/sweep/` would render as a bar labelled with
the pretraining hyperparameter it inherited from whichever YAML supplied its
architecture — a sweep point it is not. A separate tree also leaves
`find_checkpoints` alone. Names are deterministic, so a re-run overwrites rather
than accumulating near-duplicate baselines.

The name keys on the CONFIG STEM, not on `cfg.encoder`: two configs can share an
encoder and differ in architecture (`transformer_infonce_mpfull_final` is 4 layers
/ patch 50 / Post-LN, `transformer_vicreg_mpfull_final` is 8 / 200 / Pre-LN, and
both report `encoder="transformer"`), so an encoder-keyed name would let the second
silently overwrite the first at the same `(lr, seed)`. Overwriting is correct for a
RE-RUN of one cell and wrong for a different architecture. Cells produced before
this change carry the old short `<encoder>_...` name; nothing reads the directory
name, so the two conventions coexist harmlessly until those cells are re-run.

`lr_enc` and `lr_head` should be passed EQUAL in this mode. Their split exists to
protect pretrained features with a slower backbone LR; from scratch there are none
to protect, and the module's 1e-5 default is a finetuning LR that would leave the
backbone barely moved — which measures random features plus a linear head, not a
supervised baseline. No flag enforces this: it is the caller's choice, and
`scripts/supervised_baseline.slurm` makes it.

LABELED-DATA-SIZE SWEEP (`--n-train-frac`) fits a target on a seeded FRACTION of
the train split's labelled rows instead of all of them — val/test never move, so
the sweep changes what is fit on, not what is scored on. The draw
(`subsample_train_mask`) depends only on `(mask, frac, seed)`, never on which
checkpoint is being finetuned, so every arm — every SSL checkpoint and the
from-scratch baseline alike — trains on the EXACT SAME rows at a given
`(fraction, seed)` point. That is what keeps a data-efficiency curve comparable
point-for-point rather than confounding "less data" with "a different sample of
data per arm." Results at `frac < 1.0` are redirected to `runs/pdf/datafrac/`, never
`runs/pdf/sweep/` or `runs/pdf/supervised/` — same reasoning as SUPERVISED MODE's separate tree:
a reduced-data point must not collide with, or be mistaken for, the full-data
result at the same run/config. `frac=1.0` (the default) is untouched: same output
path and same rows as if the flag were never passed, so every existing full-data
result stays exactly reproducible.

CHEMISTRY AXIS (`--fit-registry`) fits on ONE registry and scores on ANOTHER. The
protocol table above varies the INSTRUMENT with chemistry held fixed — clean and
augmented are two channels of the same CHILI-3K rows. This flag adds the missing
axis: point it at the CHILI-100K metal-oxide registry and the fit set changes
chemistry while evaluation stays pinned to the same CHILI-3K test rows, in both
channels:

  cross_clean  finetune on the FIT registry's clean train -> score on eval clean test
  cross_aug    the SAME model                             -> score on eval AUG test

so the row effect (in-registry vs cross) is the chemistry shift, the column effect
(clean vs aug) is still exactly the instrument shift, and the interaction is
whether a chemistry shift changes what the instrument shift costs.

**The two registries are NOT row-aligned and must not be assumed to be.** With one
registry, targets and the per-target NaN mask are read from the clean channel and
reused positionally for the augmented one — legal only because the alignment
assertion holds. Across registries there is no such correspondence, so the fit side
carries its own mask and its own `y`, and only the EVAL side's mask indexes the test
tensors. The within-registry assertion still runs on the eval registry, because
that is what makes clean and aug comparable.

**Classification is refused in this mode**, rather than silently mis-scored. The
class vocabulary would have to span two registries whose label sets do not agree
(CHILI-3K's `target_oxidation` takes 5 values locked to its 12 prototypes; COD
binary oxides reach stoichiometries those prototypes cannot produce), leaving the
head with units it never trains on. Regression targets have no such problem.

`--n-train-frac` is the tool for matching `n_train` across the two fit sets — a
chemistry effect measured against a differently-sized fit set is confounded with a
data-volume effect. Pass the fraction that lands the fit registry on the same
labelled train count the in-registry arm used.

LR SEARCH (`--out-root`) writes each run's `finetune.json` to
`<out-root>/<run>_enc<lr_enc>_head<lr_head>/` instead of into the run dir. Built
for precondition 2 — giving the 22 SSL finetunes the same 5-point val-selected LR
search the scratch arm got — where the same checkpoint is finetuned at several
LRs and every result would otherwise land on one path, last writer winning.

Same reasoning as SUPERVISED MODE's and LABELED-DATA-SIZE's separate trees, with
one addition: `runs/pdf/sweep/*/finetune.json` is the epoch-matched 300-epoch grid that
`PROGRESS.md` item 6's tables and `analysis/grid2x2.py` are read off, so an LR
search that wrote in place would destroy the numbers it exists to be compared
against. BOTH LRs are in the directory name because the search carries one
DECOUPLED reference cell (the incumbent `1e-5/1e-4`) alongside the coupled grid —
naming on `lr_enc` alone would let a reader assume a cell was coupled when it was
not. The redirect is on the flag, not on a value comparison, so `--out-root` with
the incumbent LRs still lands in the new tree rather than silently in `runs/pdf/sweep/`.

`--out-root` COMPOSES with `--n-train-frac` rather than conflicting with it, and
the composition is the tuned labeled-data-size sweep: one checkpoint, at its
selected LR, at one fraction. The directory then carries both
(`<run>_enc<lr>_head<lr>_frac<f>_seed<s>`), because either alone would let two
cells of that sweep collide on one path.

REPRESENTATION-VS-READOUT (`--refit-head-aug`) answers where the transferability
damage lives. `clean_aug` says a clean-finetuned model fails on shifted data, but
it is an END-TO-END score with the encoder AND the head both fitted on clean, so
it cannot say which of them broke. This flag takes the finished clean-finetuned
model, FREEZES the encoder, fits a fresh linear head on the AUGMENTED train
split, and scores augmented test -> `tasks[<target>]["cleanfit_augrefit"]`.

  recovers to the pretrained encoder's probe score  -> the representation
      survived; the clean-fitted READOUT was the casualty.
  stays down near `clean_aug`                       -> the REPRESENTATION itself
      was moved somewhere the augmented domain no longer reaches.

The head is fitted with `downstream_eval`'s OWN probe functions (`fit_probe` /
`fit_classifier`), not a fresh implementation, because the number it produces is
only meaningful against the pretrained-encoder baseline already sitting in
`probe.json` — same closed-form OLS, same standardized logistic head, same
`score` arithmetic. A reimplementation here would make the two incomparable in
exactly the way that would not be visible in the output.

It reuses the model already in memory rather than reloading a saved checkpoint,
so the extra cost is two embedding passes and a linear fit — seconds on top of a
finetune that took minutes.

DELIBERATELY NOT PORTED from `PDF/src/finetune_downstream.py` (named so a later
session knows it was a decision, not an oversight): wandb logging; matplotlib
parity plots; `--train_data augmented2x` (two augmentation draws concatenated);
`--warm_start_head` (LP-FT); and `--eval_only` reload of a finetuned checkpoint.
None has a caller in this repo.

`import torch` comes first on purpose: importing pandas or `core.*` ahead of it
segfaults with no traceback (exit 139) — see docs/ENVIRONMENT.md.
"""

from __future__ import annotations

import torch  # MUST precede pandas / core.* — see module docstring

import argparse
import copy
import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from analysis.downstream_eval import (
    CLASSIFICATION_TARGETS, CROSS_PROTOCOLS, DEFAULT_AUG_SIGNAL, DEFAULT_REGISTRY,
    DEFAULT_SIGNAL, DEFAULT_TARGETS, PROTOCOLS, _encode, _single_threaded, embed,
    find_checkpoints, fit_classifier, fit_probe, load_probe_data, score,
    score_classifier,
)
from core.config import REPO_ROOT, TrainConfig, load_config
from core.train import build_model, pick_device, set_seed

__all__ = ["head_dim", "finetune_one", "evaluate"]

SPLITS = ("train", "val", "test")
#: Where `--config` (supervised-from-scratch) writes. Deliberately not `runs/pdf/sweep/` —
#: see the module docstring's SUPERVISED MODE note.
SUPERVISED_DIR = "runs/pdf/supervised"
#: Where `--n-train-frac < 1.0` (labeled-data-size sweep) writes. Deliberately not
#: `runs/pdf/sweep/` or `runs/pdf/supervised/` — see the module docstring's LABELED-DATA-SIZE note.
DATAFRAC_DIR = "runs/pdf/datafrac"


def subsample_train_mask(mask: torch.Tensor, frac: float, seed: int,
                         groups: np.ndarray | None = None) -> torch.Tensor:
    """Keep a seeded `frac` of `mask`'s True (labelled) rows; the rest become False.

    Call this on the TRAIN split's per-target mask only. `frac >= 1.0` returns
    `mask` unchanged (including the same tensor object) so the `frac=1.0` default
    is not just numerically a no-op but literally the untouched code path.

    **`groups` makes the axis LABELLED THINGS rather than labelled ROWS, and on
    CHILI-100K that distinction is the whole experiment.** There, five rows are five
    particle sizes of ONE crystal sharing ONE `crystal_system`; a row-level draw
    therefore moves "views per crystal" while a plot captioned "labelled set size"
    implies otherwise — the same objection that rejected augmenting RRUFF to
    manufacture labelled rows (`scripts/xrd_rruff_ladder.slurm`). With `groups`, a
    seeded `frac` of the DISTINCT groups is drawn and **every row of a chosen group is
    kept**: labelling a crystal really does hand you all of its particles for free, so
    row count stays a consequence of the axis rather than being the axis.

    `groups` is positional over the full mask (one entry per row, train-split order).
    `groups=None` is the original row-level draw, byte-identical — every existing
    `runs/pdf/datafrac/` cell stays reproducible.
    """
    if frac >= 1.0:
        return mask
    idx = mask.nonzero(as_tuple=True)[0]
    rng = np.random.default_rng(seed)
    if groups is None:
        keep = max(1, round(frac * len(idx)))
        chosen = idx[torch.from_numpy(rng.permutation(len(idx))[:keep])]
    else:
        g = np.asarray(groups)[idx.numpy()]
        uniq = np.unique(g)                       # sorted, so the draw is deterministic
        n_keep = max(1, round(frac * len(uniq)))
        picked = set(uniq[rng.permutation(len(uniq))[:n_keep]].tolist())
        chosen = idx[torch.from_numpy(
            np.nonzero(np.fromiter((v in picked for v in g), bool, len(g)))[0])]
    out = torch.zeros_like(mask)
    out[chosen] = True
    return out


def head_dim(cfg: TrainConfig) -> int:
    """Width of `model.encode()`'s output — the linear head's input dim.

    These are DIFFERENT config fields per encoder: `CNNEncoder.encode` pools to
    `latent_dim`, `Transformer.encode` reads the CLS token at `d_model`. They
    happen to share a default of 256, so a mixup builds a mis-sized head silently
    on the default configs and only blows up on a sweep that moves one of them.
    """
    if cfg.encoder == "cnn":
        return cfg.latent_dim
    if cfg.encoder == "transformer":
        return cfg.d_model
    raise ValueError(f"unknown encoder {cfg.encoder!r}; want 'cnn' or 'transformer'")


def _load_state(ckpt_path):
    """`(cfg, model_state, epoch)` from a `ckpt_best.pt` — weights, not a model.

    `downstream_eval.load_encoder` is the probe's version and permanently sets
    `requires_grad_(False)` on everything it returns, which a finetune would have
    to undo. Returning the raw state instead also lets each (target, channel) get
    its own FRESH `build_model(cfg)`, so no finetune inherits another's weights.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    names = {f.name for f in fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items() if k in names})
    return cfg, ckpt["model_state"], int(ckpt.get("epoch", -1))



def _warm_start(model, head, X_tr, t_tr, y_tr, classes, device, batch_size):
    """LP-FT: initialise the head to the closed-form probe on the FROZEN encoder.

    Ported from the reference `PDF/src/finetune_downstream.py`'s `--warm_start_head`,
    which this module's docstring recorded as deliberately not ported for want of a
    caller. RRUFF is the caller: at n_train = 94 a randomly initialised head emits
    large, meaningless gradients for the first epochs, and every one of them lands on
    the encoder, distorting a representation that pretraining paid for — the standard
    LP-FT feature-distortion argument, and it bites hardest exactly when n is small.
    Starting from the probe solution means the first encoder gradient it ever sees is
    already the one that best linear readout would ask for.

    Regression warm-starts against the STANDARDIZED target `t_tr`, because that is
    what the head optimises during training; `finetune_one` folds the scale back out
    afterwards. Classification composes the scaler and the logistic fit into the one
    affine map the `nn.Linear` head can hold:

        logits = A ((h - mu) / sd) + c  =  (A / sd) h + (c - (A / sd) mu)
    """
    # CALIBRATE BATCHNORM FIRST, or this whole function is a trap on a random init.
    # The probe is fitted on `eval()` embeddings, which use BatchNorm's RUNNING
    # statistics, while training consumes `train()` embeddings, which use BATCH
    # statistics. A pretrained encoder's running stats were fitted during pretraining
    # so the two agree; a randomly initialised one still carries mean 0 / var 1, and
    # they do not. Measured on RRUFF/xpdf, mean |h|:
    #
    #     cnn, random init          eval 0.0036  train 0.5374   ratio 0.01  <-- 148x
    #     cnn, pretrained           eval 0.3105  train 0.3119   ratio 1.00
    #     transformer, random init  eval 0.7843  train 0.7811   ratio 1.00  (LayerNorm)
    #
    # So the probe fits features ~148x smaller than the ones the head then receives,
    # compensates with enormous weights, and the run explodes: val loss 7404 at epoch 1
    # against 1.02 without LP-FT, test R2 -3987. Only BatchNorm + random init is
    # affected, which is why it surfaced on exactly one of the four scratch cells.
    #
    # A few forward passes in train() mode move the running stats onto the real input
    # distribution (default momentum 0.1, so ~30 updates gets ~96% of the way). This is
    # a no-op for LayerNorm models, and for pretrained ones it re-estimates BN on the
    # DOWNSTREAM domain rather than the pretraining one — a change, but the defensible
    # direction, and it keeps both arms on one code path.
    model.train()
    with torch.no_grad():
        for _ in range(5):
            for i in range(0, len(X_tr), batch_size):
                chunk = X_tr[i:i + batch_size]
                if len(chunk) > 1:            # BatchNorm errors on a size-1 batch
                    model.encode(chunk.unsqueeze(1).to(device))

    model.eval()
    with torch.no_grad():
        H = embed(model, X_tr, device, batch_size).double()
        if classes is None:
            # `.reshape(-1)`: the regression target is a (N, 1) COLUMN since joint
            # heads arrived, and `fit_probe` does its own `unsqueeze(1)`. Joint
            # heads never reach here — `finetune_one` refuses warm_start_head for
            # them, because a closed-form joint head is separable and unbuilt.
            W = fit_probe(H, t_tr.reshape(-1).double(), ridge=True)  # (d+1, 1), bias last
            head.weight.copy_(W[:-1].T.to(head.weight.dtype).to(device))
            head.bias.copy_(W[-1].to(head.bias.dtype).to(device))
            return
        clf, scaler, _, _ = fit_classifier(H.float(), y_tr, classes)
        A = np.asarray(clf.coef_) / np.asarray(scaler.scale_)          # (k, d)
        c = np.asarray(clf.intercept_) - A @ np.asarray(scaler.mean_)  # (k,)
        if A.shape[0] != head.out_features:
            # sklearn emits a single row for a 2-class fit; the head has one unit per
            # class in the shared vocabulary. Leave the random init rather than
            # silently mis-map rows onto classes.
            return
        head.weight.copy_(torch.from_numpy(A).to(head.weight.dtype).to(device))
        head.bias.copy_(torch.from_numpy(c).to(head.bias.dtype).to(device))


def finetune_one(cfg, state, classes, X_tr, y_tr, X_val, y_val, *, device, epochs,
                 lr_enc, lr_head, patience, batch_size, seed, warm_start_head=False):
    """Finetune one (encoder + linear head). Returns `(model, head, log, best_epoch)`.

    `classes is None` -> regression (one head unit per target column, MSE);
    otherwise classification (`len(classes)` units, cross-entropy on class codes).
    That is the only branch — the loop below is the same for both.

    **`y_tr` may be `(N,)` or `(N, T)`.** A 2-D target is a JOINT head: ONE encoder
    and one `(T, d)` linear map trained against all T targets at once, which is
    what `finetune_cv` uses to ask whether a single model can read a whole
    synthesis recipe rather than one parameter of it. There is no `joint` branch to
    read, and that is deliberate — single-target regression IS the `T=1` case once
    `y` is viewed as a matrix, so the module got SMALLER when joint arrived (the
    old `pred.squeeze(1)` went away and nothing replaced it). A fork would have
    duplicated the training loop, the val loop, the best-state restore and the
    divergence guard, and an early-stopping fix would then reach one copy.

    The loss keeps `MSELoss`'s default `reduction="mean"`, which averages over
    `B*T`, so a joint val loss sits at the same ~1.0 scale as every single-target
    `log` in `runs/pdf/sweep/` and `finetune_cv`'s LR selection stays comparable across the
    two. The trade is real and one-sided by choice: head row *t* then sees 1/T of
    the gradient a single-target head would, while the ENCODER — the shared
    representation actually under test — keeps a single-target-comparable gradient,
    so `lr_enc` keeps its meaning. If the head is undertrained the symptom shows in
    `best_epoch` pinned at `epochs` on most folds, and the answer is a larger
    `lr_head`, not an auto-scaler nobody asked for.

    A joint head REFUSES three things up front, because each would otherwise fail
    far from its cause: classification (cross-entropy over T independent label sets
    is a different model, with no caller), `warm_start_head` (`fit_probe` is
    single-RHS on purpose — a joint closed-form head is separable and so was never
    built), and a non-finite target. That last one matters most: T columns feed ONE
    loss, so a single NaN makes every epoch fail `val_loss < best_loss` and the run
    dies 70 lines below at the divergence raise, blaming the learning rate.

    `state is None` is the SUPERVISED-FROM-SCRATCH baseline: the encoder keeps the
    random init `build_model` gave it and everything else about the loop is
    unchanged. That is the entire difference, which is the point — the baseline and
    the finetunes it is compared against differ in initial weights and in nothing
    else.

    `set_seed(seed)` fires immediately before the model and head are built, so a
    finetune is reproducible independent of its position in the sweep: running one
    (target, channel) alone gives the same numbers as running it inside the full
    grid. Pretrained weights overwrite the encoder's random init, so with `state`
    the seed is really only felt by the head init and the shuffle order — but with
    `state=None` it also sets the ENCODER's init, which is why the baseline is far
    more seed-sensitive than a finetune and must be run at several seeds.

    Two param groups: the encoder backbone at `lr_enc`, the head at `lr_head`.
    `proj_head.*` is EXCLUDED — `.encode()` never calls it, so those parameters
    receive no gradient and handing them to Adam would only be misleading.

    Early stopping restores the BEST-val state, not the last: both state dicts are
    deep-copied on every improvement and loaded back before returning. Without
    that, `patience` epochs of degradation are what gets scored.

    REGRESSION TARGETS ARE STANDARDIZED for the loss, PER COLUMN, on the TRAIN
    split's mean and std (never val — that would leak). Unlike the probe's
    closed-form OLS, which is equivariant to an affine target map, gradient descent
    is not: the target's physical scale sets the loss scale and therefore what
    `lr_head` means. On `target_np_size` (nanoparticle size in A, ~7-57, mean 33) an
    unstandardized head starts with an MSE near 1100 and spends its first epochs
    learning the bias. Standardizing puts every target's loss near 1.0 at init, so
    one `lr_head` is defensible across targets whose units differ.

    **Per column, not globally, and on a joint head that is the difference between
    a measurement and a fiction.** The Au registry's eight targets span an sd of
    137.6 (`target_mixing_speed`) to 0.095 (`target_vol_PVP`) — 1,450x. A scalar
    `float(y_tr.std())` over that matrix returns ~130, which scales the six volume
    columns down to sd ~7e-4; they then contribute ~5e-7 of the MSE and the model
    fits mixing_speed alone while reporting eight numbers. Nothing raises:
    `float(...)` on a 2-D tensor happily returns the global scalar. Hence
    `.mean(0)` / `.std(0)`, and hence the per-column affine-equivariance check in
    `tests/test_finetune.py` — the older uniform-scale check passes a global bug.

    The normalization is folded back into the returned head's weights, ROW BY ROW —
    `y = (W.h + b)*std + mean = (W*std).h + (b*std + mean)` — so the head this
    returns predicts in ORIGINAL units per column and `evaluate` needs no stored
    stats. Metrics stay in A; only `log`'s train/val losses are in standardized
    units, which is the space actually being optimized.

    Classification is untouched: cross-entropy on class codes has no target scale.
    """
    joint = y_tr.ndim == 2 and y_tr.shape[1] > 1
    if joint:
        if classes is not None:
            raise ValueError("joint head over classification targets is not built: "
                             "cross-entropy over independent label sets is a "
                             "different model and has no caller here")
        if warm_start_head:
            raise ValueError("warm_start_head has no joint form: fit_probe is "
                             "single-RHS on purpose (a closed-form joint head is "
                             "separable, so it was never built)")
        for name, t in (("y_tr", y_tr), ("y_val", y_val)):
            bad = (~torch.isfinite(t)).any(0).nonzero().flatten().tolist()
            if bad:
                raise ValueError(f"{name} is non-finite in column(s) {bad}; T columns "
                                 f"feed ONE loss, so this would surface as a "
                                 f"divergence blamed on the learning rate")

    set_seed(seed)
    model = build_model(cfg).to(device)
    if state is not None:
        model.load_state_dict(state)

    if classes is None:
        criterion = nn.MSELoss()
        # A (N,) target is the T=1 case: view it as a column and everything below —
        # head width, per-column stats, the fold-back — is written once.
        Y_tr = y_tr if y_tr.ndim == 2 else y_tr.unsqueeze(1)
        Y_val = y_val if y_val.ndim == 2 else y_val.unsqueeze(1)
        # Train-split stats only, PER COLUMN. A constant target has no scale to
        # remove and would divide by zero; its R2 is an undefined 0/0 either way.
        y_mean, y_std = Y_tr.mean(0), Y_tr.std(0)
        y_std = torch.where(y_std < 1e-8, torch.ones_like(y_std), y_std)
        t_tr, t_val = (Y_tr - y_mean) / y_std, (Y_val - y_mean) / y_std
        n_out = Y_tr.shape[1]
    else:
        criterion = nn.CrossEntropyLoss()
        t_tr = torch.from_numpy(_encode(y_tr, classes))
        t_val = torch.from_numpy(_encode(y_val, classes))
        n_out = len(classes)
    head = nn.Linear(head_dim(cfg), n_out).to(device)

    if warm_start_head:
        _warm_start(model, head, X_tr, t_tr, y_tr, classes, device, batch_size)

    def batch_loss(x, t):
        # (B, L) -> (B, 1, L): the encoders take a channel dim, the registry does not.
        pred = head(model.encode(x.unsqueeze(1).to(device)))
        return criterion(pred, t.to(device))

    # drop_last: the encoder is full of BatchNorm, which errors on a size-1 batch.
    # The remainder is data-dependent once the per-target NaN mask has been applied,
    # so this is not left to chance.
    loader_tr = DataLoader(TensorDataset(X_tr, t_tr), batch_size=batch_size,
                           shuffle=True, drop_last=True)
    loader_val = DataLoader(TensorDataset(X_val, t_val), batch_size=batch_size,
                            shuffle=False, drop_last=False)
    if len(loader_tr) == 0:
        raise ValueError(f"0 train batches: {len(X_tr)} labeled rows at batch_size={batch_size}")

    optimizer = torch.optim.Adam([
        {"params": [p for n, p in model.named_parameters() if not n.startswith("proj_head")],
         "lr": lr_enc},
        {"params": head.parameters(), "lr": lr_head},
    ])

    log = []
    best_loss, best_epoch, no_improve = float("inf"), 0, 0
    best_model = best_head = None
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        total, n = 0.0, 0
        for x, t in loader_tr:
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(x, t)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
        train_loss = total / n

        model.eval()
        head.eval()
        total, n = 0.0, 0
        per_target = torch.zeros(n_out, dtype=torch.float64) if joint else None
        with torch.no_grad():
            for x, t in loader_val:
                total += batch_loss(x, t).item() * len(x)
                if joint:
                    # A second forward rather than a hand-rolled loss: `batch_loss`
                    # stays the one definition of what is being optimized. Joint
                    # only, and val is a fifth of the rows, so the cost is ~10% of
                    # an epoch.
                    pred = head(model.encode(x.unsqueeze(1).to(device))).cpu()
                    per_target += ((pred - t) ** 2).double().sum(0)
                n += len(x)
        val_loss = total / n

        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if joint:
            # One stopping epoch now serves T targets whose single-target optima
            # ranged from epoch 1 to epoch 100 on identical Au folds. Recording the
            # per-column val loss is what lets a joint R2 drop be attributed to the
            # shared representation rather than to a stopping epoch that was wrong
            # for that column. Absent for T=1, so single-target logs are unchanged.
            entry["val_loss_per_target"] = [v / n for v in per_target.tolist()]
        log.append(entry)
        if val_loss < best_loss:
            best_loss, best_epoch, no_improve = val_loss, epoch, 0
            best_model = copy.deepcopy(model.state_dict())
            best_head = copy.deepcopy(head.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_model is None:
        # `val_loss < best_loss` is False when val_loss is NaN, so a finetune that
        # diverges on every epoch never records a best state. Say so here rather
        # than dying on `load_state_dict(None)` on the next line.
        blame = ""
        if joint:
            per_t = log[-1]["val_loss_per_target"]
            worst = max(range(len(per_t)), key=lambda j: per_t[j])
            blame = f"; worst column is {worst} (val {per_t[worst]:.4g})"
        raise ValueError(
            f"finetune diverged: val_loss non-finite for all {len(log)} epochs "
            f"(last train_loss={log[-1]['train_loss']:.4g}){blame}; "
            f"try a lower lr_enc/lr_head"
        )

    model.load_state_dict(best_model)
    head.load_state_dict(best_head)
    if classes is None:
        # Fold the target normalization into the restored head so it predicts in
        # original units. AFTER the restore, not before: folding first would be
        # overwritten by the state dict loaded on top of it. Row by row: weight is
        # (T, d) against a (T,) std, so the scale has to be a column to broadcast
        # down the rows rather than across the features.
        std, mean = y_std.to(device), y_mean.to(device)
        with torch.no_grad():
            head.weight.mul_(std.unsqueeze(1))
            head.bias.mul_(std).add_(mean)
    return model, head, log, best_epoch


@torch.no_grad()
def evaluate(model, head, X, y, device, batch_size, classes=None, modal=None):
    """Score a finetuned (model, head) — the same metric keys `probe.json` carries.

    Regression: `r2`/`mae`, computed in torch with `downstream_eval.score`'s exact
    arithmetic (R² against `y`'s own mean, in float64).

    Classification: `f1_weighted`/`f1_per_class`/`f1_baseline`/`accuracy`/`classes`/
    `modal_class`, matching `downstream_eval.score_classifier` — including its rule
    that the per-class list is restricted to classes PRESENT in `y`, since a class
    with zero true instances scores a meaningless 0/0. `modal` is the class CODE of
    the fit split's modal class: the baseline has to be the one this head could
    itself have fallen back on, not one recomputed from the scored split.
    """
    model.eval()
    head.eval()
    out = torch.cat([
        head(model.encode(X[i : i + batch_size].unsqueeze(1).to(device))).cpu()
        for i in range(0, len(X), batch_size)
    ])

    if classes is None:
        if out.shape[1] != 1:
            # squeeze(1) is a no-op on (N, T>1) and `y - pred` would then broadcast
            # (N,) against (N, T) into a plausible-looking wrong number. A joint
            # head is scored per column by its caller, never here.
            raise ValueError(f"evaluate scores one target; head has {out.shape[1]} "
                             f"outputs — score a joint head per column instead")
        pred, y = out.squeeze(1).double(), y.double()
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return {"r2": float(1 - ss_res / ss_tot), "mae": float((y - pred).abs().mean()),
                # Per-row output, for the appendix parity panels. Additive: every
                # existing key keeps its meaning, and a reader of the figure can
                # recompute `r2`/`mae` from these two lists alone.
                "y_true": [float(v) for v in y],
                "y_pred": [float(v) for v in pred]}

    from sklearn.metrics import f1_score

    truth = _encode(y, classes)
    pred = out.argmax(dim=1).numpy()
    kw_all = dict(labels=list(range(len(classes))), zero_division=0)
    present = sorted(set(truth.tolist()))  # codes actually occurring in y
    with _single_threaded():
        return {
            "f1_weighted": float(f1_score(truth, pred, average="weighted", **kw_all)),
            "f1_per_class": [float(v) for v in
                             f1_score(truth, pred, average=None, labels=present, zero_division=0)],
            "f1_baseline": float(f1_score(truth, np.full_like(truth, modal),
                                          average="weighted", **kw_all)),
            "accuracy": float((pred == truth).mean()),
            "classes": [float(classes[c]) for c in present],
            "modal_class": float(classes[modal]),
            # Per-row output, for the appendix confusion panels. Stored as class
            # VALUES, not codes: `classes` above is narrowed to the present set,
            # so a code would not index it, and values are self-describing.
            "y_true": [float(classes[c]) for c in truth],
            "y_pred": [float(classes[c]) for c in pred],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs/pdf/sweep", help="dir whose */ckpt_best.pt are finetuned")
    parser.add_argument("--filter", default="*", help="glob over run-dir names, e.g. 'transformer_*'")
    parser.add_argument("--config", default=None,
                        help="SUPERVISED MODE: a configs/pretrain/*.yaml read for its ARCHITECTURE "
                             f"only. Trains from random init (no checkpoint) into {SUPERVISED_DIR}/. "
                             "Pass --lr-enc == --lr-head; the 1e-5 default is a finetuning LR. "
                             "Mutually exclusive with --runs/--filter")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY,
                        help="downstream registry parquet — the EVALUATION registry")
    parser.add_argument("--signal", default=DEFAULT_SIGNAL, help="signal_<name>_x/_y to finetune on")
    parser.add_argument("--aug-signal", default=DEFAULT_AUG_SIGNAL,
                        help="augmented channel for the shift protocols; skipped if absent")
    parser.add_argument("--fit-registry", default=None,
                        help="CHEMISTRY AXIS: fit on THIS registry's clean train/val and score "
                             "on --registry's test rows (protocols cross_clean/cross_aug). "
                             "Regression targets only")
    parser.add_argument("--fit-signal", default=None,
                        help="channel to read from --fit-registry (default: --signal)")
    parser.add_argument("--target", nargs="+", default=DEFAULT_TARGETS,
                        help="target_* column(s); each gets its own finetune per fit channel")
    parser.add_argument("--device", default="auto", help='"auto" | "cpu" | "cuda"')
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr-enc", type=float, default=1e-5, help="encoder backbone LR")
    parser.add_argument("--lr-head", type=float, default=1e-4, help="linear head LR")
    parser.add_argument("--patience", type=int, default=15, help="early-stopping patience on val loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start-head", action="store_true",
                        help="LP-FT: initialise the head from the closed-form probe on the "
                             "frozen encoder instead of at random. Matters at small n_train, "
                             "where random-head gradients distort the pretrained encoder.")
    parser.add_argument("--n-train-frac", type=float, default=1.0,
                        help="finetune each target on a seeded FRACTION of the train split's "
                             "labeled rows (val/test untouched) -- the labeled-data-size sweep. "
                             "1.0 (default) is every row, byte-identical to omitting the flag. "
                             f"<1.0 redirects output to {DATAFRAC_DIR}/ so it can never overwrite "
                             "the full-data result for the same run/config.")
    parser.add_argument("--group-col", default=None,
                        help="registry column naming the LABELED UNIT for --n-train-frac. "
                             "Without it the fraction is of train ROWS; with it, of "
                             "distinct groups, keeping every row of a chosen group. On "
                             "CHILI-100K (5 particle sizes per crystal, one label) pass "
                             "`group_key` or the x-axis reads 'views per crystal'.")
    parser.add_argument("--out-root", default=None,
                        help="LR SEARCH: write to <out-root>/<run>_enc<lr_enc>_head<lr_head>/ "
                             "instead of the run dir, so one checkpoint can be finetuned at "
                             "several LRs without the cells overwriting each other -- and "
                             "without touching runs/pdf/sweep/*/finetune.json")
    parser.add_argument("--refit-head-aug", action="store_true",
                        help="after the CLEAN finetune, freeze the encoder and fit a fresh "
                             "linear head on the AUGMENTED train split -> tasks.*."
                             "cleanfit_augrefit. Separates 'the representation moved' from "
                             "'the clean-fitted readout does not transfer'. Needs an "
                             "augmented channel; ignored without one")
    parser.add_argument("--save-weights", action="store_true",
                        help="also write each finetuned encoder+head. OFF by default: a copy is "
                             "~6 MB (CNN) / ~15 MB (Transformer), so a full sweep is 1-3 GB in runs/pdf/sweep/")
    args = parser.parse_args()
    if not (0.0 < args.n_train_frac <= 1.0):
        raise SystemExit(f"--n-train-frac must be in (0, 1], got {args.n_train_frac}")
    if args.fit_registry:
        # Refuse rather than mis-score: a class vocabulary spanning two registries
        # whose label sets disagree leaves the head with units it never trains on.
        # See the module docstring's CHEMISTRY AXIS note.
        bad = [t for t in args.target if t in CLASSIFICATION_TARGETS]
        if bad:
            raise SystemExit(
                f"--fit-registry is regression-only; drop {bad} (the class vocabulary "
                f"does not survive a change of registry)"
            )
        if args.refit_head_aug:
            # cleanfit_augrefit refits the head on the FIT registry's augmented
            # channel, which the chemistry-axis fit registry does not carry.
            raise SystemExit("--refit-head-aug is not defined with --fit-registry")

    # `jobs` is (out_dir, ckpt_path | None) — the checkpoint is loaded inside the loop,
    # not here, so a sweep never holds more than one state dict at a time. A None
    # checkpoint is supervised mode, and `sup_cfg` is then the architecture for it.
    sup_cfg = None
    if args.config:
        # Refuse rather than silently ignore: --runs/--filter select checkpoints, and
        # supervised mode loads none, so passing both means one of them is a mistake.
        if args.runs != "runs/pdf/sweep" or args.filter != "*":
            raise SystemExit(
                "--config is supervised mode (random init, no checkpoint), so --runs/--filter "
                "have nothing to select. Drop them, or drop --config."
            )
        sup_cfg = load_config(args.config)
        # Named by the CONFIG STEM, not by `cfg.encoder`. Two configs can share an
        # encoder and differ in architecture — `transformer_infonce_mpfull_final`
        # (4 layers / patch 50 / Post-LN) and `transformer_vicreg_mpfull_final`
        # (8 / 200 / Pre-LN) both report encoder="transformer", so an encoder-named
        # dir would have had the second silently overwrite the first at the same
        # (lr, seed). Deterministic naming is meant to overwrite RE-RUNS of the same
        # cell, never a different architecture.
        stem = Path(args.config).stem
        # `--out-root` applies HERE TOO (s32). The checkpoint path already honoured it;
        # supervised mode did not, and the deterministic `{stem}_lr{lr}_seed{seed}` name
        # is only safe to overwrite when the OTHER coordinates match. They do not across
        # datasets: `runs/pdf/supervised/transformer_vicreg_mpfull_final_lr0.0001_seed42` is
        # the CHILI scratch control RESULTS.md Q1 reads, and an RRUFF scratch run at the
        # same (stem, lr, seed) would have silently destroyed it — the exact failure
        # `scripts/ssl_lr_search.slurm` redirects to runs/pdf/ssl_lr/ to avoid.
        out = REPO_ROOT / (args.out_root or SUPERVISED_DIR) / f"{stem}_lr{args.lr_enc:g}_seed{args.seed}"
        out.mkdir(parents=True, exist_ok=True)  # deterministic name: a re-run overwrites
        jobs = [(out, None)]
        signal_len = sup_cfg.signal_len
    else:
        ckpts = find_checkpoints(args.runs, args.filter)
        if not ckpts:
            raise SystemExit(f"no */ckpt_best.pt under {args.runs} matching {args.filter!r}")
        jobs = [(p.parent, p) for p in ckpts]
        # signal_len is a property of the pretraining grid, so it comes from the first
        # checkpoint's config rather than being a flag; a mismatch across runs raises below.
        first = torch.load(ckpts[0], map_location="cpu", weights_only=False)
        signal_len = int(first["config"]["signal_len"])
        del first

    device = pick_device(args.device)
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = REPO_ROOT / registry_path

    # All three splits, unlike the probe: the finetune early-stops on val.
    clean, r_range = load_probe_data(registry_path, args.signal, args.target, signal_len, SPLITS)
    data = {"clean": clean}

    has_aug = f"signal_{args.aug_signal}_y" in set(pq.read_schema(registry_path).names)
    if has_aug:
        aug, _ = load_probe_data(registry_path, args.aug_signal, args.target, signal_len, SPLITS)
        # Both reads filter the same frame the same way, so the row order must match —
        # targets are channel-independent and are taken from the clean channel only.
        for s in SPLITS:
            if aug[s][2] != clean[s][2]:
                raise ValueError(f"{args.aug_signal} rows are not aligned with {args.signal} in {s!r}")
        data["aug"] = aug

    # CHEMISTRY AXIS. The fit side is a whole separate registry, so it gets its own
    # entry in `data` and — unlike clean/aug — is deliberately NOT alignment-checked
    # against it. Only train/val are read: the test rows always come from the eval
    # registry, which is what pins evaluation while the fit set moves.
    fit_registry_path = None
    if args.fit_registry:
        fit_registry_path = Path(args.fit_registry)
        if not fit_registry_path.is_absolute():
            fit_registry_path = REPO_ROOT / fit_registry_path
        cross, cross_r_range = load_probe_data(
            fit_registry_path, args.fit_signal or args.signal, args.target, signal_len,
            ("train", "val"))
        # The grids must agree even though the rows do not: `channel_contract` checks
        # each registry against the pretraining grid independently, so a disagreement
        # here means one of them slid off it and the encoder would see two r axes.
        if cross_r_range != r_range:
            raise ValueError(
                f"fit registry grid {cross_r_range} != eval registry grid {r_range}"
            )
        data["cross"] = cross

    if fit_registry_path is not None:
        protocols = [p for p, (_, test) in CROSS_PROTOCOLS.items() if test in data]
        fit_channels = ["cross"]
        protocol_table = CROSS_PROTOCOLS
    else:
        protocols = list(PROTOCOLS) if has_aug else ["clean_clean"]
        fit_channels = [c for c in ("clean", "aug") if c in data]
        protocol_table = PROTOCOLS

    finetune_cfg = {"epochs": args.epochs, "lr_enc": args.lr_enc, "lr_head": args.lr_head,
                    "patience": args.patience, "batch_size": args.batch_size, "seed": args.seed,
                    "n_train_frac": args.n_train_frac,
                    # The UNIT the fraction counts. Without it a reader cannot tell a
                    # crystal-level rung from a row-level one, and the two mean
                    # different things at the same `n_train_frac`.
                    "group_col": args.group_col,
                    # Recorded because it CHANGES THE RESULT, not just the code path:
                    # measured A/B at fixed seed on RRUFF/xpdf/n_atoms, random head
                    # -> test R2 -0.120, LP-FT -> +0.322. A run JSON that does not say
                    # which one it was is not reproducible from its own provenance.
                    "warm_start_head": args.warm_start_head}

    # train/val describe the FIT side, test the EVAL side. Identical in the
    # single-registry path (clean and aug are two channels of the same rows).
    fit_data = data[fit_channels[0]]

    # The labelled unit for --n-train-frac, positional over the TRAIN split's rows.
    # Read from the FIT registry (the rows being subsampled), keyed by material_id
    # rather than by position: `load_probe_data` filters and orders rows itself, so a
    # positional join against the parquet would silently misalign.
    train_groups = None
    if args.group_col:
        if args.n_train_frac >= 1.0:
            raise SystemExit("--group-col does nothing without --n-train-frac < 1.0")
        gsrc = fit_registry_path or registry_path
        gdf = pd.read_parquet(gsrc, columns=["material_id", args.group_col])
        gmap = dict(zip(gdf["material_id"], gdf[args.group_col]))
        ids_train = fit_data["train"][2]
        missing = [i for i in ids_train if i not in gmap]
        if missing:
            raise SystemExit(f"--group-col {args.group_col!r}: {len(missing)} train id(s) "
                             f"absent from {gsrc.name}, e.g. {missing[:3]}")
        train_groups = np.array([gmap[i] for i in ids_train])
        print(f"--group-col {args.group_col}: {len(np.unique(train_groups))} distinct "
              f"group(s) over {len(train_groups)} train row(s) "
              f"({len(train_groups) / len(np.unique(train_groups)):.2f} rows/group)")
    n = {"train": len(fit_data["train"][0]), "val": len(fit_data["val"][0]),
         "test": len(clean["test"][0])}
    print(
        f"device={device} registry={registry_path.name} signal={args.signal}"
        f"{' +' + args.aug_signal if has_aug else ' (no augmented channel)'}\n"
        + (f"fit_registry={fit_registry_path.name} "
           f"fit_signal={args.fit_signal or args.signal}\n" if fit_registry_path else "")
        + f"targets={', '.join(args.target)}  protocols={', '.join(protocols)}\n"
        + f"grid={signal_len} pts over [{r_range[0]:.2f}, {r_range[1]:.2f}] A  "
        + f"n_train={n['train']} n_val={n['val']} n_test={n['test']}\n"
        + f"finetune: {args.epochs} epochs max, lr_enc={args.lr_enc:g} lr_head={args.lr_head:g} "
        + f"patience={args.patience} batch={args.batch_size} seed={args.seed}\n"
        + (f"SUPERVISED from-scratch baseline ({sup_cfg.encoder}) -> {jobs[0][0].name}\n"
           if sup_cfg else "")
        + (f"n_train_frac={args.n_train_frac:g} (labeled-data-size sweep) -> {DATAFRAC_DIR}/\n"
           if args.n_train_frac < 1.0 else "")
        + (f"LR search -> {args.out_root}/ (runs/pdf/sweep/ untouched)\n" if args.out_root else "")
        + f"{len(jobs)} run(s) x {len(args.target)} target(s) x {len(fit_channels)} "
        f"fit channel(s) = {len(jobs) * len(args.target) * len(fit_channels)} finetunes\n"
    )

    rows = []
    for base_dir, ckpt_path in jobs:
        run = base_dir.name
        # frac=1.0 (the default) writes to the SAME place as always -- base_dir,
        # already `runs/<run>/` or `runs/pdf/supervised/<...>/`. frac<1.0 is redirected
        # to its own tree so it can never collide with, or be mistaken for, the
        # full-data result. See the module docstring's LABELED-DATA-SIZE note.
        # --out-root is the LR search's tree: one dir per (run, lr pair), so the
        # cells of a search cannot overwrite each other OR runs/pdf/sweep/. See the module
        # docstring's LR SEARCH note.
        if args.out_root:
            # The two redirects COMPOSE rather than conflict: the tuned
            # labeled-data-size sweep is one checkpoint at one LR at one fraction,
            # so the name has to carry both or its cells collide.
            frac = "" if args.n_train_frac >= 1.0 else f"_frac{args.n_train_frac:g}_seed{args.seed}"
            out_dir = (REPO_ROOT / args.out_root
                       / f"{run}_enc{args.lr_enc:g}_head{args.lr_head:g}{frac}")
            out_dir.mkdir(parents=True, exist_ok=True)
        elif args.n_train_frac < 1.0:
            out_dir = REPO_ROOT / DATAFRAC_DIR / f"{run}_frac{args.n_train_frac:g}_seed{args.seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = base_dir
        # state=None is the from-scratch baseline; everything below is identical.
        cfg, state, epoch = (
            (sup_cfg, None, None) if ckpt_path is None else _load_state(ckpt_path)
        )
        if cfg.signal_len != signal_len:
            raise ValueError(
                f"{run}: signal_len={cfg.signal_len} but the registry grid is {signal_len} — "
                f"finetune one grid at a time (use --filter)"
            )

        tasks, labelled, best_epochs, logs = {}, {}, {}, {}
        for t in args.target:
            # Per-target masks, on the FIT rows and the EVAL rows separately. Within
            # one registry these are the same object read twice: targets are
            # channel-independent and the aug row order was asserted equal above. With
            # --fit-registry the two sides are different registries with no row
            # correspondence at all, so conflating them would index the eval tensors
            # with the fit registry's mask.
            fit_masks = {s: torch.isfinite(fit_data[s][1][t]) for s in ("train", "val")}
            # Data-efficiency sweep: shrink TRAIN only, to a seeded subset shared by
            # every arm at this (fraction, seed) -- see subsample_train_mask. This is
            # also how n_train is matched across two fit registries.
            fit_masks["train"] = subsample_train_mask(
                fit_masks["train"], args.n_train_frac, args.seed, train_groups)
            fit_y = {s: fit_data[s][1][t][fit_masks[s]] for s in ("train", "val")}

            test_mask = torch.isfinite(clean["test"][1][t])
            y_test = clean["test"][1][t][test_mask]
            labelled[t] = {"train": int(fit_masks["train"].sum()),
                           "val": int(fit_masks["val"].sum()),
                           "test": int(test_mask.sum())}

            if t in CLASSIFICATION_TARGETS:
                # One vocabulary for the whole task — the union over all three splits —
                # so per-class lists stay positionally comparable across protocols.
                # (--fit-registry refuses classification, so both sides are one registry.)
                classes = sorted(set().union(set(fit_y["train"].tolist()),
                                             set(fit_y["val"].tolist()),
                                             set(y_test.tolist())))
                modal = int(np.bincount(_encode(fit_y["train"], classes),
                                        minlength=len(classes)).argmax())
            else:
                classes, modal = None, None

            fits = {}
            for c in fit_channels:
                print(f"  {run} / {t} / fit={c} ...", end=" ", flush=True)
                model, head, log, best = finetune_one(
                    cfg, state, classes,
                    data[c]["train"][0][fit_masks["train"]], fit_y["train"],
                    data[c]["val"][0][fit_masks["val"]], fit_y["val"],
                    device=device, epochs=args.epochs, lr_enc=args.lr_enc,
                    lr_head=args.lr_head, patience=args.patience,
                    batch_size=args.batch_size, seed=args.seed,
                    warm_start_head=args.warm_start_head)
                fits[c] = (model, head)
                best_epochs.setdefault(t, {})[c] = best
                logs.setdefault(t, {})[c] = log
                print(f"best epoch {best}/{len(log)} (val {log[best - 1]['val_loss']:.4g})")
                if args.save_weights:
                    torch.save({"model_state": model.state_dict(), "head_state": head.state_dict(),
                                "target": t, "fit_channel": c, "best_epoch": best,
                                "classes": classes, "config": asdict(cfg)},
                               out_dir / f"finetune_{t}_{c}.pt")

            # clean_clean and clean_aug are the SAME finetuned model on two test sets;
            # so are cross_clean and cross_aug. The test tensors always come from the
            # EVAL registry, indexed by its own mask.
            tasks[t] = {}
            for name, (fit, test) in protocol_table.items():
                if name not in protocols:
                    continue
                model, head = fits[fit]
                tasks[t][name] = evaluate(model, head, data[test]["test"][0][test_mask],
                                          y_test, device, args.batch_size, classes, modal)

            # REPRESENTATION vs READOUT. Same encoder that just produced clean_aug,
            # frozen, with a head refitted on the augmented channel — so this and
            # clean_aug differ ONLY in where the head came from, and this and the
            # pretrained encoder's probe.json differ ONLY in the encoder.
            if args.refit_head_aug and "aug" in data and "clean" in fits:
                enc = fits["clean"][0]
                enc.eval()
                h_tr = embed(enc, data["aug"]["train"][0][fit_masks["train"]],
                             device, args.batch_size)
                h_te = embed(enc, data["aug"]["test"][0][test_mask], device, args.batch_size)
                if classes is None:
                    tasks[t]["cleanfit_augrefit"] = score(
                        h_te, y_test, fit_probe(h_tr, fit_y["train"]))
                else:
                    tasks[t]["cleanfit_augrefit"] = score_classifier(
                        fit_classifier(h_tr, fit_y["train"], classes), h_te, y_test)
                k = "f1_weighted" if classes is not None else "r2"
                print(f"    refit head on aug (encoder frozen): {k}="
                      f"{tasks[t]['cleanfit_augrefit'][k]:.4f}  vs clean_aug "
                      f"{tasks[t]['clean_aug'][k]:.4f}")

        # A supervised run inherited `loss`/`cov_weight`/`temperature` from whichever
        # YAML supplied its architecture; they describe a pretraining run that never
        # happened, so they are recorded as null rather than as plausible numbers.
        # `pretrained` is the field to branch on — the others are only labels.
        pretrained = state is not None
        result = {
            "run": run,
            "pretrained": pretrained,
            "epoch": epoch,
            "encoder": cfg.encoder,
            "loss": cfg.loss if pretrained else "supervised",
            "latent_dim": cfg.latent_dim,
            "cov_weight": cfg.cov_weight if pretrained else None,
            "temperature": cfg.temperature if pretrained else None,
            # `registry` is the EVALUATION registry in every mode. With the chemistry
            # axis the fit side is a different file, and a reader that assumes one
            # registry would attribute the fit rows to the wrong dataset.
            "registry": str(registry_path.relative_to(REPO_ROOT)),
            "signal": args.signal,
            "aug_signal": args.aug_signal if has_aug else None,
            "fit_registry": (str(fit_registry_path.relative_to(REPO_ROOT))
                             if fit_registry_path else None),
            "fit_signal": (args.fit_signal or args.signal) if fit_registry_path else None,
            "protocols": protocols + (["cleanfit_augrefit"]
                                      if args.refit_head_aug and has_aug else []),
            "signal_len": signal_len,
            "r_range": list(r_range),
            "n_train": n["train"],
            "n_val": n["val"],
            "n_test": n["test"],
            "n_labelled": labelled,
            "classification_targets": [t for t in args.target if t in CLASSIFICATION_TARGETS],
            "finetune_cfg": finetune_cfg,
            "best_epoch": best_epochs,
            "log": logs,
            "tasks": tasks,
        }
        (out_dir / "finetune.json").write_text(json.dumps(result, indent=2) + "\n")
        rows.append(result)
        # The headline protocol is the first one that ran, not `clean_clean` by name:
        # the chemistry axis never produces `clean_clean` and this line would KeyError
        # on it. `protocols` is ordered by its table, so [0] is the in-distribution
        # cell in both modes (clean_clean, or cross_clean).
        head_p = protocols[0]
        print(f"  [{head_p}] " + "  ".join(
            f"{t}: " + (f"F1={m[head_p]['f1_weighted']:.4f}" if t in CLASSIFICATION_TARGETS
                        else f"R2={m[head_p]['r2']:.4f} MAE={m[head_p]['mae']:.3f}")
            for t, m in tasks.items()) + "\n")

    # One table per task — the same layout probe.json's ranking prints, so the two
    # can be read side by side. Never a mean across targets.
    width = max(len(r["run"]) for r in rows)
    for target in args.target:
        clf = target in CLASSIFICATION_TARGETS
        key, metric = ("f1_weighted", "weighted F1") if clf else ("r2", "R²")
        # Ranked on the first protocol that ran — `clean_clean` in the single-registry
        # mode, `cross_clean` on the chemistry axis, which never produces the former.
        rank_p = protocols[0]
        first = rows[0]["tasks"][target][rank_p]
        lab = rows[0]["n_labelled"][target]
        # The trivial-classifier baseline depends only on the labels, so it is the
        # same for every run — a header line, not a repeated column. Classes are
        # rounded for DISPLAY only; finetune.json keeps the exact values. Guarded on
        # `clf`: a regression task's metrics have no `classes` key at all.
        note = ""
        if clf:
            classes_desc = (f"[{', '.join(f'{c:g}' for c in first['classes'])}]"
                            if len(first["classes"]) <= 10 else f"{len(first['classes'])} classes")
            note = f" — modal-class baseline {first['f1_baseline']:.4f} over {classes_desc}"
        print(f"\n── finetuned, ranked by {metric} on {target} / {rank_p} "
              f"(test, n={lab['train']}/{lab['val']}/{lab['test']}){note} " + "─" * 12)
        # R² and MAE in every regression column — the probe's layout, kept
        # identical here so the two tables still read side by side.
        print(f"{'run':<{width}} {'encoder':>12} {'loss':>8}"
              + "".join(f"{p:>{13 if clf else 15}}" for p in protocols))
        if not clf:
            print(f"{'':<{width}} {'':>12} {'':>8}" + f"{'R²':>8}{'MAE':>7}" * len(protocols))
        for r in sorted(rows, key=lambda r: r["tasks"][target][rank_p][key], reverse=True):
            m = r["tasks"][target]
            cells = "".join(
                f"{m[p][key]:>13.4f}" if clf else f"{m[p]['r2']:>8.4f}{m[p]['mae']:>7.3f}"
                for p in protocols)
            print(f"{r['run']:<{width}} {r['encoder']:>12} {r['loss']:>8}{cells}")


if __name__ == "__main__":
    main()
