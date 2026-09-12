"""tools/chili_predictions.py — the per-row prediction store behind the appendix pair.

`plot_chili_parity.py` and `plot_chili_confusion.py` are two views of ONE dataset:
the same three schemes, the same two fit channels, the same 360-row augmented CHILI-3K
test split. Only the targets differ — regression to the first, categorical to the
second. This module is that dataset, so the two plotters cannot disagree about what
they are drawing or about which run a number came from.

WHERE THE ROWS COME FROM. Nothing here reads a published run. Figures 2-4 are backed
by JSONs that stored only aggregates, and no downstream run saved weights, so the
predictions had to be made again into a separate tree (`scripts/parity_predictions.slurm`
documents why, and the caveats that follow from it). The probe is the exception and is
the wiring check: closed-form on frozen embeddings, it reproduced all eight published
cells exactly, so a probe panel here IS the published probe.

THE STORED METRIC IS NEVER TRUSTED. `load` recomputes R² / weighted-F1 from the rows
and raises if it disagrees with the scalar the run reported. A panel annotation can
therefore never drift from the points underneath it — the failure mode that
`tools/plot_au_parity.py` guards against the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PRETRAIN_RUN = "2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372"

#: (paper name, paperstyle ARM_COLORS key, JSON path). Order is the ramp's order —
#: increasing use of pretraining — and is the column order of both figures.
SCHEMES = (
    ("baseline",   "supervised", REPO / "runs/pdf/parity/baseline"),
    ("probe",      "probe",      REPO / "runs/pdf/sweep" / PRETRAIN_RUN / "probe_chili_parity.json"),
    ("fine-tuned", "finetuned",  REPO / "runs/pdf/parity/finetuned"),
)

#: (protocol key, column label). `clean_aug` is fitted on CHILI's own single instrument
#: realization, `aug_aug` on re-simulated spectra; BOTH are evaluated on the augmented
#: test split, so the pair isolates the fit channel and nothing else.
CHANNELS = (("clean_aug", "fixed"), ("aug_aug", "varied"))

#: (target key, panel title, axis unit). Regression first, categorical second — the
#: row order of the two figures, and the split between them.
REGRESSION = (("target_np_size", "Nanoparticle size", "Å"),
              ("target_mo_bond", "M–O distance", "Å"))
CATEGORICAL = (("target_cn", "Coordination number"),
               ("target_oxidation", "Oxidation state"))


def _resolve(path: Path) -> Path:
    """A scheme's JSON. Directories are globbed, because the finetune trees name their
    run dir from the LR pair (`<run>_enc1e-05_head0.0001`) and hardcoding that string
    here would silently miss a rerun at a different rate rather than reporting it."""
    if path.suffix == ".json":
        return path
    hits = sorted(path.glob("*/finetune.json"))
    if len(hits) != 1:
        raise SystemExit(
            f"{path.relative_to(REPO) if path.is_relative_to(REPO) else path}: expected "
            f"exactly one */finetune.json, found {len(hits)}"
            + ("  — run scripts/parity_predictions.slurm" if not hits else
               "\n  " + "\n  ".join(str(h) for h in hits)))
    return hits[0]


def classes_of(y_true, y_pred):
    """The class VALUES present, in numeric order — the axis order of a confusion
    matrix. Oxidation state includes 8/3, so these are floats, and sorting them
    numerically is what puts that class between 2 and 4 rather than at the end."""
    return sorted(set(y_true.tolist()) | set(y_pred.tolist()))


def _wf1(y_true, y_pred):
    """Weighted F1. The stored labels are class VALUES stored as floats (8/3 is
    2.666...), which sklearn types as `continuous` and refuses, so both vectors are
    encoded to positional codes first. Encoding cannot reorder or merge classes —
    `classes_of` is a sorted set over both vectors — so this is the same score
    `downstream_eval.score_classifier` computed, recovered from the rows."""
    from sklearn.metrics import f1_score
    classes = classes_of(y_true, y_pred)
    code = {v: i for i, v in enumerate(classes)}
    t = np.array([code[v] for v in y_true.tolist()])
    p = np.array([code[v] for v in y_pred.tolist()])
    return float(f1_score(t, p, average="weighted",
                          labels=list(range(len(classes))), zero_division=0))


def _r2(y_true, y_pred):
    return float(1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())


def load(targets, schemes=None):
    """`{(scheme, protocol, target): {"y_true", "y_pred", "score", "n"}}`.

    `score` is R² for a regression target and weighted F1 for a categorical one,
    RECOMPUTED from the rows and checked against the value the run reported.
    """
    wanted = [s for s in SCHEMES if schemes is None or s[0] in schemes]
    out = {}
    for name, _, path in wanted:
        d = json.loads(_resolve(path).read_text())
        for target in targets:
            if target not in d["tasks"]:
                raise SystemExit(f"{name}: '{target}' not scored in {_resolve(path)}")
            for proto, _ in CHANNELS:
                m = d["tasks"][target][proto]
                if "y_true" not in m:
                    raise SystemExit(
                        f"{name}/{target}/{proto} has no per-row output — this JSON predates "
                        f"the prediction-persisting change; rerun it")
                yt, yp = np.asarray(m["y_true"]), np.asarray(m["y_pred"])
                is_reg = "r2" in m
                got = _r2(yt, yp) if is_reg else _wf1(yt, yp)
                ref = m["r2"] if is_reg else m["f1_weighted"]
                if abs(got - ref) > 1e-6:
                    raise SystemExit(
                        f"{name}/{target}/{proto}: score recomputed from rows ({got:.6f}) "
                        f"disagrees with the stored value ({ref:.6f})")
                out[(name, proto, target)] = {"y_true": yt, "y_pred": yp,
                                              "score": got, "n": len(yt)}
    return out


def row_label(title):
    """A row header is drawn ROTATED, so its length is spent against the panel's
    HEIGHT (~0.7 in at six columns) rather than the figure's width. "Coordination
    number" at 8 pt is ~1.3 in and ran into the row above it, so multi-word titles
    are broken at their last space into two stacked lines."""
    if " " not in title:
        return title
    head, _, tail = title.rpartition(" ")
    return f"{head}\n{tail}"


def columns(schemes=None):
    """The (scheme, colour key, protocol, channel label) column order shared by both
    figures — schemes outermost so a scheme's two channels sit adjacent, which is the
    comparison Figure 2 measures."""
    return [(name, key, proto, lab)
            for name, key, _ in SCHEMES if schemes is None or name in schemes
            for proto, lab in CHANNELS]
