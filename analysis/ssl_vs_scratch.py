"""analysis/ssl_vs_scratch.py — the 2x2 this repo exists to run, as one figure.

    python -m analysis.ssl_vs_scratch

Renders `DESIGN.md` -> *The experiment this repo exists to run*: does SSL
pretraining on a large domain-agnostic corpus (MP) buy anything that in-domain
supervised training on the target system does not? **Evaluation is held fixed on
augmented CHILI** in every cell, so cells differ only in how the model got there:

               fit on CLEAN CHILI      fit on AUGMENTED CHILI
  SSL          runs/pdf/sweep/*  clean_aug       runs/pdf/sweep/*  aug_aug
  from scratch runs/pdf/supervised/* clean_aug     runs/pdf/supervised/* aug_aug

Nothing is recomputed — this reads `finetune.json` files `analysis/finetune.py`
already wrote.

WHY A DOT PLOT AND NOT BARS. Every point is one trained model, drawn
individually. The SSL arm is 9-11 runs whose spread is large and task-dependent
(the s18 probe measured this), and a bar of the mean would hide exactly the thing
that decides whether "SSL competes" is true of the method or of one lucky run.
Bars would also imply a zero baseline that R2 and F1 do not have.

HOW TO READ IT (the three effects `DESIGN.md` names):
- **Row effect** — scratch vs SSL *within one fit block*: the value of pretraining.
- **Column effect** — the clean block vs the aug block: the size of the domain gap.
- **Interaction** — the actual claim. If SSL+clean approaches scratch+aug while
  scratch+clean does not, pretraining SUBSTITUTES for having in-domain training
  data. The two rows to compare for that are the FIRST and the LAST in each panel.

WHAT IS DELIBERATELY NOT HERE:
- **The frozen probe.** `DESIGN.md`'s grid is init x fit-channel with gradient
  finetuning throughout; the probe is a third factor, not a third row of this one.
- **`clean_clean`.** It is not a cell of this grid, both arms saturate in-domain,
  and reading it gives the OPPOSITE impression. See `PROGRESS.md` -> item 6.
- **`target_metal`.** Dropped per `DESIGN.md`: every arm scores ~0.00-0.03, so it
  separates nothing.
- **The 27 aug_sweep runs.** They have checkpoints only, no downstream eval; they
  belong to the augmentation experiment, not this one.

This module imports neither torch nor pandas, so the import-order segfault in
`docs/ENVIRONMENT.md` cannot apply to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Five, not six — see the module docstring.
TARGETS = ("target_np_size", "target_mo_bond", "target_mean_bond",
           "target_oxidation", "target_cn")
CLASSIFICATION = frozenset({"target_oxidation", "target_cn"})

#: (fit channel -> the protocol that scores it on the AUGMENTED test set).
#: There is no `aug_clean`; evaluation never moves off augmented in this grid.
FIT_PROTOCOL = {"clean": "clean_aug", "aug": "aug_aug"}

#: A finetune whose best epoch landed this close to its own ceiling never
#: early-stopped, so its score is a LOWER BOUND. Same rule as finetune_curves.py.
CEILING_FRACTION = 0.9

ARM_STYLE = {
    "scratch":        dict(color="#a33", marker="o", label="from scratch"),
    "SSL loss_sweep": dict(color="#3b6ea5", marker="o", label="SSL (MP-20, loss_sweep)"),
    "SSL mpfull":     dict(color="#4d9a5a", marker="s", label="SSL (full-MP, mpfull_final)"),
    "excluded":       dict(color="0.6", marker="x", label="arch-mismatched (NOT comparable)"),
}
ROW_ORDER = ["scratch", "SSL loss_sweep", "SSL mpfull", "excluded"]


def metric_key(target: str) -> str:
    return "f1_weighted" if target in CLASSIFICATION else "r2"


def classify(run_name: str, encoder: str) -> str:
    """Which arm a run belongs to — by CONFIG FAMILY, never pooled across corpora.

    `loss_sweep` is pretrained on MP-20 and `mpfull_final` on full-MP, so merging
    them would confound corpus size with pretraining-vs-not (settled 2026-07-31).

    The one architectural exclusion is `transformer_vicreg_mpfull_final`
    (num_layers 8 / patch 200 / stride 100 / Pre-LN). Every OTHER transformer in
    the repo — the 9 loss_sweep runs and `transformer_infonce_mpfull_final` — is
    4 / 50 / 25 / Post-LN, which is also what the scratch transformer is built
    from. It is drawn, greyed and labelled, rather than dropped: it is the most
    shift-robust model in the figure and hiding it would be the bigger distortion.
    """
    if "mpfull_final" not in run_name:
        return "SSL loss_sweep"
    if encoder == "transformer" and "vicreg" in run_name:
        return "excluded"
    return "SSL mpfull"


def load(runs_dir: Path, supervised_dir: Path, epochs: int | None = None) -> dict:
    """`(encoder, arm) -> [point, ...]`, a point per (run, target, fit channel).

    A point carries `truncated`, which is per (run, target, fit channel) — a run
    can converge on one target and hit the ceiling on another, so it cannot be a
    property of the run as a whole. It also carries `epochs`, the run's own
    ceiling, because a directory can hold results from DIFFERENT ceilings at the
    same time and nothing in the filename says so.

    `epochs=N` keeps only runs whose ceiling is N. **This is a correctness guard,
    not a convenience.** `analysis/finetune.py` rewrites `finetune.json` in place,
    so while an epoch-matched resweep is draining, `runs/pdf/sweep/` legitimately holds a
    MIXTURE — on 2026-07-31 it sat at `{100: 2, 300: 20}` for over an hour. Reading
    it then silently blends two experiments into one figure, with no visible tell:
    the files look identical, and a stale `finetune_ep100.json` beside them makes
    the mixture look deliberate. `main` refuses to plot a mixture it was not told
    to expect.
    """
    out: dict[tuple[str, str], list[dict]] = {}

    def add(path: Path, arm_of):
        d = json.loads((path / "finetune.json").read_text())
        if not all(t in d["tasks"] for t in TARGETS):
            return  # stage-1 LR-selection leftovers carry one target only
        ceiling = d["finetune_cfg"]["epochs"]
        if epochs is not None and ceiling != epochs:
            return
        enc = d["encoder"]
        arm = arm_of(path.name, enc)
        for t in TARGETS:
            for fit, proto in FIT_PROTOCOL.items():
                if proto not in d["tasks"][t]:
                    continue
                out.setdefault((enc, arm), []).append({
                    "run": path.name, "target": t, "fit": fit, "epochs": ceiling,
                    "value": d["tasks"][t][proto][metric_key(t)],
                    "truncated": d["best_epoch"][t][fit] >= CEILING_FRACTION * ceiling,
                })

    for p in sorted(runs_dir.iterdir()):
        if p.is_dir() and (p / "finetune.json").exists():
            add(p, classify)
    for p in sorted(supervised_dir.iterdir()):
        if p.is_dir() and (p / "finetune.json").exists():
            add(p, lambda n, e: "scratch")
    return out


def ceilings_by_arm(data: dict) -> dict:
    """`(encoder, arm) -> {ceiling: n_runs}` — what budget each arm actually ran at.

    Reported rather than assumed. The whole point of the epoch-matched resweep is
    that the two arms share a ceiling; if they do not, that IS precondition 1 and
    the figure must say so instead of quietly comparing across budgets.
    """
    out: dict[tuple[str, str], dict[int, int]] = {}
    for key, pts in data.items():
        runs = {}
        for p in pts:
            runs[p["run"]] = p["epochs"]
        counts: dict[int, int] = {}
        for e in runs.values():
            counts[e] = counts.get(e, 0) + 1
        out[key] = counts
    return out


def panel(ax, data, encoder, target, show_ylabels):
    """One (encoder, target): four stacked bands, fit=clean above fit=aug."""
    rows, labels = [], []
    y = 0.0
    for fit in ("clean", "aug"):
        for arm in ROW_ORDER:
            pts = [p for p in data.get((encoder, arm), [])
                   if p["target"] == target and p["fit"] == fit]
            if not pts:
                continue
            rows.append((y, arm, pts))
            labels.append((y, f"{arm.replace('SSL ', '')}  (n={len(pts)})"))
            y += 1.0
        y += 0.7  # gap between the two fit blocks

    for y0, arm, pts in rows:
        st = ARM_STYLE[arm]
        vals = np.array([p["value"] for p in pts])
        trunc = np.array([p["truncated"] for p in pts])
        # Deterministic jitter: points at identical values must stay distinguishable,
        # but the figure must not change between runs.
        jit = (np.arange(len(vals)) % 5 - 2) * 0.09
        for mask, face in ((~trunc, st["color"]), (trunc, "none")):
            if not mask.any():
                continue
            if st["marker"] == "x":
                # An unfilled marker has no face to hollow out, so the truncated/
                # converged split cannot be shown by fill. It never needs to be:
                # `x` is only the arch-mismatched arm, which is excluded anyway.
                ax.scatter(vals[mask], y0 + jit[mask], s=34, marker="x",
                           c=st["color"], linewidths=1.2, zorder=3, clip_on=False)
            else:
                ax.scatter(vals[mask], y0 + jit[mask], s=34, marker=st["marker"],
                           facecolors=face, edgecolors=st["color"], linewidths=1.2,
                           zorder=3, clip_on=False)
        if len(vals) > 1:  # spread bar: min-max, so the reader sees range not SE
            ax.plot([vals.min(), vals.max()], [y0, y0], color=st["color"],
                    alpha=0.35, lw=1.0, zorder=2)

    ax.set_yticks([y for y, _ in labels])
    ax.set_yticklabels([t for _, t in labels] if show_ylabels else [], fontsize=6)
    ax.set_ylim(max(y for y, _ in labels) + 0.9, -0.9)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.set_xlabel("weighted F1" if target in CLASSIFICATION else "R²", fontsize=7)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_title(target.replace("target_", ""), fontsize=9)

    # The two fit blocks, named inside the panel so a reader never has to hunt.
    # `mid` is where the aug block starts: the rows were built clean-then-aug over
    # the same arm list, so it is exactly half — computed, not assumed, because an
    # encoder missing an arm in one block would break the halving.
    if rows:
        n_clean = sum(1 for _, _arm, pts in rows if pts[0]["fit"] == "clean")
        ax.axhline(rows[n_clean][0] - 0.85, color="0.85", lw=0.8)
        for start, name in ((0, "fit = CLEAN"), (n_clean, "fit = AUG")):
            ax.text(0.985, rows[start][0] - 0.62, name, transform=ax.get_yaxis_transform(),
                    fontsize=6.5, color="0.3", va="bottom", ha="right", style="italic")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=REPO_ROOT / "runs/pdf/sweep")
    ap.add_argument("--supervised", type=Path, default=REPO_ROOT / "runs/pdf/supervised")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "figures" / "ssl_vs_scratch.png")
    ap.add_argument("--epochs", type=int, default=None,
                    help="keep only runs whose finetune_cfg.epochs is N. Omit and the "
                         "run dirs must already agree; a mixture is refused, not averaged")
    args = ap.parse_args()

    data = load(args.runs, args.supervised, args.epochs)
    if not data:
        where = f"{args.runs} or {args.supervised}"
        raise SystemExit(f"no finetune.json under {where}"
                         + (f" at epochs={args.epochs}" if args.epochs else ""))

    # REFUSE a mixture rather than average across it. `analysis/finetune.py` rewrites
    # finetune.json in place, so a draining resweep leaves runs/pdf/sweep/ genuinely mixed and
    # nothing in the files marks which is which. Averaging two epoch budgets into one
    # arm is not a caveat, it is a wrong number.
    ceilings = ceilings_by_arm(data)
    mixed = {k: v for k, v in ceilings.items() if len(v) > 1}
    if mixed:
        lines = "\n".join(f"    {e}/{a}: " + ", ".join(f"{n} run(s) @ {c} epochs"
                                                       for c, n in sorted(v.items()))
                          for (e, a), v in sorted(mixed.items()))
        raise SystemExit(
            "REFUSING to plot: these arms mix epoch budgets, so a point's score would\n"
            "depend on how long its run was allowed rather than on the arm it belongs to.\n"
            f"{lines}\n"
            "  This is normal while an epoch-matched resweep is still draining — wait for\n"
            "  it, or pass --epochs N to select one budget explicitly."
        )

    encoders = [e for e in ("cnn", "transformer") if any(k[0] == e for k in data)]
    n_trunc = sum(p["truncated"] for k, v in data.items() for p in v if k[1] != "scratch")
    n_ssl = sum(len(v) for k, v in data.items() if k[1] != "scratch")
    for (enc, arm), pts in sorted(data.items()):
        ceil = next(iter(ceilings[(enc, arm)]))  # single-valued: the guard above proved it
        print(f"  {enc:12s} {arm:16s} {len(pts) // (len(TARGETS) * 2)} run(s), "
              f"{len(pts)} points, {ceil} epochs")
    print(f"SSL points hitting the epoch ceiling: {n_trunc}/{n_ssl}")

    # Per-ARM ceilings are single-valued now, but the two arms can still differ from
    # each other — that IS precondition 1, and the caption must state whichever is true
    # rather than hardcoding the pre-resweep story.
    ssl_ceils = {c for k, v in ceilings.items() if k[1] != "scratch" for c in v}
    scr_ceils = {c for k, v in ceilings.items() if k[1] == "scratch" for c in v}
    if ssl_ceils == scr_ceils:
        budget_note = (f"Both arms ran the SAME {next(iter(ssl_ceils))}-epoch ceiling "
                       f"(precondition 1 CLOSED); {n_trunc}/{n_ssl} SSL points still "
                       f"reached it (hollow).")
    else:
        budget_note = (f"CAVEAT: the SSL arm ran {sorted(ssl_ceils)} epochs and "
                       f"{n_trunc}/{n_ssl} of its points hit that ceiling (hollow = lower "
                       f"bound); the scratch arm ran {sorted(scr_ceils)}. Budgets DIFFER "
                       f"— precondition 1 is open and this favors scratch.")

    fig, axes = plt.subplots(len(encoders), len(TARGETS),
                             figsize=(4.1 * len(TARGETS), 4.6 * len(encoders)),
                             squeeze=False)
    for r, enc in enumerate(encoders):
        for c, target in enumerate(TARGETS):
            panel(axes[r][c], data, enc, target, show_ylabels=(c == 0))
        axes[r][0].set_ylabel(enc.upper(), fontsize=12, labelpad=10)

    handles = [plt.Line2D([], [], marker=s["marker"], color=s["color"], ls="none",
                          markersize=6, label=s["label"]) for s in ARM_STYLE.values()]
    handles.append(plt.Line2D([], [], marker="o", color="0.3", ls="none", markersize=6,
                              markerfacecolor="none", label="hollow = hit epoch ceiling (LOWER BOUND)"))
    # Legend at the BOTTOM: the suptitle is three lines and they collide at the top.
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.002),
               fontsize=8.5, ncol=5, framealpha=0.9)

    fig.suptitle(
        "Can SSL pretraining compete with in-domain supervised training?  "
        "— evaluation held fixed on AUGMENTED CHILI in every cell\n"
        "Each dot is one trained model. Compare 'scratch' vs 'SSL' WITHIN a fit block "
        "(value of pretraining); compare the CLEAN block vs the AUG block (size of the "
        "domain gap).\n"
        + budget_note
        + " Note the scratch arm still got an LR search the SSL arm never had "
          "(precondition 2, open, favors scratch).",
        fontsize=10.5, y=0.985, va="top",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.905))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
