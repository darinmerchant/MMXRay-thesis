"""analysis/chem_lr_select.py — settle Q7 section 3b: does SSL pretraining help under a
chemistry shift, once BOTH arms are tuned on the fit set they are actually fitting?

    python -m analysis.chem_lr_select

s38 measured pretrained vs random-init on the CHILI-100K fit set and the scratch arm
won by up to 0.414 R2. That reading was withdrawn: the arms did not share a backbone LR
(pretrained `lr_enc=1e-5`, scratch 1e-3/1e-4 — 10-100x on exactly the weights a
chemistry shift must move). `scripts/chem_lr_search.slurm` sweeps 5 coupled LRs for
BOTH arms; this reads those cells, selects each arm's LR on val loss, and prints the
comparison at each arm's own optimum. Only then do the arms differ in initialisation
alone, which is the only form in which the question has an answer.

SELECTION RULE, following `analysis/ssl_lr_select.py`: minimum `val_loss` at
`best_epoch`, on CHILI-100K's val split, per (arm, checkpoint, target). **Test is never
read.** `tasks.*.cross_clean.r2` is scored on CHILI-3K TEST rows, and selecting on it
would hand this control the split it exists to report.

Selection is PER TARGET, unlike `ssl_lr_select` which used `target_np_size` as a proxy
to keep stage 1 cheap. Here every cell already ran all three targets, and each target is
a separate finetune with its own `best_epoch`, so a proxy would throw away information
that has already been paid for.

THE INCUMBENT IS CARRIED, NOT SWEPT. `1e-5/1e-4` (pretrained, from
`runs/pdf/chem_transfer_val20/`) and the old scratch LRs (from `runs/pdf/chem_scratch/`, seed 42)
are DECOUPLED reference points, not grid cells: an optimum landing on one says the sweep
found nothing better, not that the grid was too narrow. Edges are judged over the 5
coupled points only — an argmin at 3e-5 or 3e-3 means the grid did not bracket the
minimum and that cell's selection is not a search result.

⚠️ `val_loss` at `best_epoch` is a MINIMUM over a noisy curve, i.e. an order statistic.
`ssl_lr_select.py` measured re-runs of the identical computation moving it by a median
9.4% (max 30.3%) on GPU nondeterminism, enough to move one argmin a full grid step. Read
a selection whose two best cells are within ~10% as a tie, not a winner.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from core.config import REPO_ROOT

NOISE = 0.0143
TARGETS = ("target_np_size", "target_mo_bond", "target_mean_bond")
#: The coupled grid `scripts/chem_lr_search.slurm` declares. Edges are judged against
#: THIS, not against whichever cells happen to be on disk.
GRID = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
STEMS = ("cnn_infonce_mpfull_final", "transformer_infonce_mpfull_final",
         "transformer_vicreg_mpfull_final")


def _stem(run: str) -> str | None:
    for s in STEMS:
        if s in run:
            return s
    return None


def _lr(run_dir: str) -> float | None:
    """The encoder LR out of the run-dir name (`..._enc0.001_head0.001_...`)."""
    m = re.search(r"_enc([0-9.e+-]+?)_head", run_dir)
    return float(m.group(1)) if m else None


def read(root: Path) -> dict[tuple[str, str, float], dict]:
    """(stem, target, lr) -> {val_loss, r2_clean, r2_aug, best_epoch}."""
    out: dict[tuple[str, str, float], dict] = {}
    for f in sorted(root.glob("**/finetune.json")):
        j = json.loads(f.read_text())
        stem, lr = _stem(j["run"]), _lr(f.parent.name)
        if stem is None or lr is None:
            continue
        for target in TARGETS:
            log = j.get("log", {}).get(target, {}).get("cross")
            best = j.get("best_epoch", {}).get(target, {}).get("cross")
            task = j.get("tasks", {}).get(target, {})
            if not log or best is None:
                continue
            out[(stem, target, lr)] = {
                "val_loss": min(e["val_loss"] for e in log),
                "best_epoch": best,
                "r2_clean": task.get("cross_clean", {}).get("r2"),
                "r2_aug": task.get("cross_aug", {}).get("r2"),
            }
    return out


def select(cells, stem, target):
    """Argmin val_loss over the COUPLED grid only. Returns (lr, cell, note)."""
    have = {lr: cells[(stem, target, lr)] for lr in GRID if (stem, target, lr) in cells}
    if not have:
        return None, None, "no cells"
    lr = min(have, key=lambda k: have[k]["val_loss"])
    note = ""
    if len(have) < len(GRID):
        note += f"[{len(have)}/{len(GRID)} cells] "
    if lr in (GRID[0], GRID[-1]):
        note += "EDGE — grid did not bracket the minimum; not a search result "
    # A near-tie is a tie: see the docstring's 9.4% note.
    second = sorted(have.values(), key=lambda c: c["val_loss"])
    if len(second) > 1 and second[1]["val_loss"] < second[0]["val_loss"] * 1.10:
        note += "(within 10% of runner-up — read as a tie) "
    return lr, have[lr], note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained-root", default="runs/pdf/chem_lr_pretrained")
    ap.add_argument("--scratch-root", default="runs/pdf/chem_lr_scratch")
    ap.add_argument("--pretrained-incumbent", default="runs/pdf/chem_transfer_val20",
                    help="the decoupled 1e-5/1e-4 cell, carried as a reference point")
    ap.add_argument("--scratch-incumbent", default="runs/pdf/chem_scratch",
                    help="the s38 scratch cells at their CHILI-3K-selected LR")
    args = ap.parse_args()

    arms = {}
    for name, root in (("pretrained", args.pretrained_root),
                       ("scratch", args.scratch_root)):
        cells = read(REPO_ROOT / root)
        if not cells:
            raise SystemExit(f"no cells under {root} — has the sweep landed?")
        arms[name] = cells
        print(f"{name:11s} {len({k[0] for k in cells})} checkpoints x "
              f"{len({k[2] for k in cells})} LRs from {root}")

    inc = {"pretrained": read(REPO_ROOT / args.pretrained_incumbent),
           "scratch": read(REPO_ROOT / args.scratch_incumbent)}

    print("\n=== selected LR per (arm, checkpoint, target), on CHILI-100K val loss ===")
    for stem in STEMS:
        print(f"\n--- {stem} ---")
        print(f"  {'target':18s} {'arm':11s} {'sel LR':>8} {'val_loss':>9} "
              f"{'cross_clean':>12} {'incumbent':>10}  note")
        for target in TARGETS:
            for arm in ("pretrained", "scratch"):
                lr, cell, note = select(arms[arm], stem, target)
                if lr is None:
                    print(f"  {target:18s} {arm:11s} {'--':>8}  {note}")
                    continue
                ic = [c for (s, t, _), c in inc[arm].items() if s == stem and t == target]
                istr = f"{ic[0]['r2_clean']:>10.3f}" if ic and ic[0]["r2_clean"] is not None else f"{'--':>10}"
                print(f"  {target:18s} {arm:11s} {lr:>8.0e} {cell['val_loss']:>9.4f} "
                      f"{cell['r2_clean']:>12.3f} {istr}  {note}")

    print("\n=== THE ANSWER: cross_clean at each arm's OWN selected LR ===")
    print(f"  {'checkpoint':34s} {'target':18s} {'pretrained':>11} {'scratch':>9} {'delta':>8}")
    verdict = []
    for stem in STEMS:
        for target in TARGETS:
            lp, cp, _ = select(arms["pretrained"], stem, target)
            ls, cs, _ = select(arms["scratch"], stem, target)
            if cp is None or cs is None:
                continue
            d = cp["r2_clean"] - cs["r2_clean"]
            verdict.append(d)
            mark = "" if abs(d) <= NOISE else ("  pretrained" if d > 0 else "  scratch")
            print(f"  {stem:34s} {target:18s} {cp['r2_clean']:>11.3f} "
                  f"{cs['r2_clean']:>9.3f} {d:>+8.3f}{mark}")
    if verdict:
        wins = sum(1 for d in verdict if d > NOISE)
        losses = sum(1 for d in verdict if d < -NOISE)
        print(f"\n  pretraining wins {wins}/{len(verdict)} cells beyond the {NOISE} floor, "
              f"loses {losses}, ties {len(verdict) - wins - losses}")
        print("  Both arms are now tuned on the fit set they fit, so this differs in "
              "INITIALISATION alone.")


if __name__ == "__main__":
    main()
