"""tools/plot_appendix_rruff_labels.py — RRUFF's crystal-system label distribution.

    /opt/anaconda3/envs/mmxray/bin/python -m tools.plot_appendix_rruff_labels

ONE PANEL, ONE TARGET. The RRUFF registry carries ten `target_*` columns, but the
paper predicts on exactly one — crystal system. The other nine were measured
near-unlearnable or are auxiliary (`data/builders/rruff.py`: cell parameters top
out at R2 = 0.16 fit directly on the raw signal, and that is a property of the
data at n=148, not of any model), so plotting them here would present nine panels
of distributions nothing is scored on. `analysis/target_distributions.py` still
draws all ten per split when the data itself needs checking.

Same conventions as `tools/plot_appendix_labels.py` (A.3.3), for the same reasons
recorded there: ALL 148 MINERALS POOLED with no split breakdown, the y axis is
SHARE of labeled samples, ONE HUE (slot 1) because nothing is distinguished by
colour, no legend, and NO n ON THE FIGURE — the count and the modal-floor pair
print to stdout so a caption can quote them without re-deriving. The pooled modal
weighted F1 and the fit-split convention (train-modal scored on test, the same
call `analysis/downstream_eval.py` makes) are printed side by side.

TICKS ARE THE CLASS NAMES, IN IUCr CODE ORDER. The target column holds the
numeric IUCr code (targets are read as float32 downstream), and the registry
carries the readable twin `crystal_system` precisely so nothing has to decode an
integer by hand — the code→name mapping is read off the data and cross-checked,
rather than imported from `data/builders/rruff.py`, which would drag `core.*`
and torch into a module that otherwise needs neither. The order is the code
order, triclinic → cubic, because it is the symmetry ladder and alphabetizing it
would shuffle a meaningful axis.

NO `import torch` GUARD, same as `plot_appendix_labels.py` and for the same
reason: the import-order segfault (`docs/TRAPS.md`) only bites a process that
imports torch at all, and this one never does. Add the guard back the moment
this file imports anything from `core`.

Reads the registry only — no checkpoint, no run directory, nothing trained.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from analysis import paperstyle as ps

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data" / "downstream" / "rruff" / "rruff_registry.parquet"

TARGET = "target_crystal_system"


def _nice_top(m: float) -> tuple[float, float]:
    """(axis top, tick step) clearing `m` with ~10% headroom, on a round step.

    Same rule as `plot_appendix_labels.py`: computed rather than hardcoded so a
    registry change cannot silently clip a bar off the top of the panel.
    """
    for step in (0.02, 0.05, 0.1, 0.2):
        top = float(np.ceil(m * 1.10 / step) * step)
        if top / step <= 6:  # more than six gridless ticks is a ruler, not an axis
            return top, step
    raise ValueError(f"no round step fits a maximum of {m}")


def build(registry: Path, out_stem: str, sub: str | None) -> None:
    ps.use()
    df = pd.read_parquet(registry, columns=["split", TARGET, "crystal_system"])
    codes = df[TARGET].astype(int)

    # The code→name mapping comes off the data; a code with two names would mean
    # the registry's readable twin desynced from the target, which must not pass.
    by_code = dict(zip(codes, df["crystal_system"]))
    for c, name in zip(codes, df["crystal_system"]):
        if by_code[c] != name:
            raise ValueError(f"code {c} maps to both {by_code[c]!r} and {name!r}")
    order = sorted(by_code)  # IUCr order, triclinic → cubic

    share = np.array([float((codes == c).mean()) for c in order])

    fig, ax = plt.subplots(figsize=(ps.WIDTH, 2.2))
    ax.bar(np.arange(len(order)), share, 0.6, color=ps.SLOTS[0], zorder=3)
    # Rotated because "monoclinic" and "orthorhombic" touch when horizontal at the
    # shared 8 pt tick size — the names stay full and the type scale stays uniform.
    ax.set_xticks(np.arange(len(order)), [by_code[c] for c in order],
                  rotation=30, ha="right", rotation_mode="anchor")
    ax.set_title("Crystal system", loc="left", pad=3)
    ax.set_ylabel("share of labeled samples")
    top, step = _nice_top(float(share.max()))
    ax.set_ylim(0, top)
    ax.set_yticks(np.arange(0, top + step / 2, step))

    # stdout, never drawn — the figure carries nothing but its name and the bars.
    modal = codes.value_counts().idxmax()
    pooled = f1_score(codes, np.full(len(codes), modal), average="weighted",
                      zero_division=0)
    train_modal = codes[df.split == "train"].value_counts().idxmax()
    ev = codes[df.split == "test"]
    paper = f1_score(ev, np.full(len(ev), train_modal), average="weighted",
                     zero_division=0)
    print(f"  {TARGET} n={len(codes)} modal={by_code[modal]} "
          f"share={float((codes == modal).mean()):.3f} pooled_wF1={pooled:.3f} "
          f"paper_wF1(train-modal on test)={paper:.3f}", flush=True)
    for c in order:
        print(f"    {by_code[c]:14s} {int((codes == c).sum()):3d}  "
              f"{float((codes == c).mean()):.3f}", flush=True)

    ps.layout(fig)
    ps.save(fig, out_stem, sub=sub)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=str(REGISTRY))
    ap.add_argument("--out", default="appendix_rruff_labels")
    ap.add_argument("--sub", default="appendix")
    args = ap.parse_args()
    build(Path(args.registry), args.out, args.sub)


if __name__ == "__main__":
    main()
