"""analysis/chili_theta_probe.py — the theta-decodability table on CHILI-3K's varied channel.

    python -m analysis.chili_theta_probe

The corpus table (`analysis/theta_probe.py`) asks whether the instrument is linearly
decodable from the frozen embedding of banked PRETRAINING views. This is the same
question on structures the encoder never saw: CHILI-3K's `signal_xpdf_aug` channel —
one Debye-simulated view per nanoparticle at instrument params drawn from the
pretraining `PDF_RANGES` (`tools/augment_chili.py`), with the four draws stored as
registry columns. No new simulation: the views and their theta already exist.

SAME ESTIMATOR, ONE-VIEW DESIGN. `theta_probe.probe` is imported, not re-derived:
ridge on the frozen embedding, lambda by group-matched inner CV, pooled out-of-fold
R2, 5 folds x 3 repeats. The corpus measurement validated the 1-view design against
the 8-view one (its docstring), and one view per row is all this channel has.

CV IS GROUPED ON THE PARENT CRYSTAL, NOT THE ROW. CHILI-3K's 3180 rows are 53 metals
x 12 prototypes x 5 particle sizes; the 5 sizes of one (metal, prototype) share one
parent cell and are near-duplicate signals, so they must land on one side of every
fold split — the direct analogue of the corpus probe's material grouping (636 groups).

THE RAW BASELINE IS THE VARIED SIGNAL ITSELF (min-max normalized, the encoder-input
convention), because that is the channel carrying theta; the clean native channel is
constant in theta by construction and can decode nothing. Random-init control per the
corpus protocol: one untrained copy of the exemplar geometry, same seed convention.

`import torch` comes first on purpose: importing pandas or `core.*` ahead of it
segfaults with no traceback (exit 139) — see docs/ENVIRONMENT.md.
"""

from __future__ import annotations

import torch  # MUST precede pandas / core.* — see module docstring

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.downstream_eval import channel_contract, embed, load_encoder
from analysis.theta_probe import _row, probe
from core.train import build_model, pick_device

__all__ = ["load_chili_views"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "data" / "downstream" / "chili" / "chili_registry.parquet"
DEFAULT_RUN = REPO_ROOT / "runs" / "pdf" / "sweep" / \
    "2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372"
DEFAULT_OUT = REPO_ROOT / "analysis" / "out" / "chili_theta_probe.json"

#: Draw order of `core.transforms.draw_pdf_params` — the order the aug_* columns were
#: filled in, kept so this table's rows line up with every other theta artifact.
PARAM_NAMES = ["uiso", "qmax", "qbroad", "qmin"]


def load_chili_views(registry: Path, signal: str, signal_len: int) -> torch.Tensor:
    """`(3180, signal_len)` normalized rows of one registry signal channel.

    The registry's r-grid is CHECKED against the pretraining grid and SLICED, never
    resampled — the same contract as `downstream_eval.load_probe_data`, minus the
    split filter (invariance is measured on every row, not a split).
    """
    df = pd.read_parquet(registry, columns=[f"signal_{signal}_x", f"signal_{signal}_y"])
    grid = np.asarray(df[f"signal_{signal}_x"].iloc[0], dtype=float)
    expected, normalize = channel_contract(signal, signal_len)
    if len(grid) < signal_len or not np.allclose(grid[:signal_len], expected,
                                                 atol=(expected[1] - expected[0]) / 100):
        raise SystemExit(f"{registry}: signal_{signal} grid does not lead with the "
                         f"pretraining grid — a resample would change every number")
    return torch.stack([
        normalize(torch.from_numpy(np.asarray(v, dtype=np.float32)[:signal_len].copy()))
        for v in df[f"signal_{signal}_y"]
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN,
                        help="run dir whose ckpt_best.pt is probed (the thesis exemplar)")
    parser.add_argument("--control-seed", type=int, default=0,
                        help="torch seed for the random-init control weights")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="auto", help='"auto" | "cpu" | "cuda"')
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    device = pick_device(args.device)
    model, cfg, epoch = load_encoder(args.run / "ckpt_best.pt", device)

    X = load_chili_views(args.registry, "xpdf_aug", cfg.signal_len)
    meta = pd.read_parquet(args.registry,
                           columns=["target_metal", "crystal_type",
                                    *(f"aug_{p}" for p in PARAM_NAMES)])
    theta = meta[[f"aug_{p}" for p in PARAM_NAMES]].to_numpy(dtype=np.float64)
    groups = (meta["target_metal"].astype(int).astype(str) + "|"
              + meta["crystal_type"]).to_numpy()
    print(f"{args.registry.name}: {len(X)} views x 1, {len(np.unique(groups))} parent-"
          f"crystal groups, params {PARAM_NAMES}, encoder {cfg.encoder} on {device}",
          flush=True)

    raw = probe(X, theta, groups, PARAM_NAMES, args.folds, args.repeats)
    print("  raw signal done", flush=True)

    H = embed(model, X, device, args.batch_size)
    pretrained = probe(H, theta, groups, PARAM_NAMES, args.folds, args.repeats)
    print(f"  {args.run.name} done", flush=True)

    torch.manual_seed(args.control_seed)
    control = build_model(cfg).to(device).eval()
    for p in control.parameters():
        p.requires_grad_(False)
    Hc = embed(control, X, device, args.batch_size)
    randominit = probe(Hc, theta, groups, PARAM_NAMES, args.folds, args.repeats)
    print("  random-init control done", flush=True)

    record = {
        "config": {"registry": str(args.registry), "signal": "xpdf_aug", "n_views": 1,
                   "n_rows": len(X), "grouped_on": "parent crystal (metal, prototype)",
                   "n_groups": int(len(np.unique(groups))), "run": args.run.name,
                   "epoch": epoch, "encoder": cfg.encoder, "folds": args.folds,
                   "repeats": args.repeats, "control_seed": args.control_seed},
        "raw_baseline": raw,
        "pretrained": pretrained,
        "randominit": randominit,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)

    head = " ".join(f"{p:>8}" for p in PARAM_NAMES)
    print(f"\n{'representation':<44} {head} {'mean':>8}")
    for name, scores in (("raw signal", raw), ("random-init encoder", randominit),
                         (args.run.name, pretrained)):
        print(_row(name, scores, PARAM_NAMES))
    print(f"\nPooled out-of-fold R2 per instrument parameter on CHILI-3K xpdf_aug, "
          f"parent-crystal-grouped {args.folds}-fold x {args.repeats} repeats, ridge "
          "(lambda by grouped inner CV). Read a null against the raw row.")


if __name__ == "__main__":
    main()
