"""analysis/chili_invariance_hist.py — the view-invariance histograms on CHILI-3K.

    python -m analysis.chili_invariance_hist

The corpus figure (`analysis/plot_invariance_hist.py`) draws same-material against
different-material cosine distances over banked pretraining view pairs. This is the
same figure on structures the encoder never saw, from channels that already exist in
the CHILI-3K registry — no new simulation:

  view A  `signal_xpdf`      CHILI's clean native xPDF, nominal instrument
  view B  `signal_xpdf_aug`  the same nanoparticle re-simulated at one draw from the
                             pretraining `PDF_RANGES` (`tools/augment_chili.py`)

TWO PROTOCOL DEVIATIONS FROM THE CORPUS FIGURE, both forced by what exists and both
belonging in the caption. (1) The pair is (nominal, production draw), not two
production draws — the corpus positive pair straddles two draws, this one anchors at
the clean signal. (2) The 3180 rows are 636 parent crystals x 5 particle sizes, so
the different-material histogram contains same-crystal different-size pairs; they are
different physical particles, and all rows are kept by decision (2026-09-03).

Everything else is imported from the corpus module — `cosine_distances`, `draw`, the
styles — so the two figures cannot drift apart in measurement or rendering. Loading
is `chili_theta_probe.load_chili_views`, the registry-channel counterpart of
`analysis.latent.load_views` (grid checked and sliced, min-max per row).

`import torch` comes first on purpose: importing pandas or `core.*` ahead of it
segfaults with no traceback (exit 139) — see docs/ENVIRONMENT.md.
"""

from __future__ import annotations

import torch  # MUST precede pandas / core.* — see module docstring

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from analysis.chili_theta_probe import DEFAULT_REGISTRY, DEFAULT_RUN, load_chili_views
from analysis.downstream_eval import channel_contract, embed, load_encoder
from analysis.plot_invariance_hist import (INPUT_LABEL, _style, cosine_distances, draw,
                                           median_examples, pair_stats)
from core.train import build_model, pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN,
                        help="run dir whose ckpt_best.pt is drawn (the thesis exemplar)")
    parser.add_argument("--control-seed", type=int, default=0,
                        help="torch seed for the random-init control weights")
    parser.add_argument("--device", default="auto", help='"auto" | "cpu" | "cuda"')
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default="figures/chili_invariance_hist.pdf")
    parser.add_argument("--paper", action="store_true",
                        help="thesis render: paperstyle sizing, ps.save under figures/ "
                             "with --out's stem; without it, the diagnostic canvas")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, cfg, _ = load_encoder(args.run / "ckpt_best.pt", device)
    A = load_chili_views(args.registry, "xpdf", cfg.signal_len)
    B = load_chili_views(args.registry, "xpdf_aug", cfg.signal_len)
    print(f"{args.registry.name}: {len(A)} materials, views (xpdf, xpdf_aug), "
          f"encoder {cfg.encoder} on {device}", flush=True)

    torch.manual_seed(args.control_seed)
    control = build_model(cfg).to(device).eval()
    for p in control.parameters():
        p.requires_grad_(False)

    reps = [
        (INPUT_LABEL["pdf"], A, B),
        ("Untrained encoder", embed(control, A, device, args.batch_size),
         embed(control, B, device, args.batch_size)),
        ("Pretrained encoder", embed(model, A, device, args.batch_size),
         embed(model, B, device, args.batch_size)),
    ]
    panels = [(title, *cosine_distances(HA, HB)) for title, HA, HB in reps]

    # Same extended stats as plot_invariance_hist.main, same rank estimator.
    from analysis.latent import participation_ratio, spectrum
    for (title, HA, _HB), (_t, same, diff) in zip(reps, panels):
        st = pair_stats(same, diff)
        print(f"  {title:<24} median same {np.median(same):.2e}  "
              f"median diff {np.median(diff):.2e}  ratio {np.median(diff) / np.median(same):.1f}x  "
              f"AUROC {st['auroc']:.4f}  same>p10(diff) {100 * st['tail_frac']:.1f}%  "
              f"eff.rank {participation_ratio(spectrum(HA)):.1f}", flush=True)

    if args.paper:
        from analysis import paperstyle as ps
        ps.use()
    else:
        _style()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The example row's positive pair is (nominal xpdf, production redraw) — the same
    # (A_i, B_i) convention the histograms measure. x is the pretraining r-grid the
    # registry channel was checked against and sliced to.
    draw(panels, out, examples=median_examples(A, B),
         x=channel_contract("xpdf", cfg.signal_len)[0], modality="pdf",
         paper=args.paper)


if __name__ == "__main__":
    main()
