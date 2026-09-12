"""analysis/invariance_decay.py — how far does theta move the representation?

    python -m analysis.invariance_decay --filter '*transformer_vicreg_mpfull_final*'

Reads the ladders `tools/simulate_ladder.py` wrote and measures, per instrument
parameter, how far the encoder's embedding moves as that parameter is swept — against
the same measurement on the raw signal.

THIS IS THE INVARIANCE MEASUREMENT. `analysis/latent.py` and
`analysis/latent_theta.py` measure the geometry of the representation at the
production draw: how far apart two views land, how much variance theta accounts for,
whether the space is collapsed. None of them is invariance in the sense the method
claims, because all of them are ratios of two scales and a ratio can be improved by
moving either one. Invariance is movement per unit of theta, which is a dose-response
curve, which is this.

THE Y-QUANTITY IS DISPLACEMENT FROM THE IN-RANGE MEAN, NOT FROM A REFERENCE VIEW, and
this is the one substantive departure from the reference repo's version of the figure
(`PDF_XRD_Fusion/PDF/Experiments/03_ssl_augmented/evaluate.ipynb`, cell 27). That one
measures cosine distance to a single randomly-drawn training view. Inside the
augmentation window that is really "how far from theta_0", so the curve's shape there
is set by wherever theta_0 happened to land — which is what puts a spurious dip in its
uiso panel. Here the reference is each material's MEAN embedding over the rungs inside
the production range, renormalized to the sphere. The curve then reads as "how far has
theta pushed this material from where it normally sits", it has no arbitrary constant
to defend, and it is the same KIND of quantity as the within-material reference line
(both are spreads), so comparing the two is exact rather than approximate.

⚠️ INPUTS ARE MIN-MAX NORMALIZED BEFORE ENCODING, and the reference repo's figure did
NOT do this. Its training dataloader normalizes every spectrum in `__getitem__`; its
evaluate notebook passes raw G(r) straight to `encode()`. Min-max shifts as well as
scales, and cosine distance is not invariant to a shift, so those curves were measured
on inputs the encoder had never seen. Reproduced here would have silently changed every
number. See docs/TRAPS.md.

THE INPUT-SPACE CURVE IS THE BASELINE, not the random-init encoder. A near-flat latent
curve has two explanations — the encoder discarded that axis, or the axis never moved
the signal — and only the input curve separates them. That matters concretely for
`qmax`, which barely perturbs G(r) inside [15, 30] because PDF resolution goes as
2pi/qmax and nothing degrades until qmax drops below ~10 (docs/TRAPS.md). A random-init
control was measured and rejected for this role: it collapses everything toward a point
(within 0.0007, between 0.0023 on MP-20), so it is destructive rather than neutral, and
its absolute displacements are small for a reason that has nothing to do with
invariance.

AGGREGATION IS MEDIAN AND INTERQUARTILE, not mean and standard deviation. The
reference figure's +-1sd bands go negative, which is meaningless for a distance and
fatal on the log axis the plotter uses.

`mean` AND `sd` ARE RECORDED ANYWAY (2026-08-27), so that claim is checkable from the
JSON rather than only assertable from this docstring, and MEASURING IT CONFIRMED IT:
on `cnn_vicreg_mpfull_final`, `mean - sd` is negative on 13 of 16 `qmin` latent rungs
and 12 of 16 `qmin` input rungs. `qmin` is the panel carrying the out-of-range
inversion, so the one band that cannot be drawn is the one over the figure's honest
half. The distributions are right-skewed there (mean/median 1.67 latent, 2.04 input),
which is why the two aggregations diverge at all.

`gmean` AND `gsd` ARE THE DRAWABLE STANDARD DEVIATION, and are what the plotters use
for a band. Mean and sd of `log d`, exponentiated: the band is `gmean x/ gsd` rather
than `mean +- sd`, so it is MULTIPLICATIVE, cannot reach zero, and is symmetric on the
log axis the figure actually has — which is the axis these distances are spread evenly
on. `gsd` is dimensionless: it is a FACTOR (1.9 means "x/1.9"), not a distance, so it
must never be quoted beside `sd` as if the two were the same units.

All four moments come off an array `curve` already returns and `measure` used to
discard, so recording them costs nothing but the JSON's size.

Writes `<run>/invariance_decay.json`. Plot-free, per the `aug_sweep.py` /
`plot_aug_sweep.py` split: a figure change must never risk the measurement.

`import torch` comes first on purpose: importing pandas or `core.*` ahead of it
segfaults with no traceback (exit 139) — see docs/ENVIRONMENT.md.
"""

from __future__ import annotations

import torch  # MUST precede pandas / core.* — see module docstring

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.downstream_eval import embed, find_checkpoints, load_encoder

__all__ = ["unit", "displacement", "curve", "reference_lines", "measure"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LADDER = REPO_ROOT / "runs/pdf/latent" / "invariance"

#: Quantiles of the per-material displacement drawn as the band. p50 is the line.
QUANTILES = (0.25, 0.5, 0.75)


def minmax_rows(g: np.ndarray) -> np.ndarray:
    """Per-signal min-max to [0, 1] over the last axis — the encoder-input convention.

    `core.transforms.minmax_normalize` refuses anything but a 1-D signal by design, so
    the batched form lives here rather than looping a 16k-row array through it.
    """
    lo = g.min(axis=-1, keepdims=True)
    hi = g.max(axis=-1, keepdims=True)
    return (g - lo) / (hi - lo + 1e-8)


def unit(X: np.ndarray) -> np.ndarray:
    """Rows L2-normalized, in float64. Cosine distance is defined on the sphere."""
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


def displacement(P: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """`1 - cos(P_ml, ref_m)` for points `(n_levels, n_materials, d)` against `(n_materials, d)`."""
    return 1.0 - (unit(P) * unit(ref)[None, :, :]).sum(axis=-1)


def curve(P: np.ndarray, in_range: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(quantile_curves, per_material_displacement)` for one ladder in one space.

    The reference is the mean over the IN-RANGE rungs, per material. Averaging first
    and normalizing after is deliberate: the mean of unit vectors is not a unit vector,
    and normalizing each rung before averaging would weight every rung equally on the
    sphere rather than letting the mean sit where the points actually are.
    """
    ref = P[in_range].mean(axis=0)
    d = displacement(P, ref)                      # (n_levels, n_materials)
    return np.quantile(d, QUANTILES, axis=1), d


#: Below this a cosine distance is "identical at input precision" — the same constant
#: and the same reason as `analysis/plot_invariance_hist.FLOOR`. It matters here because
#: `log` of a perfectly invariant rung (~1e-16, or marginally negative from float error)
#: is either -37 or undefined, and one such material would dominate a log-space mean.
LOG_FLOOR = 1e-7


def moments(d: np.ndarray) -> dict:
    """Per-rung spread over materials, arithmetic and geometric.

    `mean`/`sd` are recorded so a reader can check that `mean - sd` runs negative on
    this data rather than taking the module docstring's claim on trust. `gmean`/`gsd`
    are the pair a plotter should draw: the band is `gmean x/ gsd`, which cannot reach
    zero and is symmetric on a log axis. `ddof=1` throughout — the 256 materials are a
    sample of the corpus, not the corpus.
    """
    log_d = np.log(np.clip(d, LOG_FLOOR, None))
    return {"mean": d.mean(axis=1).tolist(), "sd": d.std(axis=1, ddof=1).tolist(),
            "gmean": np.exp(log_d.mean(axis=1)).tolist(),
            "gsd": np.exp(log_d.std(axis=1, ddof=1)).tolist(),
            "n_at_log_floor": int((d < LOG_FLOOR).sum())}


def reference_lines(H_refs: np.ndarray, H_nominal: np.ndarray, rng) -> dict:
    """The two horizontal lines, in whichever space `H_*` are embeddings of.

    within  mean cosine distance between the two production-range draws of one
            material — literally the positive pair the encoder was trained on, so it
            is the floor a perfectly invariant encoder would sit at.
    between mean cosine distance between DIFFERENT materials at nominal theta — the
            scale at which identity is encoded, i.e. how far theta would have to push
            a material before it landed on another one.
    """
    a, b = unit(H_refs[0]), unit(H_refs[1])
    within = float((1.0 - (a * b).sum(axis=-1)).mean())

    n = len(H_nominal)
    i = rng.integers(0, n, size=n)
    j = (i + rng.integers(1, n, size=n)) % n      # never self, no rejection loop
    U = unit(H_nominal)
    between = float((1.0 - (U[i] * U[j]).sum(axis=-1)).mean())
    return {"within_material": within, "between_material": between}


def measure(model, ladder: Path, manifest: dict, device, batch_size: int, seed: int) -> dict:
    """Both spaces, all four parameters, plus the two reference lines in each."""
    params = list(manifest["levels"])
    out: dict = {"parameters": {}}

    refs = np.load(ladder / "refs.npz")["g"]                       # (2, n_mat, grid)
    X_refs = minmax_rows(refs)
    H_refs = np.stack([embed(model, torch.from_numpy(x).float(), device, batch_size).numpy()
                       for x in X_refs])

    for p in params:
        z = np.load(ladder / f"ladder_{p}.npz")
        levels, g = z["levels"], z["g"]                            # (n_lev,), (n_lev, n_mat, grid)
        lo, hi = manifest["production_ranges"][p]
        in_range = (levels >= lo) & (levels <= hi)
        if not in_range.any():
            raise SystemExit(f"{p}: no ladder rung inside the production range — no reference")

        X = minmax_rows(g)
        if not (X.min() >= -1e-6 and X.max() <= 1 + 1e-6):
            raise SystemExit(f"{p}: normalized input outside [0, 1] — the encoder is being fed "
                             f"something it was not trained on (docs/TRAPS.md)")

        flat = torch.from_numpy(X.reshape(-1, X.shape[-1])).float()
        H = embed(model, flat, device, batch_size).numpy().reshape(len(levels), len(g[0]), -1)

        q_lat, d_lat = curve(H, in_range)
        q_inp, d_inp = curve(X, in_range)

        # Nominal is the rung the ladder pins the OTHER three parameters at, so it is
        # the natural place to read a between-material scale that no sweep has moved.
        nominal_i = int(np.argmin(np.abs(levels - manifest["nominal"][p])))
        rng = np.random.default_rng(seed)
        out["parameters"][p] = {
            "levels": [float(v) for v in levels],
            "in_range": [bool(v) for v in in_range],
            "production_range": [float(lo), float(hi)],
            "latent": {"p25": q_lat[0].tolist(), "p50": q_lat[1].tolist(),
                       "p75": q_lat[2].tolist(), **moments(d_lat)},
            "input": {"p25": q_inp[0].tolist(), "p50": q_inp[1].tolist(),
                      "p75": q_inp[2].tolist(), **moments(d_inp)},
            "reference_latent": reference_lines(H_refs, H[nominal_i], rng),
            "reference_input": reference_lines(X_refs, X[nominal_i],
                                               np.random.default_rng(seed)),
        }
        print(f"  [{p}] latent p50 in-range max {q_lat[1][in_range].max():.5f}, "
              f"out-of-range max {q_lat[1][~in_range].max():.5f}", flush=True)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDER,
                        help="dir written by tools/simulate_ladder.py")
    parser.add_argument("--runs", default="runs/pdf/sweep", help="dir whose */ckpt_best.pt are measured")
    parser.add_argument("--filter", default="*", help="glob over run-dir names")
    parser.add_argument("--device", default="auto", help='"auto" | "cpu" | "cuda"')
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0, help="between-material pair draw")
    args = parser.parse_args()

    from core.train import pick_device

    manifest = json.loads((args.ladder / "manifest.json").read_text())
    device = pick_device(args.device)
    ckpts = find_checkpoints(args.runs, args.filter)
    if not ckpts:
        raise SystemExit(f"no */ckpt_best.pt under {args.runs} matching {args.filter!r}")

    grid_len = np.load(args.ladder / "refs.npz")["g"].shape[-1]
    print(f"{manifest['registry']}: {manifest['n_materials']} materials, grid {grid_len}, "
          f"{len(ckpts)} checkpoints on {device}", flush=True)

    for ckpt in ckpts:
        model, cfg, epoch = load_encoder(ckpt, device)
        # The Transformer bakes signal_len into its positional embedding; a mismatch
        # otherwise fails deep in the forward naming neither the ladder nor the run.
        if cfg.encoder == "transformer" and cfg.signal_len != grid_len:
            raise SystemExit(f"{ckpt.parent.name}: signal_len {cfg.signal_len} != ladder grid "
                             f"{grid_len}")
        print(f"{ckpt.parent.name}", flush=True)
        record = {"run": ckpt.parent.name, "epoch": epoch, "encoder": cfg.encoder,
                  "ladder": str(args.ladder), **{k: v for k, v in manifest.items()
                                                 if k != "material_ids"},
                  **measure(model, args.ladder, manifest, device, args.batch_size, args.seed)}
        (ckpt.parent / "invariance_decay.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"  wrote {ckpt.parent / 'invariance_decay.json'}", flush=True)


if __name__ == "__main__":
    main()
