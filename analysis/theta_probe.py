"""analysis/theta_probe.py — is the instrument linearly DECODABLE from the embedding?

    python -m analysis.theta_probe --block data/banked/material_registry__n32__55ad1ff1 \\
        --filter '*_cov*' '*_temp*' '*mpfull_final*'

The s36 invariance numbers are variance-based — θ's share of embedding variance,
displacement per unit of θ — and variance cannot see a nuisance hiding in a
low-variance direction: θ could hold 0.3% of the variance and still be perfectly
readable by a linear head. This is the standard complement: ridge-regress each
instrument parameter from the frozen embedding `h` of banked pretraining views,
material-grouped CV, pooled out-of-fold R² (the `analysis/probe_cv.py` machinery).
Low R² corroborates the invariance claim as *decodability*, not just variance
share; high R² would expose a hidden instrument direction the variance measures
missed. Either answer is a result.

TWO BASELINES, SAME ROWS, SAME ESTIMATOR. The raw normalized signal answers "can θ
be read off G(r) at all?" — where it cannot, an encoder null means *the axis never
moved the signal*, not *the encoder discarded it* (expect this on `qmax`/`qmin`,
whose input-space sensitivity is near zero in `latent_theta`'s table). One
random-init control per geometry answers what the architecture alone decodes.

GROUPED ON MATERIAL: all views of a material land on one side of every fold split,
so the probe cannot lean on memorized per-material baselines. The targets are the
bank's own per-view draws — independent uniforms per parameter
(`core.transforms.draw_pdf_params`), so per-parameter univariate probes are
well-posed, and R²'s affine invariance means raw physical units need no scaling.

LINEAR ONLY, TRAINING CORPUS ONLY — the same limits as every Q4 measurement: a null
here does not bound a nonlinear readout, and no held-out MP materials exist.

ONE SUMMARY JSON (probe_cv's shape, not latent_theta's per-run files): this is a
cross-checkpoint comparison read as one table, and ~25 rows of four numbers do not
justify 25 files.

`--space z` probes the projector output instead of `h` (same estimator, same rows,
separate default JSON), to separate two readings of a parameter that decodes from
`h`: if it also decodes from `z`, the projector is not hiding it; if it decodes
from `h` but not `z`, the projection head is what removes it and the invariance
the loss enforces lives only past the projector.

`import torch` comes first on purpose: importing pandas or `core.*` ahead of it
segfaults with no traceback (exit 139) — see docs/ENVIRONMENT.md.
"""

from __future__ import annotations

import torch  # MUST precede pandas / core.* — see module docstring

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.downstream_eval import (_single_threaded, embed, embed_z, find_checkpoints,
                                      load_encoder)
from analysis.latent import arch_name, arch_signature
from analysis.latent_theta import _check_grid, load_views
from analysis.probe_cv import grouped_folds
from core.train import build_model, pick_device

__all__ = ["probe"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "analysis" / "out" / "theta_probe.json"

#: Coarser than `_fit_ridge_gcv`'s 97-point grid: each point here costs an inner-CV
#: prediction rather than a closed form, and λ* only needs to land on the right
#: decade. Scaled by mean(s²) like the GCV grid, so it is dimensionless. The floor
#: sits at 1e-10 (was 1e-8 until 2026-09-03) so a near-interpolating fit selects an
#: interior λ rather than the grid edge; `probe` prints any edge selection.
_LAMBDAS = torch.logspace(-10, 4, 29, dtype=torch.float64)
_INNER_FOLDS = 3


def _svd_ridge(Hf: torch.Tensor, Hv: torch.Tensor):
    """One SVD of the centered fit rows → `predict(y, λ) -> val predictions`.

    The SVD depends only on the design, so all four θ targets and the whole λ grid
    reuse it: ridge coefficients are `V diag(s/(s²+λ)) Uᵀ y_c`, and the intercept
    is the fit mean (centering = unpenalized bias, same convention as
    `downstream_eval._fit_ridge_gcv`).
    """
    mu = Hf.mean(dim=0)
    U, S, Vh = torch.linalg.svd(Hf - mu, full_matrices=False)
    G = (Hv - mu) @ Vh.T  # (n_val, k) — val rows in the right singular basis

    def predict(y: torch.Tensor, lam: float) -> torch.Tensor:
        y_c = y - y.mean()
        coef = (S / (S**2 + lam)) * (U.T @ y_c)
        return G @ coef + y.mean()

    return predict, S


def probe(H: torch.Tensor, theta: np.ndarray, mat: np.ndarray, param_names: list[str],
          n_folds: int, repeats: int) -> dict:
    """`{param: {"r2_mean", "r2_per_repeat", "n_labelled"}}` — pooled OOF R² per θ.

    Ridge with **λ selected by material-grouped inner CV**, NOT by
    `downstream_eval._fit_ridge_gcv`. GCV assumes iid rows, and this design has 8
    views per material that are near-duplicates — *especially* under an invariant
    encoder, which is the thing being measured. Measured on the exemplar
    checkpoint: GCV under-regularizes and the grouped OOF R² lands at −0.5 on
    signals a properly selected λ scores ≥ 0, and the raw-vs-pretrained ordering
    flips between 8-view and 1-view designs. Inner selection on the same grouping
    as the outer folds makes the estimator honest under duplication, and the two
    designs agree again. (docs/TRAPS.md → *GCV under grouped rows*.)

    Everything is torch float64; SVDs are shared across the λ grid and all four
    targets, so the inner CV costs `(inner+1) SVDs` per outer fold, not per cell.
    """
    Y = {p: torch.from_numpy(theta[:, i].astype(np.float64))
         for i, p in enumerate(param_names)}
    H = H.double()
    out = {p: [] for p in param_names}
    edge = {p: [0, 0] for p in param_names}  # (low, high) λ-grid edge selections

    with _single_threaded():
        for rep in range(repeats):
            folds = grouped_folds(mat, n_folds, seed=rep)
            oof = {p: np.full(len(mat), np.nan) for p in param_names}
            for held in folds:
                keep = np.setdiff1d(np.arange(len(mat)), held)
                # λ*: pooled inner-val SSE over material-grouped inner splits,
                # one SVD per inner split, all λ and all targets sharing it.
                inner = grouped_folds(mat[keep], _INNER_FOLDS, seed=rep)
                sse = {p: np.zeros(len(_LAMBDAS)) for p in param_names}
                scale = None
                for iheld in inner:
                    ikeep = np.setdiff1d(np.arange(len(keep)), iheld)
                    predict, S = _svd_ridge(H[keep[ikeep]], H[keep[iheld]])
                    scale = float((S**2).mean()) if scale is None else scale
                    for p in param_names:
                        y = Y[p][keep]
                        for j, lam in enumerate(_LAMBDAS):
                            pred = predict(y[ikeep], float(lam) * scale)
                            sse[p][j] += float(((y[iheld] - pred) ** 2).sum())
                predict, S = _svd_ridge(H[keep], H[held])
                scale = float((S**2).mean())
                for p in param_names:
                    j = int(np.argmin(sse[p]))
                    if j == 0:
                        edge[p][0] += 1
                    elif j == len(_LAMBDAS) - 1:
                        edge[p][1] += 1
                    oof[p][held] = predict(Y[p][keep], float(_LAMBDAS[j]) * scale).numpy()
            for p in param_names:
                truth = Y[p].numpy()
                assert np.isfinite(oof[p]).all()
                ss_res = float(((truth - oof[p]) ** 2).sum())
                ss_tot = float(((truth - truth.mean()) ** 2).sum())
                out[p].append(1.0 - ss_res / ss_tot)

    # A high-edge selection is the shrink-to-intercept limit — the right answer for a
    # parameter with no linear signal; a low-edge one means the grid floor bound the
    # fit, which extending the grid is supposed to make impossible. Printed, never silent.
    for p in param_names:
        lo, hi = edge[p]
        if lo or hi:
            print(f"    note: {p} λ at grid edge in {lo} low / {hi} high of "
                  f"{n_folds * repeats} selections", flush=True)

    return {p: {"r2_mean": float(np.mean(v)), "r2_per_repeat": [float(x) for x in v],
                "n_labelled": len(mat)}
            for p, v in out.items()}


def _row(name: str, scores: dict, param_names: list[str]) -> str:
    vals = [scores[p]["r2_mean"] for p in param_names]
    cells = " ".join(f"{v:>8.3f}" for v in vals)
    return f"{name:<44} {cells} {np.mean(vals):>8.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block", required=True,
                        help="banked block dir; whichever shard_*.npy are present are used")
    parser.add_argument("--runs", default="runs/pdf/sweep", help="dir whose */ckpt_best.pt are measured")
    parser.add_argument("--filter", nargs="+", default=["*"],
                        help="globs over run-dir names, unioned")
    parser.add_argument("--n-materials", type=int, default=512,
                        help="default matches latent_theta's draw, so the probe complements "
                             "the s36 measurement on the SAME point set")
    parser.add_argument("--n-views", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--control-seed", type=int, default=0,
                        help="torch seed for the random-init control weights")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="auto", help='"auto" | "cpu" | "cuda"')
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--space", choices=("h", "z"), default="h",
                        help='"h" (encoder representation, the default) or "z" (projector output)')
    parser.add_argument("--out", type=Path, default=None,
                        help="default theta_probe.json for --space h, theta_probe_z.json for z")
    args = parser.parse_args()
    if args.out is None:
        args.out = DEFAULT_OUT if args.space == "h" else DEFAULT_OUT.with_name("theta_probe_z.json")
    embed_fn = embed if args.space == "h" else embed_z

    device = pick_device(args.device)
    ckpts = sorted({p for pattern in args.filter for p in find_checkpoints(args.runs, pattern)})
    if not ckpts:
        raise SystemExit(f"no */ckpt_best.pt under {args.runs} matching {args.filter}")

    X, mat, theta, info = load_views(args.block, args.n_materials, args.n_views,
                                     args.sample_seed)
    param_names = info["param_names"]
    print(f"{info['block']}: shards {info['shards_present']} -> {info['n_materials']} "
          f"materials x {info['n_views_per_material']} views, params {param_names}, "
          f"{len(ckpts)} checkpoints on {device}", flush=True)

    # Raw signal first, before any encoder, so a failure here is not mistaken for a
    # checkpoint problem — and because every other row is read against it.
    raw = probe(X, theta, mat, param_names, args.folds, args.repeats)
    print(f"  raw signal done", flush=True)

    models: dict[str, dict] = {}
    control_cfgs: dict[str, tuple[tuple, object]] = {}
    for ckpt in ckpts:
        model, cfg, epoch = load_encoder(ckpt, device)
        _check_grid(cfg, info, ckpt.parent.name)
        name, sig = arch_name(cfg), arch_signature(cfg)
        if control_cfgs.setdefault(name, (sig, cfg))[0] != sig:
            raise SystemExit(f"two different geometries both name their control {name!r}: "
                             f"{control_cfgs[name][0]} vs {sig} — widen arch_name()")
        H = embed_fn(model, X, device, args.batch_size)
        models[ckpt.parent.name] = {
            "encoder": cfg.encoder, "arch": name, "epoch": epoch, "pretrained": True,
            "params": probe(H, theta, mat, param_names, args.folds, args.repeats),
        }
        print(f"  {ckpt.parent.name} done", flush=True)

    controls: dict[str, dict] = {}
    for name, (_, cfg) in sorted(control_cfgs.items()):
        torch.manual_seed(args.control_seed)
        model = build_model(cfg).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        H = embed_fn(model, X, device, args.batch_size)
        controls[f"randominit_{name}"] = {
            "encoder": cfg.encoder, "arch": name, "epoch": None, "pretrained": False,
            "params": probe(H, theta, mat, param_names, args.folds, args.repeats),
        }
        print(f"  randominit_{name} done", flush=True)

    record = {
        "config": {**info, "runs": args.runs, "filter": args.filter, "folds": args.folds,
                   "repeats": args.repeats, "control_seed": args.control_seed,
                   "space": args.space,
                   "ridge": "lambda by material-grouped inner CV "
                            f"({_INNER_FOLDS} folds, {len(_LAMBDAS)}-point grid)",
                   "grouped_on": "material"},
        "raw_baseline": raw,
        "models": models,
        "controls": controls,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)

    head = " ".join(f"{p:>8}" for p in param_names)
    print(f"\n{'run':<44} {head} {'mean':>8}")
    print(_row("raw signal", raw, param_names))
    order = lambda kv: (kv[1]["arch"], "vicreg" not in kv[0], kv[0])
    current_arch = None
    for run, rec in sorted({**models, **controls}.items(), key=order):
        if rec["arch"] != current_arch:
            current_arch = rec["arch"]
            print(f"-- {current_arch}")
        print(_row(run, rec["params"], param_names))
    print("\nPooled out-of-fold R² per instrument parameter, material-grouped "
          f"{args.folds}-fold x {args.repeats} repeats, ridge (λ by grouped inner CV).\n"
          "Read a null against the raw row: where raw itself is ~0 the axis barely "
          "moves the signal, and an encoder null says nothing about discarding.")


if __name__ == "__main__":
    main()
