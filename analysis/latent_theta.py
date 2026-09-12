"""analysis/latent_theta.py — WHICH instrument axes did the encoder discard?

    python -m analysis.latent_theta --block data/banked/material_registry__n32__55ad1ff1 \
        --filter '*transformer_vicreg_mpfull_final*'

The companion to `analysis/latent.py`, and deliberately a separate module. That one
answers "is the representation collapsed, and how close do two views of a material
land" with three GLOBAL scalars on a two-view sample. This one answers the question
those scalars cannot: **the encoder was trained to ignore a four-parameter
instrument model — which of the four did it actually stop seeing?**

Nothing here is a summary of `latent.py`'s numbers at finer grain. `alignment` says
how far apart two views land; it cannot say whether the residual distance is driven
by `qmax` or by `uiso`, because a single scalar over pooled pairs has integrated
that away. Recovering it needs the per-view theta the bank already stores and
`latent.py` deliberately discards.

WHAT IT WRITES, three things, per checkpoint AND per random-init control:

  distances    within-material and between-material cosine-distance HISTOGRAMS
               (fixed bins, so two encoders overlay without renegotiating an axis)
               plus their summary stats. The invariance claim is the separation
               between the two distributions; the overlap is what a mean hides.
  sensitivity  per parameter, the slope of within-pair cosine distance on |dp|,
               with |dp| scaled by that parameter's own declared range. An axis the
               encoder ignores has slope ~0; one it still tracks has slope > 0.
  residual_pca each material's views centred on THAT material's mean, then the top
               two components of the residuals — the subspace theta moves the
               representation through, with the per-view parameter values carried
               alongside so the plotter can colour by any of them.

WHY K VIEWS PER MATERIAL AND NOT 2. `latent.py` takes a pair because alignment is
defined on a pair. Both measurements here need more: a within-material DISTRIBUTION
needs many pairs per material, and a residual needs a per-material mean that two
points can only ever straddle. K=8 gives C(8,2)=28 pairs per material and a mean
worth subtracting, at 4x the forward passes of a pair.

THE UNIVARIATE FIT IS LICENSED BY THE SAMPLER, NOT ASSUMED. Regressing distance on
one parameter at a time would be wrong if the four co-varied, since each slope would
carry its correlates. They do not: `core/transforms.draw_pdf_params` draws all four
independently, so within-pair |dp| columns are uncorrelated by construction and the
univariate slope IS the marginal effect. This is a property of how the bank was
built — if a later sampler ever couples two parameters, this module needs a joint
fit and the docstring is the warning.

|dp| IS SCALED BY THE DECLARED RANGE, which is what makes four slopes comparable.
`uiso` and `qmax` differ by orders of magnitude in their units, so raw slopes rank
the parameters by unit choice rather than by the encoder's behaviour. Dividing by
`meta["aug_ranges"]`'s own width makes every slope "cosine distance per full sweep
of this parameter" — one axis, four bars, no hidden normalization.

COSINE, AND ON THE SPHERE, to stay on `latent.py`'s footing: both modules measure
`h` (never the projection `z`, because every downstream probe and finetune in this
repo reads `h`), and both put the distance measurement on L2-normalized embeddings.
The residual PCA is the one exception and is UNNORMALIZED on purpose — it is a
statement about where variance sits in the axes a downstream head sees, the same
asymmetry `latent.py` argues for its effective rank.

FOUR CAVEATS, all structural:
 1. Measured on TRAINING data. Pretraining splits are `(1.0, 0.0, 0.0)`, so there
    are no held-out MP materials to measure on.
 2. Shards are CONTIGUOUS material ranges (`tools/bank_pdf.py:_shard_ids`), so a
    partial block is biased by registry ordering rather than a random draw.
 3. A `*_mpfull_final` checkpoint measured on the MP-20 block is on a DIFFERENT
    corpus from the one it pretrained on, not merely a subset of it. Whether that
    matters is untested; it is stamped into every record as `block`.
 4. n=1 per checkpoint. No pretraining seeds exist, so nothing here has an error
    bar — the same limitation `RESULTS.md` carries for the selected model.

THE RANDOM-INIT CONTROL IS NOT OPTIONAL. A patch-embedding transformer with a
smoothing front end is already partly invariant to a resolution parameter before
any training happens, so a low slope on its own says nothing about what pretraining
achieved. Every quantity here is meant to be read as pretrained-against-control,
which is why the control is computed in the same run rather than left to the caller.

NEITHER IS THE INPUT-SPACE BASELINE, and it is the one that stops this module
over-claiming. A near-zero latent slope has two completely different explanations:
the encoder learned to discard that axis, or the axis never moved the signal in the
first place. Those are indistinguishable in latent space alone — and the second is
known to happen here, because PDF resolution goes as 2pi/`qmax` and the production
range [15, 30] sits entirely above the ~10 A^-1 where that starts to bite (the
`qmax` flat-then-cliff result in docs/TRAPS.md). So the raw min-max-normalized
signals go through the identical pairing, scaling and estimator as a third row.
Read a parameter as DISCARDED only where the input row is large and the latent row
is not; where both are ~0, nothing was learned and nothing should be claimed.

Reads `shard_*.npy` DIRECTLY rather than through `BankedPDFPretrainDataset`, which
refuses a block whose shards do not cover `0..n_ids-1` — and a few rsync'd shards
are exactly that. The bank stores RAW physical signal on purpose, so normalization
is applied here: the read-path convention the encoders were trained under
(DESIGN.md -> Signal normalization).

NORMALIZATION IS DISPATCHED ON THE BLOCK'S MODALITY, exactly as
`core/dataset_base.py` and `analysis/latent.py` do it — `minmax_normalize` for PDF,
`max_normalize` for XRD, defaulting to `pdf` for blocks banked before the key
existed. This module has its OWN `load_views` (K views plus their theta, where
`latent.py` takes a pair), so the fix had to land in both; getting it wrong is
silent, and here it would corrupt the input-space baseline the docstring above calls
load-bearing. See docs/TRAPS.md.

THE PARAMETER LIST IS THE BLOCK'S, NOT THIS MODULE'S. `param_names` is
`meta["aug_ranges"]`, so a PDF block yields the four PDF parameters and an XRD block
the six ranged XRD ones with no code change. Everything downstream — the univariate
licence above, the range scaling, the probe in `analysis/theta_probe.py` — is
written against that list rather than against four hardcoded names. The XRD bank's
parquet also carries `pref_h/k/l` and `cheb_0..3`, which `aug_ranges` deliberately
omits: the preferred direction is a categorical dressed as three indicators and is
meaningless on rows where `max_texture` is ~0, and `cheb_3` is zeroed by
`BKG_ORDER`, so both would be uninterpretable as regression targets.

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

from analysis.downstream_eval import _single_threaded, embed, find_checkpoints, load_encoder
from analysis.latent import CONTROLS_ROOT, arch_name, arch_signature
from core.config import TrainConfig
from core.train import build_model, pick_device
from core.transforms import max_normalize, minmax_normalize

__all__ = ["load_views", "within_pairs", "between_pairs", "cosine_distance",
           "histogram", "sensitivity", "residual_pca", "measure_h", "measure"]

#: Cosine distance on unit vectors is bounded to [0, 2], so the bin edges can be
#: fixed rather than data-derived. That is what lets a pretrained encoder and its
#: control overlay without one of them silently rescaling the other's axis.
#:
#: LOG-SPACED, and that is not a plotting preference. The quantity spans five
#: decades here — a pretrained encoder puts two views of one material at ~1e-3 and
#: two different materials at ~7e-1 — so 200 LINEAR bins on [0, 2] are 0.01 wide and
#: swallow the entire within-material distribution in bin 0. The separation this
#: module exists to measure would be invisible in its own histogram. Values are
#: clipped into the range: an exactly-identical pair gives distance 0, which has no
#: log bin, and there is no floor below which the distinction still matters.
DIST_BINS = 200
DIST_RANGE = (1e-6, 2.0)


def load_views(block, n_materials, n_views_per_material, seed):
    """K banked views of each sampled material -> `(X, mat, theta, info)`.

    `X` is `(M*K, grid_len)` float32, normalized per row by the block's own modality
    convention (see module docstring); `mat` is the material index each row belongs
    to; `theta` is `(M*K, P)` — the parameters that view was simulated at, read from
    the shard's own parquet.

    Which K views a material contributes is seeded from its `global_index`, not from
    its position in the sample, so the draw is unchanged by how many shards are on
    disk or by `n_materials` — two runs over different shard subsets stay comparable
    on their overlap. Same device `analysis/latent.py` uses, for the same reason.
    """
    block = Path(block)
    meta = json.loads((block / "meta.json").read_text())
    n_views = int(meta["n_views"])
    ranges = meta["aug_ranges"]
    param_names = list(ranges)
    # Same dispatch, same default, as core/dataset_base.py and analysis/latent.py.
    modality = meta.get("modality", "pdf")
    normalize = max_normalize if modality == "xrd" else minmax_normalize

    if n_views_per_material > n_views:
        raise SystemExit(f"--n-views {n_views_per_material} exceeds the block's {n_views} banked views")

    shard_paths = sorted(block.glob("shard_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    if not shard_paths:
        raise FileNotFoundError(f"no shard_*.npy in {block}")

    arrays: list[np.ndarray] = []
    params: list[np.ndarray] = []
    entries: list[tuple[int, int, int]] = []  # (shard position, row of view 0, global_index)
    grid_len: int | None = None
    for pos, npy in enumerate(shard_paths):
        # The params parquet is written in worker-COMPLETION order while a view's
        # .npy row is fixed by (global_index, view) — sort, never assume the two
        # files line up positionally (same trap as core/dataset_base.py).
        df = pd.read_parquet(npy.with_suffix(".parquet")).sort_values(["global_index", "view"])
        gidx = df["global_index"].to_numpy()
        n_ids_shard, remainder = divmod(len(df), n_views)
        expected = np.repeat(np.arange(gidx[0], gidx[0] + n_ids_shard), n_views)
        if remainder or not np.array_equal(gidx, expected):
            raise ValueError(f"{npy.name}: params are not exactly {n_views} views of a contiguous id range")

        arr = np.load(npy, mmap_mode="r")
        rows, width = arr.shape
        if rows != len(df):
            raise ValueError(f"{npy.name}: {rows} banked views but {len(df)} param rows — shard is truncated")
        if grid_len is None:
            grid_len = width
        elif width != grid_len:
            raise ValueError(f"{npy.name}: grid_len {width} != {grid_len} in an earlier shard")

        arrays.append(arr)
        params.append(df[param_names].to_numpy(dtype=np.float32))
        entries += [(pos, i * n_views, int(gidx[i * n_views])) for i in range(n_ids_shard)]

    take = len(entries) if n_materials is None else min(n_materials, len(entries))
    picked = sorted(np.random.default_rng(seed).choice(len(entries), size=take, replace=False).tolist())

    K = n_views_per_material
    X = torch.empty((take * K, grid_len), dtype=torch.float32)
    mat = np.repeat(np.arange(take), K)
    theta = np.empty((take * K, len(param_names)), dtype=np.float32)
    for out_m, e in enumerate(picked):
        pos, row_base, global_index = entries[e]
        views = np.random.default_rng(seed + global_index).choice(n_views, size=K, replace=False)
        for k, v in enumerate(views):
            row = row_base + int(v)
            # np.array (not asarray) copies out of the read-only memmap and casts a
            # float16 block up, exactly as the banked dataset's read path does.
            raw = np.array(arrays[pos][row], dtype=np.float32)
            X[out_m * K + k] = normalize(torch.from_numpy(raw))
            theta[out_m * K + k] = params[pos][row]

    info = {
        "block": block.name,
        "modality": modality,
        "shards_present": [int(p.stem.split("_")[1]) for p in shard_paths],
        "n_materials_available": len(entries),
        "n_materials": take,
        "n_views_per_material": K,
        "sample_seed": seed,
        "grid_len": grid_len,
        "param_names": param_names,
        "aug_ranges": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
    }
    return X, mat, theta, info


def within_pairs(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every `(i, j)`, i<j, whose rows share a material. `M*C(K,2)` pairs."""
    i_all, j_all = [], []
    for m in np.unique(mat):
        rows = np.flatnonzero(mat == m)
        i, j = np.triu_indices(len(rows), k=1)
        i_all.append(rows[i])
        j_all.append(rows[j])
    return np.concatenate(i_all), np.concatenate(j_all)


def between_pairs(mat: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """`n` random `(i, j)` drawn from DIFFERENT materials.

    Sampled rather than enumerated: the full cross-material set is ~(M*K)^2/2, which
    at the default sample is ~8M pairs to compute a histogram that converges in
    thousands. Rejection rather than a stratified construction, because with
    M=512 materials a random pair collides on material only ~0.2% of the time.
    """
    i = rng.integers(0, len(mat), size=n)
    j = rng.integers(0, len(mat), size=n)
    same = mat[i] == mat[j]
    while same.any():
        j[same] = rng.integers(0, len(mat), size=int(same.sum()))
        same = mat[i] == mat[j]
    return i, j


def cosine_distance(H: torch.Tensor, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """`1 - cos(h_i, h_j)` in [0, 2]. Pairwise on rows, never a full matrix."""
    U = torch.nn.functional.normalize(H.to(torch.float64), dim=1)
    return (1.0 - (U[i] * U[j]).sum(dim=1)).numpy()


def histogram(d: np.ndarray) -> dict:
    """Fixed log-spaced counts plus the summary stats a caption needs.

    Quantiles are taken on the UNCLIPPED values — clipping is a binning device and
    must not reach the numbers a caption quotes.
    """
    edges = np.geomspace(*DIST_RANGE, DIST_BINS + 1)
    counts, edges = np.histogram(np.clip(d, *DIST_RANGE), bins=edges)
    q = np.quantile(d, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "counts": [int(c) for c in counts],
        "bin_edges": [float(e) for e in edges],
        "n": int(len(d)),
        "mean": float(d.mean()),
        "quantiles": {k: float(v) for k, v in zip(("p5", "p25", "p50", "p75", "p95"), q)},
    }


def sensitivity(d: np.ndarray, dtheta: np.ndarray, param_names, widths) -> dict:
    """Per parameter: slope of within-pair distance on range-scaled `|dp|`.

    One univariate least-squares fit per column — legitimate because the sampler
    draws the four parameters independently (see module docstring), so no column
    carries another's effect. Pearson r is reported beside the slope because the
    slope alone cannot distinguish "strong dependence" from "noisy dependence with a
    wide |dp| spread", and the second is what a near-invariant axis looks like.
    """
    out = {}
    for p, name in enumerate(param_names):
        x = dtheta[:, p] / widths[p]
        xc, dc = x - x.mean(), d - d.mean()
        var = float((xc ** 2).sum())
        slope = float((xc * dc).sum() / var) if var > 0 else float("nan")
        denom = float(np.sqrt(var * (dc ** 2).sum()))
        out[name] = {
            "slope": slope,
            "pearson_r": float((xc * dc).sum() / denom) if denom > 0 else float("nan"),
            "mean_abs_dp_scaled": float(x.mean()),
        }
    return out


def residual_pca(H: torch.Tensor, mat: np.ndarray, n_components: int = 2):
    """Per-material-centred embeddings -> top-`k` component scores + variance shares.

    Centring each material on ITS OWN mean removes the between-material variance,
    which is ~all of the total and would otherwise dominate every component — a
    straight PCA of `H` draws a map of which materials differ, not of what theta
    does. What remains is exactly the subspace the instrument model moves the
    representation through.

    Unnormalized, unlike the distance measurements: this is a statement about where
    variance sits in the axes a downstream head reads.

    Also returns `within_fraction` = trace(cov of residuals) / trace(cov of H) — the
    share of latent variance that is instrument nuisance rather than structure. Not
    plotted; it is the one-number anchor for the histogram panel and falls out of the
    centring for free.
    """
    X = H.to(torch.float64).numpy()
    R = X - np.stack([X[mat == m].mean(axis=0) for m in np.unique(mat)])[mat]

    total_var = float(((X - X.mean(axis=0)) ** 2).sum() / max(len(X) - 1, 1))
    within_var = float((R ** 2).sum() / max(len(X) - 1, 1))

    with _single_threaded():
        U, S, _ = np.linalg.svd(R - R.mean(axis=0), full_matrices=False)
    scores = U[:, :n_components] * S[:n_components]
    explained = (S ** 2 / max((S ** 2).sum(), 1e-300))[:n_components]
    return scores, {
        "explained_variance_ratio": [float(v) for v in explained],
        "within_fraction": within_var / total_var if total_var > 0 else float("nan"),
    }


def measure_h(H, mat, theta, info, *, rng, max_pca_points):
    """Everything this module writes, from an already-computed point set.

    Split out from `measure` so the INPUT SIGNALS can go through the identical
    computation — see `main`, where that baseline is what makes the per-parameter
    panel interpretable rather than merely suggestive.
    """
    wi, wj = within_pairs(mat)
    bi, bj = between_pairs(mat, len(wi), rng)
    d_within = cosine_distance(H, wi, wj)
    d_between = cosine_distance(H, bi, bj)

    widths = [hi - lo for lo, hi in (info["aug_ranges"][n] for n in info["param_names"])]
    dtheta = np.abs(theta[wi] - theta[wj])

    scores, pca_info = residual_pca(H, mat)
    keep = np.sort(rng.choice(len(scores), size=min(max_pca_points, len(scores)), replace=False))

    return {
        "latent_dim": int(H.shape[1]),
        "distances": {
            "within": histogram(d_within),
            "between": histogram(d_between),
            # The single number the two histograms are usually reduced to. Kept so a
            # reader who wants `latent.py`'s framing does not have to re-derive it.
            "separation": float(d_between.mean() - d_within.mean()),
        },
        "sensitivity": sensitivity(d_within, dtheta, info["param_names"], widths),
        "residual_pca": {
            **pca_info,
            "n_points": int(len(keep)),
            "scores": [[float(a), float(b)] for a, b in scores[keep]],
            "theta": [[float(v) for v in row] for row in theta[keep]],
        },
    }


@torch.no_grad()
def measure(model, X, mat, theta, info, device, batch_size, *, rng, max_pca_points):
    """`measure_h` on one encoder's embeddings of `X`."""
    H = embed(model, X, device, batch_size)
    return measure_h(H, mat, theta, info, rng=rng, max_pca_points=max_pca_points)


def _check_grid(cfg: TrainConfig, info: dict, run: str) -> None:
    """The Transformer bakes `signal_len` into its positional embedding, so a block
    on another r-grid fails deep in the forward with a shape error naming neither
    the block nor the run."""
    if cfg.encoder == "transformer" and cfg.signal_len != info["grid_len"]:
        raise SystemExit(
            f"{run}: signal_len {cfg.signal_len} != block grid_len {info['grid_len']} — "
            f"this checkpoint was not pretrained on {info['block']}"
        )


def _write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"  wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block", required=True,
                        help="banked block dir; whichever shard_*.npy are present are used")
    parser.add_argument("--runs", default="runs/pdf/sweep", help="dir whose */ckpt_best.pt are measured")
    parser.add_argument("--filter", nargs="+", default=["*"],
                        help="globs over run-dir names, unioned")
    parser.add_argument("--controls-root", default=CONTROLS_ROOT,
                        help="where the random-init controls are written")
    parser.add_argument("--n-materials", type=int, default=512,
                        help="materials sampled from the shards present; all of them if fewer")
    parser.add_argument("--n-views", type=int, default=8,
                        help="banked views per material; needs >=3 for a meaningful residual")
    parser.add_argument("--max-pca-points", type=int, default=2000,
                        help="residual scores stored per record, subsampled — a scatter saturates "
                             "long before the full sample and the JSON should stay readable")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--pair-seed", type=int, default=0,
                        help="seed for the between-material pair draw and the PCA subsample")
    parser.add_argument("--control-seed", type=int, default=0,
                        help="torch seed for the random-init control weights")
    parser.add_argument("--device", default="auto", help='"auto" | "cpu" | "cuda"')
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.n_views < 3:
        raise SystemExit("--n-views must be >= 3: two views straddle their own mean, so the "
                         "residual would be +-d/2 by construction and carry no shape")

    device = pick_device(args.device)
    ckpts = sorted({p for pattern in args.filter for p in find_checkpoints(args.runs, pattern)})
    if not ckpts:
        raise SystemExit(f"no */ckpt_best.pt under {args.runs} matching {args.filter}")

    X, mat, theta, info = load_views(args.block, args.n_materials, args.n_views, args.sample_seed)
    print(f"{info['block']}: shards {info['shards_present']} -> {info['n_materials']} of "
          f"{info['n_materials_available']} materials x {info['n_views_per_material']} views, "
          f"grid_len {info['grid_len']}, params {info['param_names']}, "
          f"{len(ckpts)} checkpoints on {device}", flush=True)

    # The input-space row, computed before any encoder so a failure here is not
    # mistaken for a checkpoint problem. `X` stands in for `H`: same pairs, same
    # range scaling, same estimator — the three rows are one column of numbers
    # rather than three protocols that happen to share a plot.
    rows: list[dict] = [{
        "run": "input_space", "pretrained": None, "epoch": None, "encoder": None,
        "arch": "input", **info,
        **measure_h(X, mat, theta, info, rng=np.random.default_rng(args.pair_seed),
                    max_pca_points=args.max_pca_points),
    }]
    out_dir = Path(args.controls_root) / "input_space"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "latent_theta.json", rows[0])

    controls: dict[str, tuple[tuple, TrainConfig]] = {}
    for ckpt in ckpts:
        model, cfg, epoch = load_encoder(ckpt, device)
        _check_grid(cfg, info, ckpt.parent.name)
        name, sig = arch_name(cfg), arch_signature(cfg)
        if controls.setdefault(name, (sig, cfg))[0] != sig:
            raise SystemExit(f"two different geometries both name their control {name!r}: "
                             f"{controls[name][0]} vs {sig} — widen arch_name()")
        record = {"run": ckpt.parent.name, "pretrained": True, "epoch": epoch,
                  "encoder": cfg.encoder, "arch": name, **info,
                  **measure(model, X, mat, theta, info, device, args.batch_size,
                            rng=np.random.default_rng(args.pair_seed),
                            max_pca_points=args.max_pca_points)}
        _write(ckpt.parent / "latent_theta.json", record)
        rows.append(record)

    for name, (_, cfg) in sorted(controls.items()):
        torch.manual_seed(args.control_seed)
        model = build_model(cfg).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        record = {"run": f"randominit_{name}", "pretrained": False, "epoch": None,
                  "encoder": cfg.encoder, "arch": name, "control_seed": args.control_seed,
                  **info,
                  **measure(model, X, mat, theta, info, device, args.batch_size,
                            rng=np.random.default_rng(args.pair_seed),
                            max_pca_points=args.max_pca_points)}
        out_dir = Path(args.controls_root) / f"randominit_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write(out_dir / "latent_theta.json", record)
        rows.append(record)

    # Slopes are printed RELATIVE to that row's own mean within-pair distance.
    # Absolute slopes are incomparable across rows here — the pretrained encoder's
    # within-distances are ~3x the control's and ~1/300th of its own
    # between-distances, so a raw slope column ranks rows by their overall scale and
    # hides the only thing being asked, which is how much of a row's residual
    # within-material distance each parameter accounts for.
    names = rows[0]["param_names"]
    head = " ".join(f"{n:>9}" for n in names)
    order = lambda r: (0 if r["arch"] == "input" else 1, r["arch"], bool(r["pretrained"]))
    print(f"\n{'run':<42} {'within':>8} {'between':>8} {'sep':>7} {'nuis':>6}  {head}")
    for r in sorted(rows, key=order):
        w = r["distances"]["within"]["mean"]
        s = " ".join(f"{r['sensitivity'][n]['slope'] / w:>9.3f}" if w > 0 else f"{'nan':>9}"
                     for n in names)
        print(f"{r['run']:<42} {w:>8.4f} "
              f"{r['distances']['between']['mean']:>8.4f} {r['distances']['separation']:>7.4f} "
              f"{r['residual_pca']['within_fraction']:>6.3f}  {s}")
    print("\nsep = mean between - mean within (higher = more invariant). "
          "nuis = share of variance that is theta rather than structure.\n"
          "Parameter columns are slope / mean within-distance: the share of a row's own "
          "residual\nspread that parameter accounts for. A parameter is DISCARDED only "
          "where input_space is\nlarge and the pretrained row is not — where both are ~0, "
          "the axis never moved the signal.")


if __name__ == "__main__":
    main()
