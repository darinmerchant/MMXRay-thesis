"""analysis/latent.py — pretraining-domain latent geometry of a saved encoder.

    python -m analysis.latent --block data/banked/material_registry__n32__55ad1ff1

Three global geometry numbers per checkpoint, measured on the PRETRAINING corpus
(MP-20 banked views) and on the representation `h` — never the projection `z`,
because `h` is what every downstream probe and finetune in this repo reads:

  effective_rank  participation ratio `(sum L)^2 / sum L^2` over the covariance
                  eigenvalues, range [1, latent_dim]. Threshold-free and
                  scale-invariant, so unlike a "components to reach 95% variance"
                  count it has no cutoff to defend and is not a step function.
  alignment       `E ||f(x) - f(x+)||^2` over the two banked views of one material
                  (Wang & Isola, alpha=2), `f` = mean-centered then L2-normalized
                  `h` (see the centering note below). LOWER = closer views.
  uniformity      `log E exp(-2 ||f(x) - f(y)||^2)` over all pairs of distinct
                  materials (t=2), same `f`. LOWER = the sample spreads more evenly.

Rank and uniformity are computed on view A, alignment on the (A, B) pair, so all
three describe one consistent point set. The full covariance spectrum is saved
next to the scalars — `latent_dim` floats — so a different rank definition or a
spectrum figure later costs no re-embedding.

WHAT THIS DOES NOT DO. It never reads CHILI, never reads a target, and plots
nothing. In particular it does NOT explain the `cnn_vicreg_cov1` per-axis anomaly
(R^2 0.12 on np_size against 0.62 on mo_bond): these are GLOBAL scalars measured
in the PRETRAINING domain, and a per-axis failure in a DOWNSTREAM domain is two
steps away. A low rank here would predict that every target degrades, which is
measurably false for that run. Explaining it needs a target-conditioned quantity
(per-target variance retained in the top-k subspace, say), deliberately not built
— see `docs/MODULES.md`.

EVERYTHING IS MEAN-CENTERED, AND FOR ALIGNMENT/UNIFORMITY THAT IS A 2026-09-03
CHANGE. Effective rank is computed on mean-centered, UNNORMALIZED `h`, because it
is a statement about how variance is distributed over the axes a downstream head
actually sees; alignment and uniformity are computed on mean-centered `h`
projected to the unit sphere, each view set centered by its own sample mean.
Wang & Isola define both on uncentered `f(x)`, and the deviation is deliberate:
a ReLU CNN's `h` is non-negative, so every embedding shares a large positive
mean vector and the whole sample sits in one narrow cone of the sphere.
Uncentered, the random-init control scored the best alignment in the table
(0.0006) purely from that shared offset — a mean artifact, not view structure —
and centered it scores the worst (~1.0), which is what an encoder with no
learned invariance should score. The pre-change uncentered values live in
RESULTS.md Q4's table (measured 2026-08-06). Normalizing is also what makes the
cross-family comparison legitimate: InfoNCE and VICReg differ in whether their
PROJECTION is normalized, but neither normalizes `h`, so imposing it here treats
the two identically instead of flattering one.

FIVE CAVEATS, all structural rather than fixable in this module:
 1. Measured on TRAINING data. Pretraining splits are `(1.0, 0.0, 0.0)`
    (`core/splits.py`), so there are no held-out MP materials to measure on.
 2. The sample is drawn from whichever shards are on disk, and shards are
    CONTIGUOUS material ranges (`tools/bank_pdf.py:_shard_ids`) — so it is biased
    by registry ordering, not a random draw over MP-20.
 3. The four `*_mpfull_final` runs pretrained on full-MP, so measuring them here
    puts them on a SUBSET of their corpus rather than on it. Kept anyway: one
    common dataset is what makes the 22 numbers a comparable column, and rank
    estimates from different datasets are not on one scale.
 4. n=1 per checkpoint. No pretraining seeds exist, so nothing here has an error
    bar — the same limitation `RESULTS.md` carries for the selected model.
 5. Random-init controls run under `.eval()` like everything else, so their
    BatchNorm layers use their INITIAL running stats (0/1), having seen no data.
    That is an untrained encoder under the same protocol, not a batch-statistics
    estimate of one.

THE SVD RUNS SINGLE-THREADED, AND THAT IS LOAD-BEARING, NOT A PERFORMANCE KNOB.
This env has three OpenMP runtimes on the library path (docs/ENVIRONMENT.md), and
an unguarded `np.linalg.svd` leaves a LAPACK thread pool behind that the NEXT
torch `batch_norm` deadlocks against — silently, at 0% CPU, with no traceback.
It bites only the second and later CNN in a run (the transformer is LayerNorm, so
it sails through), which is exactly the shape that makes it look like the module
hangs at random. Same `threadpool_limits(limits=1)` remedy the probe's sklearn
head uses; measured to reproduce at 400x5000 through a 256-d CNN and not at
256x2000 through a 128-d one, so a small test cannot see it — see docs/TRAPS.md.

Reads the block's `shard_*.npy` DIRECTLY rather than through
`BankedPDFPretrainDataset`, which validates that the shards cover `0..n_ids-1`
and refuses a partial block (`core/dataset_base.py`) — and a few rsync'd shards
are exactly a partial block. The bank stores RAW physical signal on purpose, so
normalization is applied here: the read-path convention the encoders were
trained under (DESIGN.md -> Signal normalization).

NORMALIZATION IS DISPATCHED ON THE BLOCK'S MODALITY, exactly as
`core/dataset_base.py` does it — `minmax_normalize` for PDF, `max_normalize` for
XRD. An XRD row is zero-filled outside CHILI's angular coverage and has a
meaningful zero, so min-max would shift the baseline of every pattern by a
different amount (`modalities/xrd/grid.py`). Getting this wrong is silent: the
figure still draws, on inputs the encoder was never trained on — the same class of
error `analysis/invariance_decay.py` records for the reference repo. PDF blocks
predate the `modality` key and default to `pdf`, so their numbers are unchanged.

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
from core.config import TrainConfig
from core.train import build_model, pick_device
from core.transforms import max_normalize, minmax_normalize

__all__ = ["load_views", "spectrum", "participation_ratio", "alignment", "uniformity",
           "arch_name", "arch_signature", "measure"]

#: Random-init controls are not runs, so they get their own tree rather than a
#: fake directory under `runs/pdf/sweep/`.
CONTROLS_ROOT = "runs/pdf/latent"
#: Wang & Isola's `t`. Pinned, not exposed: changing it changes what the number
#: means, and no caller has needed a second value.
UNIFORMITY_T = 2.0


def load_views(block: str | Path, n_materials: int | None, seed: int):
    """Two banked views of each sampled material -> `(A, B, info)`.

    `A` and `B` are `(n, grid_len)` float32, normalized per row by the block's own
    modality convention (see module docstring). Whichever `shard_*.npy` are present
    are used; a missing shard is not an error here, only fewer materials.

    The view PAIR for a material is seeded from its `global_index`, not from its
    position in the sample — so the pair a material gets is unchanged by how many
    shards happen to be on disk or by `n_materials`, and two runs of this module
    over different shard subsets stay comparable on their overlap.
    """
    block = Path(block)
    meta = json.loads((block / "meta.json").read_text())
    n_views = int(meta["n_views"])
    # Same dispatch, same default, as core/dataset_base.py — PDF blocks were banked
    # before `modality` was recorded and must keep reading as PDF.
    modality = meta.get("modality", "pdf")
    normalize = max_normalize if modality == "xrd" else minmax_normalize

    shard_paths = sorted(block.glob("shard_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    if not shard_paths:
        raise FileNotFoundError(f"no shard_*.npy in {block}")

    arrays: list[np.ndarray] = []
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
        entries += [(pos, i * n_views, int(gidx[i * n_views])) for i in range(n_ids_shard)]

    take = len(entries) if n_materials is None else min(n_materials, len(entries))
    picked = sorted(np.random.default_rng(seed).choice(len(entries), size=take, replace=False).tolist())

    A = torch.empty((take, grid_len), dtype=torch.float32)
    B = torch.empty((take, grid_len), dtype=torch.float32)
    for out_row, e in enumerate(picked):
        pos, row_base, global_index = entries[e]
        v1, v2 = np.random.default_rng(seed + global_index).choice(n_views, size=2, replace=False)
        for dest, v in ((A, v1), (B, v2)):
            # np.array (not asarray) copies out of the read-only memmap and casts a
            # float16 block up, exactly as the banked dataset's read path does.
            raw = np.array(arrays[pos][row_base + int(v)], dtype=np.float32)
            dest[out_row] = normalize(torch.from_numpy(raw))

    info = {
        "block": block.name,
        "modality": modality,
        "shards_present": [int(p.stem.split("_")[1]) for p in shard_paths],
        "n_materials_available": len(entries),
        "n_materials": take,
        "sample_seed": seed,
        "n_views": n_views,
        "grid_len": grid_len,
    }
    return A, B, info


def spectrum(H: torch.Tensor) -> np.ndarray:
    """Covariance eigenvalues of `(n, d)` embeddings, descending. Always length `d`.

    Mean-centred and NOT L2-normalized — see the module docstring on why this half
    of the measurement is deliberately on a different footing from the other two.
    Computed in float64 via SVD rather than by forming `H^T H`, which squares the
    condition number and loses the small eigenvalues that decide the rank.

    SVD returns `min(n, d)` values, so the result is zero-padded to `d`: centring
    caps the attainable rank at `n - 1`, and those trailing eigenvalues genuinely
    are 0. Padding keeps the saved spectrum the same length as `latent_dim` for
    every checkpoint, which is what makes two of them plottable on one axis.
    """
    X = H.to(torch.float64).numpy()
    X = X - X.mean(axis=0)
    lam = np.zeros(X.shape[1])
    with _single_threaded():
        s = np.linalg.svd(X, compute_uv=False)
    lam[: len(s)] = s ** 2 / max(len(X) - 1, 1)
    return lam


def participation_ratio(lam: np.ndarray) -> float:
    """`(sum L)^2 / sum L^2` — the pinned effective-rank definition. Range [1, d].

    A fully collapsed representation has every eigenvalue 0, where the ratio is
    0/0; that returns nan rather than a number that would read as "rank 0".
    """
    denom = float((lam ** 2).sum())
    return float(lam.sum() ** 2 / denom) if denom > 0.0 else float("nan")


def _unit(H: torch.Tensor) -> torch.Tensor:
    """Center by the set's own sample mean, then project to the unit sphere.

    Centering first is what removes the shared-mean cone (module docstring); it
    also makes both scalars invariant to a constant offset of the embedding.
    """
    X = H.to(torch.float64)
    return torch.nn.functional.normalize(X - X.mean(dim=0), dim=1)


def alignment(A: torch.Tensor, B: torch.Tensor) -> float:
    """`E ||f(x) - f(x+)||^2` over positive pairs, `f` = centered, L2-normalized `h`.

    Lower = the two views of one material land closer. 0 if the encoder maps both
    views identically; 4 at the antipodal worst case on the unit sphere. Each view
    set is centered by its own mean, so scaling or offsetting `h` changes nothing.
    """
    return float(((_unit(A) - _unit(B)) ** 2).sum(dim=1).mean())


def uniformity(H: torch.Tensor, t: float = UNIFORMITY_T) -> float:
    """`log E exp(-t ||f(x) - f(y)||^2)` over all distinct pairs. Lower = more spread.

    Via logsumexp rather than `log(mean(exp(...)))`: on a well-spread
    representation most squared distances are O(1), the naive form underflows to
    -inf, and it therefore fails on exactly the cases worth measuring.
    """
    d2 = torch.pdist(_unit(H)) ** 2
    return float(torch.logsumexp(-t * d2, dim=0) - np.log(len(d2)))


def arch_name(cfg: TrainConfig) -> str:
    """Readable name of the random-init control this config needs.

    A control is only a control for encoders shaped like it, and the 22
    checkpoints span three geometries: the CNN, the 4-layer/patch-50 transformer
    every `loss_sweep` run uses, and the 8-layer/patch-200 one that
    `transformer_vicreg_mpfull_final` uses alone.
    """
    if cfg.encoder == "cnn":
        return f"cnn_h{cfg.latent_dim}"
    return f"transformer_L{cfg.num_layers}_p{cfg.patch_size}"


def arch_signature(cfg: TrainConfig) -> tuple:
    """Every field that changes `h`. `arch_name` is a shorthand over this.

    `proj_dim` is absent on purpose: only `encode()` is ever called, so the
    projection head cannot affect anything measured here. The two are checked
    against each other in `main`, so a name that silently covered two different
    geometries would be caught rather than quietly sharing one control.
    """
    if cfg.encoder == "cnn":
        return ("cnn", cfg.latent_dim)
    return ("transformer", cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.num_layers,
            cfg.patch_size, cfg.stride, cfg.norm_first, cfg.signal_len)


@torch.no_grad()
def measure(model, A: torch.Tensor, B: torch.Tensor, device, batch_size: int) -> dict:
    """The three scalars plus the spectrum they derive from, for one encoder."""
    HA = embed(model, A, device, batch_size)
    HB = embed(model, B, device, batch_size)
    if len(HA) <= HA.shape[1]:
        # Below this the participation ratio measures the SAMPLE, not the encoder:
        # centring caps the attainable rank at n-1, so a small sample fakes a
        # collapsed representation. The default 4096 is ~16x the 256-d latent.
        raise ValueError(
            f"{len(HA)} materials for a {HA.shape[1]}-d latent — effective rank is bounded by "
            f"the sample, not the representation; raise --n-materials above {HA.shape[1]}"
        )
    lam = spectrum(HA)
    return {
        "latent_dim": int(HA.shape[1]),
        "effective_rank": participation_ratio(lam),
        "alignment": alignment(HA, HB),
        "uniformity": uniformity(HA),
        "spectrum": [float(v) for v in lam],
    }


def _check_grid(cfg: TrainConfig, info: dict, run: str) -> None:
    """The Transformer bakes `signal_len` into its positional embedding, so a block
    on another r-grid would fail deep in the forward with a shape error naming
    neither the block nor the run."""
    if cfg.encoder == "transformer" and cfg.signal_len != info["grid_len"]:
        raise SystemExit(
            f"{run}: signal_len {cfg.signal_len} != block grid_len {info['grid_len']} — "
            f"this checkpoint was not pretrained on {info['block']}"
        )


def _write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"  wrote {path}", flush=True)  # 22 checkpoints is long enough that a buffered pipe looks hung


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block", required=True,
                        help="banked block dir; whichever shard_*.npy are present are used")
    parser.add_argument("--runs", default="runs/pdf/sweep", help="dir whose */ckpt_best.pt are measured")
    parser.add_argument("--filter", nargs="+", default=["*"],
                        help="globs over run-dir names, unioned. `runs/pdf/sweep/` also holds the ~59 "
                             "aug_sweep checkpoints, which were pretrained on OTHER blocks, so "
                             "the 22 want: --filter '*_cov*' '*_temp*' '*mpfull_final*'")
    parser.add_argument("--controls-root", default=CONTROLS_ROOT,
                        help="where the random-init controls are written")
    parser.add_argument("--n-materials", type=int, default=4096,
                        help="materials sampled from the shards present; all of them if fewer")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--control-seed", type=int, default=0,
                        help="torch seed for the random-init control weights")
    parser.add_argument("--device", default="auto", help='"auto" | "cpu" | "cuda"')
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = pick_device(args.device)
    ckpts = sorted({p for pattern in args.filter for p in find_checkpoints(args.runs, pattern)})
    if not ckpts:
        raise SystemExit(f"no */ckpt_best.pt under {args.runs} matching {args.filter}")

    A, B, info = load_views(args.block, args.n_materials, args.sample_seed)
    print(f"{info['block']}: shards {info['shards_present']} -> {info['n_materials']} of "
          f"{info['n_materials_available']} materials, grid_len {info['grid_len']}, "
          f"{len(ckpts)} checkpoints on {device}", flush=True)

    rows: list[dict] = []
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
                  **measure(model, A, B, device, args.batch_size)}
        _write(ckpt.parent / "latent.json", record)
        rows.append(record)

    # Controls are derived from the architectures actually matched, so `--filter
    # 'cnn_*'` produces the CNN control and nothing else.
    for name, (_, cfg) in sorted(controls.items()):
        torch.manual_seed(args.control_seed)
        model = build_model(cfg).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        record = {"run": f"randominit_{name}", "pretrained": False, "epoch": None,
                  "encoder": cfg.encoder, "arch": name, "control_seed": args.control_seed,
                  **info, **measure(model, A, B, device, args.batch_size)}
        out_dir = Path(args.controls_root) / f"randominit_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write(out_dir / "latent.json", record)
        rows.append(record)

    # Controls first within each architecture: every row is read against one.
    print(f"\n{'run':<42} {'arch':<24} {'eff_rank':>9} {'align':>8} {'uniform':>9}")
    for r in sorted(rows, key=lambda r: (r["arch"], r["pretrained"], -r["effective_rank"])):
        print(f"{r['run']:<42} {r['arch']:<24} {r['effective_rank']:>9.2f} "
              f"{r['alignment']:>8.4f} {r['uniformity']:>9.4f}")


if __name__ == "__main__":
    main()
