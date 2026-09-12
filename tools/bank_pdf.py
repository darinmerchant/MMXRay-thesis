"""tools/bank_pdf.py — pre-simulate N views per material into a block. PDF or XRD.

`--modality` picks the transform (`pdf` -> `PDFSimulate`, `xrd` -> `XRDSimulate`);
everything else — sharding, seeding, the shard layout, the raw-amplitude rule —
is shared. The name keeps its `pdf`, as `verify_bank` and every SLURM header
already import it by that name; it stopped being literal in step 5 of the XRD arm.

TWO NAMING RULES, both enforced by pins (tests/test_dataset_bank.py, T0.1's AST
check) and both consequences of the `n_ids` incident (`db4252c`):
  - `modality` goes in the block DIRECTORY NAME (`xrd__{dataset}__n{N}__{hash}`),
    NEVER in `_config_hash`'s payload — `tools/gen_sweep_configs.py:29` recomputes
    block paths from that hash, so an added payload key silently repoints every
    generated sweep config.
  - `pdf` produces NO prefix, so every existing block name stays byte-identical.

DESIGN.md -> Throughput: raw sim-on-the-fly is too costly (~75 min / MP-20 epoch,
saturates ~20 sims/s), so we bank N views/material ONCE and sample 2 at read time.
This writes ONE block for one (registry, aug-config, N, seed); shard it across a
SLURM array so each task fits a wall-clock.

The banked G(r) is RAW physical amplitude — un-normalized on purpose. The
encoder-input min-max normalization (core.transforms.minmax_normalize) is a
READ-path step (see DESIGN.md -> Signal normalization), so the future bank-read
dataset applies it, not this tool. Keeping the bank raw means a normalization-
scheme change (e.g. the standardization ablation) is a read-path edit, not a
~10 h re-bank.

A block directory holds, per shard:
    shard_<i>.npy       (n_views_in_shard, GRID_LEN) array of RAW G(r)
    shard_<i>.parquet   one row per view: material_id, global_index, view, + drawn params
and, written once by shard 0:
    meta.json           registry, aug ranges, r-grid, n_views, seed, config hash, n_ids

Views are seeded per (global_index, view) — global over the FULL id list — so shards
never collide and a re-run reproduces bit-for-bit. Set OMP/BLAS threads to 1 in the
job (see scripts/bank_pdf.slurm); the pool forks, so workers inherit the loaded
registry (copy-on-write) and the capped BLAS env -> ~20 sims/s on a full node.

Example (one shard of 20):
    python -m tools.bank_pdf --registry data/registry/material_registry.parquet \
        --out data/banked --n-views 32 --shards 20 --shard 0

One point of the augmentation-strength sweep — `--range` overrides a param's draw
range, and the block hash keys on it, so each point gets its own block dir:
    python -m tools.bank_pdf --registry data/registry/material_registry_sweep12k.parquet \
        --n-views 12 --range qbroad=0.0175,0.0925
Endpoints come from `core.transforms.ranges_at`, not from typing them by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from core.registry import MaterialRegistry
from core.transforms import PDF_RANGES, XRD_RANGES, PDFSimulate, Transform, XRDSimulate

REPO_ROOT = Path(__file__).resolve().parents[1]

#: modality -> (transform class, its draw ranges, its block-dir prefix).
#: The empty pdf prefix is load-bearing: existing block names stay byte-identical.
_MODALITIES = {
    "pdf": (PDFSimulate, PDF_RANGES, ""),
    "xrd": (XRDSimulate, XRD_RANGES, "xrd__"),
}

# Filled once per worker process (fork-inherited registry + transform).
_REG: MaterialRegistry | None = None
_TF: Transform | None = None
_N: int = 0
_SEED: int = 0


def _config_hash(tf: Transform, n_views: int, seed: int, grid_len: int, n_ids: int) -> str:
    """Stable id for the block: aug ranges + grid + N + seed + #materials. Same
    inputs -> same hash, so a re-run lands in the same block and sweep variants get
    distinct ones.

    `n_ids` is in the payload because it is NOT otherwise in the block path
    (`{registry}__n{N}__{hash}`): without it a `--sample 200` smoke test and a full
    run over the same registry and ranges would share a directory and interleave
    shards. Added session 23 — the pre-existing MP-20 production block
    `material_registry__n32__55ad1ff1` was named under the old payload, so its
    directory name no longer matches what this function would now generate. Nothing
    reads a block by hash (`BankedPDFPretrainDataset` takes a path), so that block
    keeps working; only a re-bank of it would land in a new directory.
    """
    # Reads `tf._grid` and `tf._ranges` from EITHER transform — XRDSimulate carries
    # an empty `_grid` for exactly this line. The payload KEYS stay these six for
    # both modalities (T0.1's AST check pins them); modality itself is expressed in
    # the directory-name prefix, never here.
    payload = json.dumps(
        {"ranges": tf._ranges, "grid": tf._grid, "grid_len": grid_len,
         "n_views": n_views, "seed": seed, "n_ids": n_ids},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def _parse_range(spec: str) -> tuple[str, tuple[float, float]]:
    """`name=lo,hi` -> (name, (lo, hi)), for `--range`.

    A ZERO-WIDTH `name=c,c` is legal and is how a param is switched off: the draw
    still happens (and still consumes one number from the seeded stream), it just
    always returns `c`. See core.transforms.draw_pdf_params.
    """
    name, sep, bounds = spec.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"--range {spec!r}: expected NAME=LO,HI")
    # Name membership is checked in main(), against the MODALITY's ranges —
    # argparse runs this type callable before --modality is known.
    try:
        lo, hi = (float(v) for v in bounds.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--range {spec!r}: expected NAME=LO,HI") from None
    if lo > hi:
        raise argparse.ArgumentTypeError(f"--range {spec!r}: lo ({lo}) > hi ({hi})")
    return name, (lo, hi)


def _seed(global_index: int, view: int) -> int:
    return _SEED + global_index * _N + view


def _sim_material(task: tuple[int, str]) -> tuple[int, np.ndarray, list[dict]]:
    """Simulate all N views of one material. Returns (global_index, (N, L) G(r),
    per-view param dicts). Record fetched once; each view is an independent draw."""
    gidx, mid = task
    record = _REG.get(mid)
    ys, params = [], []
    for v in range(_N):
        res = _TF(record, seed=_seed(gidx, v))
        ys.append(res.y)
        params.append(res.params)
    return gidx, np.stack(ys), params


def _shard_ids(ids: list[str], shards: int, shard: int) -> tuple[list[int], list[str]]:
    """Contiguous slice of the global id list for this array task."""
    bounds = np.linspace(0, len(ids), shards + 1, dtype=int)
    lo, hi = bounds[shard], bounds[shard + 1]
    return list(range(lo, hi)), ids[lo:hi]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", required=True, help="path to a *.parquet registry")
    ap.add_argument("--modality", choices=sorted(_MODALITIES), default="pdf",
                    help="which forward model to bank (default pdf)")
    ap.add_argument("--out", default="data/banked", help="base dir for blocks (default data/banked)")
    ap.add_argument("--n-views", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1, help="total array tasks over the id list")
    ap.add_argument("--shard", type=int, default=0, help="this task's index in [0, shards)")
    ap.add_argument("--workers", type=int, default=0, help="pool size; 0 = os.cpu_count()")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--sample", type=int, default=0, help="cap ids (quick test); 0 = all")
    ap.add_argument(
        "--range", dest="ranges", action="append", type=_parse_range, metavar="NAME=LO,HI",
        help="override one param's draw range (repeat per param); unset params keep "
             "core.transforms.PDF_RANGES. A zero-width NAME=c,c switches that param off. "
             "The block hash keys on the ranges, so every sweep point lands in its own "
             "block dir. Compute grid points with core.transforms.ranges_at rather than "
             "typing endpoints by hand.",
    )
    args = ap.parse_args()

    global _REG, _TF, _N, _SEED
    _N, _SEED = args.n_views, args.seed
    tf_cls, ranges, prefix = _MODALITIES[args.modality]
    overrides = dict(args.ranges or [])
    for name in overrides:
        if name not in ranges:
            raise SystemExit(
                f"--range {name}: not a {args.modality} param; expected one of {list(ranges)}"
            )
    _TF = tf_cls(**{f"{name}_range": lo_hi for name, lo_hi in overrides.items()})
    if overrides:
        print(f"range overrides: { {k: tuple(v) for k, v in overrides.items()} }", flush=True)
    _REG = MaterialRegistry.load(REPO_ROOT / args.registry if not Path(args.registry).is_absolute() else args.registry)

    ids = _REG.material_ids
    if args.sample:
        ids = ids[: args.sample]

    # Probe one sim to learn the grid (length + r-axis) before allocating.
    probe = _TF(_REG.get(ids[0]), seed=0)
    grid_len = len(probe.y)
    r_axis = probe.x

    block_hash = _config_hash(_TF, _N, _SEED, grid_len, len(ids))
    dataset = Path(args.registry).stem
    block = REPO_ROOT / args.out / f"{prefix}{dataset}__n{_N}__{block_hash}"
    block.mkdir(parents=True, exist_ok=True)

    gidxs, shard_ids = _shard_ids(ids, args.shards, args.shard)
    print(f"block {block.name}: shard {args.shard}/{args.shards} = {len(shard_ids)} ids "
          f"x {_N} views, grid {grid_len}, dtype {args.dtype}", flush=True)

    arr = np.empty((len(shard_ids) * _N, grid_len), dtype=args.dtype)
    rows: list[dict] = []
    tasks = list(zip(gidxs, shard_ids))
    nproc = args.workers or None
    done = 0
    # fork so workers inherit the loaded registry (COW) + capped BLAS env; also
    # makes globals (_REG/_TF/_N/_SEED) visible without re-loading. Linux defaults
    # to fork; pinning it keeps macOS (spawn-default) correct for local tests too.
    with get_context("fork").Pool(processes=nproc) as pool:
        for gidx, ys, params in pool.imap_unordered(_sim_material, tasks, chunksize=4):
            local = gidx - gidxs[0]
            arr[local * _N:(local + 1) * _N] = ys.astype(args.dtype)
            for v, p in enumerate(params):
                rows.append({"material_id": ids[gidx], "global_index": gidx, "view": v, **p})
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(shard_ids)} materials", flush=True)

    np.save(block / f"shard_{args.shard}.npy", arr)
    pd.DataFrame(rows).to_parquet(block / f"shard_{args.shard}.parquet", index=False)

    if args.shard == 0:
        # `modality` is recorded HERE, additively — meta.json is not the hash
        # payload, so this cannot repoint anything. The r_* keys keep their names
        # for every existing reader; for xrd they carry the 2theta axis (10, 80,
        # 0.014), and r_axis.npy likewise holds TT_GRID.
        meta = {
            "registry": args.registry, "dataset": dataset, "modality": args.modality,
            "n_ids": len(ids),
            "n_views": _N, "seed": _SEED, "shards": args.shards, "dtype": args.dtype,
            "grid_len": grid_len, "r_min": float(r_axis[0]), "r_max": float(r_axis[-1]),
            "r_step": float(r_axis[1] - r_axis[0]), "aug_ranges": _TF._ranges,
            "config_hash": block_hash,
        }
        (block / "meta.json").write_text(json.dumps(meta, indent=2))
        np.save(block / "r_axis.npy", r_axis)

    print(f"wrote shard {args.shard} -> {block}", flush=True)


if __name__ == "__main__":
    main()
