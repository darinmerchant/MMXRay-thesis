"""tools/carve_mp_aug3k.py — a 3,000-material MP set, one banked view each, for the augmentation-PCA figure.

    # SuperCloud only — the mpfull PDF bank is 80 GB and exists nowhere else:
    ~/.conda/envs/mmx/bin/python -m tools.carve_mp_aug3k

Writes a DOWNSTREAM-REGISTRY-SHAPED parquet (`signal_mpaug_x/_y` + `aug_*` columns
+ `split`) rather than a bespoke npz, so `analysis/downstream_eval.load_probe_data`
and `analysis/finetune.py` read it with no new code: the figure's fine-tuned column
is then `finetune.py --registry <this> --signal mpaug --target aug_qmin`, not a
second training loop written to visualize one figure.

WHY THE mpfull BANK AND NOT THE LOCAL 45k ONE. The encoder in the figure's frozen
column is `cnn_vicreg_mpfull_final`, pretrained on exactly this block, so its views
are the distribution the encoder saw — the same "training corpus only" footing as
RESULTS.md -> Q4 -> *The theta-probe*, and no sim->real gap is smuggled in. The 45k
`material_registry` bank also cannot reach MP's own crystal-system distribution:
it holds 4.2% triclinic against full-MP's 11.3%.

STRATIFIED TO MP'S OWN DISTRIBUTION, largest-remainder, from
`data/registry/mpfull_symmetry.parquet` (symprec 0.1, `tools/derive_mpfull_symmetry.py`).
The point of stratifying rather than sampling uniformly is that the figure is read
for structure-vs-instrument organization: a set whose crystal systems are skewed
away from MP would leave "is that gradient chemistry or sampling?" open.

ONE VIEW PER MATERIAL, drawn from the block's 32. All four instrument parameters
are therefore ONE joint draw per row (`core.transforms.draw_pdf_params` draws them
independently), so the four augmentation panels are literally the same 3,000 points
recoloured, and no augmentation panel can be read against a different point set.

RAW G(r) IS WHAT IS STORED. The bank is raw by convention (DESIGN.md -> Signal
normalization) and min-max normalization happens at the READ path; storing
normalized signal here would double-normalize under `load_probe_data`.

SPLITS ARE THIS FILE'S OWN, 70/10/20, stratified by crystal system. `val` exists
because a gradient finetune early-stops on it; the figure's R^2 stamps are fit on
`train` and scored on `test`. The mpfull registry's own `split` column describes
PRETRAINING and is deliberately not reused: every row here was in pretraining.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "data" / "banked" / "material_registry_mpfull__n32__55ad1ff1"
SYMMETRY = REPO / "data" / "registry" / "mpfull_symmetry.parquet"
OUT = REPO / "data" / "downstream" / "mp_aug3k" / "mp_aug3k.parquet"

SIGNAL = "mpaug"
PARAMS = ("uiso", "qmax", "qbroad", "qmin")
#: 70/10/20 — train fits the ridge and the finetunes, val early-stops them, test is scored.
SPLIT_FRACS = {"train": 0.70, "val": 0.10, "test": 0.20}


def largest_remainder(props: pd.Series, n: int) -> pd.Series:
    """Integer counts summing EXACTLY to `n`, closest to `props * n`."""
    exact = props * n
    counts = np.floor(exact).astype(int)
    for idx in (exact - counts).sort_values(ascending=False).index[: n - counts.sum()]:
        counts[idx] += 1
    return counts


def assign_splits(systems: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Stratify the 70/10/20 split WITHIN each crystal system, so the test rows the
    stamps are scored on carry the same symmetry mix as the rows they were fit on."""
    out = np.empty(len(systems), dtype=object)
    for system in systems.unique():
        where = np.flatnonzero((systems == system).to_numpy())
        order = rng.permutation(where)
        n_train = int(round(len(order) * SPLIT_FRACS["train"]))
        n_val = int(round(len(order) * SPLIT_FRACS["val"]))
        out[order[:n_train]] = "train"
        out[order[n_train:n_train + n_val]] = "val"
        out[order[n_train + n_val:]] = "test"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--symmetry", type=Path, default=SYMMETRY)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"REFUSING: {args.out} exists (use --overwrite).")

    import json
    meta = json.loads((args.bank / "meta.json").read_text())
    n_views, grid_len = int(meta["n_views"]), int(meta["grid_len"])
    r_axis = np.load(args.bank / "r_axis.npy").astype(np.float32)
    print(f"{args.bank.name}: {meta['n_ids']} materials x {n_views} views, grid {grid_len}")

    sym = pd.read_parquet(args.symmetry).dropna(subset=["crystal_system"])
    props = sym.crystal_system.value_counts(normalize=True)
    counts = largest_remainder(props, args.n)
    print("MP distribution -> requested counts:")
    for system in counts.index:
        print(f"  {system:<13} {props[system] * 100:5.2f}%  ->  {counts[system]:4d}")

    rng = np.random.default_rng(args.seed)
    picked = pd.concat([
        sym[sym.crystal_system == system].sample(n=int(counts[system]), random_state=rng.integers(1 << 31))
        for system in counts.index
    ])
    picked["view"] = rng.integers(0, n_views, size=len(picked))
    want = picked.set_index("material_id")

    # Pull each chosen row out of its shard. Materials are contiguous within a shard
    # at `n_views` rows each, so the row of one view is fixed by the shard's own
    # first global_index — the same arithmetic `core.dataset_base.BankedPretrainDataset`
    # uses, restated here because that class is a two-view pretraining Dataset.
    shards = sorted(args.bank.glob("shard_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    if not shards:
        raise SystemExit(f"no shard_*.npy in {args.bank}")
    rows = []
    for npy in shards:
        params = pd.read_parquet(npy.with_suffix(".parquet"))
        hit = params[params.material_id.isin(want.index)]
        if hit.empty:
            continue
        hit = hit[hit.view.to_numpy() == want.loc[hit.material_id, "view"].to_numpy()]
        if hit.empty:
            continue
        gidx0 = int(params.global_index.min())
        arr = np.load(npy, mmap_mode="r")
        take = (hit.global_index.to_numpy() - gidx0) * n_views + hit.view.to_numpy()
        for (_, row), signal in zip(hit.iterrows(), np.asarray(arr[take], dtype=np.float32)):
            rows.append({
                "material_id": row.material_id,
                "crystal_system": want.loc[row.material_id, "crystal_system"],
                "view": int(row.view),
                "global_index": int(row.global_index),
                f"signal_{SIGNAL}_x": r_axis,
                f"signal_{SIGNAL}_y": signal,
                **{f"aug_{p}": float(row[p]) for p in PARAMS},
            })
        print(f"  {npy.name}: {len(hit)} rows ({len(rows)}/{args.n})", flush=True)

    df = pd.DataFrame(rows)
    if len(df) != args.n:
        raise SystemExit(f"carved {len(df)} rows, expected {args.n} — a shard is missing")
    df["split"] = assign_splits(df.crystal_system, rng)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")
    print(df.groupby(["split", "crystal_system"]).size().unstack(fill_value=0).to_string())
    print("\nrealized augmentation draws:")
    print(df[[f"aug_{p}" for p in PARAMS]].describe().loc[["min", "mean", "max"]].to_string())


if __name__ == "__main__":
    main()
