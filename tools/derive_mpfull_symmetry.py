"""tools/derive_mpfull_symmetry.py — sidecar symmetry labels for the full-MP registry.

`data/builders/mp_full.py` deliberately left space group / crystal system null:
"deriving them via SpacegroupAnalyzer on 133k structures is expensive with no
consumer yet". The consumer now exists — the sim-supervised crystal-system
baseline (`analysis/sim_supervised.py`, s38) trains on the banked full-MP XRD
views and needs a label per material.

**A SIDECAR, not a registry rebuild, on purpose.** The 85.4 GB cluster bank is
keyed to the registry it was banked from
(`data/banked/xrd__material_registry_mpfull__n32__ebd360d9` embeds its hash);
adding columns to `material_registry_mpfull.parquet` would produce a file the
bank no longer matches. So the labels land beside it as
`data/registry/mpfull_symmetry.parquet` (material_id, spacegroup,
crystal_system) and join on `material_id` at read time.

symprec=0.1 — the repo's one precedent (`data/builders/rruff.py`, where the
choice is audited) and Materials Project's own reporting convention, so the
derived labels match what MP itself would quote for these structures.
Structures whose analysis fails get a null row (kept, so coverage is visible)
and are excluded by the consumer.

Run (local Mac, ~15-40 min on 8 workers):
    python tools/derive_mpfull_symmetry.py [--workers 8] [--limit 500]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "data/registry/material_registry_mpfull.parquet"
OUT = REPO_ROOT / "data/registry/mpfull_symmetry.parquet"

SYMPREC = 0.1

#: IUCr ordering — identical to rruff.py/opxrd.py so "7" means cubic everywhere.
CRYSTAL_SYSTEMS = ("triclinic", "monoclinic", "orthorhombic", "tetragonal",
                   "trigonal", "hexagonal", "cubic")


def _label_one(args: tuple[str, str]) -> tuple[str, int | None, str | None]:
    """(material_id, spacegroup, crystal_system) — nulls when analysis fails."""
    mid, cif = args
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            struct = Structure.from_str(cif, fmt="cif")
            sga = SpacegroupAnalyzer(struct, symprec=SYMPREC)
            return mid, int(sga.get_space_group_number()), str(sga.get_crystal_system())
    except Exception:                    # noqa: BLE001 — a failed row is a null row,
        return mid, None, None           # counted and kept, never a dead build


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test subset")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if OUT.exists() and not args.overwrite:
        sys.exit(f"REFUSING: {OUT} already exists (use --overwrite).")

    import pandas as pd

    df = pd.read_parquet(REGISTRY, columns=["material_id", "structure"])
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} structures, {args.workers} workers, symprec={SYMPREC}", flush=True)

    rows = []
    with Pool(args.workers) as pool:
        it = pool.imap_unordered(
            _label_one, zip(df["material_id"], df["structure"]), chunksize=64)
        for i, row in enumerate(it, 1):
            rows.append(row)
            if i % 5000 == 0:
                print(f"... {i}/{len(df)}", flush=True)

    out = pd.DataFrame(rows, columns=["material_id", "spacegroup", "crystal_system"])
    out = out.sort_values("material_id").reset_index(drop=True)

    n_null = int(out["spacegroup"].isna().sum())
    unknown = sorted(set(out["crystal_system"].dropna()) - set(CRYSTAL_SYSTEMS))
    if unknown:
        raise ValueError(f"crystal system(s) outside the IUCr table: {unknown}")
    print(f"\nnulls (analysis failed): {n_null}/{len(out)}")
    print("class balance:")
    vc = out["crystal_system"].value_counts()
    for name in CRYSTAL_SYSTEMS:
        n = int(vc.get(name, 0))
        print(f"  {name:14s} {n:7d}  ({100 * n / len(out):.1f}%)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(out)} rows, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
