"""data/builders/mp_full.py — build the pretraining MaterialRegistry from full MP.

Source: the Materials Project 2019.04.01 dump — 133,420 relaxed ground-state
structures, **no atom-count cap** — from HuggingFace
`materials-toolkits/materials-project`, stored as an HDF5 of raw `cell`+`pos`+`z`
arrays (not CIFs). The dump is staged once to data/raw/mp2019/data.hdf5 and read
from there on every run (never touches the network). Each structure is
reconstructed into a pymatgen Structure and round-tripped to CIF with the same
recipe the old PDF_XRD_Fusion repo's generate_mpfull.py used, so material_ids
line up with MP-20's (`mp-<int>`) and the two pretraining sources share
identities where they overlap (~40k of MP-20's 45k rows are in this dump).

This is a **separate registry** from MP-20's material_registry.parquet (decided:
"swap = point at a different registry", DESIGN.md -> Datasets). It's the larger
pretraining pool; train.py picks which via --registry. full-MP is pretraining
data, so every row gets split="train" via core/splits.py fractions=(1,0,0) — no
val/test held out (SSL consumes the whole set).

Metadata is only what the dump cheaply provides: `energy_per_atom` (total energy
per atom, NOT a formation energy), `e_above_hull`, `pretty_formula`
(composition.reduced_formula), `n_atoms`. Space group / crystal system are NOT
source-provided here (unlike MP-20, whose CSV carried spacegroup.number) and
deriving them via SpacegroupAnalyzer on 133k structures is expensive with no
consumer yet — left null (add later if analysis/ needs stratification).

Run:
    python -m data.builders.mp_full                  # build + save (reads local dump)
    python -m data.builders.mp_full --limit 200       # smoke test on a subset
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import h5py
import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from core.registry import MaterialRegistry
from core.splits import assign

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "mp2019" / "data.hdf5"
DEFAULT_OUT = REPO_ROOT / "data" / "registry" / "material_registry_mpfull.parquet"


def _reconstruct(cell, z, frac_pos) -> tuple[str, str, int]:
    """One raw structure (cell, atomic numbers, fractional coords) -> (CIF, formula, n_atoms).

    Raises if the arrays don't form a valid structure — the caller drops and logs it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        struct = Structure(Lattice(cell), list(z), frac_pos, coords_are_cartesian=False)
        cif = str(CifWriter(struct))
    return cif, struct.composition.reduced_formula, struct.num_sites


def build(raw_path: Path, *, limit: int | None = None) -> pd.DataFrame:
    """Read the MP-2019 HDF5 dump and normalize to registry columns.

    Reconstructs each raw structure to CIF; drops (and logs) any that fail. The
    dump stores ~1k material_ids twice — genuine distinct polymorphs under one MP
    id — so repeats are KEPT, the first occurrence canonical (`mp-<id>`, so MP-20
    overlap still matches it) and later ones suffixed `mp-<id>-alt<k>` for a
    unique key.
    """
    with h5py.File(raw_path, "r") as f:
        cell = f["data/cell"][:]
        pos = f["data/pos"][:]
        z = f["data/z"][:]
        ptr = f["indexing/num_atoms"][:]  # (M+1,) cumulative atom offsets
        mid = f["data/material_id"][:]
        e_pa = f["data/energy_pa"][:]
        e_hull = f["data/energy_above_hull"][:]

    n = len(cell) if limit is None else min(limit, len(cell))
    rows = []
    failures = []
    for i in range(n):
        try:
            cif, formula, n_atoms = _reconstruct(
                cell[i], z[ptr[i] : ptr[i + 1]], pos[ptr[i] : ptr[i + 1]]
            )
        except Exception as exc:  # skip and keep going, like the old generate_mpfull
            failures.append((int(mid[i]), f"{type(exc).__name__}: {exc}"))
            continue
        rows.append({
            "material_id": f"mp-{int(mid[i])}",
            "structure": cif,
            "pretty_formula": formula,
            "energy_per_atom": float(e_pa[i]),
            "e_above_hull": float(e_hull[i]),
            "n_atoms": n_atoms,
        })
        if (i + 1) % 10000 == 0:
            print(f"  reconstructed {i + 1}/{n} ...", flush=True)

    if failures:
        print(f"dropping {len(failures)} structure(s) that failed to reconstruct: "
              f"{[m for m, _ in failures[:5]]}")

    df = pd.DataFrame(rows)
    rank = df.groupby("material_id").cumcount()  # 0 = first occurrence, 1.. = polymorphs
    n_poly = int((rank > 0).sum())
    if n_poly:
        print(f"suffixing {n_poly} repeated material_id(s) as polymorphs (mp-<id>-alt<k>; first kept as-is)")
        df["material_id"] = df["material_id"] + rank.map(lambda k: "" if k == 0 else f"-alt{k}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw", default=str(DEFAULT_RAW), help="MP-2019 data.hdf5 (staged locally)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None, help="only build the first N structures (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite an existing registry file")
    args = ap.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        sys.exit(
            f"REFUSING: raw dump not found at {raw_path}.\n"
            "Stage the MP-2019 dump there first — HuggingFace "
            "`materials-toolkits/materials-project`, or copy data.hdf5 from the old "
            "PDF_XRD_Fusion repo (PDF/Experiments/03_ssl_augmented/data/mp2019_raw/)."
        )

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        sys.exit(f"REFUSING: {out_path} already exists (use --overwrite).")

    df = build(raw_path, limit=args.limit)
    reg = MaterialRegistry(df)
    reg = assign(reg, fractions=(1.0, 0.0, 0.0))  # pretraining: everything is train

    report = reg.validate()
    print(report)
    report.raise_for_errors(context=str(out_path))

    saved = reg.save(out_path)
    size_mb = saved.stat().st_size / 1e6
    print(f"saved {saved}  ({len(reg)} rows, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
