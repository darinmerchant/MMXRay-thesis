"""data/builders/chili100k.py — build the CHILI-100K binary-metal-oxide FIT registry.

This registry exists for exactly one experiment: the **chemistry axis** of the
CHILI transfer grid. `DESIGN.md`'s 2x2 holds chemistry fixed and varies the
instrument (clean CHILI-3K vs augmented CHILI-3K). This registry adds the other
axis — fit on a chemically DIFFERENT set, evaluate on the same CHILI-3K test
rows — so the two shifts can be told apart:

    fit set                          -> eval clean-3K   -> eval aug-3K
    clean CHILI-3K (in-distribution)    clean_clean        clean_aug     (exist)
    clean CHILI-100K metal oxides       cross_clean        cross_aug     (this file)

**This is NOT a proxy for experimental validation, and must never be reported as
one.** CHILI-100K and CHILI-3K come from the same DebyeCalculator forward model
on the same grid, so every systematic error of that simulator is common-mode and
cancels. That is what makes it a clean chemistry axis (swap the structures, hold
the physics) and simultaneously what disqualifies it as a statement about real
diffractometers — see `RESULTS.md` Q5 (RRUFF) and Q6 (Au), where CHILI rank was
measured NOT to transfer to measured data.

Source: the raw CHILI-100K HDF5 files, staged by `scripts/unpack_chili100k.slurm`
under `data/raw/chili100k/cod_output_080124/<COD-ID>.h5` (20,882 files, one per
COD entry, five nanoparticle sizes each = 104,410 graphs).

**We read the `.h5` directly and never instantiate `CHILI(dataset='CHILI-100K')`.**
That class runs a torch_geometric processing step which writes ~100k small files
onto Lustre. The `.h5` carries everything this builder needs, so the `.pt` step
buys nothing and costs a filesystem.

Per (COD entry, particle size) -> one row (base_mode = ``both``):

    material_id       chili100k-<COD-ID>-<size_idx>   (size_idx 0-4, ascending)
    structure         CIF of the periodic PARENT cell, rebuilt from
                      UnitCellGraph/NodeFeatures[:,0] (Z) +
                      UnitCellGraph/FractionalCoordinates +
                      GlobalLabels/CellParameters — the same recipe as
                      `chili.py::_parent_structure`, different source fields.
    signal_xpdf_{x,y} CHILI native xPDF — (r, G), 6000 pts, r = 0 -> 59.99 step
                      0.01. VERIFIED bit-identical in origin and step to
                      CHILI-3K's, which is what lets `downstream_eval`'s read path
                      slice rather than resample (see `_print_balance`).
    target_np_size    nanoparticle size in ANGSTROM
    target_mo_bond    minimum metal-oxygen distance in the parent cell, ANGSTROM
    target_mean_bond  mean nearest-neighbour M-O distance, ANGSTROM
    group_key         composition (reduced formula) — the split unit
    metadata          cod_id, overlap_3k, crystal_system, space_group_number,
                      space_group_symbol, n_atoms, cell_a..cell_gamma
    has_pdf           True

**The three targets are IMPORTED from `data/builders/chili.py`, not reimplemented.**
`_mo_bond`, `_cn_and_bond` and `_group_key` are pure `Structure -> float` functions.
Reimplementing them here would let the two registries' labels drift apart under any
future edit, and a cross-registry comparison whose two sides label differently
measures the labelling difference, not the chemistry. The import is the whole point.

Deliberately NOT built:

- **The XRD channel.** The experiment is instantiated on PDF (`RESULTS.md` ->
  *What is not established*, item 6). `ScatteringData/<size>/XRD` is right there as
  (2, 580) on the same Q grid CHILI-3K uses; add it when a caller needs it.
- **`target_oxidation` / `target_cn` / `target_metal`.** All three are
  classification, and the class vocabularies do not survive the crossing: CHILI-3K's
  `oxidation` takes 5 values locked to its 12 prototypes, while COD binary oxides
  reach stoichiometries (M2O3 — Al2O3, Cr2O3, Fe2O3 are all here) that CHILI-3K's
  prototypes cannot produce. A head sized to the union would carry units it never
  trains on. `RESULTS.md` already records that `oxidation`/`cn` are coarsenings of
  `crystal_type` rather than independent tasks, so they are the worst targets to
  ask a chemistry-transfer question with.
- **An augmented channel — by this builder.** It is added by `tools/augment_chili.py`
  (2026-08-19; its chili100k mode, staged by `scripts/augment_chili100k.slurm` at the
  ~1.9x-the-3K-job cost measured here: sum n^2 6.17e10 vs 3.23e10). The channel is a
  FIT arm, not an eval arm: fit on aug-100K train via `downstream_eval --fit-registry
  ... --fit-signal xpdf_aug`, evaluate on CHILI-3K as ever — training data drawn from
  the pretraining augmentation distribution while the eval domain stays clean. Until
  then it was deliberately absent, because no cell of the 2x2 asked for it.

Row selection, in three stages (counts measured on the staged data, 2026-08-11):

    all CHILI-100K                                        20,882 structures
    exactly 2 elements, one of them O                      1,298
    the partner is a metal (METALLOIDS EXCLUDED)              942  -> 4,710 particles
    - reduced formula in a CHILI-3K TEST row                 898  -> 4,490 particles

**The leakage rule is composition-in-3K-test, and it is exact rather than
conservative.** CHILI-3K's splits are group-disjoint on composition, so a formula
is entirely in its test split or entirely absent from it — dropping those formulas
removes precisely the rows that could leak and nothing else. 44 structures go
(CuO 18, RuO2 7, Li2O2 6, HfO2 5, CrO2 3, Na2O 2, CdO2 2, AlO 1). The looser
matches survive and are RECORDED rather than dropped, in `overlap_3k`:

    none        no CHILI-3K parent cell shares this reduced formula
    formula     a 3K cell shares the formula (but that formula is not in 3K test)
    formula_sg  ... and the space group
    cell        ... and every cell length within 1% and every angle within 0.5 deg

so a stricter exclusion is a `--max-overlap` away instead of a rebuild.

Split: group-disjoint on ``group_key`` via `core.splits.assign`. **Only train and
val are consumed downstream** — the test rows come from CHILI-3K — but all three
are built so the registry stays self-describing and uniform with every other one.

**WATCH THE SPLIT BALANCE — `core/splits.py` SPLITS BY GROUP COUNT, NOT BY ROWS.**
Only **110** reduced formulas survive (118 pass the metal filter; the leak rule takes
8), and MgO alone is 202 of the 898 structures — 22.5%, with Fe3O4 at 82 and CaO at
58 behind it. 110 groups at 0.8/0.1/0.1 is **88/11/11 GROUPS**, and because the groups
are that uneven the ROW proportions drift a long way from what was asked. Measured at
seed 0: **4025/245/220 rows = 0.896/0.055/0.049**.

Seed 0 is the default and was KEPT after surveying seeds 0-7, rather than shopping for
a prettier val — selecting a split on a criterion invented after seeing the data is
worse practice than accepting the default. The survey is still worth knowing, because
two draws are disqualifying rather than merely ugly:

    seed 1  ->  val is 92% MgO
    seed 2  ->  train collapses to 1945 rows, BELOW the 2530 that `--n-train-frac`
                needs to match CHILI-3K's fit set. That lower bound is the real
                constraint on any re-seed, not the val balance.

Seed 0's val is 245 rows / 11 formulas / 32.7% Co3O4, and the early-stopping confound
that follows from it is recorded in `RESULTS.md` -> Q7. `_print_balance` prints the
largest group per split for exactly this reason.

**TO WIDEN VAL, MOVE THE TEST SPLIT INTO IT — `--fractions 0.8 0.2 0.0` — AND CHANGE
NOTHING ELSE.** `n_train = round(0.8 * 110) = 88` groups either way, and `assign`
shuffles once and takes the first `n_train`, so **the train split is bit-identical**
(4025 rows, same `material_id` set) while val goes 245 -> 465 rows, 11 -> 22 formulas,
largest group 32.7% -> 26.9%, and the old val is a strict SUBSET of the new one. That
is what makes it a clean control on the early-stopping confound: the fit set does not
move, so a changed result cannot be a changed-fit-set result. `--n-train-frac` stays
0.6286, so the cells stay comparable to `runs/archive/chem_transfer/` one for one.

Do NOT reach for `--fractions 0.85 0.15 0.0` (an earlier plan said to): it makes val
LUMPIER, not less — 335 rows / 16 formulas / **36.9% MnO2** — and moves 6 formulas into
train, which changes the fit set and destroys the attribution. Nor re-seed: `--seed`
changes train too, and seeds 1 and 2 are outright disqualifying (above).

Run:
    python -m data.builders.chili100k                  # build + save
    python -m data.builders.chili100k --limit 50       # smoke test on a subset

    # re-split an existing registry — no .h5 scan, no Lustre, ~2 s:
    python -m data.builders.chili100k --resplit data/downstream/chili100k/chili100k_registry.parquet \
      --fractions 0.8 0.2 0.0 --out data/downstream/chili100k/chili100k_registry_val20.parquet

`--resplit` exists because the scan is the whole cost and the split is not part of it:
`assign` is a pure function of (group order, seed, fractions), so re-splitting the
built parquet gives BYTE-IDENTICAL rows to a rebuild at the same flags while skipping
a multi-hour pass over 20,882 .h5 on Lustre (`scripts/build_chili100k.slurm`).

Cost: dominated by one CrystalNN pass per structure (~0.12 s), so ~2 min for the
898 parent cells — the five sizes of one COD entry share a parent cell and are
labelled once, which is why this is far cheaper than 4,490 x 0.12 s.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from core.registry import MaterialRegistry
from core.splits import assign
from data.builders.chili import _cn_and_bond, _group_key, _mo_bond
from modalities.pdf.simulate import RMIN, RSTEP

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "chili100k" / "cod_output_080124"
DEFAULT_OUT = (REPO_ROOT / "data" / "downstream" / "chili100k"
               / "chili100k_registry.parquet")
DEFAULT_CHILI3K = (REPO_ROOT / "data" / "downstream" / "chili"
                   / "chili_registry.parquet")
GROUP_COL = "group_key"

#: Native length of the stored xPDF, and the prefix the read path consumes.
#: Kept as constants so a source file on a different reduction fails the shape
#: check instead of quietly producing a registry on a grid nothing else shares.
N_GR = 6000
N_R = 5000
R_GRID = RMIN + np.arange(N_R) * RSTEP

#: Everything that is NOT a metal for the purposes of "binary metal oxide".
#: The metalloids are on this list ON PURPOSE — SiO2 is 203 of the 1,298
#: two-element oxides and calling quartz a metal oxide would be the single
#: largest error in the selection. Including Ge/As/Sb/Te for the same reason.
NONMETALS = frozenset({
    "H", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar",
    "Ge", "As", "Se", "Br", "Kr", "Sb", "Te", "I", "Xe", "At", "Rn", "He",
})

#: Tolerances for the `cell`-level CHILI-3K overlap tier. Loose enough to absorb
#: the difference between a COD redetermination and CHILI-3K's ionic-radius-derived
#: lattice constants; tight enough that two genuinely different polymorphs of one
#: formula do not collapse into one match.
CELL_LTOL = 0.01   # fractional, on a/b/c
CELL_ATOL = 0.5    # degrees, on alpha/beta/gamma

OVERLAP_LEVELS = ("none", "formula", "formula_sg", "cell")


class _Chili3kIndex:
    """What CHILI-3K looks like, for the leak rule and the overlap census.

    Built from the CHILI-3K registry rather than from its raw `.pt`, so this
    builder needs only a parquet and never a second raw tree. `cells` is keyed by
    (formula, space group) because that is the coarsest key a cell comparison can
    be conditioned on — comparing lattice parameters across different formulas is
    meaningless.
    """

    CELL_COLS = ("cell_a", "cell_b", "cell_c", "cell_alpha", "cell_beta", "cell_gamma")

    def __init__(self, registry_path: Path):
        df = pd.read_parquet(
            registry_path,
            columns=[GROUP_COL, "split", "space_group_number", *self.CELL_COLS],
        )
        #: The leak rule. 3K splits are group-disjoint on composition, so a formula
        #: is wholly in test or wholly not — this set IS the leak, exactly.
        self.test_comps = set(df.loc[df["split"] == "test", GROUP_COL])
        parents = df.drop_duplicates(subset=[GROUP_COL, "space_group_number", *self.CELL_COLS])
        self.comps = set(parents[GROUP_COL])
        self.comp_sg = set(zip(parents[GROUP_COL], parents["space_group_number"].astype(int)))
        self.cells: dict[tuple[str, int], list[tuple[float, ...]]] = defaultdict(list)
        for row in parents.itertuples(index=False):
            key = (getattr(row, GROUP_COL), int(row.space_group_number))
            self.cells[key].append(tuple(float(getattr(row, c)) for c in self.CELL_COLS))
        self.n_parents = len(parents)

    def overlap(self, formula: str, sg: int, cell: tuple[float, ...]) -> str:
        """Tiered match against CHILI-3K's parent cells -> an `OVERLAP_LEVELS` value."""
        if formula not in self.comps:
            return "none"
        if (formula, sg) not in self.comp_sg:
            return "formula"
        for other in self.cells[(formula, sg)]:
            lengths_ok = all(
                abs(cell[i] - other[i]) <= CELL_LTOL * max(cell[i], other[i]) for i in range(3)
            )
            angles_ok = all(abs(cell[i] - other[i]) <= CELL_ATOL for i in range(3, 6))
            if lengths_ok and angles_ok:
                return "cell"
        return "formula_sg"


def _parent_structure(h5):
    """Periodic parent cell from the HDF5 — `chili.py::_parent_structure`'s recipe.

    Same three ingredients as CHILI-3K (Z, fractional coordinates, cell parameters)
    read from `UnitCellGraph`/`GlobalLabels` instead of a torch_geometric `Data`,
    and guarded identically so a malformed file drops one row rather than producing
    a nonsense CIF.
    """
    from pymatgen.core import Lattice, Structure

    cp = np.asarray(h5["GlobalLabels/CellParameters"][()], dtype=float).ravel()
    z = np.asarray(h5["UnitCellGraph/NodeFeatures"][()])[:, 0].round().astype(int)
    frac = np.asarray(h5["UnitCellGraph/FractionalCoordinates"][()], dtype=float)
    if cp.shape != (6,):
        raise ValueError(f"CellParameters {cp.shape}, expected (6,)")
    if not ((z >= 1).all() and (z <= 118).all()):
        raise ValueError(f"NodeFeatures[:,0] not valid atomic numbers: {np.unique(z)}")
    if frac.shape != (len(z), 3):
        raise ValueError(f"FractionalCoordinates {frac.shape} inconsistent with {len(z)} species")
    return Structure(Lattice.from_parameters(*cp), [int(v) for v in z], frac)


def _is_binary_metal_oxide(h5) -> bool:
    """Exactly two elements, one of them oxygen, the other a metal.

    Read off `GlobalLabels/ElementsPresent` (atomic numbers) rather than the parent
    structure, because it is the cheap check — it decides ~95% of the files without
    building a pymatgen `Structure` or opening any other group.
    """
    from pymatgen.core import Element

    z = [int(round(v)) for v in h5["GlobalLabels/ElementsPresent"][()]]
    if len(z) != 2 or 8 not in z:
        return False
    partner = Element.from_Z([v for v in z if v != 8][0])
    return partner.symbol not in NONMETALS


def _sizes(h5) -> list[str]:
    """The five particle-size group names, ascending by diameter.

    The names are the diameters themselves ('13.64Å'), and they differ per COD
    entry, so they are sorted numerically rather than lexically — '9.4Å' must not
    sort after '13.6Å'. `DiscreteParticleGraphs` and `ScatteringData` are asserted
    to carry the same set, since a row pairs one group's atoms with the other's G(r).
    """
    particles, scattering = list(h5["DiscreteParticleGraphs"]), list(h5["ScatteringData"])
    if set(particles) != set(scattering):
        raise ValueError(f"size groups disagree: {sorted(particles)} vs {sorted(scattering)}")
    return sorted(particles, key=lambda k: float(k.rstrip("Å")))


def _xpdf(h5, size: str) -> tuple[np.ndarray, np.ndarray]:
    """`ScatteringData/<size>/xPDF` stored as [r, G] -> (x, y) float32 arrays.

    **float32 arrays, not Python lists of floats.** `chili.py` uses `.tolist()`, which
    is fine at 3180 rows and is not here: a 6000-element list of Python floats is
    ~32 B/value against numpy's 4, and at 4,490 rows x 2 arrays that is the difference
    between ~200 MB and ~16 GB peak RSS (measured, job 5375316). The source data is
    float32 in the HDF5 anyway, and the read path casts to torch float32, so nothing
    is lost. The shared r-grid is deduplicated separately in `build`.
    """
    arr = np.asarray(h5[f"ScatteringData/{size}/xPDF"][()])
    if arr.shape != (2, N_GR):
        raise ValueError(f"xPDF: expected (2, {N_GR}), got {arr.shape}")
    return arr[0].astype(np.float32), arr[1].astype(np.float32)


def _cod_ids(raw_dir: Path) -> list[str]:
    """All COD ids present, from the <COD-ID>.h5 filenames, sorted numerically."""
    ids = [p.stem for p in raw_dir.glob("*.h5")]
    if not ids:
        raise RuntimeError(f"no *.h5 files in {raw_dir}")
    return sorted(ids, key=int)


def _text(dataset) -> str:
    """A scalar HDF5 string label as `str`, whether h5py hands back bytes or str."""
    value = dataset[()]
    return value.decode() if isinstance(value, bytes) else str(value)


def build(raw_dir: Path, index: _Chili3kIndex, *, limit: int | None = None) -> pd.DataFrame:
    """Read the staged CHILI-100K .h5 and normalize to registry columns.

    Labelling happens ONCE per COD entry — the five particle sizes of one entry share
    a parent cell, so `_mo_bond`/`_cn_and_bond` run 898 times, not 4,490. That is the
    difference between a two-minute build and a ten-minute one.
    """
    from pymatgen.analysis.local_env import CrystalNN

    cnn = CrystalNN()  # built once and reused: it is the builder's whole runtime cost
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    stats = {"files": 0, "binary_metal_oxide": 0, "leak_dropped": 0}
    grid: np.ndarray | None = None  # the one shared r-axis; see the row loop

    cod_ids = _cod_ids(raw_dir)
    if limit is not None:
        cod_ids = cod_ids[:limit]

    # Progress, because this loop is silent for ~15 min on Lustre and a silent job is
    # indistinguishable from a hung one — which is exactly the ambiguity that cost
    # nine hours when the CHILI-100K download died (docs/TRAPS.md). Every 2000 files,
    # so it is ~10 lines in a log rather than a scroll.
    for cod_id in cod_ids:
        stats["files"] += 1
        if stats["files"] % 2000 == 0:
            print(f"  ... {stats['files']}/{len(cod_ids)} files, "
                  f"{len(rows)} rows so far", flush=True)
        try:
            with h5py.File(raw_dir / f"{cod_id}.h5", "r") as h5:
                if not _is_binary_metal_oxide(h5):
                    continue
                stats["binary_metal_oxide"] += 1

                struc = _parent_structure(h5)
                formula = _group_key(struc)
                # THE LEAK RULE. Checked before any expensive labelling: a formula in
                # CHILI-3K's test split cannot appear in a set we fit on, and there is
                # no point running CrystalNN on a row that is about to be discarded.
                if formula in index.test_comps:
                    stats["leak_dropped"] += 1
                    continue

                cif = struc.to(fmt="cif")
                sg = int(h5["GlobalLabels/SpaceGroupNumber"][()])
                cell = (struc.lattice.a, struc.lattice.b, struc.lattice.c,
                        struc.lattice.alpha, struc.lattice.beta, struc.lattice.gamma)
                overlap = index.overlap(formula, sg, cell)
                # _mo_bond first — it is the metal-oxide guard `_cn_and_bond` relies on,
                # exactly as in chili.py. `_is_binary_metal_oxide` should have made it
                # unreachable, so a raise here means the two disagree and the row drops.
                mo_bond = _mo_bond(struc)
                _, mean_bond = _cn_and_bond(cnn, struc)

                sizes = _sizes(h5)
                for size_idx, size in enumerate(sizes):
                    particle = h5[f"DiscreteParticleGraphs/{size}"]
                    xpdf_x, xpdf_y = _xpdf(h5, size)
                    # Every row's r-axis is the same 6000 points. Keep ONE array and
                    # reference it, the way au_selfdriving.py does — storing a distinct
                    # copy per row is 4,490 identical arrays for no information. Sharing
                    # it would also HIDE a file whose grid differed, so each one is
                    # checked against the kept grid before being dropped; that check is
                    # per-file, where `_print_balance`'s is only on row 0.
                    if grid is None:
                        grid = xpdf_x
                    elif not np.array_equal(xpdf_x, grid):
                        raise ValueError(f"{cod_id}/{size}: r-grid differs from the first file")
                    rows.append({
                        "material_id": f"chili100k-{cod_id}-{size_idx}",
                        "structure": cif,
                        "signal_xpdf_x": grid,
                        "signal_xpdf_y": xpdf_y,
                        "target_np_size": float(particle["NP size (Å)"][()]),
                        "target_mo_bond": mo_bond,
                        "target_mean_bond": mean_bond,
                        "has_pdf": True,
                        "group_key": formula,
                        "cod_id": cod_id,
                        "overlap_3k": overlap,
                        "crystal_system": _text(h5["GlobalLabels/CrystalSystem"]),
                        "space_group_symbol": _text(h5["GlobalLabels/SpaceGroupSymbol"]),
                        "space_group_number": sg,
                        "n_atoms": int(particle["NodeFeatures"].shape[0]),
                        "cell_a": float(cell[0]), "cell_b": float(cell[1]),
                        "cell_c": float(cell[2]), "cell_alpha": float(cell[3]),
                        "cell_beta": float(cell[4]), "cell_gamma": float(cell[5]),
                    })
        except Exception as e:  # noqa: BLE001 — skip and keep going, like chili.py
            failures.append((cod_id, f"{type(e).__name__}: {e}"))
            continue

    print(f"scanned {stats['files']} file(s): "
          f"{stats['binary_metal_oxide']} binary metal oxide(s), "
          f"{stats['leak_dropped']} dropped for a formula in CHILI-3K test, "
          f"{stats['binary_metal_oxide'] - stats['leak_dropped'] - len(failures)} kept")
    if failures:
        print(f"dropping {len(failures)} entr(ies) that failed to build: {failures[:5]}")
    return pd.DataFrame(rows)


def _print_balance(reg: MaterialRegistry, index: _Chili3kIndex) -> None:
    """What the build measured — printed so the caveats are seen, not just filed."""
    df = reg.frame
    order = [s for s in ("train", "val", "test") if s in df["split"].unique()]

    # THE load-bearing check, ported from au_selfdriving.py. This registry stores a
    # longer grid than the encoder reads and relies on `load_probe_data` SLICING it.
    # If the origin or step ever disagreed with CHILI-3K's, the cross-registry fit
    # would be comparing two different r axes and the whole experiment would be void.
    r = np.asarray(df["signal_xpdf_x"].iloc[0], dtype=np.float64)
    dev = np.abs(r[:N_R] - R_GRID).max()
    print(f"\nr-grid: {len(r)} pts over [{r[0]:.2f}, {r[-1]:.2f}] A")
    print(f"  leading {N_R} pts vs RMIN + arange({N_R}) * RSTEP: max dev {dev:.3e}")
    if not np.allclose(r[:N_R], R_GRID, atol=RSTEP / 100):
        raise ValueError(
            f"stored r-grid does not match the pretraining grid (max dev {dev:.3e}) — "
            f"downstream_eval slices rather than resamples and would refuse this registry"
        )
    print(f"  -> slice, not resample: [{r[0]:.2f}, {r[N_R - 1]:.2f}] A is the read path's view")

    # THE leak assertion (the reason this registry exists in this shape).
    leaked = set(df[GROUP_COL]) & index.test_comps
    assert not leaked, f"composition(s) also in the CHILI-3K TEST split: {sorted(leaked)[:5]}"
    print(f"\nno formula shared with CHILI-3K's test split: OK "
          f"({len(index.test_comps)} test compositions checked)")

    census = df.drop_duplicates("cod_id")["overlap_3k"].value_counts()
    print(f"overlap_3k census (structures, of {df['cod_id'].nunique()}): "
          f"{ {k: int(census.get(k, 0)) for k in OVERLAP_LEVELS} }")
    print(f"  ('cell' rows share a parent cell with a NON-test CHILI-3K row — recorded, "
          f"not dropped; filter on this column for a stricter exclusion)")

    print(f"\ncrystal_system balance (% of each split's rows):")
    print((pd.crosstab(df["crystal_system"], df["split"], normalize="columns") * 100)
          [order].round(1).to_string())

    print("\ntarget ranges:")
    for col in [c for c in df.columns if c.startswith("target_")]:
        print(f"  {col:20s} [{df[col].min():8.3f}, {df[col].max():8.3f}]  "
              f"nulls={int(df[col].isna().sum())}")

    print(f"\nsplit sizes (rows): {df['split'].value_counts().to_dict()}")
    # SPLIT LUMPINESS. 110 formulas is few enough that one compound can dominate a
    # split; MgO alone is ~22% of the registry. A val split that is mostly one
    # compound makes early stopping a statement about that compound, so the largest
    # group per split is printed rather than left to be discovered later.
    for s in order:
        sizes = df.loc[df["split"] == s, GROUP_COL].value_counts()
        n = int(sizes.sum())
        print(f"  {s:5s}: {sizes.size:3d} formula(s), largest {sizes.index[0]} "
              f"= {int(sizes.iloc[0])} rows ({100 * sizes.iloc[0] / n:.1f}% of the split)")

    per_group = df.groupby(GROUP_COL)["split"].nunique()
    assert (per_group == 1).all(), "a composition leaked across splits!"
    print(f"group-disjoint on {GROUP_COL!r}: OK ({per_group.size} compositions, none split)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--chili3k-registry", default=str(DEFAULT_CHILI3K),
                    help="source of the test compositions the leak rule drops")
    ap.add_argument("--fractions", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="split fractions; the test split is unused downstream")
    ap.add_argument("--limit", type=int, default=None, help="only scan the first N .h5 files")
    ap.add_argument("--seed", type=int, default=0, help="split seed")
    ap.add_argument("--resplit", default=None, metavar="PARQUET",
                    help="re-split an already-built registry instead of scanning the .h5 "
                         "(same rows, new `split` column) — see the module docstring")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        sys.exit(f"REFUSING: {out_path} already exists (use --overwrite).")

    index = _Chili3kIndex(Path(args.chili3k_registry))
    print(f"CHILI-3K index: {index.n_parents} parent cells, {len(index.comps)} compositions, "
          f"{len(index.test_comps)} of them in its test split")

    if args.resplit:
        # The leak check and every balance print below run again on the loaded frame,
        # so a re-split is validated exactly as hard as a build — this path skips the
        # scan, not the guards.
        src = Path(args.resplit)
        if src.resolve() == out_path.resolve():
            sys.exit(f"REFUSING: --resplit and --out are the same file ({src}).")
        df = pd.read_parquet(src)
        print(f"re-splitting {src} ({len(df)} rows, {df[GROUP_COL].nunique()} compositions) "
              f"— no .h5 read")
    else:
        df = build(Path(args.raw_dir), index, limit=args.limit)
    reg = MaterialRegistry(df)
    reg = assign(reg, fractions=tuple(args.fractions), group_by=GROUP_COL, seed=args.seed)

    report = reg.validate()
    print(report)
    report.raise_for_errors(context=str(out_path))
    _print_balance(reg, index)

    saved = reg.save(out_path)
    print(f"\nsaved {saved}  ({len(reg)} rows, {saved.stat().st_size / 1e6:.1f} MB, "
          f"signals={reg.signal_names})")


if __name__ == "__main__":
    main()
