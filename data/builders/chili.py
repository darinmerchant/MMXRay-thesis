"""data/builders/chili.py — build the CHILI-3K downstream MaterialRegistry.

Source: **only** the CHILI-3K processed `.pt` files (torch_geometric `Data`),
staged under `data/raw/chili/processed/`. Nothing reads the old repo's
hand-made `metadata.csv` — everything is derived from the canonical `.pt`, so
the whole registry (including the split-grouping key) is built by this repo's
own infrastructure. Each sample is a nanoparticle carved from a bulk parent
cell, and each `.pt` carries the parent cell plus CHILI's own simulated signals.

Per sample -> one registry row (base_mode = ``both``: structure + measured signals):

    material_id       chili-<sample_idx>   (sample_idx from the data_<idx>.pt filename)
    structure         CIF of the periodic PARENT cell, rebuilt from
                      unit_cell_x[:,0] (Z) + unit_cell_pos_frac + cell_params
                      (the exact recipe the reference re-simulates from).
    signal_xrd_{x,y}  CHILI native XRD  — (Q, I), 580 pts, Q-space Debye model.
    signal_xpdf_{x,y} CHILI native xPDF — (r, G), 6000 pts.
    target_np_size    nanoparticle size in ANGSTROM (d.y['np_size'] as-is; the
                      reference divides by 10 for nm only at simulation time).
    target_mo_bond    minimum metal-oxygen distance in the parent cell, ANGSTROM
                      — the reference's `nn_MO_bond_A`, recomputed from the CIF
                      rather than ported from its hand-made CSV (see _mo_bond).
    target_oxidation  formal metal charge from stoichiometry, O = -2 (see _oxidation).
    target_cn         metal coordination number, CrystalNN; NaN if sites disagree.
    target_mean_bond  mean nearest-neighbour M-O distance, ANGSTROM (see _cn_and_bond).
    target_metal      atomic number of the metal element (see _metal).
    group_key         composition (reduced formula) — the split unit.
    metadata          crystal_type, crystal_system, space_group_number,
                      space_group_symbol, n_atoms, cell_a..cell_gamma.
    has_xrd/has_pdf   both True.

The last three port the ground-truth targets of Na Narong et al., npj Comput. Mater.
11, 98 (2025) — oxidation state, coordination number, mean nearest-neighbour bond
length — onto CHILI. Their recipe labels the periodic MP cell with pymatgen
(BVAnalyzer for oxidation state, CrystalNN for the other two), and so does this one,
with one substitution: bond valence is replaced by formal stoichiometric charge,
because CHILI's cells are hypothetical (Al2O, AuO3 and B3O4 are all present) and a
bond-valence sum is not meaningful on them.

**READ THIS BEFORE INTERPRETING THE TWO CLASSIFICATION TARGETS.** CHILI-3K is
combinatorial: 53 metals x 12 prototypes x 5 nanoparticle sizes = 3180, and each
prototype carries exactly one stoichiometry (RockSalt/Wurtzite/ZincBlende/NiAs/CsCl
-> MO, Rutile/Fluorite/CdCl2/CdI2 -> MO2, AntiFluorite -> M2O, Spinel -> M3O4,
ReO3 -> MO3). So `target_oxidation` and `target_cn` are both deterministic functions
of `crystal_type` — two coarsenings of one 12-way prototype variable, not two
independent tasks, and a probe scoring well on them has identified the prototype
rather than resolved a local environment. `target_mean_bond` is the one of the three
that varies with the METAL (1.65-3.32 A measured) and is not prototype identification.
Neither classification target leaks across splits — both are group-constant — but
neither is as hard as the paper's version. (Both couplings verified on all 3180 rows,
not a sample: `groupby("crystal_type")[target].nunique().max() == 1` for each.)

`target_mean_bond` has its own caveat, measured s20 on the rebuilt registry: it
correlates with the pre-existing `target_mo_bond` at **r = 0.999**, and the two are
equal to 1e-6 on 66.8% of rows (mean gap 0.0034 A, max 0.1845 A). CHILI's prototypes
are high-symmetry, so most metal sites have every neighbour at one distance and the
mean simply IS the minimum; only the distorted prototypes separate them. It is kept
because it is the paper's actual definition and makes that comparison explicit, but
it is close to a restatement of `target_mo_bond` and should not be read as an
independent task. A frozen probe scores them 0.6307 vs 0.6241 R2 — the same number.

`target_metal` (session 21, not from the paper) is the counterpart with the OPPOSITE
property: it is the one classification target that is INDEPENDENT of `crystal_type`.
The 53 metals and 12 prototypes are fully crossed, so metal identity is not a
coarsening of the prototype variable the way oxidation/CN are — a probe scoring it
well has actually recognized the metal, not just the structure. 53 classes, encoded
as the metal's atomic number rather than an arbitrary index, so the label is a real
physical quantity like the other targets (see `_metal`).

**`target_metal` has a real per-class TEST-COVERAGE gap, caused by the split, not the
target.** The 12 prototypes collapse to only 5 distinct compositions per metal (one
per stoichiometry class: the 5 MO-forming prototypes share one `group_key`, the 4
MO2-forming ones share another, and M2O/M3O4/MO3 are one prototype each) — measured
exactly 5/metal, 265 = 53*5 groups total. Splitting 5 groups 80/10/10 per metal is
enough variance that, measured on the built registry, **29 of the 53 metals have
ZERO rows in `test`**. Weighted F1 is unaffected (it is implicitly restricted to
classes with test support), but a per-class F1 breakdown will show ~29 classes never
evaluated — `_print_balance` reports the exact count at build time. This is a property
of composition-level group-disjoint splitting applied to a 53-way target, not a bug.

Split: group-disjoint on ``group_key`` via `core.splits.assign` (default
0.8/0.1/0.1). ``group_key`` = ``composition.reduced_formula`` — deliberately
COMPOSITION-level, not structure-level, even though the only target built so
far (``np_size``) only needs structure-level grouping. Composition is a strict
superset: grouping by it also keeps every polymorph of one formula (and thus
every parent structure) together. ``target_mo_bond`` (session 18) vindicated
that choice — it is constant per PARENT STRUCTURE (not per composition, as the
old repo assumed), and composition-level grouping contains structure-level
grouping, so it landed leak-safe without re-splitting. The three targets added in
session 20 needed no re-split for the same reason: all three are constant across a
parent structure's 5 nanoparticle sizes, and ``target_oxidation`` is constant across
a whole composition exactly. NOTE: `splits.py` is not stratified; the builder prints
the per-split class balance for ``crystal_system`` and both classification targets so
that cost is visible.

Run:
    python -m data.builders.chili                  # build + save
    python -m data.builders.chili --limit 200       # smoke test on a subset

Cost: still diffpy-free, but no longer instant — CrystalNN is ~0.12 s per structure,
so a full build spends ~6.5 min on `target_cn`/`target_mean_bond` on top of reading
the 3180 .pt files.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from core.registry import MaterialRegistry
from core.splits import assign

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "chili"
DEFAULT_OUT = REPO_ROOT / "data" / "downstream" / "chili" / "chili_registry.parquet"
GROUP_COL = "group_key"


def _parent_structure(d):
    """Periodic parent cell — the reference's `parent_structure` recipe."""
    from pymatgen.core import Lattice, Structure

    cp = np.asarray(d.y["cell_params"], dtype=float).ravel()          # a,b,c,al,be,ga
    z = np.asarray(d.y["unit_cell_x"])[:, 0].astype(int)             # col 0 = atomic number
    frac = np.asarray(d.y["unit_cell_pos_frac"], dtype=float)        # (n_uc, 3)
    if not ((z >= 1).all() and (z <= 118).all()):
        raise ValueError(f"unit_cell_x[:,0] not valid atomic numbers: {np.unique(z)}")
    if frac.shape != (len(z), 3):
        raise ValueError(f"pos_frac {frac.shape} inconsistent with {len(z)} species")
    return Structure(Lattice.from_parameters(*cp), [int(v) for v in z], frac)


def _mo_bond(struc) -> float:
    """Minimum metal-oxygen distance in the parent cell, in ANGSTROM.

    This is exactly the reference's ``nn_MO_bond_A`` — not a reinterpretation of
    it. Verified session 18 against the old repo's `augmented_metadata.csv` on
    chili-0/5/10/100/1000: agreement to ~1e-5 A, so the old hand-made CSV did not
    need porting. Every CHILI-3K composition contains oxygen, so "not O" == metal.
    `distance_matrix` already applies the periodic minimum image.

    Constant per PARENT STRUCTURE, not per composition (measured s18: a formula
    with 5 polymorphs has 5 distinct values). The old repo's note that it is
    ~constant per composition does not hold. Composition-level `group_key` is a
    strict superset of structure-level, so the split stays leak-safe anyway — see
    the module docstring.
    """
    is_o = np.array([sp.symbol == "O" for sp in struc.species])
    if is_o.all() or not is_o.any():
        raise ValueError(
            f"parent cell is not a metal oxide ({struc.composition.reduced_formula}); "
            f"target_mo_bond is undefined"
        )
    return float(struc.distance_matrix[np.ix_(~is_o, is_o)].min())


def _oxidation(struc) -> float:
    """Formal metal charge from stoichiometry with O = -2:  M_xO_y -> 2y/x.

    Stands in for the paper's `BVAnalyzer`, which assigns oxidation state by bond
    valence. That is the right tool on the real MP structures the paper used; it is
    the wrong one here, because CHILI's cells are enumerated rather than observed and
    a bond-valence sum on Al2O or AuO3 is either a failure or a fiction. Formal charge
    is exact for a binary oxide, never fails, and drops no rows.

    Over CHILI-3K this takes five values — 1, 2, 8/3, 4, 6 — one per stoichiometry.
    The 8/3 class (M3O4) is a stoichiometric AVERAGE, not a per-site oxidation state:
    such a structure is physically one M(2+) and two M(3+). It is kept as its own
    class deliberately; read it as a stoichiometry label, not a valence.

    Callers reach this only after `_mo_bond`, which already raised if the cell is not
    a metal oxide, so both amounts below are guaranteed non-zero.
    """
    amt = struc.composition.get_el_amt_dict()
    return 2.0 * amt["O"] / sum(v for el, v in amt.items() if el != "O")


def _metal(struc) -> float:
    """Atomic number of the metal element — a classification target orthogonal to
    `crystal_type`, unlike `_oxidation`/`_cn_and_bond` (see module docstring).

    Encoded as the atomic number (13.0 for Al, 50.0 for Sn, ...) rather than an
    arbitrary class index, so it is a real physical quantity like every other
    target — and so it slots into the existing float-column / `CLASSIFICATION_TARGETS`
    machinery in `analysis/downstream_eval.py` with no new code path.

    Callers reach this only after `_mo_bond`, which already confirmed the cell has
    exactly one non-oxygen element.
    """
    metals = [el for el in struc.composition.elements if el.symbol != "O"]
    if len(metals) != 1:
        raise ValueError(f"expected exactly one non-O element, got {metals}")
    return float(metals[0].Z)


def _cn_and_bond(cnn, struc) -> tuple[float, float]:
    """One CrystalNN pass -> (uniform metal-site CN or NaN, mean-of-means M-O bond).

    Both of the paper's CrystalNN targets come from a single pass because CrystalNN
    is the whole cost of this builder (~0.12 s per structure); computing them apart
    would double it for nothing.

    CN is the coordination number shared by every metal site, or NaN where the sites
    disagree. The paper DROPS disagreeing structures; here the row survives and is
    merely unlabelled for this one target, so the other three targets keep it — the
    probe masks per target. Across CHILI-3K only Spinel disagrees ({4, 6}, 265 rows);
    the values that occur are 4, 6 and 8, so CN = 5 never appears and this is a
    three-class task, not the paper's {4, 5, 6}.

    The bond length is the paper's "mean of the means": the mean neighbour distance of
    each metal site, then averaged over sites with EQUAL WEIGHT. Weighting per site
    rather than per bond is the point — a plain mean over all bonds would let
    6-coordinate sites outweigh 4-coordinate ones. Distances are CrystalNN's own
    `nn_distance`, not recomputed from the sites, so the neighbour image CrystalNN
    chose is the one measured.

    Every CrystalNN neighbour of a metal site is oxygen (measured: 560 sites across
    120 structures, no exceptions), so these are M-O quantities despite CrystalNN
    itself being species-blind.

    A metal site with NO neighbour at all yields (NaN, NaN) rather than raising. Its
    solid-angle criterion can find nothing in a sparse or degenerate cell, and that is
    a failure of THESE TWO targets only — raising would drop the row and take
    `np_size`, `mo_bond` and `oxidation` down with it. `_print_balance` reports how
    many rows end up unlabelled, so a silent epidemic is visible at build time.
    """
    cns, site_means = [], []
    for i, sp in enumerate(struc.species):
        if sp.symbol == "O":
            continue
        nn = cnn.get_nn_info(struc, i)
        if not nn:
            return float("nan"), float("nan")
        cns.append(len(nn))
        site_means.append(float(np.mean([x["site"].nn_distance for x in nn])))
    cn = float(cns[0]) if len(set(cns)) == 1 else float("nan")
    return cn, float(np.mean(site_means))


def _group_key(struc) -> str:
    """Composition (reduced formula): the split-grouping unit.

    Coarser than one parent structure — every polymorph of a formula, and so
    every nanoparticle size of every one of those structures, shares a key.
    Chosen over a per-structure fingerprint because it stays leak-safe for a
    composition-constant target (e.g. bond length) as well as a
    structure-level one (np_size); see module docstring.
    """
    return struc.composition.reduced_formula


def _channel(d, key: str) -> tuple[list[float], list[float]]:
    """A native signal stored as [grid, values] in d.y[key] -> (x, y) lists."""
    arr = np.asarray(d.y[key], dtype=float)  # (2, N): row 0 = grid, row 1 = intensity
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"{key}: expected (2, N), got {arr.shape}")
    return arr[0].tolist(), arr[1].tolist()


def _sample_indices(processed: Path) -> list[int]:
    """All sample_idx present, from the data_<idx>.pt filenames, sorted."""
    idxs = [int(p.stem.split("_")[1]) for p in processed.glob("data_*.pt")]
    if not idxs:
        raise RuntimeError(f"no data_*.pt files in {processed}")
    return sorted(idxs)


def build(raw_dir: Path, *, limit: int | None = None) -> pd.DataFrame:
    """Read the staged CHILI-3K .pt files (only) and normalize to registry columns."""
    import torch
    from pymatgen.analysis.local_env import CrystalNN

    processed = raw_dir / "processed"
    indices = _sample_indices(processed)
    if limit is not None:
        indices = indices[:limit]

    cnn = CrystalNN()  # built once and reused: it is the builder's whole runtime cost
    rows = []
    failures = []
    for s_idx in indices:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = torch.load(processed / f"data_{s_idx}.pt", weights_only=False)
                struc = _parent_structure(d)
                cif = struc.to(fmt="cif")
                xrd_x, xrd_y = _channel(d, "xrd")
                xpdf_x, xpdf_y = _channel(d, "xPDF")
                sg = int(d.y["space_group_number"])
                # inside the try on purpose: a non-oxide has no M-O bond, and one
                # such sample must drop like any other bad row, not abort the build.
                # _mo_bond runs first — it is the metal-oxide guard the other two rely on.
                mo_bond = _mo_bond(struc)
                oxidation = _oxidation(struc)
                metal = _metal(struc)
                cn, mean_bond = _cn_and_bond(cnn, struc)
        except Exception as e:  # noqa: BLE001 — skip and keep going, like mp20
            failures.append((s_idx, f"{type(e).__name__}: {e}"))
            continue
        rows.append({
            "material_id": f"chili-{s_idx}",
            "structure": cif,
            "signal_xrd_x": xrd_x,
            "signal_xrd_y": xrd_y,
            "signal_xpdf_x": xpdf_x,
            "signal_xpdf_y": xpdf_y,
            "target_np_size": float(d.y["np_size"]),
            "target_mo_bond": mo_bond,
            "target_oxidation": oxidation,
            "target_cn": cn,
            "target_mean_bond": mean_bond,
            "target_metal": metal,
            "has_xrd": True,
            "has_pdf": True,
            "group_key": _group_key(struc),
            "crystal_type": str(d.y["crystal_type"]),
            "crystal_system": str(d.y["crystal_system"]),
            "space_group_symbol": str(d.y["space_group_symbol"]),
            "space_group_number": sg,
            "n_atoms": int(d.y["n_atoms"]),
            "cell_a": float(struc.lattice.a), "cell_b": float(struc.lattice.b),
            "cell_c": float(struc.lattice.c), "cell_alpha": float(struc.lattice.alpha),
            "cell_beta": float(struc.lattice.beta), "cell_gamma": float(struc.lattice.gamma),
        })

    if failures:
        print(f"dropping {len(failures)} sample(s) that failed to build: {failures[:5]}")
    return pd.DataFrame(rows)


def _print_balance(reg: MaterialRegistry) -> None:
    """Show per-split class balance — the cost of not stratifying.

    `crystal_system` is the original check. The two classification targets are shown
    for the same reason and one more: an F1 score is unreadable without knowing what
    the modal class was, and the probe's baseline is only trustworthy if the classes
    are distributed similarly in train and test.
    """
    df = reg.frame
    order = [s for s in ("train", "val", "test") if s in df["split"].unique()]
    # target_metal is excluded from the crosstab loop: 53 classes would print a
    # 53-row table for a target whose whole point is even coverage, not the coarse
    # imbalance crystal_system/oxidation/cn can have. Its balance is summarized
    # instead, right after the loop.
    for col in ("crystal_system", "target_oxidation", "target_cn"):
        ct = pd.crosstab(df[col], df["split"], normalize="columns") * 100
        print(f"\n{col} balance (% of each split's rows):")
        print(ct[order].round(1).to_string())
        if df[col].isna().any():
            print(f"  ({int(df[col].isna().sum())} rows unlabeled for {col}, excluded above)")
    all_metals = df["target_metal"].unique()
    per_metal_test = df.loc[df["split"] == "test", "target_metal"].value_counts().reindex(
        all_metals, fill_value=0
    )
    print(f"\ntarget_metal balance: {len(all_metals)} metals, "
          f"test support {int(per_metal_test.min())}-{int(per_metal_test.max())} rows/metal "
          f"({int((per_metal_test == 0).sum())} metal(s) absent from test)")
    unlabelled = {c: int(df[c].isna().sum()) for c in df.columns if c.startswith("target_")}
    print("\nunlabelled rows per target:", unlabelled)
    print("split sizes:", df["split"].value_counts().to_dict())
    # group-disjointness assertion (the whole point)
    per_group = df.groupby(GROUP_COL)["split"].nunique()
    assert (per_group == 1).all(), "a composition leaked across splits!"
    print(f"group-disjoint on {GROUP_COL!r}: OK ({per_group.size} compositions, none split)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None, help="only build the first N samples")
    ap.add_argument("--seed", type=int, default=0, help="split seed")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        sys.exit(f"REFUSING: {out_path} already exists (use --overwrite).")

    df = build(Path(args.raw_dir), limit=args.limit)
    reg = MaterialRegistry(df)
    reg = assign(reg, group_by=GROUP_COL, seed=args.seed)

    report = reg.validate()
    print(report)
    report.raise_for_errors(context=str(out_path))
    _print_balance(reg)

    saved = reg.save(out_path)
    print(f"\nsaved {saved}  ({len(reg)} rows, {saved.stat().st_size / 1e6:.1f} MB, "
          f"signals={reg.signal_names})")


if __name__ == "__main__":
    main()
