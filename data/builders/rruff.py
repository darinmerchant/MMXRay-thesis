"""data/builders/rruff.py — build the RRUFF downstream MaterialRegistry.

148 **experimentally measured** mineral XRD patterns, staged under
`data/raw/rruff/` from the reference repo's `XRD/Data/Downstream/RRUFF/`. This is
the only real-measurement registry in the repo: everything else (MP, CHILI) is
simulated, so this is where a sim-pretrained encoder finally meets data it cannot
have seen the generating process for.

Per mineral -> one registry row (base_mode = ``signal``, no structure):

    material_id        the RRUFF mineral name ("Quartz", "Calcite", ...); already
                       unique across the 148 rows, one sample each.
    signal_xrd_{x,y}   the measured pattern, resampled from RRUFF's native
                       linspace(5, 90, 8500) onto this repo's `TT_GRID`
                       ([10, 80] deg, 4999 pts). Stored AS DELIVERED, not rescaled
                       here — but note the source `.pt` is ALREADY max-normed, so
                       unlike every other registry this channel is not raw physical
                       intensity and never can be; RRUFF ships no absolute scale.
                       Harmless downstream (`max_normalize` is idempotent), but do
                       not read it as counts. Its per-row max averages **0.962**,
                       not 1.0, because cropping to [10, 80] deg drops the global
                       maximum for minerals whose strongest peak sits in [5, 10] or
                       [80, 90]; the read path renormalizes.
    signal_xpdf_{x,y}  the *virtual* PDF: a sine Fourier transform of the SAME
                       resampled pattern, on the diffpy pretraining r-grid
                       (0-50 A, 5000 pts). See `modalities/pdf/from_xrd.py`.
    target_cell_{a,b,c,alpha,beta,gamma}
                       the 6 unit-cell parameters, parsed from each row's `cif`
                       in `test.csv` at BUILD time.
    target_cation_cn   site-averaged CrystalNN coordination number over the
                       CATION sites (positive oxidation state, assigned by bond
                       valence). NaN for the 21 minerals where valences cannot
                       be assigned — see `derive_structure_targets`.
    metal_elements,    which constituents pymatgen calls metals, and how many —
    n_metals           METADATA, not targets (measured s35: 15 minerals have no
                       metal, 92 have one, 41 have two; metalloids Si/As/Sb/Te/Se
                       do not count, so quartz has a cation but no metal).
    split              train/test, 80/20, seeded. No val — see Splits below.

**No `structure` column, on purpose.** `test.csv` does carry a CIF per mineral,
and an earlier plan for this registry stored it so a diffpy G(r) could be
simulated alongside the virtual one as an in-distribution control. That was cut:
a diffpy channel would be *simulated* data, and the entire reason to run RRUFF is
to test on measured data. Storing the CIF would also leave a live simulation path
on a registry that must never be simulated from (DESIGN.md -> base modes: real
downstream data is ``signal`` mode, "no simulation, no augmentation"). The CIFs
are read once, for the cell parameters, and not carried.

Consequence to state plainly: because there is no in-distribution PDF control,
a weak PDF-encoder score on this registry is NOT attributable — it mixes the
sim->real gap with the Qmax truncation gap of the virtual PDF, and this registry
alone cannot separate them.

**Targets are the P1-reduced cell.** The CIFs in `test.csv` are pymatgen-generated
and P1-reduced, so the parsed cell is the reduced setting, which for several
minerals is NOT the conventional one (the reference repo's `build_metadata.py`
checked this against literature values for ~7 minerals and kept the reduced parse
anyway). `--check-cells` reprints that comparison. `spacegroup.number` in
`test.csv` is all 1 for the same reason and is unusable as a target, so it is
dropped rather than carried.

**Splits: 80/20 train/test, no val.** Every material_id is unique (one sample per
mineral), so group-disjoint, stratified and plain random splitting all collapse to
the same thing here — `core/splits.assign` is called with its default
`group_by=material_id` and the degeneracy is not worth a special case. The val
fraction is 0.0 because the probe this feeds selects ridge lambda by GCV on the
training set and never reads a held-out val split; at n=148 spending 20-30 rows on
a split nothing consumes would be a straight loss of training data.

Run: python data/builders/rruff.py [--overwrite]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.registry import MATERIAL_ID, SPLIT, MaterialRegistry
from core.splits import assign
from modalities.pdf.from_xrd import two_theta_to_q, virtual_pdf
from modalities.pdf.simulate import RMIN, RSTEP
from modalities.xrd.grid import TT_GRID

DEFAULT_RAW_DIR = Path("data/raw/rruff")
DEFAULT_OUT = Path("data/downstream/rruff/rruff_registry.parquet")

#: RRUFF's native 2theta grid. The reference repo's `PV_GRID`; the stored signals
#: are (1, 8500) with no x-axis of their own, so the grid is a constant here.
#: It COVERS [10, 80], so resampling onto TT_GRID is pure interpolation — there is
#: no zero-fill, unlike the CHILI XRD path (`grid.resample_chili_xrd`).
PV_LO, PV_HI, PV_N = 5.0, 90.0, 8500
PV_GRID = np.linspace(PV_LO, PV_HI, PV_N)

#: The diffpy pretraining r-grid, rebuilt from `simulate.py`'s own constants so a
#: change there cannot silently desync this registry. `downstream_eval.py` checks
#: the stored grid against `RMIN + arange(n) * RSTEP` and refuses a mismatch.
N_R = 5000
R_GRID = RMIN + np.arange(N_R) * RSTEP

CELL_KEYS = ("a", "b", "c")
ANGLE_KEYS = ("alpha", "beta", "gamma")
CELL_TARGET_COLS = tuple(f"target_cell_{k}" for k in CELL_KEYS + ANGLE_KEYS)
#: Session-32 additions. The cell parameters turned out to be a near-unlearnable
#: task here — a ridge fit DIRECTLY on the raw 5000-point virtual PDF, with no
#: encoder in the path at all, scores R2 = 0.16 on `cell_a`, <= 0.05 on `cell_b/c`
#: and ~0.00 on all three angles. That is a property of the DATA (n=118, Qmax
#: 5.24 A^-1), not of any model, so no probe or finetune fix reaches it.
#: The same ridge/logreg sweep found these three carry signal:
#:   crystal_system  weighted F1 0.354 vs a 0.031 modal-class baseline — 11x, and
#:                   the physically right ask, since a diffraction pattern encodes
#:                   symmetry far more directly than one specific edge length.
#:   n_atoms         R2 0.175   n_elem  R2 0.114   — weak, but both above every
#:                   cell parameter, and free on the CIF pass already being done.
#: Session-35 addition. `target_cation_cn` is the MEAN CrystalNN coordination
#: number over all cation sites, not "the CN of the cation": 72 of the 127
#: labelable minerals have more than one cation ELEMENT (aegirine holds Na at
#: CN 8, Fe at 6, Si at 4), so a single-cation label would need a selection
#: convention, and the CHILI-style "one cation, all sites agree" rule keeps only
#: 49 minerals — 31 of them CN 6, a 63% modal baseline. The site-weighted mean is
#: defined whenever bond valence resolves (127 of 148) and becomes a regression
#: target like `target_n_atoms`. Unlike CHILI's `target_cn` (4.0/6.0/8.0 classes)
#: this is a genuine float — do not feed it to a classification head.
EXTRA_TARGET_COLS = ("target_crystal_system", "target_n_atoms", "target_n_elem",
                     "target_cation_cn")
TARGET_COLS = CELL_TARGET_COLS + EXTRA_TARGET_COLS

#: Loose symprec on purpose. `test.csv`'s CIFs are P1-reduced, so the stored
#: `spacegroup.number` is 1 for every row and is unusable — but the SYMMETRY IS
#: STILL RECOVERABLE from the coordinates, which is the whole point of this target.
#:
#: **What this buys is one row, not the target** *(measured s33, all 148 CIFs)*. The
#: original note here claimed a tight default "would hand most of them back as
#: triclinic". That is FALSE: at symprec=0.01 only 10 of 148 are triclinic and 147/148
#: already resolve past P1. Sweeping 0.01 -> 0.2 moves exactly ONE label — Nickeline,
#: orthorhombic at 0.01 and hexagonal at 0.1, where hexagonal (NiAs, P6_3/mmc) is the
#: published answer. So 0.1 is still the right choice; the reason is just far narrower
#: than it was written to be, and the label is robust rather than tolerance-tuned.
#:
#: The lever that actually costs a label is pymatgen's OTHER tolerance,
#: `angle_tolerance`, left at its default 5 deg: Tugarinovite's cell has beta = 91.13
#: deg, inside that window, so it reads orthorhombic `Pnnm` where the published MoO2 is
#: monoclinic `P2_1/c`. Tightening to <= 2 deg fixes it and moves nothing else. Left at
#: the default so this builder makes no undocumented departure from pymatgen — see
#: `docs/TRAPS.md` for the full audit and the other two mislabelled minerals.
SYMPREC = 0.1

#: `target_*` columns are read as float32 by `analysis/downstream_eval.load_probe_data`,
#: so a classification target has to arrive NUMERIC — CHILI does the same thing
#: (`target_metal` is an atomic number, `target_cn` a float holding 4/6/8). The code
#: is the IUCr ordering, identical to the range table in `data/builders/mp20.py`, so
#: the two registries agree on what "3" means. The readable string is kept alongside
#: as the metadata column `crystal_system` (as CHILI does) — nothing has to decode
#: an integer by hand to see the class balance.
CRYSTAL_SYSTEMS = ("triclinic", "monoclinic", "orthorhombic", "tetragonal",
                   "trigonal", "hexagonal", "cubic")
CRYSTAL_SYSTEM_CODE = {name: i + 1 for i, name in enumerate(CRYSTAL_SYSTEMS)}

#: Literature *conventional* cells (A, deg) for the `--check-cells` printout only.
#: Never labels — they exist to show how the P1-reduced parse relates to the
#: setting a crystallographer would quote. Ported from the reference builder.
REFERENCE_CELLS = {
    "Quartz": (4.913, 4.913, 5.405, 90, 90, 120),
    "Calcite": (4.989, 4.989, 17.062, 90, 90, 120),
    "Rutile": (4.593, 4.593, 2.959, 90, 90, 90),
    "Fluorite": (5.463, 5.463, 5.463, 90, 90, 90),
    "Corundum": (4.759, 4.759, 12.991, 90, 90, 120),
    "Galena": (5.936, 5.936, 5.936, 90, 90, 90),
    "Gold": (4.078, 4.078, 4.078, 90, 90, 90),
}


def parse_cell(cif: str) -> list[float]:
    """(a, b, c, alpha, beta, gamma) from CIF text; raises if any is missing.

    Raises rather than returning NaN: a cell parameter that fails to parse means
    the CIF is not the shape this builder assumes, and silently emitting a null
    target would hide that behind a row that still looks fine.
    """

    def grab(pattern: str) -> float:
        m = re.search(pattern, cif)
        if m is None:
            raise ValueError(f"pattern {pattern!r} not found in CIF")
        return float(m.group(1))

    lengths = [grab(rf"_cell_length_{k}\s+([\d.]+)") for k in CELL_KEYS]
    angles = [grab(rf"_cell_angle_{k}\s+([\d.]+)") for k in ANGLE_KEYS]
    return lengths + angles


def derive_structure_targets(cif: str) -> tuple[str, int, int, float, str, int]:
    """(crystal_system, n_atoms, n_elements, cation_cn, metal_elements, n_metals).

    Raises on an unparseable structure rather than emitting nulls — same reasoning
    as `parse_cell`: a row that silently loses its label still looks fine downstream.

    `cation_cn` is the one deliberate exception: bond valence CANNOT assign
    oxidation states to 21 of the 148 minerals (every native element — Au, Ag,
    diamond, Si... — has no cation at all, plus pyrite-type persulfides and a few
    hydrates), so for exactly the `ValueError("Valences cannot be assigned!")`
    that BVAnalyzer raises there, the label is NaN by construction, not by
    accident. Any other exception still propagates.
    """
    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    struct = Structure.from_str(cif, fmt="cif")
    system = SpacegroupAnalyzer(struct, symprec=SYMPREC).get_crystal_system()
    metals = sorted(el.symbol for el in struct.composition.elements if el.is_metal)
    try:
        decorated = BVAnalyzer().get_oxi_state_decorated_structure(struct)
    except ValueError:
        cation_cn = float("nan")
    else:
        cnn = CrystalNN()
        cns = [cnn.get_cn(decorated, i) for i, site in enumerate(decorated)
               if site.specie.oxi_state > 0]
        cation_cn = float(np.mean(cns))
    return (system, len(struct), len(struct.composition.elements),
            cation_cn, ",".join(metals), len(metals))


def simulate_diffpy_pdfs(cifs) -> np.ndarray:
    """`(N, 5000)` diffpy G(r) from the CIFs — the IN-DISTRIBUTION control channel.

    Everything else in this registry is measured. This one is NOT, deliberately: it is
    the reference the virtual PDF is missing. `signal_xpdf` is a Fourier transform of a
    real pattern and is capped by Cu K-alpha at Qmax = 5.24 A^-1, roughly 3-6x below the
    15-30 the encoders were pretrained on. That leaves any weak score on this registry
    unattributable — sim->real gap and Qmax truncation gap are entangled and cannot be
    separated from `signal_xpdf` alone. This channel breaks the tie: same minerals, same
    targets, same splits, but PDFs the encoder IS in distribution for. A score that
    stays low HERE is the encoder; a score that recovers here was the truncation.

    Params are `simulate_pdf`'s own defaults (uiso 0.005, qmax 22.5, qbroad 0.03,
    qmin 1.0) — the mid-range of the pretraining augmentation ranges. FIXED, not drawn:
    a control wants one reference point, not a sample.

    **Read this before quoting anything off this channel.** It is simulated from a
    P1-reduced CIF, so it is not "what RRUFF would look like with a better source" — it
    is what the *database structure* predicts, free of every experimental effect
    (texture, strain, real instrument broadening, background) that makes the measured
    arm hard. It is an upper reference, not a fair comparison.
    """
    from modalities.pdf.simulate import simulate_pdf

    out, failed = [], []
    for mid, cif in cifs.items():
        try:
            r, g = simulate_pdf(cif)
        except Exception as exc:                      # noqa: BLE001 - report, don't mask
            failed.append(f"{mid}: {type(exc).__name__}")
            out.append(np.full(N_R, np.nan))
            continue
        if len(r) < N_R or not np.allclose(r[:N_R], R_GRID, atol=RSTEP / 100):
            raise ValueError(f"{mid}: diffpy returned an unexpected r grid ({len(r)} pts)")
        out.append(np.asarray(g[:N_R], dtype=np.float64))
    if failed:
        raise ValueError(f"{len(failed)} CIF(s) failed to simulate, e.g. {failed[:5]}")
    return np.stack(out)


def _load_signals(raw_dir: Path) -> dict[str, np.ndarray]:
    """material_id -> (8500,) measured intensity, from `test_pv_xrd.pt`."""
    import torch

    raw = torch.load(raw_dir / "test_pv_xrd.pt", weights_only=False)
    out = {}
    for mid, tensor in raw.items():
        y = np.asarray(tensor, dtype=np.float64).ravel()
        if y.size != PV_N:
            raise ValueError(f"{mid}: signal has {y.size} points, expected {PV_N}")
        out[str(mid)] = y
    return out


def build(raw_dir: Path) -> pd.DataFrame:
    """Assemble the registry frame. No split column yet — `assign` adds it."""
    meta = pd.read_csv(raw_dir / "test.csv")
    signals = _load_signals(raw_dir)

    if meta["material_id"].duplicated().any():
        dupes = meta.loc[meta["material_id"].duplicated(), "material_id"].tolist()
        raise ValueError(f"duplicate material_id(s) in test.csv: {dupes[:5]}")
    missing = set(meta["material_id"]) - set(signals)
    if missing:
        raise ValueError(f"{len(missing)} id(s) in test.csv have no signal, e.g. {sorted(missing)[:5]}")

    # --- XRD: RRUFF's native grid -> TT_GRID. Pure interpolation; [5, 90] covers
    # [10, 80], so `np.interp`'s left/right clamps are never reached.
    xrd = np.stack([
        np.interp(TT_GRID, PV_GRID, signals[mid]) for mid in meta["material_id"]
    ])

    # --- PDF: sine FT of that SAME resampled pattern, not of the native one, so
    # the two channels are two views of one input and the comparison between the
    # XRD and PDF encoders is not confounded by different Q coverage.
    xpdf = virtual_pdf(TT_GRID, xrd, R_GRID)

    print("simulating the diffpy control channel (148 CIFs)...", flush=True)
    xpdf_diffpy = simulate_diffpy_pdfs(dict(zip(meta["material_id"].astype(str), meta["cif"])))

    cells = np.array([parse_cell(c) for c in meta["cif"]], dtype=np.float64)
    derived = [derive_structure_targets(c) for c in meta["cif"]]

    frame = pd.DataFrame({
        "material_id": meta["material_id"].astype(str),
        "signal_xrd_x": [TT_GRID.astype(np.float32)] * len(meta),
        "signal_xrd_y": list(xrd.astype(np.float32)),
        "signal_xpdf_x": [R_GRID.astype(np.float32)] * len(meta),
        "signal_xpdf_y": list(xpdf.astype(np.float32)),
        "signal_xpdf_diffpy_x": [R_GRID.astype(np.float32)] * len(meta),
        "signal_xpdf_diffpy_y": list(xpdf_diffpy.astype(np.float32)),
    })
    for i, col in enumerate(CELL_TARGET_COLS):
        frame[col] = cells[:, i]
    systems = [d[0] for d in derived]
    unknown = sorted(set(systems) - set(CRYSTAL_SYSTEM_CODE))
    if unknown:
        raise ValueError(f"crystal system(s) not in the IUCr table: {unknown}")
    frame["target_crystal_system"] = np.array(
        [CRYSTAL_SYSTEM_CODE[s] for s in systems], dtype=np.float64)
    frame["crystal_system"] = systems   # readable twin, metadata not target
    frame["target_n_atoms"] = np.array([d[1] for d in derived], dtype=np.float64)
    frame["target_n_elem"] = np.array([d[2] for d in derived], dtype=np.float64)
    frame["target_cation_cn"] = np.array([d[3] for d in derived], dtype=np.float64)
    frame["metal_elements"] = [d[4] for d in derived]
    frame["n_metals"] = np.array([d[5] for d in derived], dtype=np.int64)
    frame["has_xrd"] = True
    frame["has_pdf"] = True
    return frame


def carve_val(reg: MaterialRegistry, val_frac: float, seed: int) -> MaterialRegistry:
    """Move `val_frac` of the WHOLE set from train into val, leaving test untouched.

    The probe needs no val split (it picks ridge lambda by GCV), but
    `analysis/finetune.py` early-stops on one and refuses to run without it. Rather
    than re-splitting three ways — which would move the test rows and make the
    finetune numbers incomparable to the probe's — this carves val out of train
    only. **Test is untouched by construction**, not by arithmetic that happens to
    round the same way, so the two runs are scored on the identical 30 minerals.
    """
    df = reg.frame
    train_ids = df.loc[df[SPLIT] == "train", MATERIAL_ID].to_numpy()
    n_val = int(round(val_frac * len(df)))
    if not 0 < n_val < len(train_ids):
        raise ValueError(f"val_frac {val_frac} gives {n_val} of {len(train_ids)} train rows")
    picked = np.random.default_rng(seed + 1).permutation(train_ids)[:n_val]
    mapping = dict(zip(df[MATERIAL_ID], df[SPLIT]))
    mapping.update({mid: "val" for mid in picked})
    return reg.with_split(mapping)


def _print_checks(reg: MaterialRegistry, *, check_cells: bool) -> None:
    """What the build measured — printed so the caveats are seen, not just filed."""
    df = reg.frame
    q = two_theta_to_q(TT_GRID)
    print(f"\nQ coverage from 2theta [{TT_GRID[0]:.0f}, {TT_GRID[-1]:.0f}] deg: "
          f"[{q[0]:.3f}, {q[-1]:.3f}] A^-1")
    print(f"  -> real-space resolution ~pi/Qmax = {np.pi / q[-1]:.2f} A; "
          f"termination ripple period ~2pi/Qmax = {2 * np.pi / q[-1]:.2f} A")
    print(f"  -> pretraining qmax range is 15-30 A^-1, so this is "
          f"{15.0 / q[-1]:.1f}-{30.0 / q[-1]:.1f}x BELOW it (out of distribution)")

    xrd = np.stack(df["signal_xrd_y"].to_numpy())
    xpdf = np.stack(df["signal_xpdf_y"].to_numpy())
    print(f"\nsignal_xrd  {xrd.shape}  range [{xrd.min():.4f}, {xrd.max():.4f}]  "
          f"baseline p5 = {np.percentile(xrd, 5):.5f} (of max 1.0 — why no background step)")
    print(f"signal_xpdf {xpdf.shape}  range [{xpdf.min():.3f}, {xpdf.max():.3f}]")
    assert np.isfinite(xrd).all() and np.isfinite(xpdf).all(), "non-finite signal"

    # G(0) = 0 identically: every term of the sine transform carries sin(Q*0).
    assert np.allclose(xpdf[:, 0], 0.0, atol=1e-9), "G(0) should be exactly 0"
    print(f"  G(0) = 0 for all rows (sine transform identity): OK")

    # DISCRIMINABILITY GUARD — the check that caught the DC bug (see
    # `modalities/pdf/from_xrd.py`). If [S(Q)-1] ever regains a DC term, every
    # mineral transforms to nearly the same curve and this jumps to ~0.999 while
    # every other check here still passes. A near-constant channel would train and
    # score without complaint, so it has to be measured, not assumed.
    cc = np.corrcoef(xpdf)
    iu = np.triu_indices(len(xpdf), k=1)
    mean_cc = float(cc[iu].mean())
    print(f"  mean pairwise corr between the {len(xpdf)} virtual PDFs = {mean_cc:.4f} "
          f"(~0.999 would mean one shared curve, i.e. no material information)")
    assert mean_cc < 0.5, (
        f"virtual PDFs are {mean_cc:.4f} correlated with each other — the channel "
        f"carries almost no per-material signal; check the S(Q) normalization"
    )

    print("\ntarget ranges:")
    for col in TARGET_COLS:
        if col == "target_crystal_system":
            vc = df["crystal_system"].value_counts().to_dict()
            print(f"  {col:22s} {len(vc)} classes, nulls={int(df[col].isna().sum())}  {vc}")
        else:
            print(f"  {col:22s} [{df[col].min():8.3f}, {df[col].max():8.3f}]  "
                  f"nulls={int(df[col].isna().sum())}")
    print("\nmetal constituents (metadata): n_metals",
          df["n_metals"].value_counts().sort_index().to_dict(),
          f"— {int((df['n_metals'] == 0).sum())} minerals have no metal at all")
    print("split sizes:", df["split"].value_counts().to_dict())

    if check_cells:
        print("\nP1-reduced parse vs literature CONVENTIONAL cell (labels are the "
              "reduced parse; this is orientation, not validation):")
        lut = df.set_index("material_id")
        for name, ref in REFERENCE_CELLS.items():
            if name not in lut.index:
                print(f"  {name:10s} (not in dataset)")
                continue
            got = tuple(lut.loc[name, list(CELL_TARGET_COLS)])
            match = "==" if np.allclose(got, ref, atol=0.02) else "!="
            print(f"  {name:10s} parsed=({', '.join(f'{v:7.3f}' for v in got)})")
            print(f"  {'':10s} lit   =({', '.join(f'{v:7.3f}' for v in ref)})  {match}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=0, help="split seed")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="carve this fraction of the WHOLE set out of train into val, "
                         "leaving test untouched; needed by analysis/finetune.py")
    ap.add_argument("--check-cells", action="store_true",
                    help="print the P1-reduced vs conventional cell comparison")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        sys.exit(f"REFUSING: {out_path} already exists (use --overwrite).")

    reg = MaterialRegistry(build(Path(args.raw_dir)))
    # (train, val, test) — val is 0.0 deliberately; see the module docstring.
    reg = assign(reg, fractions=(0.8, 0.0, 0.2), seed=args.seed)
    if args.val_frac:
        reg = carve_val(reg, args.val_frac, args.seed)

    report = reg.validate()
    print(report)
    report.raise_for_errors(context=str(out_path))
    _print_checks(reg, check_cells=args.check_cells)

    saved = reg.save(out_path)
    print(f"\nsaved {saved}  ({len(reg)} rows, {saved.stat().st_size / 1e6:.1f} MB, "
          f"signals={reg.signal_names}, modes={reg.base_modes.value_counts().to_dict()})")


if __name__ == "__main__":
    main()
