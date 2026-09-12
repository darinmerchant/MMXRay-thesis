"""tools/augment_chili.py — one augmented xPDF view per CHILI-3K sample.

Adds a SECOND signal channel to the CHILI downstream registry: the same
nanoparticles, re-simulated once each at randomly drawn instrument params. The
clean channel (`signal_xpdf_*`, CHILI's own native xPDF) stays untouched, so a
probe can be trained on one and evaluated on the other — the sim->real domain
shift proxy from the reference repo.

    signal_xpdf_aug_x/_y   augmented G(r), 6000 pts over r = 0..59.99 A
    aug_uiso/_qmax/_qbroad/_qmin   the drawn params, one row each

`--oob` produces a THIRD channel, `signal_xpdf_oob_*` (+ `oob_*` param columns),
identical in every way except that the params are drawn from `PDF_OOB_RANGES` —
the one-sided out-of-band segments the invariance ladder measured as outside the
trained range. The in-band channel shifts the eval domain by exactly the
distribution the SSL augmentations practiced; this one shifts it by instrument
settings they never saw. Same seed, so draws are quantile-paired with the
in-band channel (same material, same relative severity).

THE SAME TOOL SERVES THE CHILI-100K BINARY-OXIDE FIT REGISTRY (2026-08-19) —
point `--registry` at `data/downstream/chili100k/chili100k_registry.parquet` and
`--raw-dir` at `data/raw/chili100k/cod_output_080124`. There the augmented
channel is a FIT set, not an eval set: fit on aug-100K train, evaluate on
CHILI-3K (`analysis/downstream_eval.py --fit-registry ... --fit-signal
xpdf_aug`), so the training data is drawn from the pretraining augmentation
distribution while the eval domain stays clean. Rows are
`chili100k-<COD>-<size_idx>`, five particle sizes per COD `.h5`; `_row_source`
maps either id format to (raw file, particle, seed offset). Cost ~1.9x the
CHILI-3K job (measured sum n^2 6.17e10 vs 3.23e10 — `data/builders/chili100k.py`),
and the `.h5` live only on the cluster: run `scripts/augment_chili100k.slurm`
there, rsync the stage back, merge on the Mac.

WHY A SEPARATE TOOL, not part of `data/builders/chili.py`: the builder is cheap
(minutes, diffpy-free) and gets re-run whenever a target changes. This step is a
~3 h Debye sum over 3180 particles and needs `debyecalculator`. Coupling them
would make every target tweak cost an afternoon.

Physics: `modalities/pdf/debye.py` — the Debye sum over the particle's ACTUAL
atoms, not diffpy on the parent cell, because only that keeps the particle-size
envelope that `target_np_size` is read from (see that module's docstring).
Atom positions come from the raw `.pt` (`pos_abs` + `x[:,0]` = Z), which the
registry does not carry — it stores only the parent CIF.

Reproducible: params are drawn from `seed + sample_idx` via
`core.transforms.draw_pdf_params`, so they are RECOMPUTED at merge time rather
than persisted. Nothing to keep in sync, and a re-run reproduces bit-for-bit.
(The reference's version called bare `random.uniform` with no seed set — its
augmented set cannot be regenerated.)

Scheduling: cost and MEMORY both scale as n_atoms**2 — the largest particle here
is 14793 atoms, taking ~7 min and 9.1 GB in one process (measured; DebyeCalculator's
`batch_size` does not cap it). So work runs in ascending size order and the pool
narrows as particles grow, trading workers for headroom; each worker then gets a
proportional share of the torch threads.

Run:
    python -m tools.augment_chili                     # generate into a stage dir (resumable)
    python -m tools.augment_chili --merge --overwrite # fold the stage into the registry
    ... --oob [--merge --overwrite]                   # same, for the out-of-band channel

    # CHILI-100K binary oxides (generate on the cluster: the .h5 are only there)
    python -m tools.augment_chili \\
        --registry data/downstream/chili100k/chili100k_registry.parquet \\
        --raw-dir data/raw/chili100k/cod_output_080124 [--merge --overwrite]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from core.registry import MATERIAL_ID
from core.transforms import PDF_OOB_RANGES, PDF_RANGES, draw_pdf_params
from modalities.pdf.debye import XPDF_RMAX, XPDF_RMIN, XPDF_RSTEP, simulate_pdf_debye
from modalities.pdf.simulate import RMIN as PRETRAIN_RMIN, RSTEP as PRETRAIN_RSTEP

REPO_ROOT = Path(__file__).resolve().parents[1]

# Two r-grids meet here, and only one of the three numbers is allowed to differ.
# analysis/downstream_eval.py feeds this bank to encoders pretrained on the 0-50 A
# grid by TRUNCATING to the first 5000 points — it never resamples — so the origin
# and the step must match and only rmax may be longer. Checked at generation time
# rather than at eval time, because that is 26 GPU-hours earlier.
assert (XPDF_RMIN, XPDF_RSTEP) == (PRETRAIN_RMIN, PRETRAIN_RSTEP), (
    f"xPDF grid ({XPDF_RMIN}, {XPDF_RSTEP}) must share origin and step with the "
    f"pretrain grid ({PRETRAIN_RMIN}, {PRETRAIN_RSTEP}) — see analysis/downstream_eval.py"
)

DEFAULT_REGISTRY = REPO_ROOT / "data" / "downstream" / "chili" / "chili_registry.parquet"
DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "chili" / "processed"
SIGNAL = "xpdf_aug"       # in-band (default): params from PDF_RANGES
OOB_SIGNAL = "xpdf_oob"   # --oob: params from PDF_OOB_RANGES
def _prefix(signal: str) -> str:
    """Param-column prefix ("aug_uiso" / "oob_uiso") — the signal's own suffix,
    so the two channels' drawn params can never overwrite each other."""
    return signal.removeprefix("xpdf_")


def _default_stage(registry_path: Path, signal: str) -> Path:
    """Next to the registry it augments, so two datasets' stages cannot mix."""
    return registry_path.parent / f"{signal}_stage"

#: Bytes of peak RSS per atom PAIR, measured s19: 9.10 GB at 14793 atoms. Used
#: only to decide how many particles may be in flight at once.
BYTES_PER_PAIR = 9.10e9 / 14793**2


def _particle(pt_path: Path) -> tuple[list[str], np.ndarray]:
    """(chemical symbols, absolute coords) of the nanoparticle in one CHILI .pt."""
    import torch
    from ase.data import chemical_symbols

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = torch.load(pt_path, weights_only=False)
    z = d.x[:, 0].numpy().astype(int)  # col 0 = atomic number, as in data/builders/chili.py
    return [chemical_symbols[v] for v in z], d.pos_abs.numpy().astype(float)


def _particle_h5(h5_path: Path, size_idx: int) -> tuple[list[str], np.ndarray]:
    """(chemical symbols, absolute coords) of one particle size in a CHILI-100K .h5.

    Same two fields the CHILI `.pt` carries (`AbsoluteCoordinates` is what its
    `pos_abs` was built from), read straight from the raw HDF5. `_sizes` is
    imported from the builder rather than re-sorting here — '9.4Å' must not sort
    after '13.6Å', and one implementation of that rule is enough.
    """
    import h5py
    from ase.data import chemical_symbols

    from data.builders.chili100k import _sizes

    with h5py.File(h5_path, "r") as h5:
        g = h5[f"DiscreteParticleGraphs/{_sizes(h5)[size_idx]}"]
        z = np.asarray(g["NodeFeatures"][()])[:, 0].round().astype(int)
        pos = np.asarray(g["AbsoluteCoordinates"][()], dtype=float)
    return [chemical_symbols[int(v)] for v in z], pos


def _row_source(material_id: str) -> tuple[str, int | None, int]:
    """(raw filename, particle size index or None, seed offset) for one row.

    Two id formats meet here:

        chili-<i>               CHILI-3K — one particle per .pt, seed offset i
        chili100k-<cod>-<size>  CHILI-100K — five sizes per COD .h5; offset
                                8*cod + size_idx keeps every row on its own
                                draw (size_idx < 8 always)

    The offset is a pure function of the id so `merge` can RECOMPUTE the drawn
    params instead of persisting them — the same contract the 3K path has
    always had, extended rather than forked.
    """
    parts = material_id.split("-")
    if parts[0] == "chili100k":
        cod, size_idx = int(parts[1]), int(parts[2])
        return f"{cod}.h5", size_idx, 8 * cod + size_idx
    return f"data_{parts[1]}.pt", None, int(parts[1])


def _workers_for(n_atoms: int, max_workers: int, budget_bytes: float) -> int:
    """How many particles of this size fit in the memory budget at once."""
    return max(1, min(max_workers, int(budget_bytes // (BYTES_PER_PAIR * n_atoms**2))))


_THREADS = 1


def _init(threads: int) -> None:
    global _THREADS
    _THREADS = threads


def _one(job: tuple[str, Path, int | None, Path, int, dict]) -> tuple[str, str]:
    """Simulate one augmented view and stage it. Returns (material_id, status)."""
    material_id, raw_path, size_idx, out_path, seed, ranges = job
    try:
        import torch

        torch.set_num_threads(_THREADS)
        elements, coords = (_particle(raw_path) if size_idx is None
                            else _particle_h5(raw_path, size_idx))
        r, g = simulate_pdf_debye(elements, coords, **draw_pdf_params(seed, ranges))
        if len(r) != _grid_len() or not np.isclose(r[0], XPDF_RMIN) or not np.isclose(r[1] - r[0], XPDF_RSTEP):
            raise ValueError(f"unexpected r grid: {len(r)} pts starting {r[0]} step {r[1] - r[0]}")
        if not np.isfinite(g).all():
            raise ValueError("non-finite G(r)")
        # atomic write: a killed job must not leave a half-written .npy that a
        # resume would then trust and skip
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:  # not np.save(path) — that appends .npy to the name
            np.save(fh, g)
        tmp.replace(out_path)
        return material_id, "ok"
    except Exception as e:  # noqa: BLE001 — report and keep the run going
        return material_id, f"{type(e).__name__}: {e}"


def _grid_len() -> int:
    return int(round((XPDF_RMAX - XPDF_RMIN) / XPDF_RSTEP))


def generate(registry_path: Path, raw_dir: Path, stage: Path, *, seed: int,
             max_workers: int, budget_bytes: float, limit: int | None,
             signal: str = SIGNAL, ranges: dict = PDF_RANGES) -> None:
    """Simulate every missing augmented view into `stage`, one .npy per material."""
    stage.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(registry_path, columns=[MATERIAL_ID, "n_atoms"])
    if limit is not None:
        df = df.nsmallest(limit, "n_atoms")

    (stage / "meta.json").write_text(json.dumps({
        # relpath, not relative_to: a registry outside the repo (a test fixture in
        # a temp dir) must still be recorded, not crash the run
        "registry": os.path.relpath(registry_path, REPO_ROOT),
        "signal": signal, "seed": seed, "ranges": ranges,
        "grid": {"rmin": XPDF_RMIN, "rmax": XPDF_RMAX, "rstep": XPDF_RSTEP, "n": _grid_len()},
    }, indent=2) + "\n")

    # ascending size: like-sized particles run together, so the pool can be sized
    # from the size actually in flight rather than from the worst case overall
    df = df.sort_values("n_atoms")
    jobs = []
    for material_id, n_atoms in zip(df[MATERIAL_ID], df["n_atoms"]):
        out = stage / f"{material_id}.npy"
        if out.exists():
            continue
        fname, size_idx, offset = _row_source(material_id)
        jobs.append((_workers_for(int(n_atoms), max_workers, budget_bytes),
                     (material_id, raw_dir / fname, size_idx, out, seed + offset, ranges)))
    done = len(df) - len(jobs)
    print(f"{len(df)} materials, {done} already staged, {len(jobs)} to simulate")
    if not jobs:
        return

    failures = []
    n_done = 0
    ctx = get_context("fork")
    # `_workers_for` is non-increasing in n_atoms and jobs are sorted by it, so
    # equal-k jobs are contiguous: one pool per distinct width, not per chunk.
    for k, group in itertools.groupby(jobs, key=lambda j: j[0]):
        batch = [j for _, j in group]
        # ONE thread per worker, always. This used to be `max(1, ncpu // k)`, which
        # is >1 only once k drops below ncpu — i.e. only for the largest particles,
        # at the very end of a run. Every such batch DEADLOCKED: the pool is
        # `fork`-based, and calling torch.set_num_threads(>1) in a forked child
        # spins up an OpenMP thread pool inside a process that already had one,
        # blocking forever on a mutex held at fork time. Zero CPU, no timeout, no
        # error — it just stops. (This env has three libomp copies on the path; see
        # docs/ENVIRONMENT.md. Same family as the sklearn crash in downstream_eval.)
        # Measured 2026-07-28: 3171/3180 views completed at 1 thread, then the first
        # 2-thread batch hung for 3h+ and never wrote another file.
        # Nothing is lost — memory, not CPU, is the binding constraint here, and that
        # is what `_workers_for` controls. The fast bulk of the run was already
        # 12 workers x 1 thread.
        threads = 1
        print(f"  {len(batch):5d} jobs x {k} workers x {threads} threads")
        with ctx.Pool(k, initializer=_init, initargs=(threads,)) as pool:
            for material_id, status in pool.imap_unordered(_one, batch, chunksize=1):
                n_done += 1
                if status != "ok":
                    failures.append((material_id, status))
                if n_done % 50 == 0 or n_done == len(jobs):
                    print(f"    {n_done}/{len(jobs)}  ({len(failures)} failed)", flush=True)

    if failures:
        print(f"\n{len(failures)} FAILED: {failures[:5]}")
    print(f"staged {n_done - len(failures)}/{len(jobs)} into {stage}")

    # Exit NON-ZERO if nothing was staged. `_one` swallows per-particle exceptions
    # by design — one bad particle must not kill a 3 h run — but the aggregate was
    # never checked, so a run that failed EVERY particle still exited 0 and SLURM
    # reported it `COMPLETED 0:0`, which `--dependency=afterok` then released on
    # (2026-08-05, cause: a dependency missing from the cluster env).
    # A whole-run failure is an error; a partial one stays a warning, because the run
    # is resumable and re-running fills the gaps.
    if jobs and n_done == len(failures):
        raise SystemExit(
            f"FAILED: 0 of {len(jobs)} views staged — every particle errored. "
            f"First: {failures[0][1] if failures else '?'}"
        )


def merge(registry_path: Path, stage: Path, out_path: Path, *, seed: int,
          signal: str = SIGNAL, ranges: dict = PDF_RANGES) -> None:
    """Fold the staged views into the registry as a second signal channel."""
    # The params written below are RECOMPUTED from (seed, ranges), not read from
    # the stage — so a stage generated under different settings would be silently
    # mislabelled. meta.json is the stage's own record; require agreement.
    meta = json.loads((stage / "meta.json").read_text())
    got = {k: meta.get(k) for k in ("signal", "seed", "ranges")}
    want = {"signal": signal, "seed": seed,
            "ranges": {k: list(v) for k, v in ranges.items()}}  # json turns tuples into lists
    if got != want:
        raise SystemExit(f"stage {stage} was generated as {got}, but merge was asked "
                         f"for {want} — wrong stage dir or wrong flags")

    df = pd.read_parquet(registry_path)
    grid = (XPDF_RMIN + np.arange(_grid_len()) * XPDF_RSTEP).astype(np.float32)

    missing = [m for m in df[MATERIAL_ID] if not (stage / f"{m}.npy").exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(df)} materials have no staged view "
            f"(e.g. {missing[:3]}) — run without --merge until it reports 0 failed"
        )

    ys, params = [], []
    for material_id in df[MATERIAL_ID]:
        g = np.load(stage / f"{material_id}.npy")
        if g.shape != grid.shape:
            raise ValueError(f"{material_id}: staged {g.shape} but grid is {grid.shape}")
        ys.append(g)
        # recomputed, not read back from disk — same seed, same draw, by construction
        params.append(draw_pdf_params(seed + _row_source(material_id)[2], ranges))

    # A second augmented channel must actually differ from the first: identical
    # rows mean the ranges knob never reached the simulation (same seed stream,
    # different ranges, so every curve must move).
    if signal != SIGNAL and f"signal_{SIGNAL}_y" in df.columns:
        same = sum(np.allclose(y, ref) for y, ref in zip(ys, df[f"signal_{SIGNAL}_y"]))
        if same:
            raise SystemExit(f"{same} of {len(df)} {signal} views are identical to their "
                             f"{SIGNAL} view — the stage was not simulated at {ranges}")

    df[f"signal_{signal}_x"] = [grid] * len(df)
    df[f"signal_{signal}_y"] = ys
    for name in ranges:
        df[f"{_prefix(signal)}_{name}"] = [p[name] for p in params]

    df.to_parquet(out_path, index=False)
    print(f"saved {out_path}  ({len(df)} rows, {out_path.stat().st_size / 1e6:.0f} MB, "
          f"+signal_{signal}_x/_y and {_prefix(signal)}_{'/'.join(ranges)})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY),
                    help="CHILI-3K by default; point at chili100k_registry.parquet "
                         "(with --raw-dir .../cod_output_080124) for the 100K mode")
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    ap.add_argument("--oob", action="store_true",
                    help=f"draw params from PDF_OOB_RANGES and write signal_{OOB_SIGNAL}_* "
                         f"instead of signal_{SIGNAL}_*")
    ap.add_argument("--stage", default=None,
                    help="staging dir (default: <signal>_stage next to the registry)")
    ap.add_argument("--out", default=None, help="merge target (default: in place)")
    ap.add_argument("--seed", type=int, default=0, help="base seed; view seed is seed + sample_idx")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--mem-gb", type=float, default=24.0,
                    help="peak RSS budget across workers; caps how many big particles run at once")
    ap.add_argument("--limit", type=int, default=None, help="only the N smallest particles")
    ap.add_argument("--merge", action="store_true", help="fold the stage into the registry")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    registry_path = Path(args.registry)
    signal = OOB_SIGNAL if args.oob else SIGNAL
    ranges = PDF_OOB_RANGES if args.oob else PDF_RANGES
    stage = Path(args.stage) if args.stage else _default_stage(registry_path, signal)
    if args.merge:
        out_path = Path(args.out) if args.out else registry_path
        if out_path == registry_path and not args.overwrite:
            sys.exit(f"REFUSING: would rewrite {out_path} in place (use --overwrite).")
        merge(registry_path, stage, out_path, seed=args.seed, signal=signal, ranges=ranges)
    else:
        generate(registry_path, Path(args.raw_dir), stage, seed=args.seed,
                 max_workers=args.workers, budget_bytes=args.mem_gb * 1e9, limit=args.limit,
                 signal=signal, ranges=ranges)


if __name__ == "__main__":
    main()
