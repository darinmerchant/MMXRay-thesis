"""tools/simulate_ladder.py — one instrument parameter on a ladder, the other three pinned.

    python -m tools.simulate_ladder --n-materials 256

Generates the G(r) that `analysis/invariance_decay.py` measures on: for each of the
four PDF instrument parameters, a ladder of levels spanning `PDF_BOUNDS`, with the
other three held at the simulator's own nominal values. Plus two independent
production-range draws per material, which give the within-material reference line.

WHY THIS IS A SEPARATE FILE FROM THE MEASUREMENT, and it is not a style choice. The
simulation half must pin the five BLAS/OpenMP variables at module scope, before numpy
loads, and then fan out over a `multiprocessing.Pool` — workers are SPAWNED on macOS,
so each child re-imports this module and picks the variables up before it touches
anything numeric. The measurement half must `import torch` first or it segfaults with
no traceback (docs/ENVIRONMENT.md). Those two requirements cannot hold in one module,
so the boundary is a file on disk: this writes npz, `analysis/` reads it.

Same reason `experiments/aug_range_evidence/` states it contains no torch. The
plumbing below (BLAS preamble, job-keyed cache, pool shape) is COPIED from that
directory's `_shared.py` rather than imported, because its README declares it
"EXPERIMENT, NOT ADOPTED CODE ... nothing here is imported by `core/`, `analysis/`,
or `tools/`" and importing would invert that boundary. It also hardcodes the sweep12k
registry, and this needs full-MP.

CIFs ARE EXTRACTED IN THE PARENT AND PASSED INTO THE JOBS, rather than each worker
loading the registry. `material_registry_mpfull.parquet` is 69 MB; ten spawned workers
each loading it is ~700 MB of duplicated frame to read 256 rows. Passing the CIF
strings makes the workers stateless and the memory constant. It also means the worker
calls `modalities.pdf.simulate.simulate_pdf` directly rather than going through
`core.simulate.simulate`'s dispatch seam — legitimate here because a ladder over
`uiso/qmax/qbroad/qmin` is PDF-specific by construction, so there is no modality to
dispatch on.

LEVELS ARE FRONT-LOADED INSIDE THE PRODUCTION RANGE. `PDF_RANGES` is both the
pretraining augmentation range and, via `tools/augment_chili.py`, how the downstream
evaluation set is shifted — so that window is the part of the curve that explains the
downstream results, and the region outside it is a separate (bonus) claim about
generalizing past what was trained on. Eight levels inside including both endpoints,
four below and four above reaching `PDF_BOUNDS`.

⚠️ `qmin` SNAPS to diffpy's Q-grid of step `pi/rmax = 0.0628 A^-1` (docs/TRAPS.md), so
neighbouring rungs closer than that return byte-identical G(r) and the curve would
show a staircase that is an artifact of the grid, not of the encoder. The level
spacing above clears it (0.125 below the range, 0.214 inside, 0.5 above) — and
`--check` asserts it on the simulated arrays rather than trusting the arithmetic.
"""

from __future__ import annotations

import os

# MEASURED in experiments/aug_range_evidence/_shared.py: with workers left to their own
# threading, 10 workers bought only 1.3x over one process, because each spawned its own
# thread pool onto the same cores. Set at module scope, before numpy, because OpenMP
# reads these at library-init time and macOS workers are spawned.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "data" / "registry" / "material_registry_mpfull.parquet"
DEFAULT_OUT = REPO_ROOT / "runs/pdf/latent" / "invariance"

#: `modalities/pdf/simulate.py`'s own defaults — the nominal instrument the swept
#: parameter moves away from. Matches `experiments/aug_range_evidence/_shared.NOMINAL`.
NOMINAL = {"uiso": 0.005, "qmax": 22.5, "qbroad": 0.03, "qmin": 1.0}
PARAMS = ["uiso", "qmax", "qbroad", "qmin"]

N_IN, N_OUT = 8, 4  # levels inside the production range, and on each side of it


def levels_for(param: str) -> np.ndarray:
    """The ladder for one parameter: dense inside `PDF_RANGES`, reaching `PDF_BOUNDS`.

    Both production endpoints are ON the grid, so the shaded window in the figure has
    measured points at its edges rather than interpolated ones.
    """
    from core.transforms import PDF_BOUNDS, PDF_RANGES

    lo, hi = PDF_RANGES[param]
    blo, bhi = PDF_BOUNDS[param]
    below = np.linspace(blo, lo, N_OUT + 1)[:-1]
    inside = np.linspace(lo, hi, N_IN)
    above = np.linspace(hi, bhi, N_OUT + 1)[1:]
    return np.concatenate([below, inside, above]).astype(float)


def sample_materials(registry: Path, n: int, seed: int) -> tuple[list[str], list[str]]:
    """`(ids, cifs)` for `n` materials drawn from the registry.

    A seeded draw rather than a head slice: registry row order tracks the source
    dataset's own ordering and can correlate with composition, so a head slice is a
    biased sample of chemistry — the reasoning
    `experiments/aug_range_evidence/_shared.sample_materials` gives for the same choice.
    """
    from core.registry import MaterialRegistry

    reg = MaterialRegistry.load(registry, validate=False)
    ids_all = reg.material_ids
    rng = np.random.default_rng(seed)
    picked = [ids_all[i] for i in rng.choice(len(ids_all), size=n, replace=False)]
    return picked, [reg.get(m).cif for m in picked]


def _one(job: tuple[str, dict]) -> np.ndarray:
    cif, params = job
    from modalities.pdf.simulate import simulate_pdf

    _, g = simulate_pdf(cif, **params)
    return np.asarray(g, dtype=np.float32)


def _job_key(jobs: list[tuple[str, dict]]) -> str:
    """Hash of every (cif, params) pair, so a changed grid cannot reuse a stale array."""
    blob = json.dumps([[hashlib.sha1(c.encode()).hexdigest(), sorted(p.items())]
                       for c, p in jobs], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def simulate_batch(name: str, jobs: list[tuple[str, dict]], out: Path, workers: int) -> np.ndarray:
    """`(len(jobs), 5000)` float32, cached at `out/sim_<name>.npz` on the exact job list."""
    cache = out / f"sim_{name}.npz"
    key = _job_key(jobs)
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)
        if str(stored["key"]) == key:
            print(f"  [{name}] cached ({len(jobs)} sims)", flush=True)
            return stored["g"]
        print(f"  [{name}] cache stale, re-simulating", flush=True)

    print(f"  [{name}] simulating {len(jobs)} views on {workers} workers…", flush=True)
    with mp.Pool(workers) as pool:
        g = np.stack(pool.map(_one, jobs, chunksize=8))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(cache, g=g, key=key)
    return g


def check_no_ties(g: np.ndarray, levels: np.ndarray, param: str) -> None:
    """Adjacent rungs must differ. The `qmin` Q-grid snap is the reason this exists.

    `g` is `(n_levels, n_materials, grid)`. Compared on the first material only —
    a tie is a property of the level spacing against diffpy's grid, not of the
    structure, so it either happens for every material or for none.
    """
    for i in range(len(levels) - 1):
        if np.array_equal(g[i, 0], g[i + 1, 0]):
            raise SystemExit(
                f"{param}: rungs {levels[i]:.5g} and {levels[i + 1]:.5g} gave identical G(r) — "
                f"they are closer than diffpy's grid step (pi/rmax = 0.0628 for qmin) and the "
                f"curve would show a staircase that is an artifact. Widen the spacing."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY,
                        help="corpus to draw materials from; default is full-MP, the corpus "
                             "transformer_vicreg_mpfull_final pretrained on")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-materials", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7, help="material draw seed")
    parser.add_argument("--ref-seed", type=int, default=1000,
                        help="base seed for the two production-range draws per material")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    from core.transforms import PDF_RANGES, draw_pdf_params

    ids, cifs = sample_materials(args.registry, args.n_materials, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{args.registry.name}: {len(ids)} materials, seed {args.seed}, "
          f"{args.workers} workers", flush=True)

    manifest = {
        "registry": args.registry.name,
        "n_materials": len(ids),
        "material_ids": ids,
        "sample_seed": args.seed,
        "ref_seed": args.ref_seed,
        "nominal": NOMINAL,
        "production_ranges": {k: list(v) for k, v in PDF_RANGES.items()},
        "levels": {},
    }

    for param in PARAMS:
        levels = levels_for(param)
        jobs = [(cif, {**NOMINAL, param: float(v)}) for v in levels for cif in cifs]
        g = simulate_batch(param, jobs, args.out, args.workers)
        g = g.reshape(len(levels), len(ids), -1)
        check_no_ties(g, levels, param)
        np.savez(args.out / f"ladder_{param}.npz", g=g, levels=levels)
        manifest["levels"][param] = [float(v) for v in levels]
        print(f"  [{param}] {len(levels)} levels, grid {g.shape[-1]}, no ties", flush=True)

    # Two independent production-range draws per material — the positive pair the
    # encoder was actually trained on, and therefore the reference line the latent
    # curve should be compared against.
    ref_params = [[draw_pdf_params(args.ref_seed + 2 * i + v) for i in range(len(ids))]
                  for v in (0, 1)]
    ref_jobs = [(cif, p) for view in ref_params for cif, p in zip(cifs, view)]
    gr = simulate_batch("refs", ref_jobs, args.out, args.workers).reshape(2, len(ids), -1)
    np.savez(args.out / "refs.npz", g=gr,
             params=np.array([[[p[k] for k in PARAMS] for p in view] for view in ref_params]))
    print(f"  [refs] 2 production-range draws x {len(ids)} materials", flush=True)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {args.out}/  (manifest.json, ladder_*.npz, refs.npz)")


if __name__ == "__main__":
    main()
