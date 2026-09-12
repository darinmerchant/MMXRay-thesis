"""core/dataset_base.py — pretraining Datasets: two seeded views per material.

Two datasets, one contract: both are indexed by material over the FULL registry
(pretraining doesn't use train/val/test splits) and both return
`(view1, view2, material_id, params1, params2)`, so `core/train.py` consumes
either without knowing which it got.

- `PDFPretrainDataset` — simulates both views live (diffpy per `__getitem__`).
  PDF-only, and stays so: the banked path is the production path for every
  modality, so a live XRD dataset would have no caller.
- `BankedPretrainDataset` — reads 2 of the N views pre-simulated by
  `tools/bank_pdf.py`, PDF or XRD; the block's own meta.json says which and
  picks the normalizer. **The production path**: live sim measured out at ~20
  sims/s, i.e. ~75 min per MP-20 epoch, so views are banked once and sampled
  from thereafter (DESIGN.md -> Throughput -> Banking decision).
  `BankedPDFPretrainDataset` remains as an alias — every existing caller and
  the T0.3 read-path pin import that name, and the pdf behaviour is
  byte-identical (that is what the pin holds it to).

Reproducibility (both): the per-material draw is seeded from
`(base_seed, epoch, index)`, so a given (epoch, index) always reproduces the
same pair — but `set_epoch(e)` changes it, so augmentations differ every epoch
(real contrastive pretraining behavior, not a fixed pair reused forever). Call
`set_epoch` before each epoch's DataLoader is iterated; with
`persistent_workers=True` this won't reach worker processes; see PROGRESS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.registry import MaterialRegistry
from core.transforms import PDFSimulate, max_normalize, minmax_normalize

__all__ = ["PDFPretrainDataset", "BankedPretrainDataset", "BankedPDFPretrainDataset"]


class PDFPretrainDataset(Dataset):
    """Two-view PDF pretraining dataset over an entire registry.

    `__getitem__` returns `(view1, view2, material_id, params1, params2)`:
    the two simulated `G(r)` tensors — each per-sample min-max normalized to
    [0, 1] (`core.transforms.minmax_normalize`, the encoder-input convention;
    see DESIGN.md -> Signal normalization) — the material id, and each view's
    sampled (uiso, qmax, qbroad, qmin) — kept for the invariance eval and
    debugging, not used by the contrastive loss itself.
    """

    def __init__(
        self,
        registry: MaterialRegistry,
        transform: PDFSimulate | None = None,
        *,
        base_seed: int = 0,
    ) -> None:
        self._registry = registry
        self._ids = registry.material_ids
        self._transform = transform if transform is not None else PDFSimulate()
        self._base_seed = base_seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._ids)

    def _seed(self, index: int, view: int) -> int:
        # unique per (epoch, index, view); stride by 2*len(ids) per epoch so
        # no (epoch, index, view) triple ever collides with another.
        return self._base_seed + (self._epoch * len(self._ids) + index) * 2 + view

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, dict, dict]:
        material_id = self._ids[index]
        record = self._registry.get(material_id)
        v1 = self._transform(record, seed=self._seed(index, 0))
        v2 = self._transform(record, seed=self._seed(index, 1))
        return (
            minmax_normalize(torch.from_numpy(v1.y).float()),
            minmax_normalize(torch.from_numpy(v2.y).float()),
            material_id,
            v1.params,
            v2.params,
        )


class BankedPretrainDataset(Dataset):
    """Two-view pretraining dataset over a pre-simulated block (`tools/bank_pdf.py`).

    A block holds N raw G(r) views per material, sharded across array tasks:

        shard_<i>.npy       (n_ids_in_shard * N, grid_len) raw G(r); the row for
                            one view is (global_index - shard_gidx0) * N + view
        shard_<i>.parquet   one row per view: material_id, global_index, view, + params
        meta.json           n_ids, n_views, aug ranges, r-grid, config hash

    `__getitem__` draws two DISTINCT view indices for the material — C(N, 2)
    possible pairs, N=32 by default — seeded by `(base_seed, epoch, index)` and
    reseeded by `set_epoch`, exactly like the live-sim dataset above. Views are
    min-max normalized HERE because the bank stores raw physical G(r) on purpose
    (DESIGN.md -> Signal normalization), which keeps a normalization change a
    read-path edit rather than a ~10 h re-bank.

    `start`/`limit` slice the materials in global-index order — `limit` counts
    from `start`, the banked counterpart of slicing the registry frame. `start` is
    what reaches the I_NCE holdout: `tools/make_sweep_registry.py` puts the
    pretraining pool at rows [0, n_train) and the holdout at [n_train, n_ids), so
    training passes `limit=n_train` and `analysis/aug_sweep.py` passes
    `start=n_train`. Both halves are banked; only the head is ever trained on.
    """

    def __init__(self, block: str | Path, *, base_seed: int = 0, limit: int | None = None,
                 start: int = 0, extra_noise_c: tuple[float, float] | None = None) -> None:
        block = Path(block)
        if extra_noise_c is not None:
            lo, hi = (float(v) for v in extra_noise_c)
            if not 0.0 <= lo <= hi:
                raise ValueError(f"extra_noise_c must be 0 <= lo <= hi, got {extra_noise_c}")
            extra_noise_c = (lo, hi)
        self._extra_noise_c = extra_noise_c
        meta = json.loads((block / "meta.json").read_text())
        self._n_views = int(meta["n_views"])
        # Per-modality encoder-input convention (DESIGN.md -> Signal normalization):
        # XRD keeps its meaningful zero (max-norm), PDF stays min-max. Every block
        # banked before meta carried `modality` is PDF by construction.
        self._normalize = max_normalize if meta.get("modality", "pdf") == "xrd" else minmax_normalize
        # param_names come from the PARQUET COLUMNS, not from aug_ranges: the
        # ranged params are a strict subset of what was drawn (xrd banks 13
        # columns against 6 declared ranges; pdf banks exactly its 4). Column
        # order is the draw order — bank_pdf writes {**keys, **params} per row —
        # so this survives the bank -> read round trip unchanged. Set from shard
        # 0 in the loop below.
        self._param_names: list[str] | None = None

        shard_paths = sorted(block.glob("shard_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
        if not shard_paths:
            raise FileNotFoundError(f"no shard_*.npy in {block}")

        ids: list[str] = []
        gidxs: list[np.ndarray] = []
        shard_of: list[np.ndarray] = []
        self._params: list[np.ndarray] = []
        self._grid_len: int | None = None

        for s, npy in enumerate(shard_paths):
            # The params parquet is written in worker-COMPLETION order, while the
            # .npy row for a view is fixed by (global_index, view) — so sort, never
            # assume the two files line up positionally.
            df = pd.read_parquet(npy.with_suffix(".parquet")).sort_values(["global_index", "view"])
            if self._param_names is None:
                self._param_names = [c for c in df.columns
                                     if c not in ("material_id", "global_index", "view")]
            gidx = df["global_index"].to_numpy()
            n_ids_shard, remainder = divmod(len(df), self._n_views)
            expected = np.repeat(np.arange(gidx[0], gidx[0] + n_ids_shard), self._n_views)
            if remainder or not np.array_equal(gidx, expected):
                raise ValueError(f"{npy.name}: params are not exactly {self._n_views} views of a contiguous id range")

            rows, width = np.load(npy, mmap_mode="r").shape  # header read only, no data touched
            if rows != len(df):
                raise ValueError(f"{npy.name}: {rows} banked views but {len(df)} param rows — shard is truncated")
            if self._grid_len is None:
                self._grid_len = width
            elif width != self._grid_len:
                raise ValueError(f"{npy.name}: grid_len {width} != {self._grid_len} in earlier shards")

            ids.extend(df["material_id"].to_numpy()[:: self._n_views].tolist())
            gidxs.append(gidx[:: self._n_views])
            shard_of.append(np.full(n_ids_shard, s, dtype=np.int32))
            self._params.append(df[self._param_names].to_numpy(dtype=np.float32))

        # A missing/failed array task would otherwise silently shrink the dataset.
        if not np.array_equal(np.concatenate(gidxs), np.arange(int(meta["n_ids"]))):
            raise ValueError(
                f"{block.name} is incomplete: {len(ids)} materials across {len(shard_paths)} shards do not cover "
                f"global indices 0..{int(meta['n_ids']) - 1} — a shard is missing or was written twice"
            )

        if not 0 <= start < len(ids):
            raise ValueError(f"start={start} outside 0..{len(ids) - 1} for {block.name}")

        # `__getitem__` indexes all three by the SAME dataset index, so they slice
        # together or the row lookup silently addresses another material's views.
        keep = slice(start, None if limit is None else start + limit)
        self._shard_paths = shard_paths
        self._shard_of = np.concatenate(shard_of)[keep]
        # Row of view 0 within its shard: materials are contiguous, N views each.
        self._row_base = np.concatenate([np.arange(len(g)) * self._n_views for g in gidxs])[keep]
        self._ids = ids[keep]
        self._base_seed = base_seed
        self._epoch = 0
        self._arrays: list[np.ndarray] | None = None

    @property
    def grid_len(self) -> int:
        """Points per banked view — must match `TrainConfig.signal_len` for the Transformer."""
        return self._grid_len

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._ids)

    def _array(self, shard: int) -> np.ndarray:
        """The shard's memmap, opened on first use IN THIS PROCESS.

        Lazily, not in `__init__`: a `np.memmap` attribute would be serialized in
        full if the dataset were pickled to a spawned DataLoader worker. Opening
        here means each worker maps the file itself and the OS page cache — not
        RAM — holds the hot views (the MP-20 N=32 block is ~29 GB).
        """
        if self._arrays is None:
            self._arrays = [np.load(p, mmap_mode="r") for p in self._shard_paths]
        return self._arrays[shard]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, dict, dict]:
        rng = np.random.default_rng(self._base_seed + self._epoch * len(self._ids) + index)
        v1, v2 = rng.choice(self._n_views, size=2, replace=False)  # distinct: an identical pair is a free positive
        shard = int(self._shard_of[index])
        arr = self._array(shard)
        row1 = int(self._row_base[index]) + int(v1)
        row2 = int(self._row_base[index]) + int(v2)
        # np.array (not asarray) copies out of the read-only memmap and casts a
        # float16 block up — torch.from_numpy needs a writable, owned array.
        y1 = np.array(arr[row1], dtype=np.float32)
        y2 = np.array(arr[row2], dtype=np.float32)
        return (
            self._normalize(torch.from_numpy(self._add_noise(y1, shard, row1))),
            self._normalize(torch.from_numpy(self._add_noise(y2, shard, row2))),
            self._ids[index],
            self._view_params(shard, row1),
            self._view_params(shard, row2),
        )

    def _add_noise(self, y: np.ndarray, shard: int, row: int) -> np.ndarray:
        """Extra Gaussian noise on a banked view, in the simulator's own units.

        Mirrors `modalities/xrd/simulate.py` exactly — `sigma = noise_c * y.max()`,
        then `clip(y, 0, None)` — so a read-path view is distributed like a view the
        simulator would have produced at the wider range, and the normalization that
        follows sees the same shape of input either way.

        **Seeded by (shard, row), NOT by epoch**, which is the whole point: a banked
        view carries ONE fixed noise realization, so redrawing per epoch would make
        this two changes at once (a wider range AND fresh noise every epoch) and the
        comparison against an existing checkpoint could not attribute either.
        Fixed-per-view keeps the banked semantics and leaves the range as the only
        difference.

        The banked noise is still underneath: total sigma is
        `sqrt(banked^2 + drawn^2)`, so the effective floor is the banked level
        (<= 0.005) rather than `lo`. At the top of the range the banked term moves
        sigma by 0.5% and is ignored deliberately — reconstructing the exact
        complement per row would need the parquet's own `noise_c` for a correction
        smaller than the draw's spread.
        """
        if self._extra_noise_c is None:
            return y
        rng = np.random.default_rng([shard, row])
        noise_c = float(rng.uniform(*self._extra_noise_c))
        m = float(y.max())
        if m <= 0 or noise_c <= 0:
            return y
        return np.clip(y + rng.normal(0.0, noise_c * m, size=y.shape), 0.0, None).astype(np.float32)

    def _view_params(self, shard: int, row: int) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self._param_names, self._params[shard][row])}


#: Pre-rename alias (step 4 of the XRD arm). Every earlier caller — core/train.py,
#: analysis/latent.py, the T0.3 read-path pin — imports this name; behaviour on a
#: PDF block is byte-identical, which is exactly what that pin verifies.
BankedPDFPretrainDataset = BankedPretrainDataset
