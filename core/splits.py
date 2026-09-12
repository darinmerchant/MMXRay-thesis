"""core/splits.py — group-disjoint split assignment.

`assign()` writes a `split` column onto a `MaterialRegistry`: `train`/`val`/`test`,
disjoint at the *group* level (default group = `material_id`) so rows that must
travel together — e.g. CHILI's 5 nanoparticle sizes per composition — never end
up split across sets.

Pretraining is `fractions=(1.0, 0.0, 0.0)`: same code path, empty val/test.
Random only — no stratification, no k-fold. Add either only when a real
experiment needs it (see DESIGN.md → Splits).
"""

from __future__ import annotations

import numpy as np

from core.registry import MATERIAL_ID, MaterialRegistry, RegistryError

__all__ = ["assign"]


def assign(
    reg: MaterialRegistry,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    *,
    group_by: str | None = None,
    seed: int = 0,
) -> MaterialRegistry:
    """Group-disjoint random split. Returns a NEW registry with `split` set.

    `group_by` names a column whose equal values must land in the same split
    (default: `material_id`, i.e. plain per-material disjoint). `fractions` is
    `(train, val, test)` and must sum to 1.0; pretraining passes `(1.0, 0.0, 0.0)`.
    Splitting is by unique group, not by row — a group with many rows counts once.
    """
    if len(fractions) != 3:
        raise RegistryError(f"fractions must have 3 entries (train, val, test), got {fractions}")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise RegistryError(f"fractions {fractions} must sum to 1.0")

    df = reg.frame
    key = group_by or MATERIAL_ID
    if key not in df.columns:
        raise RegistryError(f"no column {key!r} to group by")
    if df[key].isna().any():
        raise RegistryError(f"group_by column {key!r} has null values")

    groups = df[key].unique()
    np.random.default_rng(seed).shuffle(groups)

    n = len(groups)
    n_train = round(fractions[0] * n)
    n_val = round(fractions[1] * n)
    # test takes the remainder, so rounding never drops or duplicates a group
    split_of_group = dict.fromkeys(groups[:n_train], "train")
    split_of_group.update(dict.fromkeys(groups[n_train : n_train + n_val], "val"))
    split_of_group.update(dict.fromkeys(groups[n_train + n_val :], "test"))

    mapping = dict(zip(df[MATERIAL_ID], df[key].map(split_of_group)))
    return reg.with_split(mapping)
