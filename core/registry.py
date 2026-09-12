"""MaterialRegistry — the modality-agnostic pivot table.

One row per material: a canonical `material_id`, a base payload, a `split`, plus
optional targets / flags / metadata.  Both simulators (PDF, XRD) are views of the
same row, so splits and pairing are shared by construction.

Base modes, derived per row from which payload is present (see DESIGN.md):

* ``structure`` — CIF text -> simulate on the fly (+ augment).
* ``signal``    — one or more measured signals -> fixed, never simulated/augmented.
* ``both``      — valid, e.g. CHILI (measured signal(s) *and* a CIF). The registry
  does not choose between them; `dataset_base.py` / the experiment does.

Signals are **named channels**: a channel ``<name>`` lives in ``signal_<name>_y``
(payload) plus optional ``signal_<name>_x`` (grid). A registry can hold several
(CHILI: ``xrd`` + ``xpdf``); `reg.signal(mid, name)` reads one.

Storage is one local parquet file with CIFs packed inline as a text column.
Loading and querying are pure in-memory table ops — this module never touches the
network, never imports pymatgen/diffpy, and knows nothing about physics or models.

Scope is deliberately minimal (`DESIGN.md` → *Essential API*): this is only what
`core/splits.py` and the `base_builder.py`s need.  Add on demand, not in advance.

Typical use::

    reg = MaterialRegistry.load("data/registry/material_registry.parquet")
    train = reg.split("train")
    cif = train.cif("mp-149")
    y = train.targets(["bond_length"])
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "MaterialRegistry",
    "MaterialRecord",
    "ValidationReport",
    "RegistryError",
    "RegistryValidationError",
    "MATERIAL_ID",
    "STRUCTURE",
    "SPLIT",
    "SIGNAL_PREFIX",
    "TARGET_PREFIX",
    "FLAG_PREFIX",
    "VALID_SPLITS",
    "BASE_MODE_STRUCTURE",
    "BASE_MODE_SIGNAL",
    "BASE_MODE_BOTH",
    "BASE_MODE_NONE",
]

# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #

MATERIAL_ID = "material_id"
STRUCTURE = "structure"
SPLIT = "split"

TARGET_PREFIX = "target_"
FLAG_PREFIX = "has_"

#: Signals are named channels: a channel ``<name>`` is the required payload
#: column ``signal_<name>_y`` plus an optional grid column ``signal_<name>_x``
#: (e.g. ``signal_xrd_y`` / ``signal_xrd_x``).  A registry may carry several
#: (CHILI: ``xrd`` + ``xpdf``).  There is no unnamed single-signal form.
SIGNAL_PREFIX = "signal_"
SIGNAL_X_SUFFIX = "_x"
SIGNAL_Y_SUFFIX = "_y"

BASE_MODE_STRUCTURE = "structure"
BASE_MODE_SIGNAL = "signal"
BASE_MODE_BOTH = "both"
BASE_MODE_NONE = "none"

VALID_SPLITS = ("train", "val", "test")

#: Columns with a fixed meaning; everything else is a target / flag / signal / metadata.
RESERVED_COLUMNS = (MATERIAL_ID, SPLIT, STRUCTURE)


class RegistryError(Exception):
    """Bad registry usage: unknown id, missing column, malformed file."""


class RegistryValidationError(RegistryError):
    """`validate()` found errors and was asked to raise."""


# --------------------------------------------------------------------------- #
# Records and validation reports
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MaterialRecord:
    """One registry row, with the column groups already split apart."""

    material_id: str
    split: str | None
    base_mode: str
    structure: str | None = None
    #: channel name -> (x, y); x may be None. Empty if the row has no signal.
    signals: dict[str, tuple[np.ndarray | None, np.ndarray]] = field(default_factory=dict)
    targets: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cif(self) -> str:
        """CIF text; raises if this row has no structure."""
        if self.structure is None:
            raise RegistryError(
                f"{self.material_id!r} is base_mode={self.base_mode!r} and has no structure"
            )
        return self.structure

    def signal(self, name: str | None = None) -> tuple[np.ndarray | None, np.ndarray]:
        """``(x, y)`` for one signal channel; ``x`` may be None.

        `name` is optional when the row has exactly one channel; with several
        (e.g. CHILI ``xrd`` + ``xpdf``) it is required.
        """
        if not self.signals:
            raise RegistryError(
                f"{self.material_id!r} is base_mode={self.base_mode!r} and has no signal"
            )
        if name is None:
            if len(self.signals) == 1:
                return next(iter(self.signals.values()))
            raise RegistryError(
                f"{self.material_id!r} has multiple signals {list(self.signals)}; pass a name"
            )
        if name not in self.signals:
            raise RegistryError(f"{self.material_id!r} has no signal {name!r}; have {list(self.signals)}")
        return self.signals[name]

    @property
    def signal_names(self) -> list[str]:
        return list(self.signals)

    @property
    def is_simulatable(self) -> bool:
        return self.structure is not None

    @property
    def has_measured_signal(self) -> bool:
        return bool(self.signals)


@dataclass
class ValidationReport:
    """Outcome of `MaterialRegistry.validate`."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self, context: str = "registry") -> ValidationReport:
        if self.errors:
            bullets = "\n  - ".join(self.errors)
            raise RegistryValidationError(
                f"{context} failed validation ({len(self.errors)} error(s)):\n  - {bullets}"
            )
        return self

    def emit_warnings(self, context: str = "registry") -> ValidationReport:
        for msg in self.warnings:
            warnings.warn(f"{context}: {msg}", stacklevel=3)
        return self

    def __str__(self) -> str:
        lines = [f"ValidationReport({self.n_rows} rows, {'OK' if self.ok else 'FAILED'})"]
        lines += [f"  ERROR   {m}" for m in self.errors]
        lines += [f"  warning {m}" for m in self.warnings]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class MaterialRegistry:
    """Local, in-memory view of a material registry table.

    Immutable in practice: every query returns plain data or a new
    `MaterialRegistry` over a filtered copy.  Mutating `.frame` in place is
    unsupported — the id lookup index would go stale.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        if MATERIAL_ID not in frame.columns:
            raise RegistryError(f"frame has no {MATERIAL_ID!r} column")
        # reset_index so row position always matches .iloc, even after filtering
        self._df = frame.reset_index(drop=True)
        self._pos: dict[str, int] | None = None

    # -------------------------------------------------------------- load/save

    @classmethod
    def load(cls, path: str | Path, *, validate: bool = True) -> MaterialRegistry:
        """Read a registry parquet. With `validate`, errors raise and warnings emit."""
        path = Path(path)
        if not path.exists():
            raise RegistryError(f"no registry at {path}")
        reg = cls(pd.read_parquet(path))
        if validate:
            report = reg.validate()
            report.emit_warnings(context=str(path))
            report.raise_for_errors(context=str(path))
        return reg

    def save(self, path: str | Path) -> Path:
        """Write to parquet, CIFs inline."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(path, index=False, compression="zstd")
        return path

    # ------------------------------------------------------------- properties

    @property
    def frame(self) -> pd.DataFrame:
        """The underlying DataFrame. Read-only by convention — copy to mutate."""
        return self._df

    @property
    def material_ids(self) -> list[str]:
        return self._df[MATERIAL_ID].tolist()

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    @property
    def target_names(self) -> list[str]:
        """Names behind the ``target_`` prefix, with the prefix stripped."""
        return [c[len(TARGET_PREFIX):] for c in self._df.columns if c.startswith(TARGET_PREFIX)]

    @property
    def flag_names(self) -> list[str]:
        return [c for c in self._df.columns if c.startswith(FLAG_PREFIX)]

    @property
    def signal_names(self) -> list[str]:
        """Signal channels present, from the ``signal_<name>_y`` columns."""
        return _signal_names(self._df.columns)

    @property
    def metadata_names(self) -> list[str]:
        """Everything that is not core, target, flag or signal — all nullable."""
        return [
            c
            for c in self._df.columns
            if c not in RESERVED_COLUMNS
            and not c.startswith((TARGET_PREFIX, FLAG_PREFIX, SIGNAL_PREFIX))
        ]

    @property
    def splits(self) -> list[str]:
        """Split values actually present (empty before `core/splits.py` runs)."""
        if SPLIT not in self._df.columns:
            return []
        return sorted(self._df[SPLIT].dropna().unique().tolist())

    @property
    def base_modes(self) -> pd.Series:
        """Per-row ``structure`` / ``signal`` / ``both`` / ``none``, from the payload."""
        has_struct = (
            _nonempty_text(self._df[STRUCTURE])
            if STRUCTURE in self._df.columns
            else _all_false(self._df)
        )
        has_signal = _all_false(self._df)
        for name in self.signal_names:
            has_signal = has_signal | _nonempty_seq(self._df[_signal_y_col(name)])
        modes = pd.Series(BASE_MODE_NONE, index=self._df.index, dtype=object)
        modes[has_signal] = BASE_MODE_SIGNAL
        modes[has_struct] = BASE_MODE_STRUCTURE
        modes[has_struct & has_signal] = BASE_MODE_BOTH
        return modes

    def __len__(self) -> int:
        return len(self._df)

    def __contains__(self, material_id: object) -> bool:
        return material_id in self._index

    def __repr__(self) -> str:
        return (
            f"MaterialRegistry(n={len(self)}, "
            f"modes={self.base_modes.value_counts().to_dict()}, "
            f"splits={self.splits or 'unassigned'}, targets={self.target_names})"
        )

    # ----------------------------------------------------------------- lookup

    @property
    def _index(self) -> dict[str, int]:
        """material_id -> row position, built once on first use."""
        if self._pos is None:
            self._pos = {mid: i for i, mid in enumerate(self._df[MATERIAL_ID])}
        return self._pos

    def _row(self, material_id: str) -> pd.Series:
        pos = self._index.get(material_id)
        if pos is None:
            raise RegistryError(f"{material_id!r} not in registry ({len(self)} rows)")
        return self._df.iloc[pos]

    def get(self, material_id: str) -> MaterialRecord:
        """Materialise one row as a `MaterialRecord`."""
        row = self._row(material_id)
        cif = row[STRUCTURE] if STRUCTURE in self._df.columns else None
        cif = cif if isinstance(cif, str) and cif.strip() else None
        signals: dict[str, tuple[np.ndarray | None, np.ndarray]] = {}
        for name in self.signal_names:
            y = _as_array(row[_signal_y_col(name)])
            if y is None:
                continue
            xcol = _signal_x_col(name)
            x = _as_array(row[xcol]) if xcol in self._df.columns else None
            signals[name] = (x, y)
        split = row[SPLIT] if SPLIT in self._df.columns else None
        if cif is not None and signals:
            mode = BASE_MODE_BOTH
        elif cif is not None:
            mode = BASE_MODE_STRUCTURE
        elif signals:
            mode = BASE_MODE_SIGNAL
        else:
            mode = BASE_MODE_NONE
        return MaterialRecord(
            material_id=material_id,
            split=None if pd.isna(split) else str(split),
            base_mode=mode,
            structure=cif,
            signals=signals,
            targets={n: row[TARGET_PREFIX + n] for n in self.target_names},
            flags={n: bool(row[n]) for n in self.flag_names if not pd.isna(row[n])},
            metadata={n: row[n] for n in self.metadata_names},
        )

    def cif(self, material_id: str) -> str:
        """CIF text for one material. Raises if the row has no structure."""
        if STRUCTURE not in self._df.columns:
            raise RegistryError(f"registry has no {STRUCTURE!r} column")
        val = self._row(material_id)[STRUCTURE]
        if not isinstance(val, str) or not val.strip():
            raise RegistryError(f"{material_id!r} has no structure")
        return val

    def signal(
        self, material_id: str, name: str | None = None
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """``(x, y)`` for one signal channel; ``x`` is None if there is no grid.

        `name` is optional when the registry has exactly one channel; required
        when it has several (e.g. CHILI ``xrd`` + ``xpdf``).
        """
        channels = self.signal_names
        if not channels:
            raise RegistryError("registry has no signal channels")
        if name is None:
            if len(channels) != 1:
                raise RegistryError(f"registry has multiple signals {channels}; pass a name")
            name = channels[0]
        elif name not in channels:
            raise RegistryError(f"no signal {name!r}; have {channels}")
        row = self._row(material_id)
        y = _as_array(row[_signal_y_col(name)])
        if y is None:
            raise RegistryError(f"{material_id!r} has no {name!r} signal")
        xcol = _signal_x_col(name)
        x = _as_array(row[xcol]) if xcol in self._df.columns else None
        return x, y

    def targets(self, names: Sequence[str] | None = None) -> pd.DataFrame:
        """Target columns indexed by `material_id`; names may omit ``target_``."""
        wanted = self.target_names if names is None else [_strip_target(n) for n in names]
        unknown = [n for n in wanted if n not in self.target_names]
        if unknown:
            raise RegistryError(f"unknown target(s) {unknown}; have {self.target_names}")
        out = self._df.set_index(MATERIAL_ID)[[TARGET_PREFIX + n for n in wanted]]
        return out.rename(columns={TARGET_PREFIX + n: n for n in wanted})

    # ------------------------------------------------------------------ query

    def split(self, name: str | Sequence[str]) -> MaterialRegistry:
        """Rows in the given split(s)."""
        if SPLIT not in self._df.columns:
            raise RegistryError(f"registry has no {SPLIT!r} column — run core/splits.py first")
        names = [name] if isinstance(name, str) else list(name)
        unknown = [n for n in names if n not in self.splits]
        if unknown:
            raise RegistryError(f"no such split(s) {unknown}; have {self.splits}")
        return MaterialRegistry(self._df[self._df[SPLIT].isin(names)].copy())

    def filter(self, **predicates: Any) -> MaterialRegistry:
        """Column filters: scalar (==), iterable (isin), or callable (elementwise)."""
        mask = pd.Series(True, index=self._df.index)
        for col, want in predicates.items():
            if col not in self._df.columns:
                raise RegistryError(f"no column {col!r}; have {self.columns}")
            series = self._df[col]
            if callable(want):
                mask &= series.map(want).astype(bool)
            elif isinstance(want, (list, tuple, set, frozenset, np.ndarray, pd.Series)):
                mask &= series.isin(list(want))
            else:
                mask &= series == want
        return MaterialRegistry(self._df[mask].copy())

    def with_split(self, assignments: Mapping[str, str] | str) -> MaterialRegistry:
        """Copy with `split` set. The write path for `core/splits.py`.

        Pass a ``{material_id: split}`` mapping, or one string for every row.
        Ids absent from the mapping are left unassigned (null).
        """
        df = self._df.copy()
        if isinstance(assignments, str):
            df[SPLIT] = assignments
            return MaterialRegistry(df)
        unknown = [m for m in assignments if m not in self._index]
        if unknown:
            raise RegistryError(
                f"{len(unknown)} id(s) in assignments not in registry, e.g. {unknown[:5]}"
            )
        df[SPLIT] = df[MATERIAL_ID].map(dict(assignments))
        return MaterialRegistry(df)

    # --------------------------------------------------------------- validate

    def validate(self) -> ValidationReport:
        """Schema-level checks only — no CIF parsing, no pymatgen, no physics.

        Errors: missing/duplicate/null `material_id`; no base payload column; rows
        with *neither* payload; non-string CIFs; malformed signals; non-boolean
        flags.  A row with *both* structure and signal is valid (e.g. CHILI).

        Warnings: empty registry; missing or unassigned `split` (a fresh builder
        output is legal input to `core/splits.py`); split names outside
        `VALID_SPLITS`; all-null target columns.
        """
        rep = ValidationReport(n_rows=len(self._df))
        df = self._df
        err, warn = rep.errors.append, rep.warnings.append

        # --- core key
        ids = df[MATERIAL_ID]
        if ids.isna().any():
            err(f"{int(ids.isna().sum())} null {MATERIAL_ID}")
        if ids.duplicated().any():
            dupes = ids[ids.duplicated()].unique()[:5].tolist()
            err(f"{int(ids.duplicated().sum())} duplicate {MATERIAL_ID}, e.g. {dupes}")
        if len(df) and not ids.dropna().map(lambda v: isinstance(v, str)).all():
            warn(f"{MATERIAL_ID} should be str-typed")
        if len(df) == 0:
            warn("registry is empty")

        # --- split
        if SPLIT not in df.columns:
            warn(f"no {SPLIT!r} column (assign it with core/splits.py)")
        else:
            n_null = int(df[SPLIT].isna().sum())
            if n_null:
                warn(f"{n_null} row(s) with no split assigned")
            odd = sorted(set(df[SPLIT].dropna().unique()) - set(VALID_SPLITS))
            if odd:
                warn(f"non-standard split value(s) {odd}; expected {list(VALID_SPLITS)}")

        # --- base payload: at least one per row
        signal_names = self.signal_names
        if STRUCTURE not in df.columns and not signal_names:
            err(f"no base payload column: need {STRUCTURE!r} or a {SIGNAL_PREFIX}<name>_y column")
        else:
            modes = self.base_modes
            n_none = int((modes == BASE_MODE_NONE).sum())
            if n_none:
                bad = df.loc[modes == BASE_MODE_NONE, MATERIAL_ID].head(5).tolist()
                err(f"{n_none} row(s) with neither structure nor signal, e.g. {bad}")

            if STRUCTURE in df.columns:
                nonstr = df[STRUCTURE].dropna().map(lambda v: not isinstance(v, str))
                if nonstr.any():
                    err(f"{int(nonstr.sum())} non-string value(s) in {STRUCTURE!r}")

            for name in signal_names:
                rep.errors.extend(_validate_signals(df, name))
                if _signal_x_col(name) not in df.columns:
                    warn(f"signal {name!r} has no {_signal_x_col(name)!r} column (x grid unknown)")

        # --- flags
        for col in self.flag_names:
            vals = df[col].dropna().unique().tolist()
            if not set(vals) <= {True, False, 0, 1}:
                err(f"flag column {col!r} has non-boolean values, e.g. {vals[:5]}")

        # --- targets
        for name in self.target_names:
            if int(df[TARGET_PREFIX + name].notna().sum()) == 0:
                warn(f"target {name!r} is entirely null")

        return rep


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _all_false(df: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=df.index)


def _nonempty_text(series: pd.Series) -> pd.Series:
    return series.map(lambda v: isinstance(v, str) and bool(v.strip()))


def _nonempty_seq(series: pd.Series) -> pd.Series:
    return series.map(lambda v: _as_array(v) is not None)


def _as_array(value: Any) -> np.ndarray | None:
    """Coerce a parquet list cell / list / ndarray to a float array; None if empty."""
    if isinstance(value, np.ndarray):
        return None if value.size == 0 else value.astype(float)
    if isinstance(value, (list, tuple, pd.Series)):
        arr = np.asarray(value, dtype=float)
        return None if arr.size == 0 else arr
    return None


def _strip_target(name: str) -> str:
    return name[len(TARGET_PREFIX):] if name.startswith(TARGET_PREFIX) else name


def _signal_y_col(name: str) -> str:
    return f"{SIGNAL_PREFIX}{name}{SIGNAL_Y_SUFFIX}"


def _signal_x_col(name: str) -> str:
    return f"{SIGNAL_PREFIX}{name}{SIGNAL_X_SUFFIX}"


def _signal_names(columns: Sequence[str]) -> list[str]:
    """Channel names from the ``signal_<name>_y`` payload columns."""
    lo, hi = len(SIGNAL_PREFIX), len(SIGNAL_Y_SUFFIX)
    return [
        c[lo:-hi]
        for c in columns
        if c.startswith(SIGNAL_PREFIX) and c.endswith(SIGNAL_Y_SUFFIX) and len(c) > lo + hi
    ]


def _validate_signals(df: pd.DataFrame, name: str) -> list[str]:
    """Per-row checks for one channel: finite values, x/y lengths agree."""
    bad_finite: list[str] = []
    bad_len: list[str] = []
    ys = df[_signal_y_col(name)]
    xcol = _signal_x_col(name)
    xs = df[xcol] if xcol in df.columns else None
    for i, mid in enumerate(df[MATERIAL_ID]):
        y = _as_array(ys.iloc[i])
        if y is None:
            continue
        if not np.isfinite(y).all():
            bad_finite.append(str(mid))
        if xs is not None:
            x = _as_array(xs.iloc[i])
            if x is not None and len(x) != len(y):
                bad_len.append(str(mid))
    errors = []
    if bad_finite:
        errors.append(f"signal {name!r}: {len(bad_finite)} non-finite, e.g. {bad_finite[:5]}")
    if bad_len:
        errors.append(f"signal {name!r}: {len(bad_len)} with len(x)!=len(y), e.g. {bad_len[:5]}")
    return errors
