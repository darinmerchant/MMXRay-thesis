"""core/simulate.py — modality-agnostic CIF -> signal dispatch.

The thin seam between a MaterialRegistry row and a physics forward model. This
module knows NOTHING about Debye sums or peak profiles: it maps a modality name
(``"pdf"``, ``"xrd"``, ...) to a backend registered by
``modalities/<name>/simulate.py`` and calls it with the record's CIF. Backends
are pure functions

    (cif: str, **params) -> (x: np.ndarray, y: np.ndarray)

and are imported LAZILY — the registry, splits, and builders stay importable in
an environment without diffpy (or any other heavy simulator) installed; the
backend is only imported the first time its modality is actually simulated.

Param *sampling* (the simulator-in-the-loop augmentation) is deliberately NOT
here: `simulate` passes ``params`` straight through, so a backend is a pure
deterministic function of (cif, params). The jitter that draws those params
lives in core/transforms.py (see DESIGN.md -> Build order).

    from core.simulate import simulate
    r, g = simulate(reg.get("mp-1"), "pdf", qmax=25.0)
"""

from __future__ import annotations

import importlib
from typing import Callable

import numpy as np

from core.registry import MaterialRecord

__all__ = ["register", "simulate", "available", "SimulateError"]

Backend = Callable[..., "tuple[np.ndarray, np.ndarray]"]
_BACKENDS: dict[str, Backend] = {}


class SimulateError(RuntimeError):
    """No backend for a modality, or a backend module that failed to register one."""


def register(modality: str, fn: Backend) -> None:
    """Bind a modality name to its forward-model backend.

    Called at import time by each ``modalities/<name>/simulate.py``; the last
    registration for a name wins.
    """
    _BACKENDS[modality] = fn


def _backend(modality: str) -> Backend:
    """Return the backend for `modality`, importing its module on first use."""
    if modality not in _BACKENDS:
        try:
            importlib.import_module(f"modalities.{modality}.simulate")
        except ModuleNotFoundError as e:
            raise SimulateError(
                f"no simulate backend for modality {modality!r} "
                f"(expected modalities/{modality}/simulate.py)"
            ) from e
    if modality not in _BACKENDS:
        raise SimulateError(
            f"modalities.{modality}.simulate did not register a backend for {modality!r}"
        )
    return _BACKENDS[modality]


def simulate(record: MaterialRecord, modality: str, **params) -> tuple[np.ndarray, np.ndarray]:
    """Simulate `modality` for one registry record -> (x, y) arrays.

    ``record.cif`` is the input structure; ``params`` are forward-model settings
    passed straight to the backend (e.g. ``qmax``, ``rmin``/``rmax``/``rstep``
    for pdf). Sampling/jitter of those params is NOT done here.
    """
    return _backend(modality)(record.cif, **params)


def available() -> list[str]:
    """Modality names whose backend is already registered (imported) this session."""
    return sorted(_BACKENDS)
