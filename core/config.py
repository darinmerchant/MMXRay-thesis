"""core/config.py — the front door to `core/train.py`: YAML -> `TrainConfig`.

One flat dataclass (not nested per-encoder/per-loss configs) — an experiment
picks `encoder`/`loss` and only the matching fields apply; the rest sit at
their defaults unused. Flat is easier to read end-to-end and mirrors how the
old repo's argparse namespace worked, without needing a CLI flag per field.

`configs/pretrain/*.yaml` supplies overrides on top of the dataclass defaults;
unknown YAML keys raise `TypeError` (typo protection, for free from
`dataclass.__init__`). `resolve_config` adds git provenance for the run dir —
recorded, not enforced (see DESIGN.md -> Run tracking & experiment layout;
supersedes the old repo's `enforce_clean_tree`).
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["TrainConfig", "load_config", "resolve_config", "git_info", "REPO_ROOT"]

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    # --- data ---
    registry: str = "data/registry/material_registry.parquet"
    # Path to a banked block (tools/bank_pdf.py). Set -> read 2 of N pre-simulated
    # views per epoch (the production path); unset -> simulate both views live.
    # `registry` is unused when this is set: the block carries its own id list.
    bank: str | None = None
    limit: int | None = None  # cap #materials — smoke tests only
    signal_len: int = 5000  # our PDF grid (0-50 A / 0.01); Transformer's pos-embedding needs it
    # Extra measurement noise layered on a banked view at READ time, as (lo, hi)
    # for noise_c (std as a fraction of the row max — the simulator's own units).
    # None = off, and off is the state every run before 2026-08-12 was trained in.
    # Exists because the banked range (0.001, 0.005) covers only the clean end of
    # measured RRUFF, whose noise_c runs to 0.065 (RESULTS.md -> Q8 noise arm).
    # Re-banking to widen it would cost ~150 core-hours; noise is the one post-op
    # that composes correctly with a banked 1-D pattern (DESIGN.md -> XRD decisions).
    extra_noise_c: tuple[float, float] | None = None

    # --- encoder ---
    encoder: str = "cnn"  # "cnn" | "transformer"
    latent_dim: int = 256
    proj_dim: int = 128
    # transformer-only
    d_model: int = 256
    dim_feedforward: int = 1024
    nhead: int = 4
    num_layers: int = 4
    patch_size: int = 50
    stride: int = 25
    dropout: float = 0.1
    norm_first: bool = False

    # --- loss ---
    loss: str = "vicreg"  # "vicreg" | "infonce"
    sim_weight: float = 25.0
    var_weight: float = 25.0
    cov_weight: float = 1.0
    temperature: float = 0.07

    # --- optimization ---
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1.4e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 8
    grad_clip: float = 5.0  # 0 disables clipping (norm is still logged)
    num_workers: int = 8
    seed: int = 42

    # --- run ---
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    amp: bool = True  # only actually enabled when device resolves to cuda
    wandb: bool = False  # off by default — SuperCloud has no internet to stream to
    wandb_project: str = "mmxray-pretrain"
    out_dir: str = "runs/pdf/sweep"


def load_config(path: str | Path) -> TrainConfig:
    """Read a `configs/pretrain/*.yaml` as overrides on `TrainConfig` defaults."""
    with open(path) as f:
        overrides: dict[str, Any] = yaml.safe_load(f) or {}
    return TrainConfig(**overrides)


def git_info() -> dict[str, Any]:
    """`{git_sha, dirty}` for the current tree — recorded into every run dir.

    No `enforce_clean_tree` gate: a dirty tree is allowed, just labeled, so a
    run is never blocked, only traceable.
    """

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    sha = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {"git_sha": sha, "dirty": dirty}


def resolve_config(cfg: TrainConfig) -> dict[str, Any]:
    """The fully-resolved config written to `runs/<run>/config.yaml`."""
    return {**asdict(cfg), **git_info()}
