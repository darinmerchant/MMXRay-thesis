"""core/train.py — SSL contrastive PDF pretraining loop.

    python -m core.train --config configs/pretrain/cnn_vicreg.yaml
    python -m core.train --resume runs/pdf/sweep/2026-07-28_082017_cnn_vicreg_cov10_5bfa372

Wires a two-view pretraining dataset into an encoder + contrastive loss, and
writes one self-contained run dir under the config's `out_dir` — see DESIGN.md -> Run tracking
& experiment layout for the full design. `bank:` in the config picks which
dataset: set -> `BankedPDFPretrainDataset` (2 of N pre-simulated views, the
production path), unset -> `PDFPretrainDataset` (live diffpy per `__getitem__`).
Both expose the same tuple + `set_epoch`, so nothing below cares which it got.

Two footguns this file exists to get right (see core/dataset_base.py,
core/encoders.py):
  - the dataset yields `(L,)` -> batched `(B, L)`; encoders want `(B, 1, L)`,
    so every batch is `unsqueeze(1)`-ed before the model.
  - `persistent_workers=False` is required: `set_epoch` only reaches worker
    processes if they're re-forked each epoch, which is exactly what a
    non-persistent DataLoader does on every `for batch in loader:` — so
    `set_epoch` is called right before that loop, every epoch.

RESUME (`--resume <run_dir>`) continues a run IN PLACE — same dir, appended
log/metrics — from `ckpt_last.pt`, restoring model/optimizer/scheduler/scaler.
It takes **no `--config`**: the run dir's own `config.yaml` is the only config,
so a resumed run cannot silently become a different experiment. Three
consequences worth knowing:

  - **Every epoch reseeds globally** (`set_seed(seed + epoch)`), so epoch E's
    shuffle order and dropout depend only on `(seed, E)` and not on how the run
    got there. That is what makes a resumed epoch identical to the uninterrupted
    one; snapshotting RNG state instead would also have to survive a change of
    device and worker count. The augmentation views were already seeded this way
    (`core/dataset_base.py`), so this just extends the same rule to the loop.
  - **`epochs` cannot be raised to extend a finished run.** The cosine schedule
    is defined over `epochs * steps_per_epoch`; a longer run is a different LR
    curve, i.e. a different experiment, and belongs in a new run dir.
  - **Resuming an already-finished run is a no-op, not an error** — a requeued
    SLURM job that lost its race must exit 0.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from core.config import REPO_ROOT, TrainConfig, git_info, load_config, resolve_config
from core.dataset_base import BankedPDFPretrainDataset, PDFPretrainDataset
from core.encoders import CNNEncoder, Transformer
from core.losses import InfoNCELoss, VICRegLoss
from core.registry import MaterialRegistry

# LR schedule shape (not exposed as config knobs — nobody has needed to tune
# these yet; see PROGRESS.md if that changes).
LR_START_FACTOR = 1e-3  # warmup starts at this fraction of the target LR
LR_ETA_MIN = 1e-5  # cosine floor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    """"auto" = cuda, else cpu — MPS is deliberately excluded from auto-selection.

    `clip_grad_norm_` returns NaN on the MPS backend (torch 2.2.2, confirmed via
    tests/test_train.py: identical run on cpu gives a finite norm, on mps a NaN,
    with the forward loss itself finite in both — an MPS reduction-op bug, not a
    loop bug). Only cuda/cpu are validated (environment.yml: "CPU build... enough
    for tests"; SuperCloud training is cuda). Pass --device mps explicitly to opt
    in anyway.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.encoder == "cnn":
        return CNNEncoder(latent_dim=cfg.latent_dim, proj_dim=cfg.proj_dim)
    if cfg.encoder == "transformer":
        return Transformer(
            proj_dim=cfg.proj_dim,
            signal_len=cfg.signal_len,
            d_model=cfg.d_model,
            dim_feedforward=cfg.dim_feedforward,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            patch_size=cfg.patch_size,
            stride=cfg.stride,
            dropout=cfg.dropout,
            norm_first=cfg.norm_first,
        )
    raise ValueError(f"unknown encoder {cfg.encoder!r}; want 'cnn' or 'transformer'")


def build_loss(cfg: TrainConfig) -> nn.Module:
    if cfg.loss == "vicreg":
        return VICRegLoss(sim_weight=cfg.sim_weight, var_weight=cfg.var_weight, cov_weight=cfg.cov_weight)
    if cfg.loss == "infonce":
        return InfoNCELoss(temperature=cfg.temperature)
    raise ValueError(f"unknown loss {cfg.loss!r}; want 'vicreg' or 'infonce'")


def lr_schedule(warmup_steps: int, total_steps: int, eta_min_ratio: float):
    """Linear warmup -> cosine decay, stepped per iteration (not per epoch)."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return LR_START_FACTOR + (1.0 - LR_START_FACTOR) * (step / max(1, warmup_steps))
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return lr_lambda


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    loss_name: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    grad_clip: float,
    amp: bool,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "inv_loss": 0.0, "var_loss": 0.0, "cov_loss": 0.0, "grad_norm": 0.0}
    n_examples = 0
    n_batches = 0
    # AMP overflow accounting. A single non-finite batch poisons the epoch's mean
    # grad_norm, which is why NaN shows up in metrics.jsonl; n_skipped says whether
    # the optimizer actually skipped that step (GradScaler drops the scale when it
    # detects inf/NaN and does not apply the update).
    n_skipped = 0

    # torch.autocast validates device_type even when enabled=False, and "mps" isn't
    # a supported autocast type — amp is only ever True on cuda anyway, so any
    # supported placeholder is fine when it's off.
    autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"

    for v1, v2, *_ in loader:
        v1 = v1.unsqueeze(1).to(device, non_blocking=True)
        v2 = v2.unsqueeze(1).to(device, non_blocking=True)

        with torch.autocast(device_type=autocast_device, enabled=amp):
            z1, _ = model(v1)
            z2, _ = model(v2)
            if loss_name == "vicreg":
                loss, inv_loss, var_loss, cov_loss = criterion(z1, z2)
            else:
                loss = criterion(z1, z2)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # grads back to fp32 real scale before clipping/measuring
        max_norm = grad_clip if grad_clip > 0 else float("inf")  # inf clips nothing, still returns the norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:  # scale only drops on a detected overflow
            n_skipped += 1
        scheduler.step()  # per-iteration, not per-epoch

        bs = v1.size(0)
        totals["loss"] += loss.item() * bs
        totals["grad_norm"] += float(grad_norm)
        if loss_name == "vicreg":
            totals["inv_loss"] += inv_loss.item() * bs
            totals["var_loss"] += var_loss.item() * bs
            totals["cov_loss"] += cov_loss.item() * bs
        n_examples += bs
        n_batches += 1

    metrics = {
        "loss": totals["loss"] / n_examples,
        "grad_norm": totals["grad_norm"] / n_batches,
        "n_skipped": n_skipped,
        "amp_scale": scaler.get_scale(),
    }
    if loss_name == "vicreg":
        metrics["inv_loss"] = totals["inv_loss"] / n_examples
        metrics["var_loss"] = totals["var_loss"] / n_examples
        metrics["cov_loss"] = totals["cov_loss"] / n_examples
    return metrics


def make_run_dir(cfg: TrainConfig, config_path: str | Path, resolved: dict) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config_stem = Path(config_path).stem
    run_id = f"{timestamp}_{config_stem}_{resolved['git_sha']}"
    run_dir = REPO_ROOT / cfg.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # loud error on collision, never silently overwrite a run
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    return run_dir


def config_from_run(run_dir: str | Path) -> TrainConfig:
    """The `TrainConfig` a run was launched with, read back from its `config.yaml`.

    `resolve_config` adds git provenance keys that are not dataclass fields, so
    they are dropped here rather than blowing up `TrainConfig.__init__`.
    """
    resolved = yaml.safe_load((Path(run_dir) / "config.yaml").read_text())
    names = {f.name for f in fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in resolved.items() if k in names})


def _rewind_metrics(path: Path, last_epoch: int) -> tuple[float, int]:
    """Trim `metrics.jsonl` to `last_epoch` -> `(best loss so far, rows dropped)`.

    Two jobs, one pass, because they share a precondition. A run killed between
    the metrics append and the checkpoint save leaves a row for an epoch that is
    about to be re-run, so those rows are dropped first; what remains is exactly
    the set of epochs the resumed run inherits, and its minimum loss is the bar
    `ckpt_best.pt` must beat.

    Deliberately NOT read from `ckpt_best.pt`: that file can lag `ckpt_last.pt`
    by an epoch (last is written first), so its recorded loss is not always the
    true minimum. `metrics.jsonl` is the run's own complete record.
    """
    if not path.exists():
        return float("inf"), 0
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    keep = [r for r in rows if r["epoch"] <= last_epoch]
    if len(keep) != len(rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in keep))
    best = min((r["loss"] for r in keep), default=float("inf"))
    return float(best), len(rows) - len(keep)


def train(
    cfg: TrainConfig,
    config_path: str | Path | None = None,
    *,
    resume_from: str | Path | None = None,
) -> Path:
    if (config_path is None) == (resume_from is None):
        raise ValueError("pass exactly one of config_path (fresh run) or resume_from (continue)")

    set_seed(cfg.seed)
    if resume_from is None:
        resolved = resolve_config(cfg)
        run_dir = make_run_dir(cfg, config_path, resolved)
        ckpt = None
    else:
        run_dir = Path(resume_from)
        ckpt_path = run_dir / "ckpt_last.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{ckpt_path} — nothing to resume from")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Keep the ORIGINAL provenance so every checkpoint in this run dir carries
        # the same config; where the resume happened from is logged instead.
        resolved = ckpt["config"]
    log_path = run_dir / "log.txt"

    def log(msg: str) -> None:
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    device = pick_device(cfg.device)
    amp = cfg.amp and device.type == "cuda"

    if cfg.bank is not None:
        dataset = BankedPDFPretrainDataset(cfg.bank, base_seed=cfg.seed, limit=cfg.limit,
                                           extra_noise_c=cfg.extra_noise_c)
        # The Transformer sizes its positional embedding from `signal_len` but indexes
        # it by the ACTUAL token count, so a SHORTER bank grid trains silently against
        # a prefix of that embedding — and the wrong `signal_len` then travels in the
        # checkpoint to `analysis/finetune.py`, which builds its probe grid from it.
        if dataset.grid_len != cfg.signal_len:
            raise ValueError(
                f"bank grid is {dataset.grid_len} points but signal_len={cfg.signal_len} — "
                f"they must match; re-bank or fix the config"
            )
        source = f"bank:{Path(cfg.bank).name}"
    else:
        registry = MaterialRegistry.load(cfg.registry)
        if cfg.limit is not None:
            registry = MaterialRegistry(registry.frame.iloc[: cfg.limit].copy())
        dataset = PDFPretrainDataset(registry, base_seed=cfg.seed)
        source = f"sim:{Path(cfg.registry).name}"
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,  # BatchNorm in the proj head needs a full-size batch, every batch
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,  # required for set_epoch to reach workers — see module docstring
    )
    if len(loader) == 0:
        raise ValueError(f"0 batches: {len(dataset)} materials, batch_size={cfg.batch_size}, drop_last=True")

    log(
        f"device={device} amp={amp} encoder={cfg.encoder} loss={cfg.loss} data={source} "
        f"materials={len(dataset)} batches/epoch={len(loader)} run_dir={run_dir}"
    )

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # torch.amp.GradScaler (the non-deprecated spelling) doesn't exist in our
    # pinned torch==2.2.2 — this is the correct API for this version, not a lint miss.
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    if cfg.epochs <= cfg.warmup_epochs:
        raise ValueError(f"epochs ({cfg.epochs}) must be greater than warmup_epochs ({cfg.warmup_epochs})")
    steps_per_epoch = len(loader)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_schedule(warmup_steps, total_steps, LR_ETA_MIN / cfg.lr)
    )

    start_epoch = 1
    best_loss = float("inf")
    if ckpt is not None:
        # Order matters: optimizer state before the scheduler that references it,
        # and the model is already on `device` so loaded optimizer state lands there.
        banked = int(ckpt.get("steps_per_epoch", steps_per_epoch))
        if banked != steps_per_epoch:
            raise ValueError(
                f"{run_dir.name}: was trained at {banked} steps/epoch but this dataset gives "
                f"{steps_per_epoch} — the LR schedule is defined over total steps, so the "
                f"underlying data changed and the schedule can no longer be continued"
            )
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_loss, dropped = _rewind_metrics(run_dir / "metrics.jsonl", int(ckpt["epoch"]))
        log(
            f"resumed from {run_dir.name}/ckpt_last.pt at epoch {ckpt['epoch']} "
            f"(best_loss={best_loss:.4f}, now at git {git_info()['git_sha']})"
            + (f" — dropped {dropped} metrics row(s) past the checkpoint" if dropped else "")
        )
        if start_epoch > cfg.epochs:
            # Not an error: a requeued SLURM job that lost its race must exit 0.
            log(f"already complete ({ckpt['epoch']}/{cfg.epochs} epochs) — nothing to do")
            return run_dir

    wandb_run = None
    if cfg.wandb:
        import os

        import wandb

        os.environ.setdefault("WANDB_MODE", "offline")  # SuperCloud has no internet to stream to
        wandb_run = wandb.init(project=cfg.wandb_project, name=run_dir.name, config=resolved, dir=str(run_dir))

    metrics_path = run_dir / "metrics.jsonl"

    for epoch in range(start_epoch, cfg.epochs + 1):
        # Reseed per epoch so this epoch's shuffle order and dropout depend only on
        # (seed, epoch) — that is what makes a resumed epoch identical to the
        # uninterrupted one, without snapshotting RNG state. See module docstring.
        set_seed(cfg.seed + epoch)
        dataset.set_epoch(epoch)  # before the loop: non-persistent workers re-fork here, seeing the new epoch
        metrics = train_one_epoch(
            model, loader, criterion, cfg.loss, optimizer, scheduler, scaler, device,
            grad_clip=cfg.grad_clip, amp=amp,
        )
        log_dict = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **metrics}

        with open(metrics_path, "a") as f:
            f.write(json.dumps(log_dict) + "\n")
        if wandb_run is not None:
            wandb_run.log(log_dict, step=epoch)

        log(f"epoch {epoch:4d}/{cfg.epochs} | loss {metrics['loss']:.4f} | lr {log_dict['lr']:.2e} | grad_norm {metrics['grad_norm']:.3f}")

        # Atomic: a job killed mid-write must not leave a truncated ckpt_last.pt,
        # which is the one file --resume depends on.
        tmp_ckpt = run_dir / "ckpt_last.pt.tmp"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "steps_per_epoch": steps_per_epoch,  # --resume checks the schedule still lines up
                "config": resolved,
                "metrics": metrics,
            },
            tmp_ckpt,
        )
        tmp_ckpt.replace(run_dir / "ckpt_last.pt")

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "config": resolved, "metrics": metrics},
                run_dir / "ckpt_best.pt",
            )
            log(f"  -> new best (loss={best_loss:.4f})")

    if wandb_run is not None:
        wandb_run.finish()
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--config", help="path to a configs/pretrain/*.yaml — starts a new run")
    start.add_argument("--resume", help="path to an existing runs/<run_id> — continues it in place")
    parser.add_argument("--limit", type=int, default=None, help="override cfg.limit — quick smoke test")
    parser.add_argument("--wandb", action="store_true", help="override cfg.wandb to True")
    args = parser.parse_args()

    if args.resume:
        # No overrides on resume: the run dir's config.yaml is the only config, which
        # is what makes it impossible for a resumed run to become a different one.
        if args.limit is not None or args.wandb:
            parser.error("--limit/--wandb cannot be combined with --resume; edit the run's config.yaml")
        train(config_from_run(args.resume), resume_from=args.resume)
    else:
        cfg = load_config(args.config)
        if args.limit is not None:
            cfg.limit = args.limit
        if args.wandb:
            cfg.wandb = True
        train(cfg, args.config)
