"""core/losses.py — contrastive pretraining objectives (PDF two-view).

Two losses, both taking a pair of projected embeddings `(z1, z2)` of shape
`(B, D)` — one per augmented view of the same material (see
`core/dataset_base.py`) — and returning a scalar to minimize:

- `InfoNCELoss`  — NT-Xent: pull a view toward its partner, push away the other
  2B-2 in-batch views. Needs negatives, so it wants a reasonably large batch.
- `VICRegLoss`   — variance/invariance/covariance: no negatives, robust to small
  batches. Returns `(total, inv, var, cov)` so train.py can log the components.

Ported from PDF_XRD_Fusion/PDF/src/losses.py (reference, read-only).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["InfoNCELoss", "VICRegLoss"]


class InfoNCELoss(nn.Module):
    """NT-Xent contrastive loss over 2B views (SimCLR-style)."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.size(0)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        sim = torch.mm(z, z.T) / self.temperature  # (2B, 2B)
        sim.fill_diagonal_(float("-inf"))  # a view is never its own positive

        # view i's positive is view i+B (and vice versa)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])
        return F.cross_entropy(sim, labels)


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance regularization (Bardes et al. 2022).

    Returns `(total, inv, var, cov)`; only `total` is backpropagated, the rest
    are for logging.
    """

    def __init__(self, sim_weight: float = 25.0, var_weight: float = 25.0, cov_weight: float = 1.0) -> None:
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        B, D = z1.size()

        # invariance: the two views should map to the same point
        inv_loss = F.mse_loss(z1, z2)

        # variance: keep each dim's std >= 1 across the batch (anti-collapse)
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean()

        # covariance: decorrelate dims (off-diagonal of the feature covariance)
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (B - 1)
        cov_z2 = (z2.T @ z2) / (B - 1)
        cov_loss = (cov_z1.fill_diagonal_(0).pow(2).sum() +
                    cov_z2.fill_diagonal_(0).pow(2).sum()) / D

        total = self.sim_weight * inv_loss + self.var_weight * var_loss + self.cov_weight * cov_loss
        return total, inv_loss, var_loss, cov_loss
