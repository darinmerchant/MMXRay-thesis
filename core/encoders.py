"""core/encoders.py — 1D signal encoders for PDF/XRD pretraining.

Both encoders map a batched 1D signal `(B, 1, L)` -> `(z, h)`:

- `h` is the representation (what a downstream linear probe / finetune reads),
- `z` is `h` pushed through a projection head (what the contrastive loss sees).

`use_proj=False` returns `(h, h)` — e.g. for evaluation where the projection
head is discarded.

Our PDF grid is 0-50 A at 0.01 -> L = 5000 (core/simulate.py). `CNNEncoder` is
length-agnostic (softmax attention pool over the sequence); `Transformer` bakes
the length into its positional embedding, so it takes `signal_len` (default 5000).

Ported from PDF_XRD_Fusion/PDF/src/models.py (reference, read-only).
Checkpoint reconstruct helper is deferred to the linear-probe/finetune session
(train.py only needs to construct these and save a state_dict).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CNNEncoder", "Transformer"]


def _proj_head(in_dim: int, proj_dim: int) -> nn.Sequential:
    """Standard 3-layer BN-MLP projection head (SimCLR/VICReg style)."""
    return nn.Sequential(
        nn.Linear(in_dim, proj_dim),
        nn.BatchNorm1d(proj_dim),
        nn.ReLU(inplace=True),
        nn.Linear(proj_dim, proj_dim),
        nn.BatchNorm1d(proj_dim),
        nn.ReLU(inplace=True),
        nn.Linear(proj_dim, proj_dim),
    )


class ResBlock1d(nn.Module):
    """Two 3-wide convs + a (projected when needed) skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 1x1 projection only when shape changes (stride>1 or channel change)
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = self.skip(x) if self.skip is not None else x
        return F.relu(out + identity)


class CNNEncoder(nn.Module):
    """1D ResNet + attention pooling. Input `(B, 1, L)` -> `(z, h)`.

    Length-agnostic: the softmax attention pool collapses the sequence dim, so
    any L works (shapes below annotate L=5000).
    """

    def __init__(self, latent_dim: int = 256, proj_dim: int = 128, use_proj: bool = True) -> None:
        super().__init__()
        self.use_proj = use_proj

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),  # (B, 32, 2500)
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),                  # (B, 32, 1250)
        )
        self.layer1 = ResBlock1d(32, 64, stride=2)          # (B, 64, 625)
        self.layer2 = ResBlock1d(64, 128, stride=2)         # (B, 128, 313)
        self.layer3 = ResBlock1d(128, 256, stride=2)        # (B, 256, 157)
        self.layer4 = ResBlock1d(256, latent_dim, stride=2)  # (B, latent_dim, 79)

        self.pool_attn = nn.Linear(latent_dim, 1)  # per-position attention logit
        self.proj_head = _proj_head(latent_dim, proj_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.transpose(1, 2)          # (B, L', latent_dim)
        w = torch.softmax(self.pool_attn(x), dim=1)  # (B, L', 1)
        return (w * x).sum(dim=1)      # (B, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        z = self.proj_head(h) if self.use_proj else h
        return z, h


class Transformer(nn.Module):
    """Patch-embed transformer. Input `(B, 1, L)` -> `(z, h)`.

    Non-overlapping-ish patches of `patch_size` at `stride`, a learned CLS token,
    learned positional embedding. `signal_len` fixes the token count, so it must
    match the input L (default 5000, our PDF grid).
    """

    def __init__(
        self,
        proj_dim: int = 128,
        use_proj: bool = True,
        *,
        signal_len: int = 5000,
        d_model: int = 256,
        dim_feedforward: int = 1024,
        nhead: int = 4,
        num_layers: int = 4,
        patch_size: int = 50,
        stride: int = 25,
        dropout: float = 0.1,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.use_proj = use_proj
        self.patch_size = patch_size
        self.stride = stride

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        n_tokens = (signal_len - patch_size) // stride + 1 + 1  # +1 patch count, +1 CLS
        self.pos_embedding = nn.Embedding(n_tokens, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, norm_first=norm_first, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = _proj_head(d_model, proj_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)                                  # (B, L)
        x = x.unfold(1, self.patch_size, self.stride)     # (B, num_patches, patch_size)
        x = self.patch_embed(x)                           # (B, num_patches, d_model)

        cls = self.cls_token.expand(x.size(0), -1, -1)    # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                    # (B, 1 + num_patches, d_model)

        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_embedding(positions)

        x = self.transformer(x)
        return self.norm(x[:, 0, :])                      # read CLS -> (B, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        z = self.proj_head(h) if self.use_proj else h
        return z, h
