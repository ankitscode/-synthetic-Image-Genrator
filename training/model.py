"""MobileFaceNet (Chen et al. 2018) + ArcFace margin head, written here and trained from scratch
(no pretrained weights: InsightFace models/weights are not allowed, CLAUDE.md).

Input: RGB 112x112, normalised to [-1, 1]. Output: 128-d embedding (compare with cosine similarity).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

EMBEDDING_DIM = 128


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 1, s: int = 1, p: int = 0, groups: int = 1, act: bool = True):
        layers = [nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False), nn.BatchNorm2d(cout)]
        if act:
            layers.append(nn.PReLU(cout))
        super().__init__(*layers)


class Bottleneck(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, expansion: int):
        super().__init__()
        hidden = cin * expansion
        self.residual = stride == 1 and cin == cout
        self.body = nn.Sequential(
            ConvBlock(cin, hidden),
            ConvBlock(hidden, hidden, 3, stride, 1, groups=hidden),
            ConvBlock(hidden, cout, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) if self.residual else self.body(x)


# (expansion, out channels, repeats, first stride), as in the MobileFaceNet paper
SETTINGS = [(2, 64, 5, 2), (4, 128, 1, 2), (2, 128, 6, 1), (4, 128, 1, 2), (2, 128, 2, 1)]


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        layers: list[nn.Module] = [ConvBlock(3, 64, 3, 2, 1), ConvBlock(64, 64, 3, 1, 1, groups=64)]
        cin = 64
        for t, c, n, s in SETTINGS:
            for i in range(n):
                layers.append(Bottleneck(cin, c, s if i == 0 else 1, t))
                cin = c
        layers += [ConvBlock(cin, 512),
                   ConvBlock(512, 512, 7, 1, 0, groups=512, act=False),   # global depthwise conv (7x7 -> 1x1)
                   nn.Flatten(), nn.Dropout(0.1),
                   nn.Linear(512, embedding_dim, bias=False), nn.BatchNorm1d(embedding_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ArcFace(nn.Module):
    """Additive angular margin: logit of the true class = s * cos(theta + m)."""

    def __init__(self, embedding_dim: int, classes: int, scale: float = 32.0, margin: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale, self.margin = scale, margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, margin: float | None = None) -> torch.Tensor:
        m = self.margin if margin is None else margin
        cos = F.linear(F.normalize(embeddings.float()), F.normalize(self.weight.float())).clamp(-1 + 1e-7, 1 - 1e-7)
        target = cos.gather(1, labels[:, None])
        sin = (1 - target**2).sqrt()
        with_margin = target * math.cos(m) - sin * math.sin(m)
        # past theta = pi - m, cos(theta + m) is no longer monotonic: use the CosFace-style fallback
        with_margin = torch.where(target > math.cos(math.pi - m), with_margin, target - math.sin(math.pi - m) * m)
        logits = cos.scatter(1, labels[:, None], with_margin)
        return logits * self.scale


def normalise(batch_uint8: torch.Tensor) -> torch.Tensor:
    """(B, 112, 112, 3) uint8 -> (B, 3, 112, 112) float in [-1, 1]."""
    return batch_uint8.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
