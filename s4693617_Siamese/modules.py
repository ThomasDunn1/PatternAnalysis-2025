# modules.py
# Minimal Siamese-style encoder with a pretrained ResNet-50 backbone.
# PyTorch >= 2.0, torchvision >= 0.15 recommended.

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _resnet50_backbone(pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Return a ResNet-50 feature extractor (everything up to global pool) and its feature dim.
    Output shape (per image) after backbone+pool: [C] with C=2048.
    """
    # For modern torchvision: weights="DEFAULT" gives ImageNet-1k pretrained weights.
    net = models.resnet50(weights="DEFAULT" if pretrained else None)
    # Drop the final FC layer; keep avgpool. Children = [conv1,bn1,relu,maxpool,layer1..4,avgpool,fc]
    body = nn.Sequential(*(list(net.children())[:-1]))  # -> [B, 2048, 1, 1]
    feat_dim = 2048
    return body, feat_dim


class ProjectionHead(nn.Module):
    """
    Small MLP to map backbone features -> embedding_dim, then L2-normalize.
    """
    def __init__(self, in_dim: int, embedding_dim: int = 256, p_drop: float = 0.2):
        super().__init__()
        hidden = max(512, embedding_dim)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)                # [B, D]
        z = F.normalize(z, p=2, dim=1) # unit-norm embeddings
        return z


class SiameseEncoder(nn.Module):
    """
    Pretrained CNN backbone + projection head → normalized embeddings.

    Args
    ----
    embedding_dim : int
        Size of the output embedding (default 256).
    pretrained : bool
        If True, load ImageNet weights for the backbone.
    freeze_backbone : bool
        If True, freeze backbone weights (head still trains). Handy for quick sanity runs.
    """
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.backbone, in_dim = _resnet50_backbone(pretrained=pretrained)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = ProjectionHead(in_dim=in_dim, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # [B, 2048, 1, 1]
        z = self.head(feats)      # [B, embedding_dim], L2-normalized
        return z
