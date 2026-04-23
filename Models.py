"""
Models.py  —  Updated backbone that exposes intermediate layer outputs
======================================================================
KEY CHANGE from previous version:
  Before: backbone was an nn.Sequential that returned only the final
          layer4 output → [B, 2048, 7, 7]

  Now:    backbone is a custom module that returns THREE feature maps:
          layer2 → [B, 512,  56, 56]
          layer3 → [B, 1024, 28, 28]
          layer4 → [B, 2048,  7,  7]

  The decoder needs all three to fuse them.

WHY nn.Sequential doesn't work here:
  nn.Sequential runs layers one after another and only returns the
  LAST output. We need outputs from the MIDDLE of the network too.
  So we write a custom forward() that captures intermediate activations.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetBackbone(nn.Module):
    """
    ResNet backbone that exposes intermediate feature maps.
    Only layer4 has requires_grad=True. Everything else is frozen.
    """

    def __init__(self, resnet_model):
        super().__init__()
        m = resnet_model

        # Store each stage as a named submodule
        self.stem    = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1  = m.layer1   # output: [B, 256,  56, 56]
        self.layer2  = m.layer2   # output: [B, 512,  28, 28]  ← wait, see note
        self.layer3  = m.layer3   # output: [B, 1024, 14, 14]
        self.layer4  = m.layer4   # output: [B, 2048,  7,  7]

        # NOTE on spatial sizes (for 224×224 input):
        #   stem    : stride 2 (conv1) + stride 2 (maxpool) → 56×56
        #   layer1  : stride 1  → 56×56   channels: 256
        #   layer2  : stride 2  → 28×28   channels: 512
        #   layer3  : stride 2  → 14×14   channels: 1024
        #   layer4  : stride 2  →  7×7    channels: 2048
        #
        # We tap layer2 (28×28), layer3 (14×14), layer4 (7×7)
        # for the decoder. Updated channel numbers above.

        self.feat2_channels = 512
        self.feat3_channels = 1024
        self.feat4_channels = 2048

    def forward(self, x):
        """
        Returns
        -------
        feat2 : [B, 512,  28, 28]
        feat3 : [B, 1024, 14, 14]
        feat4 : [B, 2048,  7,  7]
        """
        x = self.stem(x)     # [B, 64,  56, 56]
        x = self.layer1(x)   # [B, 256, 56, 56]
        feat2 = self.layer2(x)    # [B, 512,  28, 28]
        feat3 = self.layer3(feat2)  # [B, 1024, 14, 14]
        feat4 = self.layer4(feat3)  # [B, 2048,  7,  7]
        return feat2, feat3, feat4


def load_backbone(backbone_name: str):
    """
    Load a pretrained ResNet backbone.
    Freezes all layers except layer4.

    Returns
    -------
    backbone    : ResNetBackbone — call backbone(x) → (feat2, feat3, feat4)
    feat_dims   : dict with keys 'feat2', 'feat3', 'feat4'
    """
    name = backbone_name.lower().strip()

    def _load(ctor):
        try:    return ctor(weights="IMAGENET1K_V1")
        except: return ctor(pretrained=True)

    ctor_map = {
        "resnet18":  models.resnet18,
        "resnet34":  models.resnet34,
        "resnet50":  models.resnet50,
        "resnet101": models.resnet101,
    }
    if name not in ctor_map:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    m = _load(ctor_map[name])

    # Freeze everything first
    for param in m.parameters():
        param.requires_grad = False

    # Unfreeze only layer4
    for param in m.layer4.parameters():
        param.requires_grad = True

    backbone = ResNetBackbone(m)

    # Sanity check
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in backbone.parameters())
    print(f"[Models] {backbone_name}: trainable {trainable:,} / {total:,} "
          f"params ({100*trainable/total:.1f}%)")

    feat_dims = {
        "feat2": backbone.feat2_channels,   # 512
        "feat3": backbone.feat3_channels,   # 1024
        "feat4": backbone.feat4_channels,   # 2048
    }
    return backbone, feat_dims