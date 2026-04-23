"""
Decoder.py  —  FPN Decoder for Few-Shot Segmentation  (v4 — Improved)
======================================================================

CHANGES FROM v3:
  1. forward_multiscale() added — returns intermediate P4 and P3 feature
     maps in addition to the final P2 output. This is needed by
     APM.SegAPM.get_multiscale_features() so that Phase 2 prototype
     extraction can pool features at three spatial scales (P4, P3, P2)
     instead of just one. Multi-scale prototyping significantly improves
     representation quality for small/thin objects.
     Reference: PFENet (Tian et al., TPAMI 2022), Section 3.2.
     https://arxiv.org/abs/2107.00509

  2. ASPP (Atrous Spatial Pyramid Pooling) module added after the FPN
     merging. ASPP captures context at multiple receptive field sizes
     (rates 1, 6, 12, 18) on the final P2 feature map. This gives
     better global context for scale-ambiguous objects.
     Reference: DeepLabV3 (Chen et al., 2017) https://arxiv.org/abs/1706.05587
     Note: ASPP is lightweight (1×1 and dilated 3×3 convs only) and adds
     ~1.5M parameters. Set USE_ASPP=False to disable if GPU memory is tight.

  3. Squeeze-and-Excitation (SE) channel attention on each FPN level.
     Recalibrates channel-wise responses before merging. Adds <0.1M params.
     Reference: Hu et al. (2018), SENet, CVPR. https://arxiv.org/abs/1709.01507

ARCHITECTURE OVERVIEW:
  Input backbone features (ResNet-50, 473×473 input):
    feat2: [B, 512,  119, 119]  (layer2, stride 4)
    feat3: [B, 1024,  60,  60]  (layer3, stride 8)
    feat4: [B, 2048,  30,  30]  (layer4, stride 16)

  FPN path:
    P4 = SE(lateral4(feat4))                        → [B, D, 30, 30]
    P3 = SE(lateral3(feat3) + upsample(P4))         → [B, D, 60, 60]
    P3 = smooth3(P3)
    P2 = SE(lateral2(feat2) + upsample(P3))         → [B, D, 119, 119]
    P2 = smooth2(P2)

  ASPP on P2 (optional):
    Rates 1, 6, 12, 18 → concat → 1×1 proj → [B, D, 119, 119]

  Final output: [B, D, 119, 119]  (D=256 by default)

  NOTE: At 224×224 input (used in some baselines), feat2=56×56 as
  documented in v3. At the standard FSS evaluation size of 473×473,
  the spatial dimensions are larger as shown above. The channel
  dimensions are always as annotated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

USE_ASPP = True   # Set False to disable ASPP (saves ~1.5M params)


# ─────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation channel attention
# ─────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Lightweight channel attention gate.
    Squeeze: global average pool → [B, C, 1, 1]
    Excitation: two FC layers with ReLU + Sigmoid
    Output: input * attention weights

    Reference: Hu et al. (2018) Squeeze-and-Excitation Networks, CVPR.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


# ─────────────────────────────────────────────────────────────────
# ASPP module
# ─────────────────────────────────────────────────────────────────

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Captures multi-scale context via parallel dilated convolutions.

    Branches:
      - 1×1 conv (rate=1)
      - 3×3 dilated conv (rate=6)
      - 3×3 dilated conv (rate=12)
      - 3×3 dilated conv (rate=18)
      - global average pool branch
    Outputs concatenated and projected back to out_channels.

    Reference: Chen et al. (2017) DeepLabV3. https://arxiv.org/abs/1706.05587
    """

    def __init__(self, in_channels, out_channels, rates=(1, 6, 12, 18)):
        super().__init__()
        mid = out_channels // len(rates)

        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, mid, 1, bias=False),
                    nn.BatchNorm2d(mid),
                    nn.ReLU(inplace=True),
                )
            else:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, mid, 3,
                              padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(mid),
                    nn.ReLU(inplace=True),
                )
            self.branches.append(branch)

        # Global average pool branch
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # Projection: concatenated branches → out_channels
        total_mid = mid * (len(rates) + 1)
        self.proj = nn.Sequential(
            nn.Conv2d(total_mid, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h, w   = x.shape[-2:]
        feats  = [b(x) for b in self.branches]
        # GAP branch: upsample back to spatial size
        gap    = F.interpolate(self.gap_branch(x), size=(h, w),
                               mode="bilinear", align_corners=False)
        feats.append(gap)
        out = self.proj(torch.cat(feats, dim=1))
        return out


# ─────────────────────────────────────────────────────────────────
# FPN Decoder
# ─────────────────────────────────────────────────────────────────

class FPNDecoder(nn.Module):
    """
    FPN-style decoder with SE attention and optional ASPP.

    Takes three backbone feature maps and fuses them into one rich
    spatial feature map. Exposes forward_multiscale() for Phase 2
    multi-scale prototype extraction.
    """

    def __init__(self, out_channels=256, use_aspp=USE_ASPP):
        super().__init__()
        self.out_channels = out_channels
        self.use_aspp     = use_aspp

        # ── Lateral 1×1 projections ─────────────────────────────
        self.lateral_layer4 = nn.Conv2d(2048, out_channels, kernel_size=1)
        self.lateral_layer3 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lateral_layer2 = nn.Conv2d(512,  out_channels, kernel_size=1)

        # ── SE channel attention after each lateral ──────────────
        self.se4 = SEBlock(out_channels)
        self.se3 = SEBlock(out_channels)
        self.se2 = SEBlock(out_channels)

        # ── Smooth 3×3 convs ─────────────────────────────────────
        self.smooth3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # ── ASPP (optional) ──────────────────────────────────────
        if use_aspp:
            self.aspp = ASPP(out_channels, out_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat2, feat3, feat4):
        """
        Standard forward pass — returns final fused feature map P2.

        Parameters
        ----------
        feat2 : [B, 512,  h2, w2]  — from backbone layer2
        feat3 : [B, 1024, h3, w3]  — from backbone layer3
        feat4 : [B, 2048, h4, w4]  — from backbone layer4

        Returns
        -------
        P2 : [B, out_channels, h2, w2]
        """
        p2, _, _ = self.forward_multiscale(feat2, feat3, feat4)
        return p2

    def forward_multiscale(self, feat2, feat3, feat4):
        """
        Returns ALL three FPN-level features: P2, P3, P4.
        Used by SegAPM.get_multiscale_features() for Phase 2 prototype
        extraction at multiple spatial scales.

        Returns
        -------
        P2 : [B, D, h2, w2]  — finest scale (fed to memory module)
        P3 : [B, D, h3, w3]  — mid scale
        P4 : [B, D, h4, w4]  — coarsest scale
        """
        # ── Step 1: Project layer4 → P4 ─────────────────────────
        P4 = self.se4(self.lateral_layer4(feat4))   # [B, D, h4, w4]

        # ── Step 2: Merge P4 + layer3 → P3 ──────────────────────
        P3 = self.lateral_layer3(feat3) + F.interpolate(
            P4, size=feat3.shape[-2:], mode="bilinear", align_corners=False
        )
        P3 = self.se3(P3)
        P3 = self.smooth3(P3)                       # [B, D, h3, w3]

        # ── Step 3: Merge P3 + layer2 → P2 ──────────────────────
        P2 = self.lateral_layer2(feat2) + F.interpolate(
            P3, size=feat2.shape[-2:], mode="bilinear", align_corners=False
        )
        P2 = self.se2(P2)
        P2 = self.smooth2(P2)                       # [B, D, h2, w2]

        # ── Step 4: ASPP on P2 (optional) ────────────────────────
        if self.use_aspp:
            P2 = self.aspp(P2)                      # [B, D, h2, w2]

        return P2, P3, P4
