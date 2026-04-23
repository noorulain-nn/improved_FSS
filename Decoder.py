"""
Decoder.py  —  Lightweight FPN-style decoder for Few-Shot Segmentation
=======================================================================
WHY THIS EXISTS
---------------
Problem with the current pipeline:
  Backbone layer4 output → [B, 2048, 7, 7]
  After bilinear upsample → [B, 2, 224, 224]

  7×7 means each cell covers a 32×32 pixel region. The model
  is completely blind to anything finer than 32 pixels. Boundaries
  come out as blobs, not sharp edges.

What this decoder does:
  It taps into THREE stages of the backbone instead of just the last one:
    layer2 → [B, 512,  56, 56]   ← fine spatial detail
    layer3 → [B, 1024, 28, 28]   ← mid-level features
    layer4 → [B, 2048,  7,  7]   ← high-level semantics

  Then progressively fuses them bottom-up (coarse → fine):
    layer4 (7×7)  → upsample → add layer3 → upsample → add layer2
    Final output: [B, 256, 56, 56]

  This gives the memory module 56×56 = 3136 spatial locations to
  compare against prototypes instead of only 49.
  Each cell now covers only an 4×4 pixel region → much sharper masks.

WHAT IS TRAINED
---------------
  • Backbone layer4       → still fine-tunes (same as before)
  • Decoder conv layers   → trained from scratch during Phase 1
  • Memory prototypes     → updated via adaptive EMA (no gradients)
  • Backbone layer1/2/3   → FROZEN (same as before)

The decoder adds only ~2.5M parameters — very lightweight.

HOW TO USE
----------
See Models.py for the updated load_backbone_with_decoder() function.
The decoder slots between the backbone and the memory module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network-style decoder.

    Takes intermediate backbone features from three stages and
    fuses them into one richer spatial feature map.

    Think of it like this:
      layer4 knows WHAT the object is  (semantics, coarse)
      layer3 knows WHERE roughly       (mid-level)
      layer2 knows the exact EDGES     (fine spatial detail)

    We combine all three knowledge levels.
    """

    def __init__(self, out_channels=256):
        """
        Parameters
        ----------
        out_channels : int
            Number of channels in the final fused feature map.
            256 is standard for FPN. Smaller = faster, less expressive.
        """
        super().__init__()

        # ── Lateral connections ──────────────────────────────────
        # Each lateral conv reduces the channel depth to out_channels.
        # "Lateral" = connecting a backbone stage horizontally into
        # the decoder pathway.
        #
        # Why 1×1 conv? It changes channel depth without touching
        # spatial resolution. No spatial mixing — just channel projection.

        self.lateral_layer4 = nn.Conv2d(2048, out_channels, kernel_size=1)
        # input: [B, 2048, 7, 7]  → output: [B, 256, 7, 7]

        self.lateral_layer3 = nn.Conv2d(1024, out_channels, kernel_size=1)
        # input: [B, 1024, 28, 28] → output: [B, 256, 28, 28]

        self.lateral_layer2 = nn.Conv2d(512, out_channels, kernel_size=1)
        # input: [B, 512, 56, 56]  → output: [B, 256, 56, 56]

        # ── Smooth convolutions ──────────────────────────────────
        # After adding two feature maps (one upsampled, one lateral),
        # we apply a 3×3 conv to smooth out the checkerboard artifacts
        # that bilinear upsampling can introduce.

        self.smooth3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # [B, 256, 28, 28] → [B, 256, 28, 28]

        self.smooth2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # [B, 256, 56, 56] → [B, 256, 56, 56]

        # Final output channels stored for reference by other modules
        self.out_channels = out_channels

        # Initialise weights with Kaiming (good default for ReLU networks)
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
        Parameters
        ----------
        feat2 : FloatTensor [B, 512,  56, 56]  — from backbone layer2
        feat3 : FloatTensor [B, 1024, 28, 28]  — from backbone layer3
        feat4 : FloatTensor [B, 2048,  7,  7]  — from backbone layer4

        Returns
        -------
        fused : FloatTensor [B, 256, 56, 56]
                Rich feature map with both semantic and spatial detail.
                The memory module will compare prototypes against this.

        Step-by-step:
          P4 = lateral(feat4)                     → [B,256, 7, 7]
          P3 = lateral(feat3) + upsample(P4)      → [B,256,28,28]
          P2 = lateral(feat2) + upsample(P3)      → [B,256,56,56]
        """

        # Step 1: Project layer4 features to 256 channels
        P4 = self.lateral_layer4(feat4)   # [B, 256, 7, 7]

        # Step 2: Upsample P4 to match layer3 spatial size (28×28),
        #         then add to projected layer3 features.
        #         "nearest" is faster than bilinear for intermediate steps.
        P3 = self.lateral_layer3(feat3) + F.interpolate(
            P4, size=feat3.shape[-2:], mode="nearest"
        )   # [B, 256, 28, 28]
        P3 = self.smooth3(P3)             # smooth artifacts

        # Step 3: Upsample P3 to match layer2 spatial size (56×56),
        #         then add to projected layer2 features.
        P2 = self.lateral_layer2(feat2) + F.interpolate(
            P3, size=feat2.shape[-2:], mode="nearest"
        )   # [B, 256, 56, 56]
        P2 = self.smooth2(P2)             # smooth artifacts
        P2 = F.normalize(P2, p=2, dim=1)   # normalize across channel dim

        return P2   # [B, 256, 56, 56]