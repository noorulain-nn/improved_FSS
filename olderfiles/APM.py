"""
APM.py  —  Adaptive Prototype Memory  (v4 — Improved)
======================================================

CHANGES FROM v3:
  1. MULTI-PROTOTYPE PER CLASS (K-means, N_PROTO prototypes)
     Each base/novel class now has N_PROTO=3 prototype vectors instead of
     one. This handles intra-class variation — e.g. "person" seen from the
     front vs side, or "chair" in different orientations.
     During inference, cosine similarity is computed against ALL prototypes
     and the MAX similarity is used (nearest-prototype assignment).
     Reference: PPNet (Liu et al., NeurIPS 2020) — Part-aware Prototype
     Network for Few-Shot Semantic Segmentation.
     https://arxiv.org/abs/2007.06309

  2. CLEANER BACKGROUND EMA
     update_from_batch() now accepts other_fg_mask (BoolTensor [B, H, W])
     from the DataLoader. Pixels marked True belong to OTHER base-class
     objects visible in the same image — they are NOT true background and
     must NOT pollute the background EMA slot.
     Pure background = mask==0 AND NOT other_fg.

  3. MULTI-SCALE PROTOTYPE IN PHASE 2
     build_novel_prototype() now accepts multi-scale feature lists:
       feat_list : list of [1, D_i, h_i, w_i] tensors per support image
                   one per FPN scale (P2, P3, P4)
     Features from each scale are masked-average-pooled independently,
     then concatenated and projected to D dimensions via a learnt linear
     (or simple concat+mean for inference).
     In practice we average across scales after normalising — simple but
     effective. For each scale separately, masked pooling is computed.
     Reference: PFENet (Tian et al., TPAMI 2022) section 3.2.
     https://arxiv.org/abs/2107.00509

MEMORY LAYOUT (unchanged):
  Slot 0          → global background (fallback)
  Slots 1..N      → foreground prototypes (N_PROTO per class × N_classes)
  Slots N+1..2N   → class-specific background (1 per class)

  For N_PROTO=3 and 15 base classes:
    Total fg slots = 15 × 3 = 45
    Total bg slots = 15
    Total = 1 + 45 + 15 = 61 slots

FIXES KEPT FROM v3:
  Fix 1 — class-specific background slots
  Fix 2 — support-derived background for novel classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Decoder import FPNDecoder

# Number of prototypes per class.
# 3 is a good default: enough to capture main modes, cheap to compute.
# Increase to 5 for harder classes; reduce to 1 to match v3 behaviour.
N_PROTO = 3


# ─────────────────────────────────────────────────────────────────
# K-means prototype extraction (no gradients — pure memory update)
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _kmeans_prototypes(features, n_proto, n_iter=10):
    """
    Extract N_PROTO prototypes from a bag of feature vectors via K-means.

    Parameters
    ----------
    features : FloatTensor [M, D]  — M feature vectors, already L2-normalised
    n_proto  : int                 — number of clusters
    n_iter   : int                 — number of Lloyd iterations

    Returns
    -------
    centres : FloatTensor [n_proto, D]  — L2-normalised cluster centres
    """
    M, D = features.shape
    if M <= n_proto:
        # Not enough vectors — pad with zeros or repeat
        pad = features.new_zeros(n_proto - M, D)
        centres = torch.cat([features, pad], dim=0)
        return F.normalize(centres, p=2, dim=1)

    # Initialise centres by picking n_proto random distinct rows
    perm    = torch.randperm(M, device=features.device)[:n_proto]
    centres = features[perm].clone()   # [n_proto, D]

    for _ in range(n_iter):
        # Assignment: cosine similarity (features already normalised)
        sim     = features @ centres.t()   # [M, n_proto]
        assigns = sim.argmax(dim=1)        # [M]

        new_centres = torch.zeros_like(centres)
        for k in range(n_proto):
            members = features[assigns == k]
            if members.shape[0] > 0:
                new_centres[k] = members.mean(dim=0)
            else:
                # Empty cluster — reinitialise to a random vector
                new_centres[k] = features[torch.randint(M, (1,))[0]]

        centres = F.normalize(new_centres, p=2, dim=1)

    return centres   # [n_proto, D]


# ─────────────────────────────────────────────────────────────────
# Memory Module
# ─────────────────────────────────────────────────────────────────

class MemoryModule(nn.Module):
    """
    Adaptive Prototype Memory with:
      - N_PROTO prototypes per class (multi-prototype, Fix from PPNet)
      - Class-specific background slots (Fix 1 from v3)
      - Support-derived novel background prototype (Fix 2 from v3)
      - Cleaner background EMA (excludes other-class pixels)

    Memory layout:
      Slot 0                             = global background (fallback)
      Slots 1  .. N*n_proto              = N_PROTO fg protos per base class
      Slots N*n_proto+1 .. N*n_proto+N   = 1 bg proto per base class
    """

    def __init__(self, num_base_classes, feature_dim, n_proto=N_PROTO):
        super().__init__()

        self.num_base_classes = num_base_classes
        self.feature_dim      = feature_dim
        self.n_proto          = n_proto

        # Slot layout
        # global bg: 1
        # fg per class: n_proto each → num_base_classes * n_proto
        # bg per class: 1 each       → num_base_classes
        self.n_fg_slots = num_base_classes * n_proto
        self.n_bg_slots = num_base_classes
        self.num_slots  = 1 + self.n_fg_slots + self.n_bg_slots

        self.memory = nn.Parameter(
            torch.randn(self.num_slots, feature_dim),
            requires_grad=False
        )
        nn.init.normal_(self.memory, mean=0.0, std=0.01)

        # Track which slots are initialised with real data
        self.slot_ready = [False] * self.num_slots

        # Fix 2 storage — built during Phase 2, used in Phase 3
        self.novel_prototypes   = {}    # cls_id → [n_proto, D] tensor
        self.novel_bg_prototype = None  # [D] tensor

        print(f"[APM] Memory layout (v4 — multi-prototype):")
        print(f"      n_proto per class  = {n_proto}")
        print(f"      Slot 0             = global background (fallback)")
        print(f"      Slots 1–{self.n_fg_slots}        "
              f"= {n_proto} fg protos × {num_base_classes} base classes")
        print(f"      Slots {self.n_fg_slots+1}–{self.num_slots-1} "
              f"= 1 bg proto × {num_base_classes} base classes")
        print(f"      Total slots = {self.num_slots}  |  "
              f"Feature dim = {feature_dim}")

    # ── Slot index helpers ────────────────────────────────────────
    def _fg_slots(self, cls):
        """Return list of n_proto slot indices for base class cls (0-based)."""
        start = 1 + cls * self.n_proto
        return list(range(start, start + self.n_proto))

    def _bg_slot(self, cls):
        """Class-specific background slot index."""
        return 1 + self.n_fg_slots + cls

    # ── Forward ───────────────────────────────────────────────────
    def forward(self, feature_map, novel_cls_id=None):
        """
        Compute cosine similarity at every spatial location.

        Phase 1 (novel_cls_id=None):
          Compare against all slots → [B, num_slots, h, w]
          The loss function extracts the relevant fg/bg pair per sample.

        Phase 3 (novel_cls_id=int):
          Compare against [bg_proto, fg_proto_0, ..., fg_proto_{n-1}]
          → [B, n_proto+1, h, w]
          Argmax: 0=background, 1..n_proto=foreground
          (any fg slot winning → predicted as foreground)
        """
        B, D, h, w = feature_map.shape
        feat_norm  = F.normalize(feature_map, p=2, dim=1)

        if novel_cls_id is None:
            # Phase 1 — all slots
            mem = F.normalize(self.memory, p=2, dim=1)  # [S, D]
        else:
            # Phase 3 — novel class: [bg, fg_proto_0, ..., fg_proto_{n-1}]
            if self.novel_bg_prototype is not None:
                bg_proto = self.novel_bg_prototype
            else:
                bg_proto = F.normalize(self.memory[0], p=2, dim=0)

            novel_protos = self.novel_prototypes[novel_cls_id]  # [n_proto, D]
            bg_norm      = F.normalize(bg_proto.unsqueeze(0), p=2, dim=1)   # [1,D]
            fg_norm      = F.normalize(novel_protos, p=2, dim=1)             # [n,D]
            mem          = torch.cat([bg_norm, fg_norm], dim=0)              # [n+1,D]

        S         = mem.shape[0]
        feat_flat = feat_norm.view(B, D, h * w)            # [B, D, h*w]
        sim       = torch.bmm(
            feat_flat.permute(0, 2, 1),
            mem.t().unsqueeze(0).expand(B, -1, -1)
        )   # [B, h*w, S]
        logits = sim.permute(0, 2, 1).view(B, S, h, w)    # [B, S, h, w]
        return logits

    # ── EMA slot update ───────────────────────────────────────────
    def _update_slot(self, feature_map, mask, slot_idx):
        """
        Masked average pooling → adaptive EMA update for one slot.
        mask: binary tensor [1, H, W] where 1=update, 0=skip.
        """
        D, h, w   = feature_map.shape[1:]
        mask_down = F.interpolate(
            mask.float().unsqueeze(1), size=(h, w), mode="nearest"
        )
        valid     = (mask_down != 255).float()
        mask_down = mask_down * valid

        if mask_down.sum() < 1:
            return

        denom     = mask_down.sum(dim=[0, 2, 3]).clamp(min=1e-6)
        proto_new = F.normalize(
            (feature_map * mask_down).sum(dim=[0, 2, 3]) / denom,
            p=2, dim=0
        )

        if not self.slot_ready[slot_idx]:
            self.memory.data[slot_idx] = proto_new
            self.slot_ready[slot_idx]  = True
        else:
            proto_old = F.normalize(self.memory.data[slot_idx], p=2, dim=0)
            sim       = F.cosine_similarity(
                proto_new.unsqueeze(0), proto_old.unsqueeze(0)
            ).item()
            alpha = max(0.0, min(1.0 - sim, 1.0))
            self.memory.data[slot_idx] = (
                (1 - alpha) * self.memory.data[slot_idx] + alpha * proto_new
            )

    # ── Phase 1 batch update ──────────────────────────────────────
    def update_from_batch(self, feature_map, binary_masks,
                          class_labels, other_fg_masks=None):
        """
        Called after each Phase 1 batch. Updates fg and bg slots.

        For fg slots (multi-prototype):
          Collect all foreground pixel features, run K-means to get
          N_PROTO centroids, update each fg slot via EMA with one centroid.

        For bg slots (cleaner):
          Use only pixels where binary_mask==0 AND NOT other_fg.
          This excludes pixels from other-class objects in the same image.

        Parameters
        ----------
        feature_map    : FloatTensor [B, D, h, w]
        binary_masks   : LongTensor  [B, H, W]   0=bg 1=fg 255=ignore
        class_labels   : list[int]   length B, values 0..num_base-1
        other_fg_masks : BoolTensor  [B, H, W] | None
                         True where another base-class object is present.
                         If None, falls back to v3 behaviour (all bg used).
        """
        B = feature_map.shape[0]
        _, D, h, w = feature_map.shape

        for i in range(B):
            feat_i  = feature_map[i].unsqueeze(0)   # [1, D, h, w]
            mask_i  = binary_masks[i].unsqueeze(0)  # [1, H, W]
            cls     = class_labels[i]

            # ── Foreground — multi-prototype via K-means ──────────
            fg_mask_i = (mask_i == 1).long()
            mask_down = F.interpolate(
                fg_mask_i.float().unsqueeze(1), size=(h, w), mode="nearest"
            ).squeeze(1)  # [1, h, w]

            fg_pixels = feat_i[0, :, mask_down[0] > 0.5]  # [D, M]
            if fg_pixels.shape[1] > 0:
                fg_feats  = F.normalize(fg_pixels.t(), p=2, dim=1)  # [M, D]
                centroids = _kmeans_prototypes(fg_feats, self.n_proto)  # [n,D]
                for k, slot_idx in enumerate(self._fg_slots(cls)):
                    proto_k = centroids[k]  # [D]
                    if not self.slot_ready[slot_idx]:
                        self.memory.data[slot_idx] = proto_k
                        self.slot_ready[slot_idx]  = True
                    else:
                        proto_old = F.normalize(
                            self.memory.data[slot_idx], p=2, dim=0
                        )
                        sim   = F.cosine_similarity(
                            proto_k.unsqueeze(0), proto_old.unsqueeze(0)
                        ).item()
                        alpha = max(0.0, min(1.0 - sim, 1.0))
                        self.memory.data[slot_idx] = (
                            (1 - alpha) * self.memory.data[slot_idx]
                            + alpha * proto_k
                        )

            # ── Background — exclude other-class pixels ───────────
            # true_bg = pixels that are labelled 0 (bg) AND are not
            # part of another base-class object in the same image.
            bg_mask_i = (mask_i == 0).long()
            if other_fg_masks is not None:
                other_fg_i = other_fg_masks[i].unsqueeze(0)  # [1, H, W]
                # Downscale other_fg to feature resolution
                ofg_down = F.interpolate(
                    other_fg_i.float().unsqueeze(1), size=(h, w),
                    mode="nearest"
                ).squeeze(1).bool()   # [1, h, w]
                # Zero out bg mask where other objects are
                bg_down = F.interpolate(
                    bg_mask_i.float().unsqueeze(1), size=(h, w),
                    mode="nearest"
                ).squeeze(1)          # [1, h, w]
                bg_down[ofg_down] = 0
                clean_bg_mask = bg_down.long()  # [1, h, w]
            else:
                clean_bg_mask = bg_mask_i

            # Update global bg slot (fallback)
            self._update_slot(feat_i, bg_mask_i, slot_idx=0)

            # Update class-specific bg slot with CLEAN background
            self._update_slot(feat_i, clean_bg_mask,
                              self._bg_slot(cls))

    # ── Phase 2: build novel prototypes ───────────────────────────
    @torch.no_grad()
    def build_novel_prototype(self, support_feat_lists, support_masks,
                               novel_cls_id):
        """
        Build multi-scale, multi-prototype representations for a novel class.

        MULTI-SCALE (Issue 6 fix):
          support_feat_lists : list of lists
            Outer list: K support images
            Inner list: [feat_P2, feat_P3, feat_P4] — 3 FPN scales
            Each feat: [1, D, h, w]
          We masked-pool each scale separately, then average across scales.

        MULTI-PROTOTYPE (Issue 3 fix):
          From the K aggregated feature vectors (one per support image),
          K-means gives N_PROTO centroids as the novel class prototypes.

        Fix 2 (v3 — unchanged):
          Background prototype built from non-masked pixels of supports.

        Parameters
        ----------
        support_feat_lists : list[list[Tensor]]  — K × n_scales feat tensors
        support_masks      : list[Tensor]        — K masks [1, H, W]
        novel_cls_id       : int
        """
        fg_vecs = []  # One pooled fg vector per support image
        bg_vecs = []  # One pooled bg vector per support image

        K = len(support_masks)

        for k in range(K):
            mask_i = support_masks[k]  # [1, H, W]
            feats_k = support_feat_lists[k]  # list of [1, D, h, w] per scale

            scale_fg_protos = []
            scale_bg_protos = []

            for feat_s in feats_k:
                _, D_s, h_s, w_s = feat_s.shape

                mask_down = F.interpolate(
                    mask_i.float().unsqueeze(1), size=(h_s, w_s),
                    mode="nearest"
                )
                valid    = (mask_down != 255).float()

                # Foreground pooling
                fg_mask  = mask_down * valid
                fg_denom = fg_mask.sum(dim=[0, 2, 3]).clamp(min=1e-6)
                fg_proto = (feat_s * fg_mask).sum(dim=[0, 2, 3]) / fg_denom

                # Background pooling (Fix 2)
                bg_mask  = (1.0 - mask_down) * valid
                bg_denom = bg_mask.sum(dim=[0, 2, 3]).clamp(min=1e-6)
                bg_proto = (feat_s * bg_mask).sum(dim=[0, 2, 3]) / bg_denom

                scale_fg_protos.append(fg_proto)
                scale_bg_protos.append(bg_proto)

            # Average across scales, then normalise
            fg_mean = torch.stack(scale_fg_protos, dim=0).mean(0)
            bg_mean = torch.stack(scale_bg_protos, dim=0).mean(0)

            fg_vecs.append(F.normalize(fg_mean, p=2, dim=0))
            bg_vecs.append(F.normalize(bg_mean, p=2, dim=0))

        # Multi-prototype: K-means over K support fg vectors
        fg_stacked = torch.stack(fg_vecs, dim=0)  # [K, D]
        n_proto_use = min(self.n_proto, K)
        centroids   = _kmeans_prototypes(fg_stacked, n_proto_use)  # [n, D]

        # Pad to n_proto if K < n_proto
        if n_proto_use < self.n_proto:
            pad = centroids[-1:].expand(self.n_proto - n_proto_use, -1)
            centroids = torch.cat([centroids, pad], dim=0)

        self.novel_prototypes[novel_cls_id] = centroids  # [n_proto, D]

        # Background: average across support images
        bg_stacked = torch.stack(bg_vecs, dim=0)  # [K, D]
        self.novel_bg_prototype = F.normalize(
            bg_stacked.mean(dim=0), p=2, dim=0
        )

        print(f"[APM] Built {self.n_proto}-proto fg + bg for novel class "
              f"{novel_cls_id} from {K} support image(s)  "
              f"({len(feats_k)} FPN scales).")
        print(f"      Fix 1: {self.num_base_classes} class-specific bg slots in memory.")
        print(f"      Fix 2: fresh support-derived bg stored.")


# ─────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────

class SegAPM(nn.Module):
    """
    SegAPM v4: ResNet backbone + FPN decoder + multi-prototype memory.

    forward() returns (logits, fused):
      Phase 1 (novel_cls_id=None):
        logits = [B, num_slots, h, w]   — all slots
      Phase 3 (novel_cls_id=int):
        logits = [B, n_proto+1, h, w]   — [bg, fg0, fg1, ..., fg_{n-1}]
        The calling code converts this to binary by taking:
          pred = (logits[:, 1:, ...].max(1)[0] > logits[:, 0, ...]).long()
    """

    def __init__(self, backbone, num_base_classes,
                 decoder_out_channels=256, n_proto=N_PROTO):
        super().__init__()
        self.backbone      = backbone
        self.decoder       = FPNDecoder(out_channels=decoder_out_channels)
        self.memory_module = MemoryModule(
            num_base_classes, decoder_out_channels, n_proto=n_proto
        )
        self.n_proto = n_proto

        print(f"[SegAPM] Total memory slots: {self.memory_module.num_slots}")
        print(f"         Decoder out channels: {decoder_out_channels}")
        print(f"         Prototypes per class: {n_proto}")

    def forward(self, x, novel_cls_id=None):
        feat2, feat3, feat4 = self.backbone(x)
        fused  = self.decoder(feat2, feat3, feat4)       # [B, D, h, w]
        logits = self.memory_module(fused, novel_cls_id)
        return logits, fused

    def get_multiscale_features(self, x):
        """
        Returns all three FPN-scale features for Phase 2 multi-scale
        prototype extraction.
        Returns: (feat_P2, feat_P3, feat_P4, fused)
          feat_P2  [B, D, h2, w2]  — finest spatial scale (from layer2)
          feat_P3  [B, D, h3, w3]  — mid scale (from layer3)
          feat_P4  [B, D, h4, w4]  — coarsest scale (from layer4)
          fused    [B, D, h2, w2]  — final merged FPN output (= P2 level)
        """
        feat2, feat3, feat4 = self.backbone(x)
        # Run through decoder internals to get intermediate features too
        fused, p4_feat, p3_feat = self.decoder.forward_multiscale(
            feat2, feat3, feat4
        )
        return p4_feat, p3_feat, fused, fused

    def freeze_everything(self):
        for param in self.parameters():
            param.requires_grad = False
        print("[SegAPM] All weights frozen for Phase 2 & 3.")