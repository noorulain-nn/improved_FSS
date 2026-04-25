"""
APM.py  —  Adaptive Prototype Memory for Few-Shot Segmentation
==============================================================

HOW THIS EXTENDS FSIC TO FSS
------------------------------
FSIC uses one memory slot per class. At both training AND test time,
the same adaptive EMA formula updates that slot whenever the class
is seen. Novel classes at test time simply update their assigned slot
via the same EMA — no special treatment.

This FSS extension keeps that exact philosophy:
  - Same adaptive EMA formula runs at Phase 1 (base) AND Phase 2 (novel)
  - The only FSS-specific addition is the fg/bg split (needed for
    binary pixel-level decisions — classification does not need this)

SLOT LAYOUT (33 slots for 15 base classes)
-------------------------------------------
  Slot 0         -> global background (fallback)
  Slots  1-15    -> foreground, one per base class
  Slots 16-30    -> class-specific background, one per base class
  Slot 31        -> WORKING fg slot (EMA builds novel proto here)
  Slot 32        -> WORKING bg slot (EMA builds novel proto here)

WHY WORKING SLOTS + DICT STORAGE
----------------------------------
  Slots 31 and 32 are TEMPORARY working slots used during Phase 2
  to run the FSIC EMA over K support images. After each novel class
  is processed, the result is CLONED into a per-class dictionary.

  This gives you:
    - FSIC-faithful EMA during prototype building (slots 31/32)
    - Per-class storage so sequential novel classes do not
      overwrite each other (dict)

  This is exactly how FSIC works conceptually:
    FSIC: EMA updates slot -> slot IS the classifier at test time
    FSS:  EMA updates slot -> clone into dict -> dict used at test time
    The EMA formula is identical. Storage is separate because
    segmentation requires one prototype per class simultaneously,
    whereas FSIC only ever tests one class at a time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Decoder import FPNDecoder


# ── EMA hyper-parameters ─────────────────────────────────────────────
ALPHA_MIN = 0.05
ALPHA_MAX = 0.30
WARMUP_N  = 5
TEMP_INIT = 10.0
TEMP_MIN  = 1.0
TEMP_MAX  = 50.0


class MemoryModule(nn.Module):

    def __init__(self, num_base_classes, feature_dim):
        super().__init__()

        self.num_base_classes = num_base_classes
        self.feature_dim      = feature_dim

        # ── Slot counts ───────────────────────────────────────────────
        # 1 global bg + 15 fg + 15 class-bg + 2 working novel slots = 33
        self.num_slots = 1 + num_base_classes + num_base_classes + 2

        # Memory matrix — EMA updated, never touched by Adam
        self.memory = nn.Parameter(
            torch.randn(self.num_slots, feature_dim),
            requires_grad=False
        )
        nn.init.normal_(self.memory, mean=0.0, std=0.01)

        # Learnable temperature — trained by Adam during Phase 1
        self.temperature = nn.Parameter(torch.tensor(TEMP_INIT))

        # Per-slot EMA update counter
        self.register_buffer(
            "n_seen", torch.zeros(self.num_slots, dtype=torch.long)
        )

        # ── Per-class novel prototype storage ─────────────────────────
        # Slots 31/32 are working slots used during Phase 2 EMA building.
        # Results are cloned here so sequential novel classes do not
        # overwrite each other. forward() reads from here during Phase 3.
        self.novel_prototypes    = {}   # cls_id -> fg proto tensor [D]
        self.novel_bg_prototypes = {}   # cls_id -> bg proto tensor [D]

        self._print_layout()

    def _print_layout(self):
        n = self.num_base_classes
        print(f"\n[MemoryModule] Slot layout ({self.num_slots} total):")
        print(f"  Slot 0         -> global background")
        print(f"  Slots  1-{n:<2}    -> base class foreground")
        print(f"  Slots {n+1}-{2*n:<2}   -> base class-specific background")
        print(f"  Slot {self.num_slots-2}         -> working novel fg slot (Phase 2 EMA)")
        print(f"  Slot {self.num_slots-1}         -> working novel bg slot (Phase 2 EMA)")
        print(f"  Novel dict     -> per-class clones after EMA (Phase 3)")
        print(f"  Temperature    = {TEMP_INIT} (learnable)")
        print(f"  EMA alpha      = [{ALPHA_MIN}, {ALPHA_MAX}]")
        print(f"  Warm-up        = {WARMUP_N} updates\n")

    # ── Slot index helpers ────────────────────────────────────────────
    def _fg_slot(self, cls):
        return 1 + cls

    def _bg_slot(self, cls):
        return 1 + self.num_base_classes + cls

    def _novel_fg_slot(self):
        return self.num_slots - 2   # slot 31

    def _novel_bg_slot(self):
        return self.num_slots - 1   # slot 32

    # ── Core FSIC adaptive EMA ────────────────────────────────────────
    def _ema_update(self, proto_new, slot_idx):
        """
        The FSIC adaptive EMA formula — shared by Phase 1 and Phase 2.

        Phase 1: called for base class fg and bg slots
        Phase 2: called for novel class working slots 31 and 32
        Formula is identical both times — this is the FSIC extension.

        Warm-up: running mean for first WARMUP_N updates (stable init)
        After:   alpha = clamp(1 - cosine_sim, ALPHA_MIN, ALPHA_MAX)
                 slot  = (1-alpha)*slot + alpha*proto_new
        """
        n = self.n_seen[slot_idx].item()

        if n < WARMUP_N:
            if n == 0:
                self.memory.data[slot_idx] = proto_new
            else:
                old = self.memory.data[slot_idx]
                self.memory.data[slot_idx] = (n * old + proto_new) / (n + 1)
        else:
            proto_old = F.normalize(
                self.memory.data[slot_idx], p=2, dim=0
            )
            sim = F.cosine_similarity(
                proto_new.unsqueeze(0),
                proto_old.unsqueeze(0)
            ).item()
            alpha = max(ALPHA_MIN, min(1.0 - sim, ALPHA_MAX))
            self.memory.data[slot_idx] = (
                (1.0 - alpha) * self.memory.data[slot_idx]
                + alpha * proto_new
            )

        self.n_seen[slot_idx] += 1

    # ── Attention-weighted masked pooling ─────────────────────────────
    def _pool_prototype(self, feature_map, mask):
        """
        Attention-weighted masked average pooling.
        Weights each spatial location by its L2 norm.
        High-norm locations are more activated and more reliable.
        Returns normalized [D] vector or None if no valid pixels.
        """
        D, h, w   = feature_map.shape[1:]
        mask_down = F.interpolate(
            mask.float().unsqueeze(1), size=(h, w), mode="nearest"
        )
        valid     = (mask_down != 255).float()
        mask_down = mask_down * valid

        if mask_down.sum() < 0.5:
            return None

        feat_norm = feature_map.norm(dim=1, keepdim=True)
        weight    = mask_down * feat_norm
        wsum      = weight.sum(dim=[0, 2, 3]).clamp(min=1e-6)
        proto     = (feature_map * weight).sum(dim=[0, 2, 3]) / wsum

        return F.normalize(proto, p=2, dim=0)

    def _simple_pool(self, feature_map, mask):
        """
        Simple masked average pooling — fallback when attention
        pool finds no valid pixels (very small objects).
        """
        D, h, w   = feature_map.shape[1:]
        mask_down = F.interpolate(
            mask.float().unsqueeze(1), size=(h, w), mode="nearest"
        ).squeeze(1)
        valid     = (mask_down != 255).float()
        mask_down = mask_down * valid
        denom     = mask_down.sum().clamp(min=1e-6)
        if denom < 0.1:
            return None
        proto = (
            feature_map.squeeze(0) * mask_down.unsqueeze(0)
        ).sum(dim=[1, 2]) / denom
        return F.normalize(proto, p=2, dim=0)

    # ── Phase 1: base class batch update ─────────────────────────────
    def update_from_batch(self, feature_map, binary_masks, class_labels):
        """
        Called after each Phase 1 training batch (no gradient).
        Updates fg slot, global bg slot, class-specific bg slot
        for each sample using the shared _ema_update formula.
        """
        B = feature_map.shape[0]

        for i in range(B):
            feat_i = feature_map[i].unsqueeze(0)
            mask_i = binary_masks[i].unsqueeze(0)
            cls    = class_labels[i]

            fg_mask = (mask_i == 1).long()
            bg_mask = (mask_i == 0).long()

            # Foreground
            fg_proto = self._pool_prototype(feat_i, fg_mask)
            if fg_proto is None:
                fg_proto = self._simple_pool(feat_i, fg_mask)
            if fg_proto is not None:
                self._ema_update(fg_proto, self._fg_slot(cls))

            # Background — global slot and class-specific slot
            bg_proto = self._pool_prototype(feat_i, bg_mask)
            if bg_proto is None:
                bg_proto = self._simple_pool(feat_i, bg_mask)
            if bg_proto is not None:
                self._ema_update(bg_proto, slot_idx=0)
                self._ema_update(bg_proto, self._bg_slot(cls))

    # ── Phase 2: novel prototype building (FSIC-faithful) ────────────
    @torch.no_grad()
    def build_novel_prototype(self, support_features, support_masks,
                               novel_cls_id):
        """
        Builds novel fg and bg prototypes using the same FSIC adaptive
        EMA as Phase 1 base training.

        Step 1: Reset working slots 31 and 32
        Step 2: Feed each support image through EMA into working slots
                (identical formula to Phase 1 base class updates)
        Step 3: Clone results into per-class dict
                (so next novel class does not overwrite this one)

        forward() reads from the dict during Phase 3 — not from slots.
        This is why sequential novel classes work correctly.

        Parameters
        ----------
        support_features : list of K tensors, each [1, D, h, w]
        support_masks    : list of K tensors, each [1, H, W]
        novel_cls_id     : int
        """
        fg_slot = self._novel_fg_slot()   # 31 — working slot
        bg_slot = self._novel_bg_slot()   # 32 — working slot

        # Step 1: Reset working slots for this novel class
        self.memory.data[fg_slot] = torch.zeros(
            self.feature_dim, device=self.memory.device
        )
        self.memory.data[bg_slot] = torch.zeros(
            self.feature_dim, device=self.memory.device
        )
        self.n_seen[fg_slot] = 0
        self.n_seen[bg_slot] = 0

        # Step 2: FSIC EMA over K support images
        updates = 0
        for feat_i, mask_i in zip(support_features, support_masks):

            fg_mask = (mask_i == 1).long()
            bg_mask = (mask_i == 0).long()

            # Foreground
            fg_proto = self._pool_prototype(feat_i, fg_mask)
            if fg_proto is None:
                fg_proto = self._simple_pool(feat_i, fg_mask)
            if fg_proto is not None:
                self._ema_update(fg_proto, fg_slot)

            # Background
            bg_proto = self._pool_prototype(feat_i, bg_mask)
            if bg_proto is None:
                bg_proto = self._simple_pool(feat_i, bg_mask)
            if bg_proto is not None:
                self._ema_update(bg_proto, bg_slot)

            updates += 1

        # Step 3: Clone into per-class dict BEFORE next class overwrites slots
        self.novel_prototypes[novel_cls_id] = (
            self.memory.data[fg_slot].clone()
        )
        self.novel_bg_prototypes[novel_cls_id] = (
            self.memory.data[bg_slot].clone()
        )

        print(f"[APM] Novel class {novel_cls_id}: "
              f"fg->slot{fg_slot}, bg->slot{bg_slot}, "
              f"{updates} support image(s) via FSIC EMA -> stored in dict")

    # ── Forward ───────────────────────────────────────────────────────
    def forward(self, feature_map, novel_cls_id=None):
        """
        Phase 1 (novel_cls_id=None):
            Compare against all 33 memory slots.
            Returns logits [B, 33, h, w].

        Phase 2/3 (novel_cls_id=int):
            Compare against [bg, fg] for this specific novel class.
            Reads from per-class dict — not from working slots.
            Returns logits [B, 2, h, w].
        """
        B, D, h, w = feature_map.shape
        feat_norm  = F.normalize(feature_map, p=2, dim=1)

        if novel_cls_id is None:
            # Phase 1 — all slots
            mem = F.normalize(self.memory, p=2, dim=1)      # [33, D]
        else:
            # Phase 3 — read from per-class dict (not working slots)
            fg_proto = F.normalize(
                self.novel_prototypes[novel_cls_id], p=2, dim=0
            )
            bg_proto = F.normalize(
                self.novel_bg_prototypes[novel_cls_id], p=2, dim=0
            )
            mem = torch.stack([bg_proto, fg_proto], dim=0)  # [2, D]

        S         = mem.shape[0]
        feat_flat = feat_norm.view(B, D, h * w)
        sim       = torch.bmm(
            feat_flat.permute(0, 2, 1),
            mem.t().unsqueeze(0).expand(B, -1, -1)
        )
        logits = sim.permute(0, 2, 1).view(B, S, h, w)

        tau = self.temperature.clamp(TEMP_MIN, TEMP_MAX)
        return logits * tau


# ── Full segmentation model ───────────────────────────────────────────
class SegAPM(nn.Module):

    def __init__(self, backbone, num_base_classes, decoder_out_channels=256):
        super().__init__()
        self.backbone      = backbone
        self.decoder       = FPNDecoder(out_channels=decoder_out_channels)
        self.memory_module = MemoryModule(num_base_classes,
                                          decoder_out_channels)

    def forward(self, x, novel_cls_id=None):
        feat2, feat3, feat4 = self.backbone(x)
        fused  = self.decoder(feat2, feat3, feat4)
        logits = self.memory_module(fused, novel_cls_id)
        return logits, fused

    def freeze_for_novel(self):
        """Freeze all weights for Phase 2/3."""
        for param in self.parameters():
            param.requires_grad = False
        print("[SegAPM] All weights frozen for Phase 2/3.")

    def freeze_everything(self):
        """Alias for backward compatibility with main_seg.py."""
        return self.freeze_for_novel()