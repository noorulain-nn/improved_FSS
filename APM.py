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
  - Novel classes get their own dedicated slots — same as base classes
  - The only FSS-specific addition is the fg/bg split (needed for
    binary pixel-level decisions — classification doesn't need this)

SLOT LAYOUT (33 slots for 15 base classes)
-------------------------------------------
  Slot 0         → global background (fallback)
  Slots  1-15    → foreground, one per base class
  Slots 16-30    → class-specific background, one per base class
  Slot 31        → novel class foreground   (FSIC-style novel slot)
  Slot 32        → novel class background   (FSS-specific addition)

WHY 33 NOT 31
--------------
Previous version had 31 slots with no dedicated novel slots.
Novel prototypes were stored outside the memory matrix in a separate
dict. This broke the FSIC philosophy — novel and base classes were
treated differently. Now novel classes have real memory slots and go
through the same EMA as base classes.

SHARED EMA (_ema_update)
-------------------------
Both Phase 1 base training and Phase 2 novel prototype building
call the same _ema_update() method. This is the FSIC adaptive EMA:
    alpha = clamp(1 - cosine_similarity, ALPHA_MIN, ALPHA_MAX)
    slot  = (1 - alpha) * slot + alpha * new_prototype
With a running-mean warm-up for the first WARMUP_N updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Decoder import FPNDecoder


# ── EMA hyper-parameters ─────────────────────────────────────────────
ALPHA_MIN = 0.05    # prototype always moves at least this much
ALPHA_MAX = 0.30    # one noisy batch can move it at most this much
WARMUP_N  = 5       # use running mean for first N updates, then EMA
TEMP_INIT = 10.0    # initial temperature
TEMP_MIN  = 1.0
TEMP_MAX  = 50.0


class MemoryModule(nn.Module):

    def __init__(self, num_base_classes, feature_dim):
        super().__init__()

        self.num_base_classes = num_base_classes
        self.feature_dim      = feature_dim

        # ── Slot counts ───────────────────────────────────────────
        self.NUM_GLOBAL_BG   = 1
        self.NUM_FG_SLOTS    = num_base_classes       # 15
        self.NUM_BG_SLOTS    = num_base_classes       # 15
        self.NUM_NOVEL_SLOTS = 2                      # 1 fg + 1 bg
        self.num_slots = (self.NUM_GLOBAL_BG
                          + self.NUM_FG_SLOTS
                          + self.NUM_BG_SLOTS
                          + self.NUM_NOVEL_SLOTS)     # 33

        # Memory matrix — EMA-updated, never touched by Adam
        self.memory = nn.Parameter(
            torch.randn(self.num_slots, feature_dim),
            requires_grad=False
        )
        nn.init.normal_(self.memory, mean=0.0, std=0.01)

        # Learnable temperature — trained by Adam
        self.temperature = nn.Parameter(torch.tensor(TEMP_INIT))

        # Per-slot update counter for warm-up logic
        self.register_buffer(
            "n_seen", torch.zeros(self.num_slots, dtype=torch.long)
        )

        # Tracks which novel class is currently loaded
        self.novel_cls_to_slot = {}

        self._print_layout()

    def _print_layout(self):
        n = self.num_base_classes
        print(f"\n[MemoryModule] Slot layout ({self.num_slots} total):")
        print(f"  Slot 0         -> global background")
        print(f"  Slots  1-{n:<2}    -> base class foreground")
        print(f"  Slots {n+1}-{2*n:<2}   -> base class-specific background")
        print(f"  Slot {self.num_slots-2}         -> novel class foreground  (FSIC-style)")
        print(f"  Slot {self.num_slots-1}         -> novel class background  (FSS addition)")
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

    # ── Core FSIC adaptive EMA — shared by Phase 1 and Phase 2 ───────
    def _ema_update(self, proto_new, slot_idx):
        """
        The FSIC adaptive EMA formula.
        Called identically for base classes (Phase 1) and
        novel classes (Phase 2) — this is the methodological extension.

        Warm-up: running mean for first WARMUP_N updates.
        After:   alpha = clamp(1 - sim, ALPHA_MIN, ALPHA_MAX)
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
        Pools feature_map over masked region using L2-norm attention.
        High-norm spatial locations contribute more to the prototype.
        Returns normalized [D] vector, or None if no valid pixels.
        """
        D, h, w = feature_map.shape[1:]

        mask_down = F.interpolate(
            mask.float().unsqueeze(1), size=(h, w), mode="nearest"
        )
        valid     = (mask_down != 255).float()
        mask_down = mask_down * valid

        if mask_down.sum() < 1:
            return None

        feat_norm = feature_map.norm(dim=1, keepdim=True)
        weight    = mask_down * feat_norm
        wsum      = weight.sum(dim=[0, 2, 3]).clamp(min=1e-6)
        proto     = (feature_map * weight).sum(dim=[0, 2, 3]) / wsum

        return F.normalize(proto, p=2, dim=0)

    # ── Phase 1: base class batch update ─────────────────────────────
    def update_from_batch(self, feature_map, binary_masks, class_labels):
        """
        Called after each Phase 1 training batch (no gradient).
        Updates fg slot, global bg slot, and class-specific bg slot
        for each sample in the batch using _ema_update.
        """
        B = feature_map.shape[0]

        for i in range(B):
            feat_i = feature_map[i].unsqueeze(0)
            mask_i = binary_masks[i].unsqueeze(0)
            cls    = class_labels[i]

            fg_mask = (mask_i == 1).long()
            bg_mask = (mask_i == 0).long()

            fg_proto = self._pool_prototype(feat_i, fg_mask)
            if fg_proto is not None:
                self._ema_update(fg_proto, self._fg_slot(cls))

            bg_proto = self._pool_prototype(feat_i, bg_mask)
            if bg_proto is not None:
                self._ema_update(bg_proto, slot_idx=0)
                self._ema_update(bg_proto, self._bg_slot(cls))

    # ── Phase 2: novel prototype building (FSIC-faithful) ────────────
    @torch.no_grad()
    def build_novel_prototype(self, support_features, support_masks,
                               novel_cls_id):
        """
        Builds novel class prototypes using the same FSIC adaptive EMA
        as Phase 1 base training.

        FSIC at test time:  novel support image -> EMA update -> slot
        FSS extension:      novel support image -> spatial masked pool
                            -> EMA update -> dedicated novel slot

        The _ema_update formula is identical in both phases.
        fg and bg are both built — necessary for binary segmentation.

        Parameters
        ----------
        support_features : list of K tensors, each [1, D, h, w]
        support_masks    : list of K tensors, each [1, H, W]
        novel_cls_id     : int
        """
        fg_slot = self._novel_fg_slot()   # 31
        bg_slot = self._novel_bg_slot()   # 32

        # Reset slots — fresh start for this novel class
        self.memory.data[fg_slot] = torch.zeros(
            self.feature_dim, device=self.memory.device
        )
        self.memory.data[bg_slot] = torch.zeros(
            self.feature_dim, device=self.memory.device
        )
        self.n_seen[fg_slot] = 0
        self.n_seen[bg_slot] = 0

        updates = 0
        for feat_i, mask_i in zip(support_features, support_masks):

            fg_mask = (mask_i == 1).long()
            bg_mask = (mask_i == 0).long()

            fg_proto = self._pool_prototype(feat_i, fg_mask)
            if fg_proto is not None:
                self._ema_update(fg_proto, fg_slot)

            bg_proto = self._pool_prototype(feat_i, bg_mask)
            if bg_proto is not None:
                self._ema_update(bg_proto, bg_slot)

            updates += 1

        self.novel_cls_to_slot[novel_cls_id] = fg_slot

        print(f"[APM] Novel class {novel_cls_id}: "
              f"fg->slot{fg_slot}, bg->slot{bg_slot}, "
              f"{updates} support image(s) via FSIC EMA")

    # ── Forward ───────────────────────────────────────────────────────
    def forward(self, feature_map, novel_cls_id=None):
        """
        Phase 1 (novel_cls_id=None): compare against all 33 slots.
        Phase 2/3 (novel_cls_id=int): binary [bg, fg] for novel class.
        Returns temperature-scaled cosine logits [B, S, h, w].
        """
        B, D, h, w = feature_map.shape
        feat_norm  = F.normalize(feature_map, p=2, dim=1)

        if novel_cls_id is None:
            mem = F.normalize(self.memory, p=2, dim=1)      # [33, D]
        else:
            bg_proto = F.normalize(
                self.memory.data[self._novel_bg_slot()], p=2, dim=0
            )
            fg_proto = F.normalize(
                self.memory.data[self._novel_fg_slot()], p=2, dim=0
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
        for param in self.parameters():
            param.requires_grad = False
        print("[SegAPM] All weights frozen for Phase 2/3.")
    def freeze_everything(self):
        return self.freeze_for_novel()   # alias for backward compatibility
        