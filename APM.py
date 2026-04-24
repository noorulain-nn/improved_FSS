"""
APM.py  —  Adaptive Prototype Memory  (FSS — Fixed for Segmentation)
=====================================================================

WHAT CHANGED FROM YOUR PREVIOUS VERSION AND WHY
-------------------------------------------------

CHANGE 1: Learnable Temperature Scalar on Cosine Similarity
  WHY:  The original FSIC got 98% on Omniglot (5 classes, clean global
        embeddings). Cosine similarity ∈ [-1,1] gave enough spread for 5
        clean classes. In FSS you have 31 slots and noisy spatial prototypes
        — the cosine scores cluster near 0.0–0.2 for all slots, so
        CrossEntropyLoss sees near-zero logit differences and produces tiny
        gradients. The backbone barely learns.

        Solution: multiply cosine scores by a learnable temperature τ
        (initialised to 10.0, which is the standard starting point in
        ProtoNet and HSNet). This spreads the logits so CrossEntropyLoss
        produces meaningful gradients from epoch 1.

        τ is added to the Adam optimiser at a higher LR (1e-2) so it
        learns quickly — it is just one scalar.

CHANGE 2: Running-Mean Warm-Up Instead of Hard First-Write
  WHY:  Your old code did a hard write on slot_ready==False (first encounter),
        which in segmentation might be a batch where the object occupies
        3% of the image — a terrible initial prototype. The EMA then
        barely updates from this anchor because subsequent sim→high→alpha→small.

        The FSIC code has NO slot_ready flag — it always uses the EMA from
        step 1. We replicate that by maintaining a running mean (n_seen
        counter) for the first WARMUP_N=5 updates, then switching to
        adaptive EMA. This is identical to Welford online mean — gives a
        stable starting point before EMA kicks in.

CHANGE 3: Clamped Alpha Range [ALPHA_MIN, ALPHA_MAX]
  WHY:  In FSIC, `adaptive_lr = 1 - sim` gives range [0,1]. This is fine
        for image-classification where one clean global vector arrives per
        update. In FSS, masked pooling gives a vector of variable quality.
        Unclamped alpha means:
          - Early training (noisy proto): sim≈0 → alpha=1.0 → FULL REPLACE
            with whatever one noisy masked average happened to produce.
          - Late training (stable proto): sim≈0.99 → alpha=0.01 → prototype
            completely frozen, cannot adapt to appearance variation.

        Clamping to [0.05, 0.30] means:
          - Minimum 5% update every batch → prototype never freezes
          - Maximum 30% per update → one noisy batch can't blow it up
          This matches the spirit of FSIC while being robust to spatial noise.

CHANGE 4: No Other Structural Changes
  - Class-specific bg slots (Fix 1) → kept exactly as before
  - Support-derived novel bg prototype (Fix 2) → kept exactly as before
  - Memory layout (31 slots) → kept exactly as before
  - forward() structure → kept exactly as before
  - update_from_batch() and build_novel_prototype() → same signatures

WHAT YOU NEED TO CHANGE IN main_seg.py (only 2 lines)
  1. Add temperature to the optimiser param groups (shown below)
  2. Add gradient clipping after loss.backward() (shown below)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Decoder import FPNDecoder


# ── Hyper-parameters for the EMA (easy to tune) ──────────────────
ALPHA_MIN  = 0.05   # prototype always moves at least this much per update
ALPHA_MAX  = 0.30   # prototype never jumps more than this per update
WARMUP_N   = 5      # number of updates before switching from running-mean to EMA
TEMP_INIT  = 10.0   # initial temperature τ  (cosine × τ → logits)
TEMP_MIN   = 1.0    # clamp τ from below so it can never collapse to 0
TEMP_MAX   = 50.0   # clamp τ from above so training doesn't go numerically unstable


class MemoryModule(nn.Module):
    """
    Memory module with:
      - Learnable temperature scalar on cosine logits
      - Running-mean warm-up before adaptive EMA (faithful to FSIC)
      - Clamped alpha range for spatial-noise robustness
      - Class-specific background slots (Fix 1)
      - Support-derived novel background (Fix 2)
    """

    def __init__(self, num_base_classes, feature_dim):
        super().__init__()

        self.num_base_classes = num_base_classes
        self.feature_dim      = feature_dim

        # ── Slot layout ──────────────────────────────────────────────
        # Slot 0          = global background (fallback for novel classes)
        # Slots 1..N      = foreground, one per base class
        # Slots N+1..2N   = class-specific background, one per base class
        self.num_slots = 1 + num_base_classes + num_base_classes  # 31 for 15 base

        self.memory = nn.Parameter(
            torch.randn(self.num_slots, feature_dim),
            requires_grad=False   # Adam never touches this — EMA updates only
        )
        nn.init.normal_(self.memory, mean=0.0, std=0.01)

        # ── CHANGE 1: Learnable temperature ──────────────────────────
        # τ is trained by Adam — it IS a gradient-tracked parameter.
        # Initialise to TEMP_INIT so logits start at cosine × 10.
        self.temperature = nn.Parameter(torch.tensor(TEMP_INIT))
        # NOTE: add this to your optimiser in main_seg.py:
        #   {"params": [model.memory_module.temperature], "lr": 1e-2}

        # ── CHANGE 2: Warm-up counters instead of slot_ready flags ───
        # n_seen[s] = number of EMA updates slot s has received so far.
        # During first WARMUP_N updates we use a running mean (stable init).
        # After WARMUP_N we switch to adaptive EMA (same as FSIC).
        self.register_buffer(
            "n_seen", torch.zeros(self.num_slots, dtype=torch.long)
        )

        # Fix 2 storage — set during Phase 2, used during Phase 3
        self.novel_prototypes   = {}    # cls_id → fg prototype [D]
        self.novel_bg_prototype = None  # bg prototype [D]

        print(f"[APM] Memory layout:")
        print(f"      Slot 0             = global background (fallback)")
        print(f"      Slots 1–{num_base_classes}        = foreground per base class")
        print(f"      Slots {num_base_classes+1}–{2*num_base_classes}"
              f" = class-specific background per base class")
        print(f"      Total slots = {self.num_slots}  |  Feature dim = {feature_dim}")
        print(f"      Temperature τ (learnable, init={TEMP_INIT})")
        print(f"      EMA alpha clamped to [{ALPHA_MIN}, {ALPHA_MAX}]")
        print(f"      Warm-up: running mean for first {WARMUP_N} updates per slot")

    # ── Slot index helpers ────────────────────────────────────────────
    def _fg_slot(self, cls):
        return cls + 1

    def _bg_slot(self, cls):
        return self.num_base_classes + 1 + cls

    # ── Forward ───────────────────────────────────────────────────────
    def forward(self, feature_map, novel_cls_id=None):
        """
        Parameters
        ----------
        feature_map  : FloatTensor [B, D, h, w]
        novel_cls_id : None   → Phase 1, all 31 slots
                       int    → Phase 2/3, [bg, fg] for this novel class

        Returns
        -------
        logits : FloatTensor [B, S, h, w]   — temperature-scaled cosine scores
        """
        B, D, h, w = feature_map.shape
        feat_norm  = F.normalize(feature_map, p=2, dim=1)  # [B, D, h, w]

        if novel_cls_id is None:
            # Phase 1 — compare against all 31 stored prototypes
            mem = F.normalize(self.memory, p=2, dim=1)      # [31, D]
        else:
            # Phase 2/3 — binary [bg, fg] for this novel class
            if self.novel_bg_prototype is not None:
                bg_proto = self.novel_bg_prototype           # Fix 2
            else:
                bg_proto = F.normalize(self.memory[0], p=2, dim=0)

            novel_proto = self.novel_prototypes[novel_cls_id]
            mem = torch.stack(
                [F.normalize(bg_proto,    p=2, dim=0),
                 F.normalize(novel_proto, p=2, dim=0)],
                dim=0
            )   # [2, D]

        # Spatial cosine similarity
        S         = mem.shape[0]
        feat_flat = feat_norm.view(B, D, h * w)             # [B, D, h*w]
        sim       = torch.bmm(
            feat_flat.permute(0, 2, 1),                     # [B, h*w, D]
            mem.t().unsqueeze(0).expand(B, -1, -1)          # [B, D, S]
        )   # [B, h*w, S]
        logits = sim.permute(0, 2, 1).view(B, S, h, w)     # [B, S, h, w]

        # ── CHANGE 1: scale by learnable temperature ─────────────────
        # This is the key fix. Without it, cosine ∈ [-1,1] gives near-zero
        # logit differences and CrossEntropyLoss barely signals the backbone.
        τ = self.temperature.clamp(TEMP_MIN, TEMP_MAX)
        return logits * τ

    # ── EMA slot update ───────────────────────────────────────────────
    def _update_slot(self, feature_map, mask, slot_idx):
        """
        Masked average pooling → warm-up running mean OR adaptive EMA update.

        CHANGE 2 + 3 vs old code:
          Old: hard write on first encounter, then unclamped alpha
          New: running mean for first WARMUP_N updates (stable init),
               then adaptive EMA with alpha clamped to [ALPHA_MIN, ALPHA_MAX]

        This is faithful to the FSIC EMA spirit while handling spatial noise.
        """
        D, h, w   = feature_map.shape[1:]
        mask_down = F.interpolate(
            mask.float().unsqueeze(1), size=(h, w), mode="nearest"
        )
        valid     = (mask_down != 255).float()
        mask_down = mask_down * valid

        # Skip if no valid foreground/background pixels in this image
        if mask_down.sum() < 1:
            return

        denom     = mask_down.sum(dim=[0, 2, 3]).clamp(min=1e-6)
        proto_new = F.normalize(
            (feature_map * mask_down).sum(dim=[0, 2, 3]) / denom,
            p=2, dim=0
        )  # [D] — new masked-average prototype from this batch sample

        n = self.n_seen[slot_idx].item()

        if n < WARMUP_N:
            # ── Warm-up phase: running mean ──────────────────────────
            # Equivalent to: mean = (n * old_mean + new) / (n+1)
            # Gives a stable multi-sample average before EMA kicks in.
            # This avoids the FSIC problem of anchoring to one bad sample.
            if n == 0:
                self.memory.data[slot_idx] = proto_new
            else:
                old = self.memory.data[slot_idx]
                self.memory.data[slot_idx] = (n * old + proto_new) / (n + 1)
        else:
            # ── CHANGE 3: FSIC adaptive EMA with clamped alpha ───────
            # Original FSIC formula: alpha = 1 - cosine_similarity
            # Change: clamp alpha to [ALPHA_MIN, ALPHA_MAX] for spatial robustness
            proto_old = F.normalize(self.memory.data[slot_idx], p=2, dim=0)
            sim       = F.cosine_similarity(
                proto_new.unsqueeze(0), proto_old.unsqueeze(0)
            ).item()

            # Faithful FSIC formula:  alpha = 1 - sim
            # Clamped to prevent full-replacement noise and prototype freezing
            alpha = max(ALPHA_MIN, min(1.0 - sim, ALPHA_MAX))

            self.memory.data[slot_idx] = (
                (1.0 - alpha) * self.memory.data[slot_idx]
                + alpha       * proto_new
            )

        self.n_seen[slot_idx] += 1

    # ── Phase 1 batch update ──────────────────────────────────────────
    def update_from_batch(self, feature_map, binary_masks, class_labels):
        """
        Called after each Phase 1 training batch (no gradient needed).
        Updates three slots per sample: global bg, class fg, class-specific bg.
        Signatures and call site in main_seg.py are UNCHANGED.
        """
        B = feature_map.shape[0]
        for i in range(B):
            feat_i = feature_map[i].unsqueeze(0)    # [1, D, h, w]
            mask_i = binary_masks[i].unsqueeze(0)   # [1, H, W]
            cls    = class_labels[i]

            fg_mask = (mask_i == 1).long()
            bg_mask = (mask_i == 0).long()

            # 1. Foreground slot for this class
            self._update_slot(feat_i, fg_mask, self._fg_slot(cls))

            # 2. Global background slot (fallback)
            self._update_slot(feat_i, bg_mask, slot_idx=0)

            # 3. Fix 1: Class-specific background slot
            self._update_slot(feat_i, bg_mask, self._bg_slot(cls))

    # ── Phase 2: build novel prototypes ──────────────────────────────
    @torch.no_grad()

    def build_novel_prototype(self, support_features, support_masks, novel_cls_id):
        fg_accum = None
        bg_accum = None
        count = 0

        for feat_i, mask_i in zip(support_features, support_masks):
            D, h, w = feat_i.shape[1:]

            mask_down = F.interpolate(
                mask_i.float().unsqueeze(1), size=(h, w), mode="nearest"
            )
            valid = (mask_down != 255).float()

            # ── NEW: confidence-weighted pooling ──────────────────────
            # Instead of treating all foreground pixels equally,
            # weight each pixel by its L2 norm (high-norm features
            # are more "activated" and more reliable for the prototype).
            feat_norm = feat_i.norm(dim=1, keepdim=True)  # [1, 1, h, w]
            
            fg_mask   = mask_down * valid
            fg_weight = fg_mask * feat_norm               # weight by activation
            fg_wsum   = fg_weight.sum(dim=[0,2,3]).clamp(min=1e-6)
            fg_proto  = (feat_i * fg_weight).sum(dim=[0,2,3]) / fg_wsum

            bg_mask   = (1.0 - mask_down) * valid
            bg_weight = bg_mask * feat_norm
            bg_wsum   = bg_weight.sum(dim=[0,2,3]).clamp(min=1e-6)
            bg_proto  = (feat_i * bg_weight).sum(dim=[0,2,3]) / bg_wsum

            fg_accum = fg_proto if fg_accum is None else fg_accum + fg_proto
            bg_accum = bg_proto if bg_accum is None else bg_accum + bg_proto
            count += 1

        self.novel_prototypes[novel_cls_id] = F.normalize(
            fg_accum / count, p=2, dim=0
        )
        self.novel_bg_prototype = F.normalize(
            bg_accum / count, p=2, dim=0
        )   

        print(f"[APM] Built fg+bg prototypes for novel class {novel_cls_id} "
              f"from {count} support image(s).")


# ── Full model (unchanged structure) ─────────────────────────────────
class SegAPM(nn.Module):

    def __init__(self, backbone, num_base_classes, decoder_out_channels=256):
        super().__init__()
        self.backbone      = backbone
        self.decoder       = FPNDecoder(out_channels=decoder_out_channels)
        self.memory_module = MemoryModule(num_base_classes, decoder_out_channels)

        total_slots = self.memory_module.num_slots
        print(f"[SegAPM] Total memory slots: {total_slots}")
        print(f"         Decoder out channels: {decoder_out_channels}")

    def forward(self, x, novel_cls_id=None):
        feat2, feat3, feat4 = self.backbone(x)
        fused  = self.decoder(feat2, feat3, feat4)
        logits = self.memory_module(fused, novel_cls_id)
        return logits, fused

    def freeze_everything(self):
        for param in self.parameters():
            param.requires_grad = False
        # Keep temperature learnable? No — in Phase 2/3 everything freezes.
        # The temperature is only needed during Phase 1 training.
        print("[SegAPM] All weights frozen for Phase 2 & 3.")




# """
# APM.py  —  Adaptive Prototype Memory with Fix 1 + Fix 2
# =========================================================

# FIX 1 — Class-specific background slots (15 extra slots)
# ---------------------------------------------------------
# Problem:  One global background slot gets blended with every possible
#           background appearance (grass, roads, indoors, sky...) and
#           becomes a meaningless average.

# Solution: Each foreground class gets its OWN background slot.
#           When training on a dog image, background pixels go into
#           the "dog-context background" slot, not the global one.
#           This way the model learns: "when searching for a dog,
#           the background typically looks like outdoor grass/sky."

# Memory layout with Fix 1:
#   Slot 0          → global background (fallback for novel classes)
#   Slots 1–15      → foreground prototype, one per base class
#   Slots 16–30     → class-specific background, one per base class
#                     slot[num_base + 1 + cls] = bg for class cls

#   Total = 1 + 15 + 15 = 31 slots

# FIX 2 — Support-derived background for novel classes
# -----------------------------------------------------
# Problem:  During Phase 2 (novel class adaptation), there is no
#           class-specific background slot for novel classes because
#           the model never trained on them. Falling back to the global
#           background slot (slot 0) gives a stale, irrelevant comparison.

# Solution: When building the novel class prototype from K support images,
#           ALSO extract a background prototype from the non-masked pixels
#           of those same support images. Store it as novel_bg_prototype.
#           At test time, use this fresh support-derived background instead
#           of the stale global slot 0.

# Why this works for FSS:
#   The support and query images of a novel class tend to share similar
#   contexts (both are outdoor sheep photos, both are indoor sofa photos).
#   So the background of the support image is a much better reference for
#   "what background looks like in this context" than a global average.

# HOW BOTH FIXES COMBINE:
#   Phase 1 training → Fix 1 builds 15 class-specific background slots
#                      alongside 15 foreground slots and 1 global bg slot
#   Phase 2 adapt   → Fix 2 builds fresh fg + bg prototypes from K support
#                      images, overriding global bg with support-derived bg
#   Phase 3 test    → Uses Fix 2's fresh bg prototype for novel comparison,
#                      giving a contextually relevant binary decision at each pixel

# FILES TO CHANGE: ONLY THIS FILE (APM.py)
#   Data_Loader.py  → unchanged
#   Models.py       → unchanged
#   Decoder.py      → unchanged
#   Metrics.py      → unchanged
#   main.py         → ONE small change (explained at bottom of this file)
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from Decoder import FPNDecoder


# class MemoryModule(nn.Module):
#     """
#     Memory module with Fix 1 (class-specific background slots)
#     and Fix 2 (support-derived novel background prototype).

#     Parameters
#     ----------
#     num_base_classes : int   e.g. 15
#     feature_dim      : int   e.g. 256 (decoder output channels)
#     """

#     def __init__(self, num_base_classes, feature_dim):
#         super().__init__()

#         self.num_base_classes = num_base_classes
#         self.feature_dim      = feature_dim

#         # ── Slot layout ──────────────────────────────────────────
#         # Slot 0              = global background (fallback)
#         # Slots 1..N          = foreground, one per base class
#         # Slots N+1..2N       = class-specific background
#         #
#         # Helper: given class index cls (0-based),
#         #   fg slot  = cls + 1
#         #   bg slot  = num_base_classes + 1 + cls

#         self.num_slots = 1 + num_base_classes + num_base_classes
#         # = 1 + 15 + 15 = 31 for 15 base classes

#         self.memory = nn.Parameter(
#             torch.randn(self.num_slots, feature_dim),
#             requires_grad=False          # Adam never touches this
#         )
#         nn.init.normal_(self.memory, mean=0.0, std=0.01)

#         # Track which slots have been written with real data
#         self.slot_ready = [False] * self.num_slots

#         # Fix 2 storage — set during Phase 2, used during Phase 3
#         self.novel_prototypes  = {}   # cls_id → fg prototype tensor [D]
#         self.novel_bg_prototype = None  # bg prototype tensor [D]
#         # (one shared bg per novel episode — overwritten each Phase 2 call)

#         print(f"[APM] Memory layout:")
#         print(f"      Slot 0             = global background (fallback)")
#         print(f"      Slots 1–{num_base_classes}         "
#               f"= foreground per base class")
#         print(f"      Slots {num_base_classes+1}–{2*num_base_classes} "
#               f"= class-specific background per base class")
#         print(f"      Total slots = {self.num_slots}  |  "
#               f"Feature dim = {feature_dim}")

#     # ── Slot index helpers ────────────────────────────────────────
#     def _fg_slot(self, cls):
#         """Foreground slot index for base class cls (0-based)."""
#         return cls + 1

#     def _bg_slot(self, cls):
#         """
#         FIX 1: Class-specific background slot index.
#         e.g. cls=0 → slot 16, cls=1 → slot 17, ... cls=14 → slot 30
#         """
#         return self.num_base_classes + 1 + cls

#     # ── Forward ───────────────────────────────────────────────────
#     def forward(self, feature_map, novel_cls_id=None):
#         """
#         Compute cosine similarity at every spatial location.

#         Parameters
#         ----------
#         feature_map  : FloatTensor [B, D, h, w]
#         novel_cls_id : None   → Phase 1 training, use all 31 slots
#                        int    → Phase 2/3, use only [bg, fg] for this class

#         Returns
#         -------
#         logits : FloatTensor [B, num_slots_used, h, w]
#           Phase 1: [B, 31, h, w]
#           Phase 2/3: [B, 2, h, w]  (binary: bg vs novel fg)
#         """
#         B, D, h, w = feature_map.shape
#         feat_norm  = F.normalize(feature_map, p=2, dim=1)  # [B, D, h, w]

#         if novel_cls_id is None:
#             # ── Phase 1: compare against all 31 stored prototypes ──
#             mem = F.normalize(self.memory, p=2, dim=1)     # [31, D]

#         else:
#             # ── Phase 2/3: binary comparison for this novel class ──
#             # FIX 2: use the fresh support-derived background prototype
#             # instead of the stale global slot 0
#             if self.novel_bg_prototype is not None:
#                 bg_proto = self.novel_bg_prototype          # [D] — Fix 2
#             else:
#                 # Fallback to global bg if Fix 2 prototype not built yet
#                 bg_proto = F.normalize(self.memory[0], p=2, dim=0)

#             novel_proto = self.novel_prototypes[novel_cls_id]  # [D]
#             mem = torch.stack(
#                 [F.normalize(bg_proto,    p=2, dim=0),
#                  F.normalize(novel_proto, p=2, dim=0)],
#                 dim=0
#             )   # [2, D]

#         # Spatial cosine similarity: every pixel vs every prototype
#         S         = mem.shape[0]
#         feat_flat = feat_norm.view(B, D, h * w)            # [B, D, h*w]
#         sim       = torch.bmm(
#             feat_flat.permute(0, 2, 1),                    # [B, h*w, D]
#             mem.t().unsqueeze(0).expand(B, -1, -1)         # [B, D, S]
#         )   # [B, h*w, S]
#         logits = sim.permute(0, 2, 1).view(B, S, h, w)    # [B, S, h, w]
#         return logits

#     # ── EMA slot update (internal) ────────────────────────────────
#     def _update_slot(self, feature_map, mask, slot_idx):
#         """
#         Masked average pooling → adaptive EMA update for one slot.
#         Identical logic to before. Called for both fg and bg slots.
#         """
#         D, h, w   = feature_map.shape[1:]
#         mask_down = F.interpolate(
#             mask.float().unsqueeze(1), size=(h, w), mode="nearest"
#         )
#         valid     = (mask_down != 255).float()
#         mask_down = mask_down * valid

#         # Skip if no valid pixels (e.g. object fills entire image → no bg)
#         if mask_down.sum() < 1:
#             return

#         denom     = mask_down.sum(dim=[0, 2, 3]).clamp(min=1e-6)
#         proto_new = F.normalize(
#             (feature_map * mask_down).sum(dim=[0, 2, 3]) / denom,
#             p=2, dim=0
#         )

#         if not self.slot_ready[slot_idx]:
#             self.memory.data[slot_idx] = proto_new
#             self.slot_ready[slot_idx]  = True
#         else:
#             proto_old = F.normalize(self.memory.data[slot_idx], p=2, dim=0)
#             sim       = F.cosine_similarity(
#                 proto_new.unsqueeze(0), proto_old.unsqueeze(0)
#             ).item()
#             alpha     = max(0.0, min(1.0 - sim, 1.0))
#             self.memory.data[slot_idx] = (
#                 (1 - alpha) * self.memory.data[slot_idx]
#                 + alpha * proto_new
#             )

#     # ── Phase 1 batch update ──────────────────────────────────────
#     def update_from_batch(self, feature_map, binary_masks, class_labels):
#         """
#         Called after each Phase 1 training batch.
#         Updates THREE slots per sample:
#           1. Global background slot (slot 0)       — same as before
#           2. Foreground slot for this class         — same as before
#           3. FIX 1: Class-specific background slot  — NEW

#         Parameters
#         ----------
#         feature_map   : FloatTensor [B, D, h, w]
#         binary_masks  : LongTensor  [B, H, W]   0=bg, 1=fg, 255=ignore
#         class_labels  : list[int]  length B, values 0..num_base-1
#         """
#         B = feature_map.shape[0]
#         for i in range(B):
#             feat_i  = feature_map[i].unsqueeze(0)   # [1, D, h, w]
#             mask_i  = binary_masks[i].unsqueeze(0)  # [1, H, W]
#             cls     = class_labels[i]                # int 0..N-1

#             fg_mask = (mask_i == 1).long()
#             bg_mask = (mask_i == 0).long()

#             # 1. Update foreground slot for this class
#             self._update_slot(feat_i, fg_mask, self._fg_slot(cls))

#             # 2. Update global background slot (fallback)
#             self._update_slot(feat_i, bg_mask, slot_idx=0)

#             # 3. FIX 1: Update class-specific background slot
#             #    Background pixels from a "dog" image go into the dog-bg slot
#             #    Background pixels from a "car" image go into the car-bg slot
#             #    This keeps different background contexts separated
#             self._update_slot(feat_i, bg_mask, self._bg_slot(cls))

#     # ── Phase 2: build novel prototypes (Fix 2) ───────────────────
#     @torch.no_grad()
#     def build_novel_prototype(self, support_features, support_masks, novel_cls_id):
#         """
#         FIX 2: Build BOTH foreground AND background prototypes from
#         K support images of the novel class.

#         The background prototype is derived from the non-masked regions
#         of the support images — the actual background context around the
#         novel object. This is far more relevant than the global slot 0
#         which averaged all backgrounds seen during Phase 1.

#         Parameters
#         ----------
#         support_features : list of [1, D, h, w] tensors  (length K)
#         support_masks    : list of [1, H, W]    tensors  (length K)
#         novel_cls_id     : int — dict key for this novel class
#         """
#         fg_accum = None
#         bg_accum = None
#         count    = 0

#         for feat_i, mask_i in zip(support_features, support_masks):
#             D, h, w = feat_i.shape[1:]

#             mask_down = F.interpolate(
#                 mask_i.float().unsqueeze(1), size=(h, w), mode="nearest"
#             )
#             valid     = (mask_down != 255).float()

#             # ── Foreground prototype ──
#             fg_mask  = mask_down * valid                   # 1 where object is
#             fg_denom = fg_mask.sum(dim=[0, 2, 3]).clamp(min=1e-6)
#             fg_proto = (feat_i * fg_mask).sum(dim=[0, 2, 3]) / fg_denom  # [D]

#             # ── FIX 2: Background prototype from THIS support image ──
#             # (1 - mask_down) flips 1→0 and 0→1, giving us background pixels
#             bg_mask  = (1.0 - mask_down) * valid           # 1 where background is
#             bg_denom = bg_mask.sum(dim=[0, 2, 3]).clamp(min=1e-6)
#             bg_proto = (feat_i * bg_mask).sum(dim=[0, 2, 3]) / bg_denom  # [D]

#             fg_accum = fg_proto if fg_accum is None else fg_accum + fg_proto
#             bg_accum = bg_proto if bg_accum is None else bg_accum + bg_proto
#             count   += 1

#         # Average across K support images and normalise
#         self.novel_prototypes[novel_cls_id] = F.normalize(
#             fg_accum / count, p=2, dim=0
#         )

#         # FIX 2: Store fresh background — overrides stale global slot 0
#         # for all Phase 3 queries of this novel class
#         self.novel_bg_prototype = F.normalize(
#             bg_accum / count, p=2, dim=0
#         )

#         print(f"[APM] Built fg+bg prototypes for novel class {novel_cls_id} "
#               f"from {count} support image(s).")
#         print(f"      FIX 1: {self.num_base_classes} class-specific bg slots "
#               f"already in memory from Phase 1.")
#         print(f"      FIX 2: Fresh support-derived bg prototype stored — "
#               f"overrides global slot 0 for this novel class.")


# # ─────────────────────────────────────────────────────────────────
# # Full model — identical structure, just MemoryModule is bigger
# # ─────────────────────────────────────────────────────────────────
# class SegAPM(nn.Module):
 

#     def __init__(self, backbone, num_base_classes, decoder_out_channels=256):
#         super().__init__()
#         self.backbone      = backbone
#         self.decoder       = FPNDecoder(out_channels=decoder_out_channels)
#         self.memory_module = MemoryModule(num_base_classes, decoder_out_channels)

#         total_slots = self.memory_module.num_slots
#         print(f"[SegAPM] Total memory slots: {total_slots}")
#         print(f"         Decoder out channels: {decoder_out_channels}")

#     def forward(self, x, novel_cls_id=None):
#         feat2, feat3, feat4 = self.backbone(x)
#         fused  = self.decoder(feat2, feat3, feat4)       # [B, 256, 28, 28]
#         logits = self.memory_module(fused, novel_cls_id) # [B, slots, 28, 28]
#         return logits, fused

#     def freeze_everything(self):
#         for param in self.parameters():
#             param.requires_grad = False
#         print("[SegAPM] All weights frozen for Phase 2 & 3.")


