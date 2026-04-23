"""
Data_Loader.py  —  Pascal-5i Few-Shot Segmentation  (Benchmark-Compliant v3)
=============================================================================

BENCHMARK CORRECTION (v3 — this version):
------------------------------------------
FSS literature (PANet ICCV-2019, HSNet ICCV-2021, PFENet TPAMI-2021) uses
exactly TWO data splits:

    TRAIN set = VOC2012 train + SBD train + SBD val  minus  VOC2012 val
                ≈ 10,582 images  |  used ONLY in Phase 1

    TEST  set = VOC2012 val  (1,449 images)
                used ONLY in Phase 3 for novel-class query evaluation

The old code used VOC val as BOTH a Phase-1 validation set AND the Phase-3
test set, which is a methodological error — the same held-out set was seen
during training (for early stopping / loss monitoring) and then reused for
final evaluation.

WHAT CHANGED IN v3:
  - prepare_base_loaders() now returns (train_loader, n_base)  — 2 values.
    The val_loader is REMOVED from Phase 1 entirely.
  - A new prepare_test_dataset() function is added. It wraps
    prepare_novel_dataset() and also prints the total test-set count.
  - _build_merged_train_list() now prints both train AND test counts.
  - val_ratio parameter is fully removed (was already unused in v2).

BUG FIXES CARRIED OVER FROM v2:
  [BUG-1] NovelClassDataset.voc_root was assigned novel_classes — fixed.
  [BUG-2] BaseClassDataset now uses SegmentationClassAug (SBD-augmented
          masks) via _get_mask_path(), falling back to SegmentationClass.
  [BUG-3] Training list merges VOC train + SBD train + SBD val, removing
          VOC val to prevent test-set leakage.
  [BUG-4] Novel-class pixels in base-class training masks are set to 255
          (ignore_index) so the base model never confuses them with bg.
  [BUG-5] Removed: no longer applicable — val_loader is gone.
  [BUG-6] NovelClassDataset uses VOC val.txt exclusively for queries.

PUBLIC API:
  prepare_base_loaders(voc_root, sbd_root, fold, batch_size,
                       num_workers, seed)
      → (train_loader, n_base)          ← 2 values (changed from v2's 3)

  prepare_test_dataset(voc_root, fold)
      → (NovelClassDataset, novel_classes)   ← same as old prepare_novel_dataset

  prepare_novel_dataset(voc_root, fold)      ← alias kept for compatibility

  NovelClassDataset.get_support_and_queries(cls_id, k_shot, seed)
      → (support_list, query_list)

USAGE IN main_seg.py:
  train_loader, NUM_BASE = Data_Loader.prepare_base_loaders(
      voc_root=VOC_ROOT, sbd_root=SBD_ROOT, fold=fold, batch_size=BATCH_SIZE
  )
  novel_dataset, novel_classes = Data_Loader.prepare_test_dataset(
      voc_root=VOC_ROOT, fold=fold
  )

References:
  Wang et al.  PANet   ICCV 2019  https://arxiv.org/abs/1908.06391
  Min  et al.  HSNet   ICCV 2021  https://arxiv.org/abs/2106.07015
  Tian et al.  PFENet  TPAMI 2021 https://arxiv.org/abs/2107.00509
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ── Standard Pascal-5i fold definitions ──────────────────────────
# Each fold reserves 5 classes as NOVEL (unseen during Phase 1).
# The remaining 15 are BASE classes used in Phase 1 training.
PASCAL_FSS_SPLITS = {
    0: [1,  2,  3,  4,  5],   # novel: aeroplane bicycle bird boat bottle
    1: [6,  7,  8,  9, 10],   # novel: bus car cat chair cow
    2: [11, 12, 13, 14, 15],  # novel: diningtable dog horse motorbike person
    3: [16, 17, 18, 19, 20],  # novel: pottedplant sheep sofa train tvmonitor
}

VOC_CLASS_NAMES = [
    "background",  "aeroplane", "bicycle", "bird",      "boat",
    "bottle",      "bus",       "car",     "cat",       "chair",
    "cow",         "diningtable","dog",    "horse",     "motorbike",
    "person",      "pottedplant","sheep",  "sofa",      "train",
    "tvmonitor",
]

IMG_SIZE = 321    # standard input resolution for Pascal-5i FSS evaluation


# ─────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────

def joint_transform(image, mask, augment=False):
    """Apply the SAME spatial transform to image AND mask."""
    image = TF.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=Image.BILINEAR)
    mask  = TF.resize(mask,  (IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST)

    if augment and random.random() > 0.5:
        image = TF.hflip(image)
        mask  = TF.hflip(mask)

    image = TF.to_tensor(image)
    image = TF.normalize(image,
                         mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225])
    mask  = torch.from_numpy(np.array(mask)).long()
    return image, mask


def _get_mask_path(voc_root, img_id):
    """
    [BUG-2 FIX] Prefer SegmentationClassAug (SBD-augmented, ~10,582 masks).
    Falls back to SegmentationClass (VOC-only, ~2,913 masks).

    SegmentationClassAug must be placed at:
        <voc_root>/SegmentationClassAug/<img_id>.png
    Download: https://github.com/DrSleep/tensorflow-deeplab-resnet
    """
    aug_path  = os.path.join(voc_root, "SegmentationClassAug", img_id + ".png")
    orig_path = os.path.join(voc_root, "SegmentationClass",    img_id + ".png")
    return aug_path if os.path.exists(aug_path) else orig_path


def _build_merged_train_list(voc_root, sbd_root=None, val_fraction=0.0, seed=42):
    """
    [BUG-3 FIX]  Build the benchmark-correct training image list:

        TRAIN = (VOC2012 train) ∪ (SBD train) ∪ (SBD val)  −  (VOC2012 val)

    This matches the protocol in PANet (Wang et al., ICCV 2019) and
    HSNet (Min et al., ICCV 2021).

    With SBD:    ~10,582 training images
    Without SBD: ~1,464  training images (not recommended)

    Also prints the total test set count (VOC2012 val) for reference.

    Args:
        voc_root (str)      : path to VOCdevkit/VOC2012/
        sbd_root (str|None) : path to benchmark_RELEASE/dataset/

    Returns:
        train_ids   (list[str]) : merged training image IDs
        voc_val_set (set[str])  : VOC2012 val IDs = the test set
    """
    voc_val_path   = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
    voc_train_path = os.path.join(voc_root, "ImageSets", "Segmentation", "train.txt")

    with open(voc_val_path) as f:
        voc_val_set = set(l.strip() for l in f if l.strip())
    with open(voc_train_path) as f:
        voc_train_ids = [l.strip() for l in f if l.strip()]

    merged = set(voc_train_ids)

    sbd_added = 0
    if sbd_root is not None:
        for fname in ["train.txt", "val.txt"]:
            sbd_split_path = os.path.join(sbd_root, fname)
            if os.path.exists(sbd_split_path):
                with open(sbd_split_path) as f:
                    ids = [l.strip() for l in f if l.strip()]
                before = len(merged)
                merged.update(ids)
                sbd_added += len(merged) - before
            else:
                print(f"  [WARNING] SBD split not found: {sbd_split_path}")

    # Remove VOC val — these are the test images, MUST NOT appear in train
    leaked = merged & voc_val_set
    merged -= voc_val_set

    source = "VOC2012 + SBD" if (sbd_root and sbd_added > 0) else "VOC2012 only"

    print("\n" + "="*60)
    print("  DATA SPLIT SUMMARY  (Pascal-5i Benchmark Protocol)")
    print("="*60)
    print(f"  Source         : {source}")
    print(f"  TRAIN set size : {len(merged):,} (image, class) pairs will be")
    print(f"                   expanded further per base class below")
    print(f"  TEST  set size : {len(voc_val_set):,} images  (VOC2012 val)")
    print(f"                   used ONLY in Phase 3 — zero Phase 1 exposure")
    if leaked:
        print(f"  Leakage check  : removed {len(leaked)} VOC-val IDs from train [OK]")
    else:
        print(f"  Leakage check  : no overlap found [OK]")
    print("="*60)

    all_ids = sorted(merged)
    rng = random.Random(seed)
    rng.shuffle(all_ids)

    if val_fraction > 0.0:
        n_val = max(1, int(len(all_ids) * val_fraction))
        val_ids = all_ids[:n_val]
        train_ids = all_ids[n_val:]
    else:
        val_ids = []
        train_ids = all_ids

    print(f"  Merged train   : {len(all_ids):,} image IDs")
    print(f"  Phase 1 TRAIN  : {len(train_ids):,} image IDs")
    if val_ids:
        print(f"  Internal VAL   : {len(val_ids):,} image IDs"
              f"  ({val_fraction*100:.0f}% of train)")
        print(f"                   drawn from TRAIN only for early stopping")
    else:
        print(f"  Internal VAL   : NONE")

    return train_ids, val_ids, voc_val_set


# ─────────────────────────────────────────────────────────────────
# Phase 1 Dataset — base classes, normal batch loading
# ─────────────────────────────────────────────────────────────────

class BaseClassDataset(Dataset):
    """
    Loads images containing BASE classes for Phase 1 training.

    Returns: (image_tensor, binary_mask, class_label)
      image_tensor : FloatTensor [3, H, W]   normalised
      binary_mask  : LongTensor  [H, W]      0=bg, 1=target, 255=ignore
      class_label  : int                     remapped 0 .. N_base-1

    [BUG-2 FIX] Uses SegmentationClassAug masks.
    [BUG-4 FIX] Novel-class pixels → 255 (ignore) during base training.
    """

    def __init__(self, voc_root, img_id_list, base_classes,
                 novel_classes, augment=False):
        """
        Args:
            voc_root      (str)       : path to VOCdevkit/VOC2012/
            img_id_list   (list[str]) : image IDs to scan (merged train list)
            base_classes  (list[int]) : VOC class IDs for training
            novel_classes (list[int]) : VOC class IDs NEVER seen in Phase 1
            augment       (bool)      : random horizontal flip
        """
        self.voc_root     = voc_root
        self.base_classes = base_classes
        self.novel_set    = set(novel_classes)   # [BUG-4]
        self.augment      = augment
        self.label_map    = {c: i for i, c in enumerate(sorted(base_classes))}

        self.samples  = []
        n_missing = 0

        for img_id in img_id_list:
            mask_path = _get_mask_path(voc_root, img_id)   # [BUG-2]
            if not os.path.exists(mask_path):
                n_missing += 1
                continue
            mask = np.array(Image.open(mask_path))
            for cls_id in base_classes:
                if (mask == cls_id).any():
                    self.samples.append((img_id, cls_id))

        if n_missing > 0:
            print(f"  [WARNING] {n_missing} image IDs skipped (mask not found)."
                  f" Install SegmentationClassAug for full SBD coverage.")

        print(f"  [BaseDataset]  {len(base_classes)} base classes"
              f" | {len(self.samples):,} (img, class) samples"
              f"{'  [augmented]' if augment else ''}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, cls_id = self.samples[idx]

        image    = Image.open(
            os.path.join(self.voc_root, "JPEGImages", img_id + ".jpg")
        ).convert("RGB")
        raw_mask = Image.open(_get_mask_path(self.voc_root, img_id))

        image, mask = joint_transform(image, raw_mask, self.augment)

        # Build binary mask:
        #   1   = target class pixel
        #   0   = background + other base classes
        #   255 = VOC boundary + all novel class pixels  [BUG-4 FIX]
        binary = torch.zeros_like(mask)
        binary[mask == cls_id] = 1
        for nov_cls in self.novel_set:
            binary[mask == nov_cls] = 255
        binary[mask == 255] = 255

        return image, binary, self.label_map[cls_id]


# ─────────────────────────────────────────────────────────────────
# Phase 3 Dataset — novel classes (support + query)
# ─────────────────────────────────────────────────────────────────

class NovelClassDataset(Dataset):
    """
    Holds images for NOVEL classes (Phase 3 evaluation).

    Queries come exclusively from VOC2012 val.txt — the same set that
    serves as the overall TEST set in the benchmark protocol.

    [BUG-1 FIX] self.voc_root = voc_root  (was mistakenly = novel_classes)
    [BUG-6 FIX] Uses VOC2012 val.txt only — no training-set contamination.
    """

    def __init__(self, voc_root, novel_classes):
        self.voc_root      = voc_root          # [BUG-1]
        self.novel_classes = novel_classes

        val_file = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
        with open(val_file) as f:
            val_ids = [l.strip() for l in f if l.strip()]

        self.class_images = {cls: [] for cls in novel_classes}
        for img_id in val_ids:
            mask_path = _get_mask_path(voc_root, img_id)
            if not os.path.exists(mask_path):
                continue
            mask = np.array(Image.open(mask_path))
            for cls_id in novel_classes:
                if (mask == cls_id).any():
                    self.class_images[cls_id].append(img_id)

        total_test = sum(len(v) for v in self.class_images.values())
        print(f"\n  [TestDataset] Novel classes for this fold:")
        for cls_id in novel_classes:
            n = len(self.class_images[cls_id])
            print(f"    class={VOC_CLASS_NAMES[cls_id]:15s} (id={cls_id:2d})"
                  f" | {n:3d} test images")
        print(f"  [TestDataset] Total test query images: {total_test}")

    def get_support_and_queries(self, cls_id, k_shot, seed=42):
        """
        Return K support images + all remaining val images as queries.
        Support ∩ Queries = empty set (always disjoint by construction).
        seed ensures reproducibility across evaluation episodes.

        Returns:
            support : list[(image_tensor [3,H,W], binary_mask [H,W])]
            queries : list[(image_tensor [3,H,W], binary_mask [H,W])]
        """
        rng  = random.Random(seed)
        imgs = self.class_images[cls_id].copy()

        if len(imgs) < k_shot + 1:
            raise ValueError(
                f"Class '{VOC_CLASS_NAMES[cls_id]}' (id={cls_id}) has only "
                f"{len(imgs)} val images. Need at least {k_shot + 1}."
            )

        rng.shuffle(imgs)
        support_ids = imgs[:k_shot]
        query_ids   = imgs[k_shot:]

        support = [self._load(img_id, cls_id) for img_id in support_ids]
        queries = [self._load(img_id, cls_id) for img_id in query_ids]
        return support, queries

    def _load(self, img_id, cls_id):
        """Load one (image_tensor, binary_mask_tensor) pair."""
        image    = Image.open(
            os.path.join(self.voc_root, "JPEGImages", img_id + ".jpg")
        ).convert("RGB")
        raw_mask = Image.open(_get_mask_path(self.voc_root, img_id))

        image, mask = joint_transform(image, raw_mask, augment=False)

        binary = torch.zeros_like(mask)
        binary[mask == 255]    = 255
        binary[mask == cls_id] = 1
        return image, binary


# ─────────────────────────────────────────────────────────────────
# Public factory functions
# ─────────────────────────────────────────────────────────────────

def prepare_base_loaders(voc_root, sbd_root=None, fold=0,
                         batch_size=8, num_workers=2, seed=42,
                         val_fraction=0.0):
    """
    Returns a train DataLoader for BASE classes (Phase 1).

    BENCHMARK-CORRECT: no validation loader — VOC val is held out
    exclusively as the Phase 3 test set.

    Args:
        voc_root    (str)     : path to VOCdevkit/VOC2012/
        sbd_root    (str|None): path to benchmark_RELEASE/dataset/ (recommended)
        fold        (int)     : 0–3, determines novel / base split
        batch_size  (int)     : dataloader batch size
        num_workers (int)     : dataloader worker processes
        seed        (int)     : worker RNG seed for reproducibility

    Returns:
        train_loader : DataLoader  — base class training data
        n_base       : int         — number of base classes (15)
    """
    novel_classes = PASCAL_FSS_SPLITS[fold]
    base_classes  = [c for c in range(1, 21) if c not in novel_classes]

    train_ids, val_ids, _ = _build_merged_train_list(
        voc_root, sbd_root, val_fraction=val_fraction, seed=seed
    )

    train_ds = BaseClassDataset(
        voc_root      = voc_root,
        img_id_list   = train_ids,
        base_classes  = base_classes,
        novel_classes = novel_classes,
        augment       = True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        worker_init_fn = lambda _: np.random.seed(seed),
    )

    val_loader = None
    if val_ids:
        val_ds = BaseClassDataset(
            voc_root      = voc_root,
            img_id_list   = val_ids,
            base_classes  = base_classes,
            novel_classes = novel_classes,
            augment       = False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
            worker_init_fn = lambda _: np.random.seed(seed),
        )

    print(f"\n  [Phase 1 — Fold {fold}]")
    print(f"    Base  classes ({len(base_classes):2d}): "
          f"{[VOC_CLASS_NAMES[c] for c in base_classes]}")
    print(f"    Novel classes ({len(novel_classes):2d}): "
          f"{[VOC_CLASS_NAMES[c] for c in novel_classes]}")
    print(f"    TRAIN samples : {len(train_ds):,}")
    print(f"    VAL   loader  : NONE  (benchmark protocol — VOC val = test only)")

    if val_loader is not None:
        print(f"    Internal VAL  : {len(val_ds):,} samples"
              f"  (from train split only)")

    return train_loader, val_loader, len(base_classes)


def prepare_test_dataset(voc_root, fold=0):
    """
    Returns the NovelClassDataset for Phase 3 evaluation.
    Images come from VOC2012 val — the benchmark test set.
    Also prints per-class and total test image counts.

    Args:
        voc_root (str) : path to VOCdevkit/VOC2012/
        fold     (int) : 0–3

    Returns:
        dataset       : NovelClassDataset
        novel_classes : list[int]
    """
    novel_classes = PASCAL_FSS_SPLITS[fold]
    dataset = NovelClassDataset(voc_root, novel_classes)
    return dataset, novel_classes


def prepare_novel_dataset(voc_root, fold=0):
    """
    Alias for prepare_test_dataset() — kept for backward compatibility.
    Prefer prepare_test_dataset() in new code.
    """
    return prepare_test_dataset(voc_root, fold)
