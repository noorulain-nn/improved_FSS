"""
Data_Loader.py  —  Pascal-5i Few-Shot Segmentation  (v4 — Improved)
=====================================================================

CHANGES FROM v3:
  1. INTERNAL VALIDATION SPLIT — benchmark-compliant early stopping
     10% of the merged TRAIN split is held out as an internal val set.
     The VOC val set (Phase 3 test) is NEVER touched in Phase 1.
     Reference: standard practice in HSNet (Min et al., ICCV 2021).

  2. STRONGER AUGMENTATION
     JointTransform adds RandomResizedCrop, ColorJitter, RandomGrayscale
     on top of the existing horizontal flip. Spatial transforms applied
     to image AND mask; colour transforms to image only.
     Reference: HSNet augmentation protocol (Min et al., ICCV 2021).

  3. CLEANER BACKGROUND — other_fg_mask returned
     BaseClassDataset returns a 4th tensor: other_fg_mask (BoolTensor).
     True where OTHER base-class objects appear. APM.update_from_batch()
     uses this to exclude those pixels from background EMA, since they
     are NOT true background — they are other foreground categories.

  4. prepare_base_loaders() returns (train_loader, val_loader, n_base).
     val_loader is None when val_fraction=0.0.

BENCHMARK PROTOCOL (unchanged):
  TRAIN set = VOC2012 train + SBD train + SBD val  minus  VOC2012 val
  TEST  set = VOC2012 val — ONLY used in Phase 3.

References:
  Wang et al.  PANet   ICCV 2019  https://arxiv.org/abs/1908.06391
  Min  et al.  HSNet   ICCV 2021  https://arxiv.org/abs/2106.07015
  Tian et al.  PFENet  TPAMI 2022 https://arxiv.org/abs/2107.00509
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# ── Standard Pascal-5i fold definitions ──────────────────────────
PASCAL_FSS_SPLITS = {
    0: [1,  2,  3,  4,  5],
    1: [6,  7,  8,  9, 10],
    2: [11, 12, 13, 14, 15],
    3: [16, 17, 18, 19, 20],
}

VOC_CLASS_NAMES = [
    "background",   "aeroplane", "bicycle",    "bird",       "boat",
    "bottle",       "bus",       "car",        "cat",        "chair",
    "cow",          "diningtable","dog",        "horse",      "motorbike",
    "person",       "pottedplant","sheep",      "sofa",       "train",
    "tvmonitor",
]

IMG_SIZE = 321


# ─────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────

class JointTransform:
    """
    Applies identical spatial transforms to image AND mask.
    Colour transforms applied to image ONLY (masks are label maps).

    Training pipeline:
      1. RandomResizedCrop (scale 0.7–1.0, ratio 0.75–1.33)
      2. RandomHorizontalFlip (p=0.5)
      3. ColorJitter (brightness/contrast/saturation 0.4, hue 0.05) — image only
      4. RandomGrayscale (p=0.05) — image only
      5. ToTensor + ImageNet normalize

    Val/test: resize only.

    Reference: HSNet (Min et al., ICCV 2021), PFENet (Tian et al., TPAMI 2022).
    """

    def __init__(self, augment=False):
        self.augment = augment
        self.color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05
        )

    def __call__(self, image, mask):
        if self.augment:
            # 1. RandomResizedCrop — same crop for image and mask
            i, j, h, w = T.RandomResizedCrop.get_params(
                image, scale=(0.7, 1.0), ratio=(0.75, 1.333)
            )
            image = TF.resized_crop(
                image, i, j, h, w, (IMG_SIZE, IMG_SIZE),
                interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.resized_crop(
                mask,  i, j, h, w, (IMG_SIZE, IMG_SIZE),
                interpolation=TF.InterpolationMode.NEAREST)

            # 2. Horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # 3. ColorJitter — image only
            image = self.color_jitter(image)

            # 4. RandomGrayscale — image only
            if random.random() < 0.05:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)
        else:
            image = TF.resize(image, (IMG_SIZE, IMG_SIZE),
                              interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.resize(mask,  (IMG_SIZE, IMG_SIZE),
                              interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])
        mask  = torch.from_numpy(np.array(mask)).long()
        return image, mask


def joint_transform(image, mask, augment=False):
    """Legacy wrapper kept for external compatibility."""
    return JointTransform(augment)(image, mask)


# ─────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────

def _get_mask_path(voc_root, img_id):
    aug_path  = os.path.join(voc_root, "SegmentationClassAug", img_id + ".png")
    orig_path = os.path.join(voc_root, "SegmentationClass",    img_id + ".png")
    return aug_path if os.path.exists(aug_path) else orig_path


def _build_merged_train_list(voc_root, sbd_root=None,
                              val_fraction=0.1, seed=42):
    """
    Build the benchmark-correct training image list and split off an
    internal validation subset for early stopping.

    TRAIN = (VOC2012 train) + (SBD train) + (SBD val)  - (VOC2012 val)
    Internal val is drawn from TRAIN only — never from VOC val.

    Returns
    -------
    train_ids   : list[str]
    val_ids     : list[str]  (empty list if val_fraction==0.0)
    voc_val_set : set[str]
    """
    voc_val_path   = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
    voc_train_path = os.path.join(voc_root, "ImageSets", "Segmentation", "train.txt")

    with open(voc_val_path)   as f:
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

    leaked = merged & voc_val_set
    merged -= voc_val_set

    all_ids = sorted(merged)
    rng = random.Random(seed)
    rng.shuffle(all_ids)

    if val_fraction > 0.0:
        n_val     = max(1, int(len(all_ids) * val_fraction))
        val_ids   = all_ids[:n_val]
        train_ids = all_ids[n_val:]
    else:
        val_ids   = []
        train_ids = all_ids

    source = "VOC2012 + SBD" if (sbd_root and sbd_added > 0) else "VOC2012 only"

    print("\n" + "="*60)
    print("  DATA SPLIT SUMMARY  (Pascal-5i Benchmark Protocol)")
    print("="*60)
    print(f"  Source              : {source}")
    print(f"  Merged train total  : {len(all_ids):,} image IDs")
    print(f"  Phase 1 TRAIN       : {len(train_ids):,}")
    if val_ids:
        print(f"  Internal VAL        : {len(val_ids):,}  ({val_fraction*100:.0f}% of train)")
        print(f"  NOTE: internal val from TRAIN split only — no benchmark leakage")
    else:
        print(f"  Internal VAL        : NONE  (val_fraction=0.0)")
    print(f"  TEST  set size      : {len(voc_val_set):,} images  (VOC2012 val)")
    print(f"                        used ONLY in Phase 3")
    if leaked:
        print(f"  Leakage check       : removed {len(leaked)} VOC-val IDs [OK]")
    else:
        print(f"  Leakage check       : no overlap found [OK]")
    print("="*60)

    return train_ids, val_ids, voc_val_set


# ─────────────────────────────────────────────────────────────────
# Phase 1 Dataset
# ─────────────────────────────────────────────────────────────────

class BaseClassDataset(Dataset):
    """
    Loads images containing BASE classes for Phase 1 training/validation.

    Returns: (image_tensor, binary_mask, other_fg_mask, class_label)

      image_tensor  : FloatTensor [3, H, W]
      binary_mask   : LongTensor  [H, W]  0=bg  1=target  255=ignore
      other_fg_mask : BoolTensor  [H, W]  True = pixel belongs to a
                                           different base class object
                                           (NOT true background!)
      class_label   : int  remapped 0..N_base-1
    """

    def __init__(self, voc_root, img_id_list, base_classes,
                 novel_classes, augment=False):
        self.voc_root  = voc_root
        self.base_set  = set(base_classes)
        self.novel_set = set(novel_classes)
        self.augment   = augment
        self.transform = JointTransform(augment=augment)
        self.label_map = {c: i for i, c in enumerate(sorted(base_classes))}

        self.samples = []
        n_missing = 0
        for img_id in img_id_list:
            mask_path = _get_mask_path(voc_root, img_id)
            if not os.path.exists(mask_path):
                n_missing += 1
                continue
            mask = np.array(Image.open(mask_path))
            for cls_id in base_classes:
                if (mask == cls_id).any():
                    self.samples.append((img_id, cls_id))

        if n_missing > 0:
            print(f"  [WARNING] {n_missing} image IDs skipped (mask not found).")

        print(f"  [BaseDataset]  {len(set(base_classes))} base classes"
              f" | {len(self.samples):,} (img, class) samples"
              f"{'  [augmented]' if augment else '  [no aug]'}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, cls_id = self.samples[idx]

        image    = Image.open(
            os.path.join(self.voc_root, "JPEGImages", img_id + ".jpg")
        ).convert("RGB")
        raw_mask = Image.open(_get_mask_path(self.voc_root, img_id))

        image, mask = self.transform(image, raw_mask)

        # Binary mask: target=1, bg=0, ignore=255
        binary = torch.zeros_like(mask)
        binary[mask == cls_id] = 1
        for nov_cls in self.novel_set:
            binary[mask == nov_cls] = 255
        binary[mask == 255] = 255

        # Other-foreground mask: pixels from OTHER base classes
        # These must NOT be used as background in EMA updates
        other_fg = torch.zeros_like(mask, dtype=torch.bool)
        for other_cls in self.base_set:
            if other_cls != cls_id:
                other_fg |= (mask == other_cls)

        return image, binary, other_fg, self.label_map[cls_id]


# ─────────────────────────────────────────────────────────────────
# Phase 3 Dataset
# ─────────────────────────────────────────────────────────────────

class NovelClassDataset(Dataset):
    """
    Holds images for NOVEL classes (Phase 3 evaluation only).
    Queries come exclusively from VOC2012 val.txt.
    """

    def __init__(self, voc_root, novel_classes):
        self.voc_root      = voc_root
        self.novel_classes = novel_classes
        self.transform     = JointTransform(augment=False)

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
        rng  = random.Random(seed)
        imgs = self.class_images[cls_id].copy()
        if len(imgs) < k_shot + 1:
            raise ValueError(
                f"Class '{VOC_CLASS_NAMES[cls_id]}' has only {len(imgs)} "
                f"val images. Need >= {k_shot + 1}."
            )
        rng.shuffle(imgs)
        support_ids = imgs[:k_shot]
        query_ids   = imgs[k_shot:]
        support = [self._load(img_id, cls_id) for img_id in support_ids]
        queries = [self._load(img_id, cls_id) for img_id in query_ids]
        return support, queries

    def _load(self, img_id, cls_id):
        image    = Image.open(
            os.path.join(self.voc_root, "JPEGImages", img_id + ".jpg")
        ).convert("RGB")
        raw_mask = Image.open(_get_mask_path(self.voc_root, img_id))
        image, mask = self.transform(image, raw_mask)
        binary = torch.zeros_like(mask)
        binary[mask == 255]    = 255
        binary[mask == cls_id] = 1
        return image, binary


# ─────────────────────────────────────────────────────────────────
# Public factory functions
# ─────────────────────────────────────────────────────────────────

def prepare_base_loaders(voc_root, sbd_root=None, fold=0,
                         batch_size=8, num_workers=2, seed=42,
                         val_fraction=0.1):
    """
    Returns train_loader, val_loader, n_base for Phase 1.

    val_loader is benchmark-compliant: carved from training split only.
    Set val_fraction=0.0 to disable internal validation entirely.
    """
    novel_classes = PASCAL_FSS_SPLITS[fold]
    base_classes  = [c for c in range(1, 21) if c not in novel_classes]

    train_ids, val_ids, _ = _build_merged_train_list(
        voc_root, sbd_root, val_fraction=val_fraction, seed=seed
    )

    train_ds = BaseClassDataset(
        voc_root=voc_root, img_id_list=train_ids,
        base_classes=base_classes, novel_classes=novel_classes,
        augment=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=lambda _: np.random.seed(seed),
        drop_last=True,
    )

    val_loader = None
    if val_ids:
        val_ds = BaseClassDataset(
            voc_root=voc_root, img_id_list=val_ids,
            base_classes=base_classes, novel_classes=novel_classes,
            augment=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    print(f"\n  [Phase 1 — Fold {fold}]")
    print(f"    Base  classes ({len(base_classes):2d}): "
          f"{[VOC_CLASS_NAMES[c] for c in base_classes]}")
    print(f"    Novel classes ({len(novel_classes):2d}): "
          f"{[VOC_CLASS_NAMES[c] for c in novel_classes]}")
    print(f"    TRAIN samples : {len(train_ds):,}")
    if val_loader:
        print(f"    Internal VAL  : {len(val_ds):,} samples  [benchmark-compliant]")
    else:
        print(f"    Internal VAL  : NONE")

    return train_loader, val_loader, len(base_classes)


def prepare_test_dataset(voc_root, fold=0):
    novel_classes = PASCAL_FSS_SPLITS[fold]
    dataset = NovelClassDataset(voc_root, novel_classes)
    return dataset, novel_classes


def prepare_novel_dataset(voc_root, fold=0):
    """Alias for prepare_test_dataset() — backward compatibility."""
    return prepare_test_dataset(voc_root, fold)
