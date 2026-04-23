"""
verify_dataset.py

Run:
    python verify_dataset.py

Prints Pascal-5i counts using the actual protocol in Data_Loader.py:
  - Phase 1 train candidates = VOC train + SBD train/val, with VOC val removed
  - Phase 1 clean train set  = candidates with no pixels from the fold's novel classes
  - Phase 3 test/query set   = VOC val images containing the fold's novel classes
"""

import os
from collections import defaultdict

import numpy as np
from PIL import Image


VOC_ROOT = "C:\\data\\VOCdevkit\\VOC2012"
SBD_ROOT = "C:\\data\\sbd\\benchmark_RELEASE"

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

PASCAL_5I = {
    0: [1, 2, 3, 4, 5],
    1: [6, 7, 8, 9, 10],
    2: [11, 12, 13, 14, 15],
    3: [16, 17, 18, 19, 20],
}


def read_ids(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def normalize_sbd_root(sbd_root):
    dataset_dir = os.path.join(sbd_root, "dataset")
    return dataset_dir if os.path.isdir(dataset_dir) else sbd_root


def get_sbd_split_file(sbd_root, split):
    return os.path.join(normalize_sbd_root(sbd_root), split + ".txt")


def get_sbd_mask_path(sbd_root, img_id):
    return os.path.join(normalize_sbd_root(sbd_root), "cls_png", img_id + ".png")


def get_voc_mask_path(voc_root, img_id):
    return os.path.join(voc_root, "SegmentationClass", img_id + ".png")


def find_mask_path(voc_root, sbd_root, img_id):
    voc_mask = get_voc_mask_path(voc_root, img_id)
    if os.path.exists(voc_mask):
        return voc_mask
    if sbd_root:
        sbd_mask = get_sbd_mask_path(sbd_root, img_id)
        if os.path.exists(sbd_mask):
            return sbd_mask
    return None


def load_mask(mask_path):
    return np.array(Image.open(mask_path))


def count_per_class(voc_root, ids):
    counts = defaultdict(int)
    for img_id in ids:
        mask_path = get_voc_mask_path(voc_root, img_id)
        if not os.path.exists(mask_path):
            continue
        mask = load_mask(mask_path)
        for class_id in range(1, 21):
            if (mask == class_id).any():
                counts[class_id] += 1
    return counts


def build_phase1_candidates(voc_root, sbd_root):
    train_ids = set(read_ids(os.path.join(voc_root, "ImageSets", "Segmentation", "train.txt")))
    val_ids = set(read_ids(os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")))

    merged = set(train_ids)
    if sbd_root:
        for split in ["train", "val"]:
            split_file = get_sbd_split_file(sbd_root, split)
            if os.path.exists(split_file):
                merged.update(read_ids(split_file))

    # Match the logic described in verify_pascal5i.py: VOC val must not appear in training.
    merged -= val_ids
    return sorted(merged)


def summarize_fold(voc_root, sbd_root, fold, val_ids, val_counts):
    novel_classes = PASCAL_5I[fold]
    candidate_ids = build_phase1_candidates(voc_root, sbd_root)

    clean_train_images = []
    phase1_samples = 0
    missing_masks = 0

    for img_id in candidate_ids:
        mask_path = find_mask_path(voc_root, sbd_root, img_id)
        if mask_path is None:
            missing_masks += 1
            continue

        mask = load_mask(mask_path)
        if any((mask == class_id).any() for class_id in novel_classes):
            continue

        present_base = [
            class_id
            for class_id in range(1, 21)
            if class_id not in novel_classes and (mask == class_id).any()
        ]
        if not present_base:
            continue

        clean_train_images.append(img_id)
        phase1_samples += len(present_base)

    test_image_ids = []
    for img_id in val_ids:
        mask_path = get_voc_mask_path(voc_root, img_id)
        if not os.path.exists(mask_path):
            continue
        mask = load_mask(mask_path)
        if any((mask == class_id).any() for class_id in novel_classes):
            test_image_ids.append(img_id)

    novel_query_hits = sum(val_counts[class_id] for class_id in novel_classes)

    print(f"\nFold {fold}:")
    print(f"  Novel classes                 : {[VOC_CLASSES[class_id] for class_id in novel_classes]}")
    print(f"  Phase1 candidate images       : {len(candidate_ids):,} (VOC train + SBD - VOC val overlap)")
    print(f"  Phase1 clean train images     : {len(clean_train_images):,}")
    print(f"  Phase1 (image, class) samples : {phase1_samples:,}")
    print(f"  Phase3 unique test images     : {len(test_image_ids):,}")
    print(f"  Phase3 novel query hits       : {novel_query_hits:,}")
    print(f"  Missing masks in train pool   : {missing_masks:,}")

    print("  Per-novel-class test counts:")
    for class_id in novel_classes:
        unique_test_count = 0
        for img_id in val_ids:
            mask_path = get_voc_mask_path(voc_root, img_id)
            if not os.path.exists(mask_path):
                continue
            mask = load_mask(mask_path)
            if (mask == class_id).any():
                unique_test_count += 1
        print(f"    {VOC_CLASSES[class_id]:>15} : {unique_test_count:,}")


train_path = os.path.join(VOC_ROOT, "ImageSets", "Segmentation", "train.txt")
val_path = os.path.join(VOC_ROOT, "ImageSets", "Segmentation", "val.txt")

train_ids = read_ids(train_path)
val_ids = read_ids(val_path)
tc = count_per_class(VOC_ROOT, train_ids)
vc = count_per_class(VOC_ROOT, val_ids)
phase1_candidates = build_phase1_candidates(VOC_ROOT, SBD_ROOT)

print("=" * 60)
print("Pascal-5i Dataset Verification")
print("=" * 60)
print(f"VOC root                     : {VOC_ROOT}")
print(f"SBD root                     : {SBD_ROOT}")
print(f"VOC train.txt                : {len(train_ids):,} images")
print(f"VOC val.txt                  : {len(val_ids):,} images")
print(f"Pascal-5i Phase1 candidates  : {len(phase1_candidates):,} images")
print(f"Total VOC-only split images  : {len(train_ids) + len(val_ids):,}")

print("\nVOC per-class presence")
print(f"{'Class':>15} | {'train':>5} | {'val':>5}")
print("-" * 33)
for class_id in range(1, 21):
    print(f"{VOC_CLASSES[class_id]:>15} | {tc[class_id]:>5} | {vc[class_id]:>5}")

print("\nActual Pascal-5i counts by fold")
for fold_id in range(4):
    summarize_fold(VOC_ROOT, SBD_ROOT, fold_id, val_ids, vc)

print("\nProtocol")
print("  Phase1 uses merged VOC+SBD training candidates after removing any VOC val overlap.")
print("  Phase1 clean train images exclude all pixels from the fold's novel classes.")
print("  Phase3 uses VOC val images that contain at least one novel-class pixel.")
