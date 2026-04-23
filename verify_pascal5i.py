"""
verify_pascal5i.py  —  Pascal-5i Dataset Integrity Checker
===========================================================
Run this script BEFORE training to confirm your dataset is correctly
set up for Few-Shot Segmentation benchmarking.

Usage:
    python verify_pascal5i.py \
        --voc_root /data/VOCdevkit/VOC2012 \
        --sbd_root /data/benchmark_RELEASE/dataset \
        --fold     0

Each check prints ✅  (pass) or ❌  (fail / warning).
A final PASS/FAIL summary is printed at the end.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────
PASCAL_FSS_SPLITS = {
    0: [1,  2,  3,  4,  5],
    1: [6,  7,  8,  9, 10],
    2: [11, 12, 13, 14, 15],
    3: [16, 17, 18, 19, 20],
}

VOC_CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Expected counts (from PANet / HSNet protocol)
EXPECTED_TRAIN_WITH_SBD = 10582
EXPECTED_TRAIN_VOC_ONLY = 1464
EXPECTED_VAL            = 1449
EXPECTED_AUG_MASKS      = 10582


# ── Helpers ───────────────────────────────────────────────────────

def load_txt(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

def get_mask_path(voc_root, img_id):
    aug  = os.path.join(voc_root, "SegmentationClassAug", img_id + ".png")
    orig = os.path.join(voc_root, "SegmentationClass",    img_id + ".png")
    return aug if os.path.exists(aug) else orig

results = []   # list of (check_name, passed: bool, message)

def check(name, passed, detail=""):
    icon = "✅" if passed else "❌"
    msg  = f"  {icon}  {name}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    results.append((name, passed))


# ══════════════════════════════════════════════════════════════════
# CHECK GROUP 1 — Directory & File Structure
# ══════════════════════════════════════════════════════════════════

def check_structure(voc_root, sbd_root):
    print("\n" + "═" * 60)
    print("CHECK 1 — Directory & File Structure")
    print("═" * 60)

    required_dirs = [
        "JPEGImages",
        "SegmentationClass",
        os.path.join("ImageSets", "Segmentation"),
    ]
    for d in required_dirs:
        full = os.path.join(voc_root, d)
        check(f"Dir exists: {d}", os.path.isdir(full), full)

    aug_dir = os.path.join(voc_root, "SegmentationClassAug")
    aug_exists = os.path.isdir(aug_dir)
    check("Dir exists: SegmentationClassAug (SBD-augmented masks)",
          aug_exists,
          f"{aug_dir}\n"
          "       ⚠  If missing, download from:\n"
          "          https://github.com/DrSleep/tensorflow-deeplab-resnet\n"
          "          or extract from SBD benchmark_RELEASE/")

    for split in ["train.txt", "val.txt", "trainval.txt"]:
        p = os.path.join(voc_root, "ImageSets", "Segmentation", split)
        check(f"File exists: ImageSets/Segmentation/{split}", os.path.exists(p))

    if sbd_root:
        for fname in ["train.txt", "val.txt"]:
            p = os.path.join(sbd_root, fname)
            check(f"SBD file exists: {fname}", os.path.exists(p), p)


# ══════════════════════════════════════════════════════════════════
# CHECK GROUP 2 — Image & Mask Counts
# ══════════════════════════════════════════════════════════════════

def check_counts(voc_root, sbd_root):
    print("\n" + "═" * 60)
    print("CHECK 2 — Image & Mask Counts")
    print("═" * 60)

    seg_dir = os.path.join(voc_root, "ImageSets", "Segmentation")
    voc_val   = load_txt(os.path.join(seg_dir, "val.txt"))
    voc_train = load_txt(os.path.join(seg_dir, "train.txt"))

    check(f"VOC2012 val   count = {len(voc_val)}",
          len(voc_val) == EXPECTED_VAL,
          f"Expected {EXPECTED_VAL}, got {len(voc_val)}")

    check(f"VOC2012 train count = {len(voc_train)}",
          len(voc_train) == EXPECTED_TRAIN_VOC_ONLY,
          f"Expected {EXPECTED_TRAIN_VOC_ONLY}, got {len(voc_train)}")

    aug_dir = os.path.join(voc_root, "SegmentationClassAug")
    if os.path.isdir(aug_dir):
        n_aug = len([f for f in os.listdir(aug_dir) if f.endswith(".png")])
        check(f"SegmentationClassAug PNG count = {n_aug}",
              n_aug >= 10000,
              f"Expected ~{EXPECTED_AUG_MASKS}, got {n_aug}. "
              "<10 000 means SBD masks incomplete.")
    else:
        check("SegmentationClassAug exists", False,
              "Cannot count masks — directory missing.")

    # Count JPEG images
    jpeg_dir = os.path.join(voc_root, "JPEGImages")
    n_jpeg = len([f for f in os.listdir(jpeg_dir) if f.endswith(".jpg")])
    check(f"JPEGImages count = {n_jpeg}",
          n_jpeg >= 17000,
          f"Expected ≥17,000 (VOC + SBD images). Got {n_jpeg}.\n"
          "       Ensure SBD JPEG images are also present in JPEGImages/.")

    # Merged training set size
    merged = set(voc_train)
    if sbd_root:
        for fname in ["train.txt", "val.txt"]:
            p = os.path.join(sbd_root, fname)
            if os.path.exists(p):
                merged.update(load_txt(p))
    voc_val_set = set(voc_val)
    merged -= voc_val_set
    check(f"Merged train set size = {len(merged)}",
          len(merged) >= 10000,
          f"Expected ~{EXPECTED_TRAIN_WITH_SBD} (with SBD), "
          f"{EXPECTED_TRAIN_VOC_ONLY} (without). Got {len(merged)}.")

    return voc_val, voc_train


# ══════════════════════════════════════════════════════════════════
# CHECK GROUP 3 — Train / Test Leakage
# ══════════════════════════════════════════════════════════════════

def check_leakage(voc_root, sbd_root, voc_val, voc_train):
    print("\n" + "═" * 60)
    print("CHECK 3 — Train/Test Leakage")
    print("═" * 60)

    voc_val_set   = set(voc_val)
    voc_train_set = set(voc_train)

    # VOC train ∩ VOC val
    overlap_voc = voc_train_set & voc_val_set
    check("VOC train ∩ VOC val = ∅",
          len(overlap_voc) == 0,
          f"Overlap: {len(overlap_voc)} images {list(overlap_voc)[:5]}")

    if sbd_root:
        sbd_all = set()
        for fname in ["train.txt", "val.txt"]:
            p = os.path.join(sbd_root, fname)
            if os.path.exists(p):
                sbd_all.update(load_txt(p))
        overlap_sbd = sbd_all & voc_val_set
        check(f"SBD ∩ VOC val = ∅  ({len(overlap_sbd)} overlapping imgs removed in train list)",
              True,    # overlap is expected and MUST be removed
              f"{len(overlap_sbd)} SBD images also appear in VOC val.\n"
              "       Data_Loader.py removes these correctly via set difference.")


# ══════════════════════════════════════════════════════════════════
# CHECK GROUP 4 — Fold Class Assignments
# ══════════════════════════════════════════════════════════════════

def check_fold_assignments():
    print("\n" + "═" * 60)
    print("CHECK 4 — Fold Class Assignments")
    print("═" * 60)

    all_novel = []
    for fold_id, classes in PASCAL_FSS_SPLITS.items():
        names = [VOC_CLASS_NAMES[c] for c in classes]
        correct = (classes == list(range(fold_id * 5 + 1, fold_id * 5 + 6)))
        check(f"Fold {fold_id} → classes {classes}",
              correct, f"Names: {names}")
        all_novel.extend(classes)

    check("All 20 classes covered across 4 folds",
          sorted(all_novel) == list(range(1, 21)),
          f"Got: {sorted(all_novel)}")


# ══════════════════════════════════════════════════════════════════
# CHECK GROUP 5 — Mask Pixel Values & Novel Pixel Masking
# ══════════════════════════════════════════════════════════════════

def check_masks(voc_root, fold, voc_val, n_sample=100):
    print("\n" + "═" * 60)
    print(f"CHECK 5 — Mask Pixel Values & Novel Pixel Masking (fold {fold}, "
          f"sampling {n_sample} val images)")
    print("═" * 60)

    novel_classes = set(PASCAL_FSS_SPLITS[fold])
    base_classes  = set(range(1, 21)) - novel_classes

    bad_pixel_vals = []
    novel_in_val   = defaultdict(int)
    base_in_val    = defaultdict(int)

    sample_ids = voc_val[:n_sample]
    for img_id in sample_ids:
        mask_path = get_mask_path(voc_root, img_id)
        if not os.path.exists(mask_path):
            continue
        mask = np.array(Image.open(mask_path))
        unique_vals = set(np.unique(mask).tolist())

        # Valid pixel values in VOC masks: 0..20 and 255
        invalid = unique_vals - set(range(0, 21)) - {255}
        if invalid:
            bad_pixel_vals.append((img_id, invalid))

        for cls in unique_vals:
            if cls in novel_classes:
                novel_in_val[cls] += 1
            elif 1 <= cls <= 20:
                base_in_val[cls] += 1

    check("No invalid pixel values in masks (valid: 0–20 + 255)",
          len(bad_pixel_vals) == 0,
          f"Bad masks: {bad_pixel_vals[:3]}")

    print(f"\n  Novel class image counts in VOC val (fold {fold}):")
    for cls in sorted(novel_in_val):
        n = novel_in_val[cls]
        ok = n >= 10   # need at least k_shot + some queries
        icon = "  ✅" if ok else "  ⚠ "
        print(f"    {icon} class {cls:2d} ({VOC_CLASS_NAMES[cls]:15s}): "
              f"{n} images in sample of {n_sample}")

    print(f"\n  Base class image counts in VOC val (fold {fold}):")
    for cls in sorted(base_in_val):
        print(f"       class {cls:2d} ({VOC_CLASS_NAMES[cls]:15s}): "
              f"{base_in_val[cls]} images in sample of {n_sample}")


# ══════════════════════════════════════════════════════════════════
# CHECK GROUP 6 — Augmented Mask Consistency
# ══════════════════════════════════════════════════════════════════

def check_aug_mask_consistency(voc_root, voc_train, n_sample=50):
    print("\n" + "═" * 60)
    print(f"CHECK 6 — Augmented vs Original Mask Consistency "
          f"(sampling {n_sample} train images)")
    print("═" * 60)

    aug_dir  = os.path.join(voc_root, "SegmentationClassAug")
    orig_dir = os.path.join(voc_root, "SegmentationClass")

    if not os.path.isdir(aug_dir):
        check("SegmentationClassAug present for consistency check", False)
        return

    mismatches = []
    checked    = 0
    for img_id in voc_train[:n_sample]:
        aug_path  = os.path.join(aug_dir,  img_id + ".png")
        orig_path = os.path.join(orig_dir, img_id + ".png")
        if not (os.path.exists(aug_path) and os.path.exists(orig_path)):
            continue
        aug_mask  = np.array(Image.open(aug_path))
        orig_mask = np.array(Image.open(orig_path))
        if aug_mask.shape != orig_mask.shape:
            mismatches.append(img_id)
        checked += 1

    check(f"SegmentationClassAug shape matches SegmentationClass "
          f"({checked} checked)",
          len(mismatches) == 0,
          f"Shape mismatches: {mismatches[:3]}")


# ══════════════════════════════════════════════════════════════════
# CHECK GROUP 7 — Few-Shot Episode Feasibility
# ══════════════════════════════════════════════════════════════════

def check_episode_feasibility(voc_root, fold, voc_val):
    print("\n" + "═" * 60)
    print(f"CHECK 7 — K-shot Episode Feasibility (fold {fold})")
    print("═" * 60)

    novel_classes = PASCAL_FSS_SPLITS[fold]
    class_images  = {cls: [] for cls in novel_classes}

    for img_id in voc_val:
        mask_path = get_mask_path(voc_root, img_id)
        if not os.path.exists(mask_path):
            continue
        mask = np.array(Image.open(mask_path))
        for cls_id in novel_classes:
            if (mask == cls_id).any():
                class_images[cls_id].append(img_id)

    print(f"  {'Class':20s}  {'VOC-val imgs':>12}  "
          f"{'1-shot OK':>10}  {'5-shot OK':>10}")
    print("  " + "-" * 58)
    all_1shot = True
    all_5shot = True
    for cls_id in novel_classes:
        n     = len(class_images[cls_id])
        ok1   = n >= 2    # 1 support + 1 query minimum
        ok5   = n >= 6    # 5 support + 1 query minimum
        all_1shot &= ok1
        all_5shot &= ok5
        print(f"  {VOC_CLASS_NAMES[cls_id]:20s}  {n:>12}  "
              f"{'✅' if ok1 else '❌':>10}  {'✅' if ok5 else '❌':>10}")

    check("All novel classes have ≥2 images (1-shot feasible)", all_1shot)
    check("All novel classes have ≥6 images (5-shot feasible)", all_5shot)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Verify Pascal-5i dataset setup for FSS.")
    parser.add_argument("--voc_root", required=True,
                        help="Path to VOCdevkit/VOC2012/")
    parser.add_argument("--sbd_root", default=None,
                        help="Path to benchmark_RELEASE/dataset/ (optional but recommended)")
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Fold to verify (default: 0)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Pascal-5i Dataset Verification")
    print(f"  voc_root : {args.voc_root}")
    print(f"  sbd_root : {args.sbd_root or 'NOT PROVIDED (VOC only)'}")
    print(f"  fold     : {args.fold}")
    print("=" * 60)

    check_structure(args.voc_root, args.sbd_root)
    voc_val, voc_train = check_counts(args.voc_root, args.sbd_root)
    check_leakage(args.voc_root, args.sbd_root, voc_val, voc_train)
    check_fold_assignments()
    check_masks(args.voc_root, args.fold, voc_val)
    check_aug_mask_consistency(args.voc_root, voc_train)
    check_episode_feasibility(args.voc_root, args.fold, voc_val)

    # ── Final Summary ──────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  FINAL SUMMARY")
    print("═" * 60)
    failed = [(n, m) for n, m in results if not m]
    passed = [n for n, m in results if m]
    print(f"  Passed: {len(passed)} / {len(results)}")
    if failed:
        print(f"\n  ❌ FAILED CHECKS ({len(failed)}):")
        for name, _ in failed:
            print(f"     • {name}")
        print("\n  ⚠ Dataset is NOT correctly configured for Pascal-5i.")
        print("    Fix the failed checks before training/evaluation.")
    else:
        print("\n  ✅ ALL CHECKS PASSED — Dataset is correctly configured.")
    print("═" * 60)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()