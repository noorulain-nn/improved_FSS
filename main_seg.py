"""
main_seg.py  —  FSS with FPN Decoder  (Benchmark-Compliant v3)
===============================================================

CHANGES FROM v2:
  1. DATA SPLIT — benchmark-correct two-split protocol
       prepare_base_loaders() now returns (train_loader, n_base)  [2 values]
       No Phase-1 validation loader — VOC val is the Phase-3 test set only
       prepare_test_dataset() replaces prepare_novel_dataset()

  2. SCHEDULER — StepLR replaced with CosineAnnealingLR
       Reference: Loshchilov & Hutter (2017), "SGDR: Stochastic Gradient
       Descent with Warm Restarts", ICLR 2017
       https://arxiv.org/abs/1608.03983
       Used in HSNet (Min et al., ICCV 2021) and most modern FSS baselines.

  3. ROC CURVES — added in Phase 3
       Pixel-level foreground probability scores are collected and passed
       to Visualizer.plot_roc_curve() after each fold.

  4. phase1_validate() is REMOVED — no validation during Phase 1 in benchmark.

  5. Visualizer.plot_training_curves() call updated (no val arguments).

EVERYTHING ELSE IS UNCHANGED:
  3-phase structure, memory module, loss, metrics, fold loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR   # ← changed from StepLR
import os
import numpy as np

import Data_Loader
import Models
import APM
import Metrics
import Visualizer

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
VOC_ROOT         = "./data/fss-data/VOCdevkit/VOC2012"
SBD_ROOT         = "./data/fss-data/sbd/benchmark_RELEASE/dataset"
NUM_FOLDS        = 4
K_SHOT           = 5
BACKBONE_NAME    = "resnet50"
DECODER_CHANNELS = 256
BATCH_SIZE       = 16
NUM_EPOCHS       = 25
LEARNING_RATE    = 3e-4      # backbone layer4 initial LR
DECODER_LR       = 2e-4      # decoder initial LR
IMG_SIZE         = 321
LR_MIN           = 1e-5      # CosineAnnealingLR eta_min
PATIENCE         = 5
VAL_FRACTION     = 0.0

N_VIS_SAMPLES    = 6         # segmentation sample rows to plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Backbone: {BACKBONE_NAME} | {K_SHOT}-shot")
print(f"Decoder: FPN out_channels={DECODER_CHANNELS}")
print(f"Scheduler: CosineAnnealingLR  T_max={NUM_EPOCHS}  eta_min={LR_MIN}")
print(f"Early stopping: patience={PATIENCE}  val_fraction={VAL_FRACTION}")
print(f"Running {NUM_FOLDS} folds...")

criterion = nn.CrossEntropyLoss(ignore_index=255)


# ─────────────────────────────────────────────────────────────────
# Shared helper — compute loss for one batch
# ─────────────────────────────────────────────────────────────────
def dice_loss(pred_logits, target, eps=1e-6):
    """
    Works for both Phase 1 (31 slots) and Phase 2/3 (2 slots).
    In Phase 1: extracts the foreground slot for the current class.
    In Phase 2/3: slot 1 is always foreground.
    """
    if pred_logits.shape[1] == 2:
        # Phase 2/3 — binary, slot 1 = foreground
        fg_idx = 1
    else:
        # Phase 1 — multi-slot, but loss is still computed per-class
        # CrossEntropy already handles slot selection via labels
        # For dice, take the max activated slot as proxy for foreground
        fg_idx = pred_logits.argmax(dim=1, keepdim=False)
        # In this case use a simplified dice on the softmax confidence
        pred_soft = torch.softmax(pred_logits, dim=1)
        # Take the probability of the correct foreground pixels
        # using the binary mask directly
        valid    = (target != 255).float()
        target_f = (target == 1).float() * valid
        # Foreground confidence = max across all slots except bg(0)
        fg_conf  = pred_soft[:, 1:].max(dim=1)[0] * valid
        
        intersection = (fg_conf * target_f).sum(dim=[1, 2])
        union        = fg_conf.sum(dim=[1, 2]) + target_f.sum(dim=[1, 2])
        dice = 1.0 - (2.0 * intersection + eps) / (union + eps)
        return dice.mean()

    pred_soft    = torch.softmax(pred_logits, dim=1)[:, fg_idx]
    valid        = (target != 255).float()
    target_f     = (target == 1).float() * valid
    pred_f       = pred_soft * valid
    intersection = (pred_f * target_f).sum(dim=[1, 2])
    union        = pred_f.sum(dim=[1, 2]) + target_f.sum(dim=[1, 2])
    dice         = 1.0 - (2.0 * intersection + eps) / (union + eps)
    return dice.mean()
    
def compute_batch_loss(model, images, masks, class_labels, novel_cls_id=None):
    logits, fused = model(images, novel_cls_id)   # [B, slots, 56, 56]

    logits_full = F.interpolate(
        logits, size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear", align_corners=False,
    )

    B    = images.shape[0]
    loss = torch.tensor(0.0, device=device)
    preds = []

    for i in range(B):
        if novel_cls_id is None:
            cls_idx  = class_labels[i].item()
            fg_slot  = cls_idx + 1
            bg_slot  = model.memory_module._bg_slot(cls_idx)
            logits_i = torch.stack(
                [logits_full[i, bg_slot], logits_full[i, fg_slot]], dim=0
            ).unsqueeze(0)
        else:
            logits_i = logits_full[i].unsqueeze(0)

        mask_i = masks[i].unsqueeze(0)
        # loss  += criterion(logits_i, mask_i)
        loss  += criterion(logits_i, mask_i) + 0.5 *dice_loss(logits_i, mask_i)
        preds.append(logits_i.argmax(dim=1).squeeze(0))

    return loss / B, preds, fused


# ─────────────────────────────────────────────────────────────────
# PHASE 1 — Train on base classes (no validation)
# ─────────────────────────────────────────────────────────────────
def phase1_train(fold, val_loader=None):
    """
    Train the backbone + FPN decoder on base classes for NUM_EPOCHS.

    Benchmark protocol: NO validation set.  Training uses the full
    merged VOC+SBD train list.  The best model is saved at the epoch
    with the lowest training loss (proxy for Phase-1 checkpoint selection).

    Parameters
    ----------
    fold : int   current cross-validation fold

    Returns
    -------
    best_train_miou : float  highest training mIoU seen across all epochs
    """
    print("\n" + "="*60)
    print(f"  PHASE 1 — Training on BASE classes  (Fold {fold})")
    print(f"  Scheduler: CosineAnnealingLR  T_max={NUM_EPOCHS}  eta_min={LR_MIN}")
    if val_loader is not None:
        print(f"  Early stopping: patience={PATIENCE} (internal val)")
    print("="*60)

    best_train_miou  = 0.0
    best_val_miou    = -1.0
    best_epoch       = 0
    no_improve       = 0
    early_stop_epoch = NUM_EPOCHS
    best_loss        = float("inf")

    train_losses = []
    train_mious  = []
    val_mious    = []
    lr_history   = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        metrics    = Metrics.SegMetrics(num_classes=2)
        epoch_loss = 0.0

        for batch_idx, (images, masks, labels) in enumerate(train_loader):
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            loss, preds, fused = compute_batch_loss(model, images, masks, labels)
            loss.backward()
            # Clip gradients — prevents decoder BN instability when τ is large early on
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            with torch.no_grad():
                model.memory_module.update_from_batch(
                    fused.detach(), masks, labels.tolist()
                )

            for i in range(images.shape[0]):
                metrics.update(preds[i].unsqueeze(0), masks[i].unsqueeze(0))

            epoch_loss += loss.item()

            if batch_idx % 30 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss {loss.item():.4f}")

        _, train_miou, _ = metrics.compute()
        avg_loss         = epoch_loss / len(train_loader)

        val_miou = 0.0
        if val_loader is not None:
            model.eval()
            val_metrics = Metrics.SegMetrics(num_classes=2)
            with torch.no_grad():
                for images_v, masks_v, labels_v in val_loader:
                    images_v = images_v.to(device)
                    masks_v  = masks_v.to(device)
                    _, preds_v, _ = compute_batch_loss(
                        model, images_v, masks_v, labels_v
                    )
                    for i in range(images_v.shape[0]):
                        val_metrics.update(
                            preds_v[i].unsqueeze(0), masks_v[i].unsqueeze(0)
                        )
            _, val_miou, _ = val_metrics.compute()
            model.train()

        train_losses.append(avg_loss)
        train_mious.append(float(train_miou))
        val_mious.append(float(val_miou))
        lr_history.append(optimizer.param_groups[0]["lr"])

        lrs = [g["lr"] for g in optimizer.param_groups]
        print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS}"
              f" | LR backbone={lrs[0]:.2e} decoder={lrs[1]:.2e}"
              f" | Train Loss={avg_loss:.4f}"
              f" | Train mIoU={train_miou*100:.2f}%"
              + (f" | Val mIoU={val_miou*100:.2f}%" if val_loader else ""))

        if val_loader is not None:
            if val_miou > best_val_miou:
                best_val_miou = float(val_miou)
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), f"phase1_best_fold{fold}.pth")
                print(f"  checkpoint saved  (val mIoU={best_val_miou*100:.2f}%)")
            else:
                no_improve += 1
                print(f"  No val improvement for {no_improve}/{PATIENCE} epochs")

        # Save checkpoint at best (lowest) training loss
        if val_loader is None and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"phase1_best_fold{fold}.pth")
            print(f"  ★ Checkpoint saved  (train loss={best_loss:.4f})")

        if train_miou > best_train_miou:
            best_train_miou = float(train_miou)

        scheduler.step()   # CosineAnnealingLR — step every epoch

        if val_loader is not None and no_improve >= PATIENCE:
            early_stop_epoch = epoch + 1
            print(f"\n  [Early Stopping] No validation improvement for "
                  f"{PATIENCE} epochs. Best epoch = {best_epoch}.")
            model.load_state_dict(
                torch.load(f"phase1_best_fold{fold}.pth", map_location=device)
            )
            break

    print(f"\n[Phase 1 Fold {fold}] Best train mIoU = {best_train_miou*100:.2f}%")
    if val_loader is not None:
        print(f"[Phase 1 Fold {fold}] Best val   mIoU = {best_val_miou*100:.2f}%"
              f"  (epoch {best_epoch})")

    # ── Plot training curves (no val) ────────────────────────────
    Visualizer.plot_training_curves(
        fold         = fold,
        train_losses = train_losses,
        train_mious  = train_mious,
        lr_history   = lr_history,
        val_mious    = val_mious if val_loader is not None else None,
        early_stop_epoch = early_stop_epoch if val_loader is not None else None,
    )

    if val_loader is not None and val_mious:
        Visualizer.plot_early_stopping(
            fold          = fold,
            train_mious   = train_mious,
            val_mious     = val_mious,
            best_epoch    = best_epoch,
            patience      = PATIENCE,
            stopped_epoch = early_stop_epoch,
        )

    return best_train_miou


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — Adapt to novel classes  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────
def phase2_adapt(novel_dataset, novel_classes, k_shot, fold):
    print("\n" + "="*60)
    print(f"  PHASE 2 — {k_shot}-shot adaptation  (Fold {fold})")
    print("="*60)

    model.load_state_dict(
        torch.load(f"phase1_best_fold{fold}.pth", map_location=device)
    )
    model.freeze_everything()
    model.eval()

    query_data = {}

    for cls_id in novel_classes:
        cls_name = Data_Loader.VOC_CLASS_NAMES[cls_id]
        print(f"\n  Adapting: {cls_name} (class {cls_id})")

        support, queries = novel_dataset.get_support_and_queries(
            cls_id, k_shot=k_shot, seed=42
        )
        query_data[cls_id] = queries

        support_feats, support_masks_list = [], []

        with torch.no_grad():
            for img, msk in support:
                img_t = img.unsqueeze(0).to(device)
                feat2, feat3, feat4 = model.backbone(img_t)
                fused = model.decoder(feat2, feat3, feat4)
                support_feats.append(fused)
                support_masks_list.append(msk.unsqueeze(0).to(device))

        model.memory_module.build_novel_prototype(
            support_feats, support_masks_list, cls_id
        )

    print("\n[Phase 2] Novel prototypes built in decoder feature space.")
    return query_data


# ─────────────────────────────────────────────────────────────────
# PHASE 3 — Test on novel classes + collect ROC scores
# ─────────────────────────────────────────────────────────────────
def phase3_test(fold, novel_classes, query_data):
    """
    Evaluate on novel-class query images from VOC val (the test set).
    Collects:
      - per-class mIoU and pixel accuracy
      - foreground probability scores for ROC curve computation
      - segmentation sample images for visual inspection

    Parameters
    ----------
    fold          : int
    novel_classes : list[int]
    query_data    : dict  cls_id → list[(q_img, q_mask)]

    Returns
    -------
    mean_novel_miou : float
    """
    print("\n" + "="*60)
    print(f"  PHASE 3 — Testing on NOVEL classes  (Fold {fold})")
    print(f"  Test set: VOC2012 val  (benchmark protocol)")
    print("="*60)

    model.eval()
    all_mious       = []
    per_class_ious  = []
    per_class_accs  = []
    class_name_list = []
    vis_samples     = []
    roc_data        = {}     # ← for ROC curves: {cls_name: {scores, labels}}

    with torch.no_grad():
        for cls_id in novel_classes:
            cls_name = Data_Loader.VOC_CLASS_NAMES[cls_id]
            queries  = query_data[cls_id]
            metrics  = Metrics.SegMetrics(num_classes=2)

            # Accumulators for this class's ROC data
            cls_scores = []
            cls_labels = []

            for q_img, q_mask in queries:
                img_t  = q_img.unsqueeze(0).to(device)
                mask_t = q_mask.unsqueeze(0).to(device)

                logits, _ = model(img_t, novel_cls_id=cls_id)
                # logits shape: [1, 2, h, w]  (slot0=bg, slot1=fg)
                logits_full = F.interpolate(
                    logits, size=(IMG_SIZE, IMG_SIZE),
                    mode="bilinear", align_corners=False,
                )
                pred = logits_full.argmax(dim=1)
                metrics.update(pred, mask_t)

                # ── Collect ROC scores ────────────────────────
                # Softmax → foreground probability → flatten
                probs    = F.softmax(logits_full, dim=1)  # [1, 2, H, W]
                fg_score = probs[0, 1].cpu().numpy().flatten()  # [H*W]
                gt_flat  = q_mask.numpy().flatten()             # [H*W]

                # Remove ignore pixels (255)
                valid     = gt_flat != 255
                cls_scores.append(fg_score[valid])
                cls_labels.append(gt_flat[valid])
                # ─────────────────────────────────────────────

            _, cls_miou, cls_acc = metrics.compute()
            all_mious.append(cls_miou)
            per_class_ious.append(float(cls_miou))
            per_class_accs.append(float(cls_acc))
            class_name_list.append(cls_name)

            # Concatenate all pixels for this class
            roc_data[cls_name] = {
                "scores": np.concatenate(cls_scores),
                "labels": np.concatenate(cls_labels),
            }

            print(f"  {cls_name:15s} (class {cls_id:2d}) | "
                  f"mIoU={cls_miou*100:.2f}%  PixAcc={cls_acc*100:.2f}%  "
                  f"({len(queries)} query images)")

            # ── Collect segmentation samples ──────────────────
            if len(vis_samples) < N_VIS_SAMPLES:
                for q_img, q_mask in queries:
                    if len(vis_samples) >= N_VIS_SAMPLES:
                        break
                    img_t  = q_img.unsqueeze(0).to(device)
                    logits, _ = model(img_t, novel_cls_id=cls_id)
                    logits_full = F.interpolate(
                        logits, size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear", align_corners=False,
                    )
                    pred_mask = logits_full.argmax(dim=1).squeeze(0)

                    sm = Metrics.SegMetrics(num_classes=2)
                    sm.update(pred_mask.unsqueeze(0), q_mask.unsqueeze(0))
                    _, sample_iou, _ = sm.compute()

                    vis_samples.append({
                        "image"     : q_img,
                        "gt_mask"   : q_mask,
                        "pred_mask" : pred_mask.cpu(),
                        "class_name": cls_name,
                        "iou"       : float(sample_iou),
                    })
            # ─────────────────────────────────────────────────

    mean_novel_miou = sum(all_mious) / len(all_mious)
    print(f"\n[Phase 3 Fold {fold}] Mean novel mIoU = {mean_novel_miou*100:.2f}%")

    # ── Save Phase 3 plots ────────────────────────────────────────
    Visualizer.plot_per_class_iou(
        fold           = fold,
        class_names    = class_name_list,
        per_class_ious = per_class_ious,
        per_class_accs = per_class_accs,
    )
    Visualizer.plot_segmentation_samples(
        fold      = fold,
        samples   = vis_samples,
        n_samples = N_VIS_SAMPLES,
    )
    Visualizer.plot_roc_curve(fold=fold, roc_data=roc_data)   # ← NEW

    return mean_novel_miou


# ─────────────────────────────────────────────────────────────────
# RUN — Loop over all folds
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fold_results = []

    for fold in range(NUM_FOLDS):
        print(f"\n\n{'#'*70}")
        print(f"#  FOLD {fold}  /  {NUM_FOLDS}")
        print(f"{'#'*70}\n")

        # ── Load data (CHANGED: 2 return values) ─────────────────
        train_loader, _, NUM_BASE = Data_Loader.prepare_base_loaders(
            voc_root    = VOC_ROOT,
            sbd_root    = SBD_ROOT,
            fold        = fold,
            batch_size  = BATCH_SIZE,
            val_fraction= VAL_FRACTION,
        )
        # Novel dataset = test set (VOC val)
        novel_dataset, novel_classes = Data_Loader.prepare_test_dataset(
            voc_root = VOC_ROOT,
            fold     = fold,
        )

        # ── Build model ───────────────────────────────────────────
        backbone, feat_dims = Models.load_backbone(BACKBONE_NAME)
        model = APM.SegAPM(
            backbone             = backbone,
            num_base_classes     = NUM_BASE,
            decoder_out_channels = DECODER_CHANNELS,
        ).to(device)

        # ── Optimiser ─────────────────────────────────────────────
        optimizer = optim.Adam([
            {
                "params": model.backbone.layer4.parameters(),
                "lr"    : LEARNING_RATE,
                "name"  : "backbone_layer4",
            },
            {
                "params"      : model.decoder.parameters(),
                "lr"          : DECODER_LR,
                "name"        : "decoder",
                "weight_decay": 1e-4,
            },
            {
                # Temperature τ: one scalar, learns fast.
                # High LR is intentional — τ needs to find its working range
                # (roughly 10–30) within the first few epochs.
                "params": [model.memory_module.temperature],
                "lr"    : 1e-2,
                "name"  : "temperature",
            },
        ])
            # ── Scheduler: CosineAnnealingLR ──────────────────────────
        # Smoothly decays LR from initial value to eta_min over T_max epochs.
        # Reference: Loshchilov & Hutter, ICLR 2017, https://arxiv.org/abs/1608.03983
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max   = NUM_EPOCHS,
            eta_min = LR_MIN,
        )

        # ── Run phases ────────────────────────────────────────────
        # phase1_train_miou = phase1_train(fold, val_loader=val_loader)
        phase1_train_miou = phase1_train(fold, val_loader=None)
        query_data        = phase2_adapt(novel_dataset, novel_classes, K_SHOT, fold)
        novel_miou        = phase3_test(fold, novel_classes, query_data)

        result = {
            "fold"       : fold,
            "phase1_miou": phase1_train_miou,
            "phase3_miou": novel_miou,
        }
        fold_results.append(result)

        print("\n" + "="*60)
        print(f"  FOLD {fold} RESULTS")
        print("="*60)
        print(f"  Phase 1 train mIoU (base)  = {phase1_train_miou*100:.2f}%")
        print(f"  Phase 3 mIoU (novel)       = {novel_miou*100:.2f}%")
        print(f"  Setting: Fold={fold} | {K_SHOT}-shot | {BACKBONE_NAME} + FPN")

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  SUMMARY ACROSS ALL FOLDS")
    print(f"{'='*60}")
    for res in fold_results:
        print(f"  Fold {res['fold']} | "
              f"P1_train={res['phase1_miou']*100:.2f}% | "
              f"P3_novel={res['phase3_miou']*100:.2f}%")

    avg_p3 = sum(r["phase3_miou"] for r in fold_results) / len(fold_results)
    std_p3 = float(np.std([r["phase3_miou"] for r in fold_results])) * 100
    print(f"\n  Mean novel mIoU (Phase 3) = {avg_p3*100:.2f}% ± {std_p3:.2f}%")
    print(f"  (averaged over {NUM_FOLDS} folds, {K_SHOT}-shot, {BACKBONE_NAME}+FPN)")

    Visualizer.plot_fold_summary(fold_results)
    print(f"\n[Visualizer] All plots saved to ./plots/")
