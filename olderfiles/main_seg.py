"""
main_seg.py  —  FSS with FPN Decoder  (v4 — Improved)
======================================================

CHANGES FROM v3:
  1. NUM_EPOCHS increased from 10 → 50 (model was not converging at 10).
     CosineAnnealingLR T_max updated accordingly.

  2. INTERNAL VALIDATION + EARLY STOPPING (benchmark-compliant).
     phase1_train() now evaluates on the internal val set (from train split)
     each epoch and tracks val mIoU. Early stopping triggers when val mIoU
     has not improved for PATIENCE consecutive epochs.
     The checkpoint is saved at the epoch with BEST VAL mIoU (not best loss).
     References:
       - Early stopping: Prechelt (1998) "Early stopping — but when?"
         Neural Networks: Tricks of the Trade, Lecture Notes in CS.
       - Benchmark-compliant val split: standard in HSNet (Min et al.,
         ICCV 2021) and PFENet (Tian et al., TPAMI 2022).

  3. CHECKPOINT BY mIoU (not loss).
     Previously saved whenever train loss decreased. Now saves whenever
     INTERNAL VAL mIoU improves. If val_loader is None (val_fraction=0),
     falls back to saving by train mIoU.

  4. MULTI-SCALE SUPPORT FEATURES in Phase 2.
     Phase 2 now calls model.get_multiscale_features() to obtain P2, P3,
     P4 feature maps for each support image. All three scales are passed
     to memory_module.build_novel_prototype(), which pools features at
     each scale before averaging — more robust prototypes for small objects.
     Reference: PFENet (Tian et al., TPAMI 2022), Section 3.2.

  5. MULTI-PROTOTYPE INFERENCE in Phase 3.
     model.memory_module.forward() now returns [B, n_proto+1, h, w] for
     novel classes. Foreground is predicted when ANY fg prototype wins over
     background. The calling code aggregates correctly.

  6. NEW VISUALIZATIONS.
     Visualizer calls added for:
       - plot_early_stopping()        — val vs train mIoU with early-stop marker
       - plot_prototype_tsne()        — t-SNE of memory slot embeddings
       - plot_per_class_iou_radar()   — radar chart of novel-class mIoU
       - plot_miou_histogram()        — per-image mIoU distribution

  7. DATACLEANER BACKGROUND.
     compute_batch_loss() and phase1_train() pass other_fg_masks from the
     DataLoader to memory_module.update_from_batch(), excluding pixels from
     other base-class objects when building the background prototype.

UNCHANGED FROM v3:
  3-phase structure, fold loop, criterion, data loading, FPN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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
N_PROTO          = 3       # prototypes per class (multi-prototype)
BATCH_SIZE       = 8
NUM_EPOCHS       = 50      # was 10 — model was not converged
LEARNING_RATE    = 1e-3    # backbone layer4 LR
DECODER_LR       = 5e-4    # decoder LR (slightly lower for stability)
IMG_SIZE         = 321
LR_MIN           = 1e-6    # CosineAnnealingLR eta_min

# Early stopping
PATIENCE         = 10      # stop if val mIoU does not improve for 10 epochs
VAL_FRACTION     = 0.1     # 10% of train split → internal validation set

N_VIS_SAMPLES    = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Backbone: {BACKBONE_NAME} | {K_SHOT}-shot")
print(f"Decoder: FPN out_channels={DECODER_CHANNELS}")
print(f"N_PROTO: {N_PROTO} prototypes per class")
print(f"Scheduler: CosineAnnealingLR  T_max={NUM_EPOCHS}  eta_min={LR_MIN}")
print(f"Early stopping: patience={PATIENCE}  val_fraction={VAL_FRACTION}")
print(f"Running {NUM_FOLDS} folds...")

criterion = nn.CrossEntropyLoss(ignore_index=255)


# ─────────────────────────────────────────────────────────────────
# Batch loss helper — Phase 1 training
# ─────────────────────────────────────────────────────────────────
def compute_batch_loss(model, images, masks, class_labels, other_fg_masks=None):
    """
    Compute per-batch cross-entropy loss for Phase 1.

    For each sample i, extracts the [bg_slot, fg_slot] logits for its
    class, then computes binary cross-entropy against the binary mask.

    Parameters
    ----------
    model         : SegAPM
    images        : FloatTensor [B, 3, H, W]
    masks         : LongTensor  [B, H, W]   0=bg 1=fg 255=ignore
    class_labels  : LongTensor  [B]
    other_fg_masks: BoolTensor  [B, H, W] | None  — from DataLoader

    Returns
    -------
    loss   : scalar tensor
    preds  : list of LongTensor [H, W]  — argmax predictions per sample
    fused  : FloatTensor [B, D, h, w]   — feature map for memory update
    """
    logits, fused = model(images)   # [B, num_slots, h, w]

    logits_full = F.interpolate(
        logits, size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear", align_corners=False,
    )

    B    = images.shape[0]
    loss = torch.tensor(0.0, device=device)
    preds = []

    for i in range(B):
        cls_idx  = class_labels[i].item()
        # Use only one fg slot (slot 0 of the n_proto set) for training loss.
        # The memory module handles multi-prototype internally via EMA+kmeans.
        fg_slots = model.memory_module._fg_slots(cls_idx)
        fg_slot  = fg_slots[0]   # primary prototype slot for loss
        bg_slot  = model.memory_module._bg_slot(cls_idx)

        logits_i = torch.stack(
            [logits_full[i, bg_slot], logits_full[i, fg_slot]], dim=0
        ).unsqueeze(0)   # [1, 2, H, W]

        mask_i = masks[i].unsqueeze(0)    # [1, H, W]
        loss  += criterion(logits_i, mask_i)
        preds.append(logits_i.argmax(dim=1).squeeze(0))

    return loss / B, preds, fused


# ─────────────────────────────────────────────────────────────────
# Phase 1 — Train on base classes
# ─────────────────────────────────────────────────────────────────
def phase1_train(fold, val_loader_ref=None):
    """
    Train backbone + decoder on base classes with early stopping.

    Checkpoint is saved at the epoch with best INTERNAL VAL mIoU.
    If val_loader is None, falls back to saving by best TRAIN mIoU.

    Parameters
    ----------
    fold           : int
    val_loader_ref : list  — [val_loader] (mutable reference so we can
                             access the loader without a global variable)
                             Pass None to disable validation.

    Returns
    -------
    best_train_miou : float
    best_val_miou   : float  (0.0 if no val loader)
    early_stop_epoch: int    (epoch where training stopped; == actual last epoch
                              if no early stopping triggered)
    train_history   : dict   keys: 'losses', 'mious', 'val_mious', 'lrs'
    """
    val_loader = val_loader_ref[0] if val_loader_ref else None

    print("\n" + "="*60)
    print(f"  PHASE 1 — Training on BASE classes  (Fold {fold})")
    print(f"  Scheduler: CosineAnnealingLR  T_max={NUM_EPOCHS}  eta_min={LR_MIN}")
    if val_loader:
        print(f"  Early stopping: patience={PATIENCE} (by internal val mIoU)")
    print("="*60)

    train_losses = []
    train_mious  = []
    val_mious    = []
    lr_history   = []

    best_val_miou   = 0.0
    best_train_miou = 0.0
    best_epoch      = 0
    no_improve      = 0   # epochs since last val improvement
    early_stop_epoch = NUM_EPOCHS

    for epoch in range(NUM_EPOCHS):
        # ── Training pass ─────────────────────────────────────────
        model.train()
        metrics    = Metrics.SegMetrics(num_classes=2)
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # DataLoader now returns 4 items: image, mask, other_fg, label
            images, masks, other_fg, labels = batch
            images   = images.to(device)
            masks    = masks.to(device)
            other_fg = other_fg.to(device)

            optimizer.zero_grad()
            loss, preds, fused = compute_batch_loss(
                model, images, masks, labels, other_fg
            )
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            # Memory update with clean background
            with torch.no_grad():
                model.memory_module.update_from_batch(
                    fused.detach(), masks, labels.tolist(),
                    other_fg_masks=other_fg
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

        # ── Validation pass (internal, benchmark-compliant) ───────
        val_miou = 0.0
        if val_loader is not None:
            model.eval()
            val_metrics = Metrics.SegMetrics(num_classes=2)
            with torch.no_grad():
                for batch in val_loader:
                    images_v, masks_v, other_fg_v, labels_v = batch
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
              f" | LR backbone={lrs[0]:.2e} decoder={lrs[-1]:.2e}"
              f" | Train Loss={avg_loss:.4f}"
              f" | Train mIoU={train_miou*100:.2f}%"
              + (f" | Val mIoU={val_miou*100:.2f}%" if val_loader else ""))

        # ── Checkpoint: save by best val mIoU (or train mIoU) ────
        if val_loader is not None:
            if val_miou > best_val_miou:
                best_val_miou = float(val_miou)
                best_epoch    = epoch + 1
                torch.save(model.state_dict(), f"phase1_best_fold{fold}.pth")
                print(f"  ★ Checkpoint saved  (val mIoU={best_val_miou*100:.2f}%)")
                no_improve = 0
            else:
                no_improve += 1
                print(f"  No val improvement for {no_improve}/{PATIENCE} epochs")
        else:
            # No val loader — save by train mIoU
            if train_miou > best_train_miou:
                best_train_miou = float(train_miou)
                best_epoch      = epoch + 1
                torch.save(model.state_dict(), f"phase1_best_fold{fold}.pth")
                print(f"  ★ Checkpoint saved  (train mIoU={best_train_miou*100:.2f}%)")

        if train_miou > best_train_miou:
            best_train_miou = float(train_miou)

        scheduler.step()

        # ── Early stopping check ──────────────────────────────────
        if val_loader is not None and no_improve >= PATIENCE:
            early_stop_epoch = epoch + 1
            print(f"\n  [Early Stopping] Val mIoU has not improved for "
                  f"{PATIENCE} epochs. Best epoch = {best_epoch}. "
                  f"Restoring best checkpoint.")
            model.load_state_dict(
                torch.load(f"phase1_best_fold{fold}.pth", map_location=device)
            )
            break

    print(f"\n[Phase 1 Fold {fold}] Best train mIoU = {best_train_miou*100:.2f}%")
    if val_loader:
        print(f"[Phase 1 Fold {fold}] Best val   mIoU = {best_val_miou*100:.2f}%"
              f"  (epoch {best_epoch})")

    history = {
        "losses":    train_losses,
        "mious":     train_mious,
        "val_mious": val_mious if val_loader else [],
        "lrs":       lr_history,
    }

    # Plots
    Visualizer.plot_training_curves(
        fold             = fold,
        train_losses     = train_losses,
        train_mious      = train_mious,
        lr_history       = lr_history,
        val_mious        = val_mious if val_loader else None,
        early_stop_epoch = early_stop_epoch if val_loader else None,
    )

    if val_loader and val_mious:
        Visualizer.plot_early_stopping(
            fold         = fold,
            train_mious  = train_mious,
            val_mious    = val_mious,
            best_epoch   = best_epoch,
            patience     = PATIENCE,
            stopped_epoch= early_stop_epoch,
        )

    return best_train_miou, best_val_miou, early_stop_epoch, history


# ─────────────────────────────────────────────────────────────────
# Phase 2 — Multi-scale adaptation to novel classes
# ─────────────────────────────────────────────────────────────────
def phase2_adapt(novel_dataset, novel_classes, k_shot, fold):
    """
    Build multi-scale, multi-prototype memory entries for novel classes.

    For each novel class:
      1. Load K support images.
      2. Extract features at THREE FPN scales (P2, P3, P4).
      3. Pass all scales to build_novel_prototype() which:
         - Masked-average-pools each scale separately
         - Averages across scales
         - Runs K-means to get N_PROTO prototypes
         - Stores fresh bg prototype (Fix 2)

    Reference: PFENet multi-scale prototype (Tian et al., TPAMI 2022).
    """
    print("\n" + "="*60)
    print(f"  PHASE 2 — {k_shot}-shot adaptation  (Fold {fold})")
    print(f"  Multi-scale prototype extraction: P2 + P3 + P4")
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

        # Collect multi-scale features per support image
        support_feat_lists = []   # K × [feat_P4, feat_P3, feat_P2]
        support_masks_list = []

        with torch.no_grad():
            for img, msk in support:
                img_t = img.unsqueeze(0).to(device)

                # get_multiscale_features returns (P4, P3, P2, fused)
                # We pass [P4, P3, P2] as the scale list
                p4, p3, p2, _ = model.get_multiscale_features(img_t)
                support_feat_lists.append([p4, p3, p2])
                support_masks_list.append(msk.unsqueeze(0).to(device))

        model.memory_module.build_novel_prototype(
            support_feat_lists, support_masks_list, cls_id
        )

    print("\n[Phase 2] Novel prototypes built (multi-scale, multi-prototype).")
    return query_data


# ─────────────────────────────────────────────────────────────────
# Phase 3 — Test on novel classes
# ─────────────────────────────────────────────────────────────────
def phase3_test(fold, novel_classes, query_data):
    """
    Evaluate on novel-class query images from VOC val.

    Multi-prototype inference:
      logits shape: [1, n_proto+1, H, W]
        Channel 0       = background similarity
        Channels 1..n   = fg prototype similarities

      Foreground prediction: pixel is foreground if ANY fg proto
      similarity exceeds bg similarity, i.e.:
        pred = (max over fg channels) > bg channel  →  1, else 0

    Collects per-image mIoU for histogram, ROC scores, and segmentation
    visualisations.
    """
    print("\n" + "="*60)
    print(f"  PHASE 3 — Testing on NOVEL classes  (Fold {fold})")
    print(f"  Test set: VOC2012 val  (benchmark protocol)")
    print("="*60)

    model.eval()
    all_mious        = []
    per_class_ious   = []
    per_class_accs   = []
    class_name_list  = []
    vis_samples      = []
    roc_data         = {}
    per_image_ious   = []

    with torch.no_grad():
        for cls_id in novel_classes:
            cls_name = Data_Loader.VOC_CLASS_NAMES[cls_id]
            queries  = query_data[cls_id]
            metrics  = Metrics.SegMetrics(num_classes=2)
            cls_scores = []
            cls_labels = []

            for q_img, q_mask in queries:
                img_t  = q_img.unsqueeze(0).to(device)
                mask_t = q_mask.unsqueeze(0).to(device)

                logits, _ = model(img_t, novel_cls_id=cls_id)
                # logits: [1, n_proto+1, h, w]
                logits_full = F.interpolate(
                    logits, size=(IMG_SIZE, IMG_SIZE),
                    mode="bilinear", align_corners=False,
                )

                # Multi-prototype inference:
                # bg = logits_full[:,0,...]
                # fg = logits_full[:,1:,...].max(dim=1)
                bg_logit = logits_full[:, 0:1, :, :]          # [1,1,H,W]
                fg_logit = logits_full[:, 1:,  :, :].max(1, keepdim=True)[0]
                                                               # [1,1,H,W]
                binary_logits = torch.cat([bg_logit, fg_logit], dim=1)
                                                               # [1,2,H,W]
                pred = binary_logits.argmax(dim=1)             # [1,H,W]

                # Per-image mIoU for histogram
                sm = Metrics.SegMetrics(num_classes=2)
                sm.update(pred, mask_t)
                _, img_miou, _ = sm.compute()
                per_image_ious.append(float(img_miou))

                metrics.update(pred, mask_t)

                # ROC scores
                probs    = F.softmax(binary_logits, dim=1)
                fg_score = probs[0, 1].cpu().numpy().flatten()
                gt_flat  = q_mask.numpy().flatten()
                valid    = gt_flat != 255
                cls_scores.append(fg_score[valid])
                cls_labels.append(gt_flat[valid])

            _, cls_miou, cls_acc = metrics.compute()
            all_mious.append(cls_miou)
            per_class_ious.append(float(cls_miou))
            per_class_accs.append(float(cls_acc))
            class_name_list.append(cls_name)

            roc_data[cls_name] = {
                "scores": np.concatenate(cls_scores),
                "labels": np.concatenate(cls_labels),
            }

            print(f"  {cls_name:15s} (class {cls_id:2d}) | "
                  f"mIoU={cls_miou*100:.2f}%  PixAcc={cls_acc*100:.2f}%  "
                  f"({len(queries)} query images)")

            # Collect visualisation samples
            if len(vis_samples) < N_VIS_SAMPLES:
                for q_img, q_mask in queries:
                    if len(vis_samples) >= N_VIS_SAMPLES:
                        break
                    img_t = q_img.unsqueeze(0).to(device)
                    logits, _ = model(img_t, novel_cls_id=cls_id)
                    logits_full = F.interpolate(
                        logits, size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear", align_corners=False,
                    )
                    bg_logit = logits_full[:, 0:1]
                    fg_logit = logits_full[:, 1:].max(1, keepdim=True)[0]
                    pred_mask = torch.cat([bg_logit, fg_logit], dim=1
                                          ).argmax(dim=1).squeeze(0)

                    smi = Metrics.SegMetrics(num_classes=2)
                    smi.update(pred_mask.unsqueeze(0), q_mask.unsqueeze(0))
                    _, samp_iou, _ = smi.compute()

                    vis_samples.append({
                        "image":      q_img,
                        "gt_mask":    q_mask,
                        "pred_mask":  pred_mask.cpu(),
                        "class_name": cls_name,
                        "iou":        float(samp_iou),
                    })

    mean_novel_miou = sum(all_mious) / len(all_mious)
    print(f"\n[Phase 3 Fold {fold}] Mean novel mIoU = {mean_novel_miou*100:.2f}%")

    # ── Save all Phase 3 plots ────────────────────────────────────
    Visualizer.plot_per_class_iou(
        fold=fold, class_names=class_name_list,
        per_class_ious=per_class_ious, per_class_accs=per_class_accs,
    )
    Visualizer.plot_segmentation_samples(
        fold=fold, samples=vis_samples, n_samples=N_VIS_SAMPLES,
    )
    Visualizer.plot_roc_curve(fold=fold, roc_data=roc_data)
    Visualizer.plot_per_class_iou_radar(
        fold=fold, class_names=class_name_list,
        per_class_ious=per_class_ious, target_miou=0.70,
    )
    Visualizer.plot_miou_histogram(
        fold=fold, per_image_ious=per_image_ious,
    )

    return mean_novel_miou


# ─────────────────────────────────────────────────────────────────
# Main loop — all folds
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fold_results = []

    for fold in range(NUM_FOLDS):
        print(f"\n\n{'#'*70}")
        print(f"#  FOLD {fold}  /  {NUM_FOLDS}")
        print(f"{'#'*70}\n")

        # ── Data loaders ──────────────────────────────────────────
        # v4: prepare_base_loaders returns 3 values (train, val, n_base)
        train_loader, val_loader, NUM_BASE = Data_Loader.prepare_base_loaders(
            voc_root     = VOC_ROOT,
            sbd_root     = SBD_ROOT,
            fold         = fold,
            batch_size   = BATCH_SIZE,
            val_fraction = VAL_FRACTION,
        )

        novel_dataset, novel_classes = Data_Loader.prepare_test_dataset(
            voc_root=VOC_ROOT, fold=fold,
        )

        # ── Build model ───────────────────────────────────────────
        backbone, feat_dims = Models.load_backbone(BACKBONE_NAME)
        model = APM.SegAPM(
            backbone             = backbone,
            num_base_classes     = NUM_BASE,
            decoder_out_channels = DECODER_CHANNELS,
            n_proto              = N_PROTO,
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
        ])

        # ── Scheduler ─────────────────────────────────────────────
        scheduler = CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS, eta_min=LR_MIN
        )

        # ── Phase 1 training with early stopping ─────────────────
        # Pass val_loader as a mutable list reference so early stopping
        # inside phase1_train() can access it without a global variable.
        val_loader_ref = [val_loader]  # [None] if VAL_FRACTION=0

        phase1_train_miou, phase1_val_miou, stop_epoch, history = \
            phase1_train(fold, val_loader_ref=val_loader_ref)

        # ── Prototype t-SNE (after training, before Phase 2) ─────
        mem_np       = model.memory_module.memory.data.cpu().numpy()
        novel_ids    = Data_Loader.PASCAL_FSS_SPLITS[fold]
        base_ids     = [c for c in range(1, 21) if c not in novel_ids]
        base_names   = [Data_Loader.VOC_CLASS_NAMES[c] for c in base_ids]
        Visualizer.plot_prototype_tsne(
            fold             = fold,
            memory_matrix    = mem_np,
            num_base_classes = NUM_BASE,
            n_proto          = N_PROTO,
            base_class_names = base_names,
        )

        # ── Phase 2 & 3 ──────────────────────────────────────────
        query_data = phase2_adapt(novel_dataset, novel_classes, K_SHOT, fold)
        novel_miou = phase3_test(fold, novel_classes, query_data)

        result = {
            "fold":          fold,
            "phase1_miou":   phase1_train_miou,
            "best_val_miou": phase1_val_miou,
            "phase3_miou":   novel_miou,
            "stop_epoch":    stop_epoch,
        }
        fold_results.append(result)

        print("\n" + "="*60)
        print(f"  FOLD {fold} RESULTS")
        print("="*60)
        print(f"  Phase 1 train mIoU (base) = {phase1_train_miou*100:.2f}%")
        if val_loader:
            print(f"  Best internal val mIoU   = {phase1_val_miou*100:.2f}%"
                  f"  (stopped @ epoch {stop_epoch})")
        print(f"  Phase 3 mIoU (novel)      = {novel_miou*100:.2f}%")
        print(f"  Setting: Fold={fold} | {K_SHOT}-shot | {BACKBONE_NAME} + FPN")

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  SUMMARY ACROSS ALL FOLDS")
    print(f"{'='*60}")
    for res in fold_results:
        print(f"  Fold {res['fold']} | "
              f"P1_train={res['phase1_miou']*100:.2f}% | "
              f"Val_mIoU={res['best_val_miou']*100:.2f}% | "
              f"P3_novel={res['phase3_miou']*100:.2f}%"
              f"  [stopped ep {res['stop_epoch']}]")

    avg_p3 = sum(r["phase3_miou"] for r in fold_results) / len(fold_results)
    std_p3 = float(np.std([r["phase3_miou"] for r in fold_results])) * 100
    print(f"\n  Mean novel mIoU (Phase 3) = {avg_p3*100:.2f}% ± {std_p3:.2f}%")
    print(f"  (averaged over {NUM_FOLDS} folds, {K_SHOT}-shot, {BACKBONE_NAME}+FPN+ASPP)")

    Visualizer.plot_fold_summary(fold_results)
    print(f"\n[Visualizer] All plots saved to ./plots/")