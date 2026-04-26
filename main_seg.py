"""
main_seg.py  —  FSS with FPN Decoder + Dense Matching
======================================================

CHANGES FROM PREVIOUS VERSION:
  1. Dense support-query matching added to Phase 3
     - Blending done in PROBABILITY SPACE (not logit space)
     - Fixes the "all pixels predicted as foreground" bug
     - DENSE_WEIGHT = 0.4 (prototype 60%, dense matching 40%)

  2. Temperature LR lowered from 1e-2 to 1e-3
     - Fixes gradient spikes in Folds 2 and 3
     - tau still learns but cannot overshoot late in training

  3. Temperature gradient hard-clamped after loss.backward()
     - Additional protection against tau oscillation

  4. phase2_adapt now returns (query_data, support_data)
     - support_data stores K support feats+masks per novel class
     - Used by Phase 3 dense matching

  5. phase3_test now accepts support_data parameter

EVERYTHING ELSE UNCHANGED.
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
BATCH_SIZE       = 16
NUM_EPOCHS       = 25
LEARNING_RATE    = 3e-4
DECODER_LR       = 2e-4
IMG_SIZE         = 321
LR_MIN           = 1e-5
PATIENCE         = 5
VAL_FRACTION     = 0.0

# Dense matching weight — blend in probability space
# 0.0 = prototype only (original behaviour)
# 0.4 = prototype 60% + dense 40%  (recommended)
# 1.0 = dense only
DENSE_WEIGHT     = 0.4

N_VIS_SAMPLES    = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Backbone: {BACKBONE_NAME} | {K_SHOT}-shot")
print(f"Decoder: FPN out_channels={DECODER_CHANNELS}")
print(f"Scheduler: CosineAnnealingLR  T_max={NUM_EPOCHS}  eta_min={LR_MIN}")
print(f"Dense matching weight: {DENSE_WEIGHT}")
print(f"Running {NUM_FOLDS} folds...")

criterion = nn.CrossEntropyLoss(ignore_index=255)


# ─────────────────────────────────────────────────────────────────
# Dice loss
# ─────────────────────────────────────────────────────────────────
def dice_loss(pred_logits, target, eps=1e-6):
    if pred_logits.shape[1] == 2:
        fg_idx = 1
        pred_soft    = torch.softmax(pred_logits, dim=1)[:, fg_idx]
        valid        = (target != 255).float()
        target_f     = (target == 1).float() * valid
        pred_f       = pred_soft * valid
        intersection = (pred_f * target_f).sum(dim=[1, 2])
        union        = pred_f.sum(dim=[1, 2]) + target_f.sum(dim=[1, 2])
        dice         = 1.0 - (2.0 * intersection + eps) / (union + eps)
        return dice.mean()
    else:
        pred_soft = torch.softmax(pred_logits, dim=1)
        valid     = (target != 255).float()
        target_f  = (target == 1).float() * valid
        fg_conf   = pred_soft[:, 1:].max(dim=1)[0] * valid
        intersection = (fg_conf * target_f).sum(dim=[1, 2])
        union        = fg_conf.sum(dim=[1, 2]) + target_f.sum(dim=[1, 2])
        dice         = 1.0 - (2.0 * intersection + eps) / (union + eps)
        return dice.mean()


def compute_batch_loss(model, images, masks, class_labels, novel_cls_id=None):
    logits, fused = model(images, novel_cls_id)

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
        loss  += criterion(logits_i, mask_i) + 0.5 * dice_loss(logits_i, mask_i)
        preds.append(logits_i.argmax(dim=1).squeeze(0))

    return loss / B, preds, fused


# ─────────────────────────────────────────────────────────────────
# Dense support-query matching helper
# ─────────────────────────────────────────────────────────────────
def dense_match_score(query_feat, support_feats, support_masks):
    """
    For each query pixel, finds its maximum cosine similarity
    against all FOREGROUND support pixels across all K support images.

    This is PANet-lite style matching — instead of one averaged
    prototype vector, we compare against every individual support
    foreground pixel and take the best match per query pixel.

    Returns values in [0, 1] — safe to blend in probability space.

    Parameters
    ----------
    query_feat    : [1, D, hq, wq]
    support_feats : list of K [1, D, hs, ws] tensors
    support_masks : list of K [1, H, W]  tensors (binary, 1=fg)

    Returns
    -------
    dense_sim : [1, 1, hq, wq]  per-pixel max foreground similarity
    """
    D  = query_feat.shape[1]
    hq = query_feat.shape[2]
    wq = query_feat.shape[3]

    q_norm = F.normalize(query_feat, p=2, dim=1)   # [1, D, hq, wq]
    q_flat = q_norm.view(1, D, hq * wq)             # [1, D, hq*wq]

    best_sim = torch.full(
        (1, hq * wq), -1.0, device=query_feat.device
    )

    for s_feat, s_mask in zip(support_feats, support_masks):
        hs, ws = s_feat.shape[2], s_feat.shape[3]

        # Downsample support mask to feature resolution
        s_mask_down = F.interpolate(
            s_mask.float().unsqueeze(1),
            size=(hs, ws), mode="nearest"
        ).squeeze(1)   # [1, hs, ws]

        fg_mask = (s_mask_down == 1).float()
        if fg_mask.sum() < 1:
            continue

        s_norm = F.normalize(s_feat, p=2, dim=1)    # [1, D, hs, ws]
        s_flat = s_norm.view(1, D, hs * ws)           # [1, D, hs*ws]

        # Zero out background support pixels
        fg_flat = fg_mask.view(1, 1, hs * ws)
        s_fg    = s_flat * fg_flat                    # [1, D, hs*ws]

        # Each query pixel vs all support fg pixels
        sim_matrix = torch.bmm(
            q_flat.permute(0, 2, 1),   # [1, hq*wq, D]
            s_fg                        # [1, D,     hs*ws]
        )   # [1, hq*wq, hs*ws]

        # Best match per query pixel, clamp negatives
        sim_max = sim_matrix.clamp(min=0).max(dim=2)[0]   # [1, hq*wq]
        best_sim = torch.max(best_sim, sim_max)

    return best_sim.view(1, 1, hq, wq)   # [1, 1, hq, wq] in [0, 1]


# ─────────────────────────────────────────────────────────────────
# PHASE 1
# ─────────────────────────────────────────────────────────────────
def phase1_train(fold, val_loader=None):
    print("\n" + "="*60)
    print(f"  PHASE 1 — Training on BASE classes  (Fold {fold})")
    print(f"  Scheduler: CosineAnnealingLR  T_max={NUM_EPOCHS}  eta_min={LR_MIN}")
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

            # Clip total gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # FIXED: hard-clamp temperature gradient separately
            # prevents tau from overshooting late in training
            if model.memory_module.temperature.grad is not None:
                model.memory_module.temperature.grad.clamp_(-0.5, 0.5)

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

        if val_loader is None and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"phase1_best_fold{fold}.pth")
            print(f"  ★ Checkpoint saved  (train loss={best_loss:.4f})")

        if train_miou > best_train_miou:
            best_train_miou = float(train_miou)

        scheduler.step()

        if val_loader is not None and no_improve >= PATIENCE:
            early_stop_epoch = epoch + 1
            model.load_state_dict(
                torch.load(f"phase1_best_fold{fold}.pth", map_location=device)
            )
            break

    print(f"\n[Phase 1 Fold {fold}] Best train mIoU = {best_train_miou*100:.2f}%")

    Visualizer.plot_training_curves(
        fold         = fold,
        train_losses = train_losses,
        train_mious  = train_mious,
        lr_history   = lr_history,
        val_mious    = val_mious if val_loader is not None else None,
        early_stop_epoch = early_stop_epoch if val_loader is not None else None,
    )

    return best_train_miou


# ─────────────────────────────────────────────────────────────────
# PHASE 2
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

    query_data   = {}
    support_data = {}   # stores support feats+masks for Phase 3 dense matching

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

        # Save for Phase 3 dense matching
        support_data[cls_id] = {
            "feats" : support_feats,        # list of K [1,256,h,w]
            "masks" : support_masks_list,   # list of K [1,H,W]
        }

    print("\n[Phase 2] Novel prototypes built. Support features cached.")
    return query_data, support_data


# ─────────────────────────────────────────────────────────────────
# PHASE 3
# ─────────────────────────────────────────────────────────────────
def phase3_test(fold, novel_classes, query_data, support_data):
    """
    Evaluate on novel-class query images.
    Combines prototype-based cosine similarity (existing)
    with dense support-query pixel matching (new).

    Blending is done in PROBABILITY SPACE:
      fg_prob = (1-DENSE_WEIGHT)*proto_fg_prob + DENSE_WEIGHT*dense_sim
    This prevents the "all foreground" bug caused by logit-space addition.
    """
    print("\n" + "="*60)
    print(f"  PHASE 3 — Testing on NOVEL classes  (Fold {fold})")
    print(f"  Test set: VOC2012 val  (benchmark protocol)")
    print(f"  Dense matching weight: {DENSE_WEIGHT}")
    print("="*60)

    model.eval()
    all_mious       = []
    per_class_ious  = []
    per_class_accs  = []
    class_name_list = []
    vis_samples     = []
    roc_data        = {}

    with torch.no_grad():
        for cls_id in novel_classes:
            cls_name = Data_Loader.VOC_CLASS_NAMES[cls_id]
            queries  = query_data[cls_id]
            metrics  = Metrics.SegMetrics(num_classes=2)

            # Get support features for this class
            s_feats = support_data[cls_id]["feats"]
            s_masks = support_data[cls_id]["masks"]

            cls_scores = []
            cls_labels = []

            for q_img, q_mask in queries:
                img_t  = q_img.unsqueeze(0).to(device)
                mask_t = q_mask.unsqueeze(0).to(device)

                # ── Step 1: prototype cosine logits ───────────────────
                logits, fused = model(img_t, novel_cls_id=cls_id)
                # logits: [1, 2, hf, wf]   fused: [1, 256, hf, wf]

                # ── Step 2: dense pixel matching ──────────────────────
                dense_sim = dense_match_score(fused, s_feats, s_masks)
                # dense_sim: [1, 1, hf, wf]  values in [0, 1]

                # ── Step 3: blend in PROBABILITY SPACE ───────────────
                # Convert prototype logits to probabilities first,
                # then blend with dense similarity scores.
                # Both signals are in [0,1] so no overflow is possible.
                proto_probs = F.softmax(logits, dim=1)  # [1, 2, hf, wf]

                fg_combined = (
                    (1.0 - DENSE_WEIGHT) * proto_probs[:, 1:2, :, :]
                    + DENSE_WEIGHT        * dense_sim.clamp(0.0, 1.0)
                )   # [1, 1, hf, wf]
                bg_combined = 1.0 - fg_combined  # [1, 1, hf, wf]

                # Stack back to [1, 2, hf, wf] for upsample + argmax
                logits_combined = torch.cat(
                    [bg_combined, fg_combined], dim=1
                )   # [1, 2, hf, wf]

                # ── Step 4: upsample and predict ─────────────────────
                logits_full = F.interpolate(
                    logits_combined,
                    size=(IMG_SIZE, IMG_SIZE),
                    mode="bilinear", align_corners=False,
                )
                pred = logits_full.argmax(dim=1)
                metrics.update(pred, mask_t)

                # ── ROC scores ────────────────────────────────────────
                # Use fg_combined (already probabilities) for ROC
                fg_full  = F.interpolate(
                    fg_combined,
                    size=(IMG_SIZE, IMG_SIZE),
                    mode="bilinear", align_corners=False,
                )
                fg_score = fg_full[0, 0].cpu().numpy().flatten()
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

            # ── Visualisation samples ─────────────────────────────────
            if len(vis_samples) < N_VIS_SAMPLES:
                for q_img, q_mask in queries:
                    if len(vis_samples) >= N_VIS_SAMPLES:
                        break
                    img_t = q_img.unsqueeze(0).to(device)

                    logits_v, fused_v = model(img_t, novel_cls_id=cls_id)
                    dense_v  = dense_match_score(fused_v, s_feats, s_masks)
                    probs_v  = F.softmax(logits_v, dim=1)
                    fg_v     = (
                        (1.0 - DENSE_WEIGHT) * probs_v[:, 1:2, :, :]
                        + DENSE_WEIGHT * dense_v.clamp(0.0, 1.0)
                    )
                    bg_v     = 1.0 - fg_v
                    comb_v   = torch.cat([bg_v, fg_v], dim=1)
                    full_v   = F.interpolate(
                        comb_v, size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear", align_corners=False,
                    )
                    pred_mask = full_v.argmax(dim=1).squeeze(0)

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

    mean_novel_miou = sum(all_mious) / len(all_mious)
    print(f"\n[Phase 3 Fold {fold}] Mean novel mIoU = {mean_novel_miou*100:.2f}%")

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
    Visualizer.plot_roc_curve(fold=fold, roc_data=roc_data)

    return mean_novel_miou


# ─────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fold_results = []

    for fold in range(NUM_FOLDS):
        print(f"\n\n{'#'*70}")
        print(f"#  FOLD {fold}  /  {NUM_FOLDS}")
        print(f"{'#'*70}\n")

        train_loader, _, NUM_BASE = Data_Loader.prepare_base_loaders(
            voc_root    = VOC_ROOT,
            sbd_root    = SBD_ROOT,
            fold        = fold,
            batch_size  = BATCH_SIZE,
            val_fraction= VAL_FRACTION,
        )
        novel_dataset, novel_classes = Data_Loader.prepare_test_dataset(
            voc_root = VOC_ROOT,
            fold     = fold,
        )

        backbone, feat_dims = Models.load_backbone(BACKBONE_NAME)
        model = APM.SegAPM(
            backbone             = backbone,
            num_base_classes     = NUM_BASE,
            decoder_out_channels = DECODER_CHANNELS,
        ).to(device)

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
                "params": [model.memory_module.temperature],
                "lr"    : 1e-3,    # FIXED: was 1e-2, caused gradient spikes
                "name"  : "temperature",
            },
        ])

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max   = NUM_EPOCHS,
            eta_min = LR_MIN,
        )

        # ── To skip Phase 1 and use existing checkpoints: ─────────────
        # Comment out phase1_train and set phase1_train_miou = 0.0
        # phase2_adapt loads the checkpoint internally.
        #
        # phase1_train_miou = 0.0  # skip training, use existing .pth
        #
        phase1_train_miou = phase1_train(fold, val_loader=None)
        query_data, support_data = phase2_adapt(
            novel_dataset, novel_classes, K_SHOT, fold
        )
        novel_miou = phase3_test(
            fold, novel_classes, query_data, support_data
        )

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