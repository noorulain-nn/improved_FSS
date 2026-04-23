"""
Visualizer.py — Plotting utilities for FSS training and evaluation  (v2)
=========================================================================
Changes from v1:
  - plot_training_curves: val parameters removed (benchmark has no Phase-1 val)
  - plot_roc_curve: NEW — pixel-level ROC per novel class + mean ROC
    (uses sklearn.metrics.roc_curve; reference: Fawcett, 2006, Pattern Rec. Lett.)
  - All other plots unchanged

Saved outputs:
  plots/fold_{N}/training_curves.png       Phase 1 train loss + mIoU + LR
  plots/fold_{N}/phase3_per_class_iou.png  Novel class bar chart
  plots/fold_{N}/segmentation_samples.png  Image | GT | Pred grid
  plots/fold_{N}/roc_curves.png            Per-class + mean ROC   ← NEW
  plots/fold_summary.png                   Cross-fold grouped bar chart

Reference:
  Fawcett T. (2006). An introduction to ROC analysis.
  Pattern Recognition Letters, 27(8), 861-874.
  https://doi.org/10.1016/j.patrec.2005.10.010
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ── Try importing sklearn — graceful fallback if not installed ────
try:
    from sklearn.metrics import roc_curve, auc
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    print("[Visualizer] sklearn not found — ROC curves disabled. "
          "Install with: pip install scikit-learn --break-system-packages")

PLOT_ROOT = "plots"


def _fold_dir(fold):
    path = os.path.join(PLOT_ROOT, f"fold_{fold}")
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Visualizer] Saved → {path}")


# ─────────────────────────────────────────────────────────────────
# 1. Training curves  (Phase 1 — train only, no val)
# ─────────────────────────────────────────────────────────────────
def plot_training_curves(fold, train_losses, train_mious, lr_history=None,
                         val_mious=None, early_stop_epoch=None):
    """
    2-panel (or 3-panel) figure showing Phase 1 training progress.
    No validation curves — benchmark protocol uses no Phase-1 val set.

      Panel 1 — Training Loss vs Epoch
      Panel 2 — Training mIoU vs Epoch
      Panel 3 — LR schedule  (optional, shown if lr_history provided)

    Parameters
    ----------
    fold         : int
    train_losses : list[float]  mean training loss per epoch
    train_mious  : list[float]  training mIoU per epoch (0–1)
    lr_history   : list[float] | None  backbone LR per epoch
    """
    n_panels = 3 if lr_history else 2
    epochs   = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    title = f"Phase 1 Training — Fold {fold}"
    if val_mious:
        title += "  (internal val for early stopping)"
    else:
        title += "  (train only)"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Panel 1 — Loss
    ax = axes[0]
    ax.plot(epochs, train_losses, "o-",
            color="#2196F3", linewidth=1.8, markersize=4, label="Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Cross-Entropy)")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _annotate_best(ax, list(epochs), train_losses, mode="min", label="best")

    # Panel 2 — mIoU
    ax = axes[1]
    ax.plot(epochs, [v * 100 for v in train_mious], "o-",
            color="#4CAF50", linewidth=1.8, markersize=4, label="Train mIoU")
    if val_mious:
        ax.plot(epochs, [v * 100 for v in val_mious], "s--",
                color="#FF9800", linewidth=1.6, markersize=4,
                label="Internal Val mIoU")
    if early_stop_epoch is not None:
        ax.axvline(early_stop_epoch, color="red", linestyle=":",
                   linewidth=1.2, label=f"Stopped @ {early_stop_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Training mIoU")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _annotate_best(ax, list(epochs), [v * 100 for v in train_mious],
                   mode="max", label="best")

    # Panel 3 — LR (optional)
    if lr_history:
        ax = axes[2]
        ax.plot(epochs, lr_history, "^-",
                color="#FF9800", linewidth=1.8, markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule (CosineAnnealing)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "training_curves.png"))


def plot_early_stopping(fold, train_mious, val_mious,
                        best_epoch, patience, stopped_epoch=None):
    """Plot train vs internal-val mIoU with early-stopping markers."""
    epochs = list(range(1, len(train_mious) + 1))
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle(f"Early Stopping Monitor — Fold {fold}  (patience={patience})",
                 fontsize=12, fontweight="bold")

    ax.plot(epochs, [v * 100 for v in train_mious], "o-",
            color="#2196F3", linewidth=1.8, markersize=4, label="Train mIoU")
    ax.plot(epochs, [v * 100 for v in val_mious], "s-",
            color="#4CAF50", linewidth=1.8, markersize=4,
            label="Internal Val mIoU")

    best_val = val_mious[best_epoch - 1] * 100
    ax.axvline(best_epoch, color="#4CAF50", linestyle="--", linewidth=1.4,
               label=f"Best val epoch = {best_epoch}  ({best_val:.2f}%)")
    ax.scatter([best_epoch], [best_val], s=120, color="#4CAF50",
               zorder=5, marker="*")

    if stopped_epoch and stopped_epoch != best_epoch:
        ax.axvline(stopped_epoch, color="red", linestyle=":",
                   linewidth=1.4, label=f"Stopped @ epoch {stopped_epoch}")

    patience_end = min(best_epoch + patience, len(epochs))
    ax.axvspan(best_epoch, patience_end, alpha=0.07, color="orange",
               label=f"Patience window ({patience} epochs)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Train vs Internal Validation mIoU")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "early_stopping.png"))


def _annotate_best(ax, epochs, values, mode="max", label="best"):
    fn   = max if mode == "max" else min
    best = fn(values)
    idx  = values.index(best)
    ax.annotate(f"{label}\n{best:.2f}",
                xy=(epochs[idx], best),
                xytext=(0, 14 if mode == "max" else -22),
                textcoords="offset points",
                ha="center", fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                color="gray")


# ─────────────────────────────────────────────────────────────────
# 2. Per-class IoU bar chart  (Phase 3)
# ─────────────────────────────────────────────────────────────────
def plot_per_class_iou(fold, class_names, per_class_ious, per_class_accs=None):
    """
    Horizontal grouped bar chart: per novel-class mIoU (+ pixel accuracy).

    Parameters
    ----------
    fold            : int
    class_names     : list[str]
    per_class_ious  : list[float]   mIoU per novel class (0–1)
    per_class_accs  : list[float]   pixel accuracy per class (0–1), optional
    """
    n       = len(class_names)
    y_pos   = np.arange(n)
    has_acc = per_class_accs is not None
    bar_h   = 0.35 if has_acc else 0.6

    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.65)))
    fig.suptitle(f"Phase 3 — Novel Class Performance  (Fold {fold})",
                 fontsize=12, fontweight="bold")

    bars_iou = ax.barh(y_pos + (bar_h / 2 if has_acc else 0),
                       [v * 100 for v in per_class_ious],
                       height=bar_h, label="mIoU (%)",
                       color="#2196F3", alpha=0.85)

    if has_acc:
        bars_acc = ax.barh(y_pos - bar_h / 2,
                           [v * 100 for v in per_class_accs],
                           height=bar_h, label="Pixel Acc (%)",
                           color="#FF9800", alpha=0.85)
        for bar in bars_acc:
            w = bar.get_width()
            ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}%", va="center", fontsize=7.5, color="#555")

    for bar in bars_iou:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%", va="center", fontsize=7.5, color="#333")

    mean_iou = np.mean(per_class_ious) * 100
    ax.axvline(mean_iou, color="red", linestyle="--", linewidth=1.2,
               label=f"Mean mIoU = {mean_iou:.1f}%")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Score (%)")
    ax.set_xlim(0, 108)
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "phase3_per_class_iou.png"))


# ─────────────────────────────────────────────────────────────────
# 3. Segmentation sample grid  (Phase 3)
# ─────────────────────────────────────────────────────────────────
def plot_segmentation_samples(fold, samples, n_samples=6):
    """
    Grid of (query image | GT mask | predicted mask) rows.

    Parameters
    ----------
    fold      : int
    samples   : list[dict]  — each dict has keys:
                  'image'      FloatTensor [3,H,W]  normalised
                  'gt_mask'    LongTensor  [H,W]
                  'pred_mask'  LongTensor  [H,W]
                  'class_name' str
                  'iou'        float
    n_samples : int  max rows to plot
    """
    samples = samples[:n_samples]
    n = len(samples)
    if n == 0:
        return

    _mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig = plt.figure(figsize=(10, 3.2 * n))
    fig.suptitle(f"Segmentation Samples — Fold {fold}", fontsize=12,
                 fontweight="bold")

    for row, s in enumerate(samples):
        img_disp = (s["image"].cpu() * _std + _mean).clamp(0, 1).permute(1, 2, 0).numpy()
        gt   = s["gt_mask"].cpu().numpy()
        pred = s["pred_mask"].cpu().numpy()

        ax_img  = fig.add_subplot(n, 3, row * 3 + 1)
        ax_gt   = fig.add_subplot(n, 3, row * 3 + 2)
        ax_pred = fig.add_subplot(n, 3, row * 3 + 3)

        ax_img.imshow(img_disp);  ax_img.set_title(f"{s['class_name']}",  fontsize=8); ax_img.axis("off")
        ax_gt.imshow(gt,   cmap="gray", vmin=0, vmax=1); ax_gt.set_title("Ground truth", fontsize=8); ax_gt.axis("off")
        ax_pred.imshow(pred, cmap="gray", vmin=0, vmax=1)
        ax_pred.set_title(f"Prediction  (IoU={s['iou']*100:.1f}%)", fontsize=8)
        ax_pred.axis("off")

    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "segmentation_samples.png"))


# ─────────────────────────────────────────────────────────────────
# 4. ROC curves  (Phase 3)  — NEW in v2
# ─────────────────────────────────────────────────────────────────
def plot_roc_curve(fold, roc_data):
    """
    Per-class pixel-level ROC curves + macro-average ROC.

    The ROC curve treats each PIXEL as an independent binary classification
    instance (foreground vs background).  For each pixel:
      score  = softmax probability of foreground  (P(class=1))
      label  = ground-truth binary mask value     (0 or 1; 255=ignore)

    Reference:
      Fawcett T. (2006). An introduction to ROC analysis.
      Pattern Recognition Letters, 27(8), 861–874.
      https://doi.org/10.1016/j.patrec.2005.10.010

    Parameters
    ----------
    fold     : int
    roc_data : dict   keyed by class_name (str)
               Each value is a dict with:
                 'scores'  : np.ndarray [N]  fg probability per pixel
                 'labels'  : np.ndarray [N]  ground-truth (0 or 1)
               Pixels with label=255 must be filtered out before passing.

    Example call from main_seg.py:
        Visualizer.plot_roc_curve(fold, roc_data)
    where roc_data is built during phase3_test like:
        roc_data[cls_name] = {'scores': np.array([...]), 'labels': np.array([...])}
    """
    if not _SKLEARN_OK:
        print("  [Visualizer] Skipping ROC — sklearn not installed.")
        return

    n_classes = len(roc_data)
    if n_classes == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Pixel-Level ROC Curves — Phase 3  (Fold {fold})",
                 fontsize=12, fontweight="bold")

    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    # ── Left panel: per-class ROC ──────────────────────────────
    ax = axes[0]
    all_fpr   = np.linspace(0, 1, 200)
    tpr_interp = []

    for i, (cls_name, data) in enumerate(roc_data.items()):
        scores = data["scores"]
        labels = data["labels"]

        if labels.sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc     = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=colors[i], linewidth=1.6,
                label=f"{cls_name}  (AUC={roc_auc:.3f})")

        # Interpolate TPR at common FPR grid for macro average
        tpr_interp.append(np.interp(all_fpr, fpr, tpr))

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random  (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-class ROC")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1.02)

    # ── Right panel: macro-average ROC ────────────────────────
    ax = axes[1]
    if tpr_interp:
        mean_tpr = np.mean(tpr_interp, axis=0)
        mean_auc = auc(all_fpr, mean_tpr)
        std_tpr  = np.std(tpr_interp,  axis=0)

        ax.plot(all_fpr, mean_tpr, color="#2196F3", linewidth=2.2,
                label=f"Macro-avg ROC  (AUC={mean_auc:.3f})")
        ax.fill_between(all_fpr,
                        np.maximum(mean_tpr - std_tpr, 0),
                        np.minimum(mean_tpr + std_tpr, 1),
                        color="#2196F3", alpha=0.15,
                        label=r"$\pm$ 1 std dev across classes")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random  (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Macro-average ROC")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1.02)

    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "roc_curves.png"))


# ─────────────────────────────────────────────────────────────────
# 5. Cross-fold summary  (after all folds)
# ─────────────────────────────────────────────────────────────────
def plot_fold_summary(fold_results):
    """
    Grouped bar chart: Phase 3 novel mIoU per fold + mean ± std.
    (Phase 1 train mIoU also shown for reference.)

    Parameters
    ----------
    fold_results : list[dict]  — each dict: {'fold', 'phase1_miou', 'phase3_miou'}
    """
    os.makedirs(PLOT_ROOT, exist_ok=True)

    folds  = [r["fold"] for r in fold_results]
    p1vals = [r["phase1_miou"] * 100 for r in fold_results]
    p3vals = [r["phase3_miou"] * 100 for r in fold_results]

    x  = np.arange(len(folds))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(folds) * 2), 5))
    fig.suptitle("Cross-Fold Summary — mIoU (%)", fontsize=13, fontweight="bold")

    bars1 = ax.bar(x - w / 2, p1vals, width=w,
                   label="Phase 1 train mIoU (base)",
                   color="#2196F3", alpha=0.85, zorder=3)
    bars3 = ax.bar(x + w / 2, p3vals, width=w,
                   label="Phase 3 mIoU (novel)",
                   color="#FF5722", alpha=0.85, zorder=3)

    for bar in list(bars1) + list(bars3):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    m3, s3 = np.mean(p3vals), np.std(p3vals)
    ax.axhline(m3, color="#BF360C", linestyle="--", linewidth=1.2,
               label=f"Mean novel mIoU = {m3:.1f}% ± {s3:.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("mIoU (%)")
    ax.set_ylim(0, max(max(p1vals), max(p3vals)) * 1.18)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    _save(fig, os.path.join(PLOT_ROOT, "fold_summary.png"))


# ─────────────────────────────────────────────────────────────────
# 6. Confusion matrix heatmap  (optional, per fold Phase 1)
# ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(fold, confusion,
                          class_names=("background", "foreground")):
    """
    Normalised confusion matrix heatmap from SegMetrics.confusion array.

    Parameters
    ----------
    fold        : int
    confusion   : np.ndarray [C, C]  raw counts from SegMetrics
    class_names : tuple[str]
    """
    row_sums = confusion.astype(np.float64).sum(axis=1, keepdims=True)
    cm_norm  = np.divide(confusion.astype(float), row_sums,
                         out=np.zeros_like(confusion, dtype=float),
                         where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle(f"Confusion Matrix — Fold {fold}  (Phase 1 last epoch)",
                 fontsize=10)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks);  ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(ticks);  ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted");  ax.set_ylabel("True")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=11)

    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "confusion_matrix.png"))
