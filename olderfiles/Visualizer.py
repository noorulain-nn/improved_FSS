"""
Visualizer.py — Plotting utilities for FSS  (v4 — Improved)
============================================================

NEW PLOTS IN v4:
  6. plot_early_stopping()    — val mIoU + train mIoU + early-stop marker
  7. plot_prototype_tsne()    — t-SNE of memory slot embeddings per fold
  8. plot_per_class_iou_radar() — radar/spider chart of novel-class mIoU
  9. plot_miou_histogram()    — per-query-image mIoU distribution
  10.plot_fold_summary()      — updated: shows val mIoU curve across folds

UPDATED PLOTS:
  1. plot_training_curves() — now accepts optional val_mious for early stopping line
  5. plot_fold_summary()    — now includes val mIoU column if available

EXISTING PLOTS (unchanged):
  2. plot_per_class_iou()       — horizontal bar chart
  3. plot_segmentation_samples() — image | GT | pred grid
  4. plot_roc_curve()           — per-class + macro-avg ROC

Saved outputs:
  plots/fold_{N}/training_curves.png
  plots/fold_{N}/early_stopping.png      ← NEW
  plots/fold_{N}/phase3_per_class_iou.png
  plots/fold_{N}/phase3_radar.png        ← NEW
  plots/fold_{N}/miou_histogram.png      ← NEW
  plots/fold_{N}/segmentation_samples.png
  plots/fold_{N}/roc_curves.png
  plots/fold_{N}/prototype_tsne.png      ← NEW
  plots/fold_summary.png

References:
  Fawcett (2006) ROC analysis. Pattern Rec. Lett. 27(8).
  Van der Maaten & Hinton (2008) t-SNE. JMLR 9.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

try:
    from sklearn.metrics import roc_curve, auc
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    print("[Visualizer] sklearn not found — ROC/t-SNE disabled. "
          "pip install scikit-learn --break-system-packages")

PLOT_ROOT = "plots"


def _fold_dir(fold):
    path = os.path.join(PLOT_ROOT, f"fold_{fold}")
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Visualizer] Saved → {path}")


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
# 1. Training curves  (Phase 1)
# ─────────────────────────────────────────────────────────────────
def plot_training_curves(fold, train_losses, train_mious,
                          lr_history=None, val_mious=None,
                          early_stop_epoch=None):
    """
    Training progress figure.

    Parameters
    ----------
    fold              : int
    train_losses      : list[float]
    train_mious       : list[float]   (0–1)
    lr_history        : list[float] | None
    val_mious         : list[float] | None   internal validation mIoU per epoch
    early_stop_epoch  : int | None           epoch where training stopped early
    """
    n_panels = 3 if lr_history else 2
    epochs   = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    title = f"Phase 1 Training — Fold {fold}"
    if early_stop_epoch:
        title += f"  [Early stop @ epoch {early_stop_epoch}]"
    else:
        title += "  (no val set, benchmark protocol)" if val_mious is None else ""
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Panel 1 — Loss
    ax = axes[0]
    ax.plot(epochs, train_losses, "o-", color="#2196F3",
            linewidth=1.8, markersize=4, label="Train loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (Cross-Entropy)")
    ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    _annotate_best(ax, epochs, train_losses, mode="min", label="best")
    if early_stop_epoch:
        ax.axvline(early_stop_epoch, color="red", linestyle=":",
                   linewidth=1.5, label=f"Early stop")

    # Panel 2 — mIoU
    ax = axes[1]
    ax.plot(epochs, [v * 100 for v in train_mious], "o-",
            color="#4CAF50", linewidth=1.8, markersize=4, label="Train mIoU")
    if val_mious is not None:
        val_epochs = list(range(1, len(val_mious) + 1))
        ax.plot(val_epochs, [v * 100 for v in val_mious], "s--",
                color="#FF9800", linewidth=1.8, markersize=4,
                label="Val mIoU (internal)")
        if early_stop_epoch:
            ax.axvline(early_stop_epoch, color="red", linestyle=":",
                       linewidth=1.5, label=f"Early stop")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mIoU (%)")
    ax.set_title("Training mIoU"); ax.legend(); ax.grid(True, alpha=0.3)
    _annotate_best(ax, epochs, [v * 100 for v in train_mious],
                   mode="max", label="best")

    # Panel 3 — LR
    if lr_history:
        ax = axes[2]
        ax.plot(epochs, lr_history, "^-", color="#FF9800",
                linewidth=1.8, markersize=4)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule (CosineAnnealing)")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "training_curves.png"))


# ─────────────────────────────────────────────────────────────────
# 2. Per-class IoU bar chart  (Phase 3)
# ─────────────────────────────────────────────────────────────────
def plot_per_class_iou(fold, class_names, per_class_ious, per_class_accs=None):
    n       = len(class_names)
    y_pos   = np.arange(n)
    has_acc = per_class_accs is not None
    bar_h   = 0.35 if has_acc else 0.6

    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.7)))
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

    ax.set_yticks(y_pos); ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Score (%)"); ax.set_xlim(0, 108)
    ax.legend(fontsize=9); ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "phase3_per_class_iou.png"))


# ─────────────────────────────────────────────────────────────────
# 3. Segmentation samples grid  (Phase 3)
# ─────────────────────────────────────────────────────────────────
def plot_segmentation_samples(fold, samples, n_samples=6):
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

        ax_img.imshow(img_disp)
        ax_img.set_title(f"{s['class_name']}", fontsize=8)
        ax_img.axis("off")

        ax_gt.imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax_gt.set_title("Ground truth", fontsize=8)
        ax_gt.axis("off")

        ax_pred.imshow(pred, cmap="gray", vmin=0, vmax=1)
        ax_pred.set_title(f"Prediction  (IoU={s['iou']*100:.1f}%)", fontsize=8)
        ax_pred.axis("off")

    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "segmentation_samples.png"))


# ─────────────────────────────────────────────────────────────────
# 4. ROC curves  (Phase 3)
# ─────────────────────────────────────────────────────────────────
def plot_roc_curve(fold, roc_data):
    if not _SKLEARN_OK:
        print("  [Visualizer] Skipping ROC — sklearn not installed.")
        return

    n_classes = len(roc_data)
    if n_classes == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Pixel-Level ROC Curves — Phase 3  (Fold {fold})",
                 fontsize=12, fontweight="bold")

    colors    = plt.cm.Set2(np.linspace(0, 1, n_classes))
    all_fpr   = np.linspace(0, 1, 200)
    tpr_interp = []

    ax = axes[0]
    for i, (cls_name, data) in enumerate(roc_data.items()):
        scores = data["scores"]
        labels = data["labels"]
        if labels.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], linewidth=1.6,
                label=f"{cls_name}  (AUC={roc_auc:.3f})")
        tpr_interp.append(np.interp(all_fpr, fpr, tpr))

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random  (AUC=0.500)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-class ROC")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

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
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Macro-average ROC")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "roc_curves.png"))


# ─────────────────────────────────────────────────────────────────
# 5. Cross-fold summary
# ─────────────────────────────────────────────────────────────────
def plot_fold_summary(fold_results):
    os.makedirs(PLOT_ROOT, exist_ok=True)

    folds  = [r["fold"] for r in fold_results]
    p1vals = [r["phase1_miou"] * 100 for r in fold_results]
    p3vals = [r["phase3_miou"] * 100 for r in fold_results]
    has_val = "best_val_miou" in fold_results[0]
    vavals  = [r.get("best_val_miou", 0) * 100 for r in fold_results]

    x = np.arange(len(folds))
    w = 0.25 if has_val else 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(folds) * 2.5), 5))
    fig.suptitle("Cross-Fold Summary — mIoU (%)", fontsize=13, fontweight="bold")

    bars1 = ax.bar(x - w, p1vals, width=w,
                   label="Phase 1 train mIoU (base)",
                   color="#2196F3", alpha=0.85, zorder=3)
    if has_val:
        bars_v = ax.bar(x, vavals, width=w,
                        label="Best internal val mIoU",
                        color="#9C27B0", alpha=0.85, zorder=3)
        bars3 = ax.bar(x + w, p3vals, width=w,
                       label="Phase 3 mIoU (novel)",
                       color="#FF5722", alpha=0.85, zorder=3)
    else:
        bars3 = ax.bar(x + w / 2, p3vals, width=w,
                       label="Phase 3 mIoU (novel)",
                       color="#FF5722", alpha=0.85, zorder=3)

    for bar in list(bars1) + list(bars3) + (list(bars_v) if has_val else []):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    m3, s3 = np.mean(p3vals), np.std(p3vals)
    ax.axhline(m3, color="#BF360C", linestyle="--", linewidth=1.2,
               label=f"Mean novel mIoU = {m3:.1f}% ± {s3:.1f}%")

    ax.set_xticks(x); ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("mIoU (%)")
    ax.set_ylim(0, max(max(p1vals), max(p3vals)) * 1.18)
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    _save(fig, os.path.join(PLOT_ROOT, "fold_summary.png"))


# ─────────────────────────────────────────────────────────────────
# 6. Early stopping plot  (NEW)
# ─────────────────────────────────────────────────────────────────
def plot_early_stopping(fold, train_mious, val_mious,
                         best_epoch, patience, stopped_epoch=None):
    """
    Plots train vs val mIoU with early stopping marker.

    Parameters
    ----------
    fold          : int
    train_mious   : list[float]  train mIoU per epoch (0–1)
    val_mious     : list[float]  val mIoU per epoch (0–1)
    best_epoch    : int          epoch with best val mIoU (1-indexed)
    patience      : int          early stopping patience
    stopped_epoch : int | None   epoch where training was actually stopped
    """
    epochs = list(range(1, len(train_mious) + 1))
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle(f"Early Stopping Monitor — Fold {fold}  "
                 f"(patience={patience})", fontsize=12, fontweight="bold")

    ax.plot(epochs, [v * 100 for v in train_mious], "o-",
            color="#2196F3", linewidth=1.8, markersize=4, label="Train mIoU")
    ax.plot(epochs, [v * 100 for v in val_mious], "s-",
            color="#4CAF50", linewidth=1.8, markersize=4, label="Internal Val mIoU")

    # Mark best val epoch
    best_val = val_mious[best_epoch - 1] * 100
    ax.axvline(best_epoch, color="#4CAF50", linestyle="--", linewidth=1.4,
               label=f"Best val epoch = {best_epoch}  ({best_val:.2f}%)")
    ax.scatter([best_epoch], [best_val], s=120, color="#4CAF50",
               zorder=5, marker="*")

    # Mark where training stopped (if different from best)
    if stopped_epoch and stopped_epoch != best_epoch:
        ax.axvline(stopped_epoch, color="red", linestyle=":",
                   linewidth=1.4, label=f"Stopped @ epoch {stopped_epoch}")

    # Patience window
    patience_end = min(best_epoch + patience, len(epochs))
    ax.axvspan(best_epoch, patience_end, alpha=0.07, color="orange",
               label=f"Patience window ({patience} epochs)")

    ax.set_xlabel("Epoch"); ax.set_ylabel("mIoU (%)")
    ax.set_title("Train vs Internal Validation mIoU")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "early_stopping.png"))


# ─────────────────────────────────────────────────────────────────
# 7. Prototype t-SNE  (NEW)
# ─────────────────────────────────────────────────────────────────
def plot_prototype_tsne(fold, memory_matrix, num_base_classes, n_proto,
                         base_class_names):
    """
    2-D t-SNE of prototype memory slots to visualise class separation.

    Reference: Van der Maaten & Hinton (2008). Visualizing High-Dimensional
    Data Using t-SNE. JMLR 9:2579-2605. https://www.jmlr.org/papers/v9/vandermaaten08a.html

    Parameters
    ----------
    fold              : int
    memory_matrix     : np.ndarray [num_slots, D]  — normalised memory
    num_base_classes  : int
    n_proto           : int  — number of fg prototypes per class
    base_class_names  : list[str]  — len = num_base_classes
    """
    if not _SKLEARN_OK:
        return
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return

    # Extract only fg slots (skip slot 0 = global bg, skip class-bg slots)
    n_fg = num_base_classes * n_proto
    fg_slots = memory_matrix[1: 1 + n_fg]   # [N_fg, D]

    if fg_slots.shape[0] < 4:
        return  # t-SNE needs at least a few points

    perp = min(30, max(5, fg_slots.shape[0] // 2))
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000,
                random_state=42, verbose=0)
    emb  = tsne.fit_transform(fg_slots)   # [N_fg, 2]

    colors = plt.cm.tab20(np.linspace(0, 1, num_base_classes))

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"Prototype t-SNE — Fold {fold}  (fg slots only)",
                 fontsize=12, fontweight="bold")

    for cls_i in range(num_base_classes):
        start = cls_i * n_proto
        pts   = emb[start: start + n_proto]
        ax.scatter(pts[:, 0], pts[:, 1],
                   color=colors[cls_i], s=80, zorder=3,
                   edgecolors="white", linewidths=0.5)
        # Label at centroid
        cx, cy = pts.mean(0)
        ax.text(cx, cy, base_class_names[cls_i],
                fontsize=7, ha="center", va="center",
                color=colors[cls_i], fontweight="bold")

    # Legend patches
    patches = [mpatches.Patch(color=colors[i], label=base_class_names[i])
               for i in range(num_base_classes)]
    ax.legend(handles=patches, fontsize=6, ncol=3,
              loc="lower right", framealpha=0.8)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "prototype_tsne.png"))


# ─────────────────────────────────────────────────────────────────
# 8. Radar / spider chart  (NEW)
# ─────────────────────────────────────────────────────────────────
def plot_per_class_iou_radar(fold, class_names, per_class_ious,
                              target_miou=0.70):
    """
    Radar (spider) chart of per-class mIoU for novel classes.
    Intuitive for thesis figures — shows which classes are weak at a glance.

    Parameters
    ----------
    fold           : int
    class_names    : list[str]
    per_class_ious : list[float]   (0–1)
    target_miou    : float         target mIoU line (default 0.70)
    """
    N = len(class_names)
    if N < 3:
        return

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    values = [v * 100 for v in per_class_ious]
    values += values[:1]

    target_vals = [target_miou * 100] * N + [target_miou * 100]

    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(polar=True))
    fig.suptitle(f"Novel Class mIoU Radar — Fold {fold}",
                 fontsize=12, fontweight="bold")

    ax.plot(angles, values, "o-", color="#2196F3", linewidth=2, markersize=5,
            label="Model mIoU (%)")
    ax.fill(angles, values, color="#2196F3", alpha=0.20)

    ax.plot(angles, target_vals, "--", color="red", linewidth=1.2,
            label=f"Target {target_miou*100:.0f}%")

    ax.set_thetagrids(np.degrees(angles[:-1]), class_names, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "phase3_radar.png"))


# ─────────────────────────────────────────────────────────────────
# 9. Per-image mIoU histogram  (NEW)
# ─────────────────────────────────────────────────────────────────
def plot_miou_histogram(fold, per_image_ious, class_names_per_image=None):
    """
    Distribution of per-query-image mIoU scores.
    Helps identify the long tail of hard/easy images.

    Parameters
    ----------
    fold                  : int
    per_image_ious        : list[float]   mIoU for each query image (0–1)
    class_names_per_image : list[str] | None  class label per image (for colouring)
    """
    vals = np.array(per_image_ious) * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Per-Image mIoU Distribution — Fold {fold}",
                 fontsize=12, fontweight="bold")

    # Left: histogram
    ax = axes[0]
    ax.hist(vals, bins=20, color="#2196F3", alpha=0.8, edgecolor="white")
    ax.axvline(vals.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {vals.mean():.1f}%")
    ax.axvline(np.median(vals), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median = {np.median(vals):.1f}%")
    ax.set_xlabel("mIoU (%)"); ax.set_ylabel("Image count")
    ax.set_title("Histogram"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right: sorted curve
    ax = axes[1]
    sorted_vals = np.sort(vals)
    ax.plot(sorted_vals, np.linspace(0, 100, len(sorted_vals)),
            color="#4CAF50", linewidth=1.8)
    ax.axhline(50, color="gray", linestyle=":", linewidth=1)
    ax.axvline(vals.mean(), color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("mIoU (%)"); ax.set_ylabel("Cumulative %")
    ax.set_title("Cumulative Distribution"); ax.grid(True, alpha=0.3)
    pct_above_50 = (vals >= 50).mean() * 100
    ax.text(0.05, 0.92, f"{pct_above_50:.0f}% of images ≥ 50% mIoU",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray"))

    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "miou_histogram.png"))


# ─────────────────────────────────────────────────────────────────
# 10. Confusion matrix  (unchanged from v3)
# ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(fold, confusion,
                           class_names=("background", "foreground")):
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
    ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=11)
    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "confusion_matrix.png"))