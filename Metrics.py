"""
Metrics.py  —  Segmentation evaluation metrics
===============================================
Replaces the original PLOT.py accuracy/F1 metrics with segmentation metrics.

Key metric for few-shot segmentation: mean IoU (Intersection over Union)
  IoU = |prediction ∩ ground_truth| / |prediction ∪ ground_truth|

We compute:
  • Per-class IoU (foreground)
  • Background IoU
  • Mean IoU (mIoU) = average over foreground classes
  • Pixel accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


# ─────────────────────────────────────────────────────────────────
# IoU computation
# ─────────────────────────────────────────────────────────────────
class SegMetrics:
    """
    Accumulates predictions over a full epoch and computes mIoU.

    Usage:
        metrics = SegMetrics(num_classes=2)   # 0=bg, 1=fg
        for batch in loader:
            ...
            metrics.update(pred_mask, gt_mask)
        iou, miou, acc = metrics.compute()
        metrics.reset()
    """

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        # confusion_matrix[i, j] = number of pixels with true class i
        #                          predicted as class j
        self.confusion = np.zeros((self.num_classes, self.num_classes),
                                  dtype=np.int64)

    def update(self, pred_mask, gt_mask):
        """
        Parameters
        ----------
        pred_mask : LongTensor [B, H, W]  — argmax of logits
        gt_mask   : LongTensor [B, H, W]  — 0=bg, 1=fg, 255=ignore
        """
        pred = pred_mask.cpu().numpy().flatten()
        gt   = gt_mask.cpu().numpy().flatten()

        # Remove ignore pixels (255)
        valid = gt != 255
        pred  = pred[valid]
        gt    = gt[valid]

        # Clip predictions to valid range (safety)
        pred = np.clip(pred, 0, self.num_classes - 1)

        # Accumulate into confusion matrix
        # np.bincount on flattened (true * C + pred)
        combined = gt.astype(np.int64) * self.num_classes + pred.astype(np.int64)
        counts   = np.bincount(combined, minlength=self.num_classes ** 2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """
        Returns
        -------
        per_class_iou : np.array [num_classes]  — IoU for each class
        miou          : float                   — mean IoU (fg classes only)
        pixel_acc     : float                   — overall pixel accuracy
        """
        # IoU for class c:
        # TP = confusion[c, c]
        # FP = confusion[:, c].sum() - TP   (other classes predicted as c)
        # FN = confusion[c, :].sum() - TP   (c predicted as other)
        # IoU = TP / (TP + FP + FN)

        per_class_iou = np.zeros(self.num_classes, dtype=np.float64)
        for c in range(self.num_classes):
            tp = self.confusion[c, c]
            fp = self.confusion[:, c].sum() - tp
            fn = self.confusion[c, :].sum() - tp
            denom = tp + fp + fn
            per_class_iou[c] = tp / denom if denom > 0 else 0.0

        # mIoU over foreground classes only (exclude background = class 0)
        fg_iou = per_class_iou[1:]
        miou   = fg_iou.mean() if len(fg_iou) > 0 else 0.0

        # Pixel accuracy
        correct = self.confusion.diagonal().sum()
        total   = self.confusion.sum()
        pixel_acc = correct / total if total > 0 else 0.0

        return per_class_iou, miou, pixel_acc


# ─────────────────────────────────────────────────────────────────
# Dice loss (optional, improves boundary quality)
# ─────────────────────────────────────────────────────────────────
def dice_loss(pred_prob, target, smooth=1.0):
    """
    Soft Dice loss for binary segmentation.

    Parameters
    ----------
    pred_prob : FloatTensor [B, H, W]  — sigmoid or softmax probability for fg
    target    : FloatTensor [B, H, W]  — binary ground truth (0 or 1)
    smooth    : float  — numerical stability term

    Returns
    -------
    loss : scalar tensor
    """
    # Flatten spatial dimensions
    pred   = pred_prob.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    loss = 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return loss


# ─────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────
def plot_training_curves(train_losses, val_losses, train_mious, val_mious, save_path=None):
    """Plot loss and mIoU curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="Train loss")
    axes[0].plot(epochs, val_losses,   label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_mious, label="Train mIoU")
    axes[1].plot(epochs, val_mious,   label="Val mIoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_title("mIoU curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_segmentation_sample(image, gt_mask, pred_mask, title="", save_path=None):
    """
    Visualise one image alongside its ground truth and predicted masks.

    Parameters
    ----------
    image     : FloatTensor [3, H, W]  — normalised image tensor
    gt_mask   : LongTensor  [H, W]
    pred_mask : LongTensor  [H, W]
    """
    # Denormalise image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_display = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    gt   = gt_mask.cpu().numpy()
    pred = pred_mask.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_display)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth mask")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Predicted mask")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_iou_histogram(val_miou, test_miou, class_names=None, save_path=None):
    """Bar chart comparing val vs test mIoU."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Val mIoU", "Test mIoU"], [val_miou * 100, test_miou * 100],
                  color=["steelblue", "coral"])
    for bar, val in zip(bars, [val_miou, test_miou]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val*100:.1f}%",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Validation vs Test mIoU")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()