"""
utils/visualization.py
=======================
Visualization utilities for brain tumor segmentation results.

Generates:
  - Side-by-side: input modality | ground truth | prediction
  - Color-coded segmentation overlays
  - Training progress plots
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (works on Colab/servers)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple


# ─── Color Map ─────────────────────────────────────────────
# Class → RGBA color
SEG_COLORS = {
    0: (0,   0,   0,   0),    # Background  — transparent
    1: (255, 0,   0,   180),  # NCR/NET     — red
    2: (0,   255, 0,   180),  # Edema       — green
    3: (255, 255, 0,   180),  # Enhancing   — yellow
}

CLASS_NAMES = {
    0: "Background",
    1: "NCR/NET (Core)",
    2: "Edema",
    3: "Enhancing Tumor",
}


def seg_to_rgb(seg_map: np.ndarray) -> np.ndarray:
    """
    Convert class label map to RGBA overlay image.

    Args:
        seg_map: (H, W) integer array
    Returns:
        (H, W, 4) uint8 RGBA array
    """
    h, w = seg_map.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_id, color in SEG_COLORS.items():
        mask = seg_map == cls_id
        overlay[mask] = color
    return overlay


def visualize_prediction(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    modality_idx: int = 1,   # 1 = T1ce (best contrast for tumors)
    save_path: Optional[str] = None,
    title: str = "",
) -> plt.Figure:
    """
    Generate a 3-panel visualization: image | ground truth | prediction.

    Args:
        image: (C, H, W) numpy array — input image (multi-channel)
        gt:    (H, W) numpy array — ground truth label map
        pred:  (H, W) numpy array — predicted label map
        modality_idx: Which channel to display as background
        save_path: If given, save figure to this path
        title: Figure title
    """
    bg = image[modality_idx]  # (H, W) — T1ce by default

    # Normalize for display
    bg_norm = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)

    gt_rgba   = seg_to_rgb(gt.astype(int))   / 255.0
    pred_rgba = seg_to_rgb(pred.astype(int)) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#1a1a2e")

    # Panel 1: Input (T1ce)
    axes[0].imshow(bg_norm, cmap="gray")
    axes[0].set_title("Input (T1ce)", color="white", fontsize=12, pad=8)

    # Panel 2: Ground Truth
    axes[1].imshow(bg_norm, cmap="gray")
    axes[1].imshow(gt_rgba, alpha=0.6)
    axes[1].set_title("Ground Truth", color="white", fontsize=12, pad=8)

    # Panel 3: Prediction
    axes[2].imshow(bg_norm, cmap="gray")
    axes[2].imshow(pred_rgba, alpha=0.6)
    axes[2].set_title("Prediction", color="white", fontsize=12, pad=8)

    # Legend
    patches = [
        mpatches.Patch(color=np.array(c[:3])/255, label=CLASS_NAMES[i])
        for i, c in SEG_COLORS.items() if i > 0
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=3,
        fontsize=10,
        facecolor="#1a1a2e",
        labelcolor="white",
        framealpha=0.7,
    )

    if title:
        fig.suptitle(title, color="white", fontsize=13, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig


def visualize_batch(
    images: torch.Tensor,
    gts: torch.Tensor,
    preds: torch.Tensor,
    n_samples: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a batch of predictions.

    Args:
        images: (B, C, H, W) tensor
        gts:    (B, H, W) tensor
        preds:  (B, H, W) tensor — argmax predictions
    """
    n = min(n_samples, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    fig.patch.set_facecolor("#1a1a2e")

    if n == 1:
        axes = axes[np.newaxis, :]

    for row in range(n):
        img_np  = images[row].cpu().numpy()    # (C, H, W)
        gt_np   = gts[row].cpu().numpy()       # (H, W)
        pred_np = preds[row].cpu().numpy()     # (H, W)

        bg = img_np[1]  # T1ce
        bg_norm = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
        gt_rgba   = seg_to_rgb(gt_np.astype(int))   / 255.0
        pred_rgba = seg_to_rgb(pred_np.astype(int)) / 255.0

        for ax in axes[row]:
            ax.axis("off")
            ax.set_facecolor("#1a1a2e")

        axes[row, 0].imshow(bg_norm, cmap="gray")
        axes[row, 1].imshow(bg_norm, cmap="gray")
        axes[row, 1].imshow(gt_rgba, alpha=0.6)
        axes[row, 2].imshow(bg_norm, cmap="gray")
        axes[row, 2].imshow(pred_rgba, alpha=0.6)

        if row == 0:
            axes[row, 0].set_title("Input (T1ce)", color="white", fontsize=11)
            axes[row, 1].set_title("Ground Truth", color="white", fontsize=11)
            axes[row, 2].set_title("Prediction",   color="white", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation curves from history dict.

    Args:
        history: e.g. {'train_loss': [...], 'val_loss': [...],
                        'train_dice_mean': [...], 'val_dice_mean': [...]}
    """
    metrics_to_plot = [
        ("loss",      "Loss",      "lower is better"),
        ("dice_mean", "Dice Mean", "higher is better"),
        ("dice_cls1", "Dice NCR",  "higher is better"),
        ("dice_cls2", "Dice Edema","higher is better"),
        ("dice_cls3", "Dice ET",   "higher is better"),
    ]

    available = [(k, t, h) for k, t, h in metrics_to_plot
                 if f"train_{k}" in history or f"val_{k}" in history]

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    fig.patch.set_facecolor("#0f0f1a")

    if n == 1:
        axes = [axes]

    for ax, (key, title, hint) in zip(axes, available):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444")

        tr = history.get(f"train_{key}", [])
        vl = history.get(f"val_{key}", [])
        if tr:
            ax.plot(tr, label="Train", color="#4fc3f7", linewidth=1.5)
        if vl:
            ax.plot(vl, label="Val",   color="#ef9a9a", linewidth=1.5)

        ax.set_title(f"{title}\n({hint})", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend(facecolor="#1a1a2e", labelcolor="white")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig
