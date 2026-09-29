"""
metrics/segmentation_metrics.py
=================================
Evaluation metrics for brain tumor segmentation.

Standard BraTS evaluation metrics:
  - Dice Score (DSC):       Primary metric, class-level and averaged
  - Hausdorff Distance 95%: Surface distance metric, secondary

BraTS official compound metrics (from 3D volumes):
  - Whole Tumor (WT):    classes 1+2+3 combined
  - Tumor Core (TC):     classes 1+3 combined
  - Enhancing Tumor (ET): class 3 only

This module handles both 2D per-slice metrics (during training)
and 3D volume-level metrics (during evaluation).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from scipy.ndimage import label as cc_label
from scipy.spatial.distance import directed_hausdorff


# ─── Per-Class Dice ────────────────────────────────────────

def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 4,
    include_background: bool = False,
    smooth: float = 1e-5,
) -> Dict[str, float]:
    """
    Compute per-class Dice score from batched predictions.

    Args:
        pred:    (B, C, H, W) logits or probabilities
        target:  (B, H, W) integer labels
        num_classes: Number of classes
        include_background: Whether to include class 0
        smooth:  Laplace smoothing

    Returns:
        Dict with keys: 'dice_cls1', 'dice_cls2', ..., 'dice_mean'
    """
    if pred.shape[1] > 1:
        probs = F.softmax(pred, dim=1)
    else:
        probs = torch.sigmoid(pred)

    # One-hot encode target
    target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    start = 0 if include_background else 1
    scores = {}
    dices = []

    for c in range(start, num_classes):
        p = probs[:, c].float()
        g = target_oh[:, c].float()
        inter = (p * g).sum()
        union = p.sum() + g.sum()
        d = (2 * inter + smooth) / (union + smooth)
        scores[f"dice_cls{c}"] = d.item()
        dices.append(d.item())

    scores["dice_mean"] = float(np.mean(dices)) if dices else 0.0
    return scores


# ─── BraTS Compound Regions ────────────────────────────────

CLASS_NAMES = {
    0: "Background",
    1: "NCR/NET",
    2: "Edema",
    3: "Enhancing Tumor",
}

def compute_brats_regions(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    smooth: float = 1e-5,
) -> Dict[str, float]:
    """
    Compute BraTS official region-level Dice scores from 2D/3D maps.

    Regions:
      WT (Whole Tumor)     = labels {1, 2, 3}
      TC (Tumor Core)      = labels {1, 3}
      ET (Enhancing Tumor) = labels {3}

    Args:
        pred_map: Predicted label map (H, W) or (H, W, D)
        gt_map:   Ground truth label map (same shape)

    Returns:
        Dict with 'dice_WT', 'dice_TC', 'dice_ET'
    """
    results = {}
    regions = {
        "WT": ([1, 2, 3], [1, 2, 3]),
        "TC": ([1, 3],    [1, 3]),
        "ET": ([3],       [3]),
    }

    for region_name, (pred_labels, gt_labels) in regions.items():
        p_bin = np.isin(pred_map, pred_labels).astype(np.float32)
        g_bin = np.isin(gt_map,   gt_labels).astype(np.float32)

        inter = (p_bin * g_bin).sum()
        union = p_bin.sum() + g_bin.sum()
        dice  = (2 * inter + smooth) / (union + smooth)
        results[f"dice_{region_name}"] = float(dice)

    return results


# ─── Hausdorff Distance 95% ────────────────────────────────

def hausdorff_distance_95(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
) -> float:
    """
    Compute 95th percentile Hausdorff Distance between two binary masks.

    Returns inf if either mask is empty.
    """
    pred_pts = np.argwhere(pred_bin)
    gt_pts   = np.argwhere(gt_bin)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]

    # Approximate 95th percentile by computing distances
    from scipy.spatial import cKDTree
    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))


# ─── Metric Aggregator ─────────────────────────────────────

class MetricTracker:
    """
    Accumulates per-batch metrics and computes epoch-level statistics.

    Usage:
        tracker = MetricTracker()
        for batch in loader:
            metrics = dice_score(pred, target)
            tracker.update(metrics)
        summary = tracker.compute()
    """

    def __init__(self):
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            if not np.isfinite(v):
                continue
            self._sums[k]   = self._sums.get(k, 0.0)   + v * n
            self._counts[k] = self._counts.get(k, 0)    + n

    def compute(self) -> Dict[str, float]:
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }

    def reset(self):
        self._sums   = {}
        self._counts = {}
