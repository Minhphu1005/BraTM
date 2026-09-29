"""
losses/combined_loss.py
========================
Loss functions for multi-class medical image segmentation.

Why Dice Loss?
  - Class-imbalance robust: tumor pixels << background pixels
  - Directly optimizes the metric we care about

Why Focal Loss?
  - Down-weights easy examples, focuses on hard misclassifications
  - Helps with the tiny ET (Enhancing Tumor) sub-region

Combined: Dice + Focal (DiceFocal) is the de-facto standard for BraTS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Dice Loss ─────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - mean_Dice_over_classes

    Args:
        num_classes:        Number of segmentation classes
        include_background: If False, class 0 (BG) excluded from Dice mean
        smooth:             Laplace smoothing to avoid division by zero
        softmax:            Apply softmax to logits before computing Dice
    """

    def __init__(
        self,
        num_classes: int = 4,
        include_background: bool = False,
        smooth: float = 1e-5,
        softmax: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth
        self.softmax = softmax

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W)   — raw model output
            targets: (B, H, W)      — integer class labels
        Returns:
            Scalar dice loss
        """
        if self.softmax:
            probs = F.softmax(logits, dim=1)
        else:
            probs = logits

        # One-hot encode targets → (B, C, H, W)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        start_cls = 0 if self.include_background else 1
        dice_sum = 0.0
        n = 0

        for c in range(start_cls, self.num_classes):
            p = probs[:, c]          # (B, H, W)
            g = targets_oh[:, c]    # (B, H, W)
            intersection = (p * g).sum()
            union = p.sum() + g.sum()
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_sum += dice
            n += 1

        return 1.0 - dice_sum / max(n, 1)


# ─── Focal Loss ────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma:              Focusing parameter (2.0 is standard)
        alpha:              Class weights tensor or None
        ignore_index:       Class index to ignore (-100 to disable)
        reduction:          'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ─── Combined Dice + Focal ─────────────────────────────────

class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal loss.

    total_loss = lambda_dice * DiceLoss + lambda_focal * FocalLoss

    Args:
        num_classes:        Number of output classes
        include_background: Include BG class in Dice calculation
        smooth:             Smoothing for Dice
        focal_gamma:        Gamma for Focal loss
        lambda_dice:        Weight for Dice term
        lambda_focal:       Weight for Focal term
    """

    def __init__(
        self,
        num_classes: int = 4,
        include_background: bool = False,
        smooth: float = 1e-5,
        focal_gamma: float = 2.0,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
    ):
        super().__init__()
        self.lambda_dice  = lambda_dice
        self.lambda_focal = lambda_focal

        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            include_background=include_background,
            smooth=smooth,
            softmax=True,
        )
        self.focal_loss = FocalLoss(gamma=focal_gamma)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: Scalar
            loss_dict:  {'dice': ..., 'focal': ..., 'total': ...}
        """
        d = self.dice_loss(logits, targets)
        f = self.focal_loss(logits, targets)
        total = self.lambda_dice * d + self.lambda_focal * f
        return total, {"dice": d.item(), "focal": f.item(), "total": total.item()}


# ─── Factory ───────────────────────────────────────────────

def build_loss(cfg: dict) -> nn.Module:
    """Build loss from config dict."""
    name = cfg.get("name", "DiceFocalLoss")

    if name == "DiceFocalLoss":
        return DiceFocalLoss(
            include_background=cfg.get("include_background", False),
            focal_gamma=cfg.get("focal_gamma", 2.0),
            lambda_dice=cfg.get("dice_weight", 1.0),
            lambda_focal=cfg.get("focal_weight", 1.0),
        )
    elif name == "DiceLoss":
        return DiceLoss(
            include_background=cfg.get("include_background", False),
        )
    elif name == "FocalLoss":
        return FocalLoss(gamma=cfg.get("focal_gamma", 2.0))
    else:
        raise ValueError(f"Unknown loss: {name}")
