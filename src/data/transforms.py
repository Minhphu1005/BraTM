"""
data/transforms.py
===================
Augmentation pipelines using Albumentations.

Why Albumentations?
  - Fastest augmentation library for 2D images
  - Handles multi-channel images natively (mask-synchronized transforms)
  - Easy to extend

Conventions:
  - image: (H, W, C) float32 numpy array
  - mask:  (H, W)    int32 numpy array
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: tuple = (240, 240)) -> A.Compose:
    """
    Augmentation pipeline for training.

    Includes:
      - Geometric: flip, rotate, scale, elastic (gentle)
      - Intensity: brightness/contrast, Gaussian noise, blur
      - Normalization: per-channel normalization already done at load time
    """
    h, w = image_size
    return A.Compose(
        [
            # ── Geometric ──────────────────────────────────────
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=0,
                border_mode=0,
                p=0.3,
            ),
            A.ElasticTransform(
                alpha=30,
                sigma=5,
                p=0.2,
                border_mode=0,
            ),
            # ── Intensity (image only, not mask) ───────────────
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4,
            ),
            A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            # ── Coarse dropout (simulate artifacts) ────────────
            A.CoarseDropout(
                max_holes=4,
                max_height=20,
                max_width=20,
                fill_value=0,
                p=0.2,
            ),
            # ── Resize/Pad to fixed size ────────────────────────
            A.PadIfNeeded(min_height=h, min_width=w, border_mode=0, p=1.0),
            A.CenterCrop(height=h, width=w, p=1.0),
        ],
        additional_targets={},
        is_check_shapes=False,
    )


def get_val_transforms(image_size: tuple = (240, 240)) -> A.Compose:
    """Minimal transforms for validation / inference (no augmentation)."""
    h, w = image_size
    return A.Compose(
        [
            A.PadIfNeeded(min_height=h, min_width=w, border_mode=0, p=1.0),
            A.CenterCrop(height=h, width=w, p=1.0),
        ],
        is_check_shapes=False,
    )
