"""
models/unet2d.py
=================
Classic 2D U-Net for multi-class brain tumor segmentation.

Architecture:
  Encoder:  4 DoubleConv blocks + MaxPool → progressively doubles features
  Bottleneck: deepest DoubleConv
  Decoder:  Bilinear upsample + skip connections → DoubleConv
  Head:     1×1 Conv → num_classes logits

Reference:
  Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image
  Segmentation." MICCAI 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ─── Building Blocks ───────────────────────────────────────

class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        mid_channels = mid_channels or out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """MaxPool → DoubleConv (encoder step)."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsample → Concat skip → DoubleConv (decoder step)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad if spatial dims don't match (edge case)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1×1 Conv → class logits."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─── U-Net 2D ──────────────────────────────────────────────

class UNet2D(nn.Module):
    """
    Standard 2D U-Net.

    Args:
        in_channels:  Number of input channels (4 for BraTS: T1,T1ce,T2,FLAIR)
        num_classes:  Number of output classes (4 for BraTS: BG,NCR,ED,ET)
        features:     Feature map sizes at each encoder depth
        bilinear:     Use bilinear upsampling (vs. ConvTranspose2d)
        dropout:      Dropout rate in DoubleConv blocks
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        features: List[int] = (32, 64, 128, 256, 512),
        bilinear: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        factor = 2 if bilinear else 1

        # ── Encoder ──────────────────────────────────────────
        self.inc   = DoubleConv(in_channels, features[0], dropout=dropout)
        self.down1 = Down(features[0], features[1], dropout=dropout)
        self.down2 = Down(features[1], features[2], dropout=dropout)
        self.down3 = Down(features[2], features[3], dropout=dropout)
        self.down4 = Down(features[3], features[4] // factor, dropout=dropout)

        # ── Decoder ──────────────────────────────────────────
        self.up1 = Up(features[4], features[3] // factor, bilinear, dropout=dropout)
        self.up2 = Up(features[3], features[2] // factor, bilinear, dropout=dropout)
        self.up3 = Up(features[2], features[1] // factor, bilinear, dropout=dropout)
        self.up4 = Up(features[1], features[0], bilinear, dropout=dropout)

        # ── Output ───────────────────────────────────────────
        self.outc = OutConv(features[0], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, num_classes, H, W)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        return self.outc(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
