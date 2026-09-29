"""
models/attention_unet2d.py
===========================
Attention U-Net: adds Attention Gates in skip connections.

The attention gate learns to suppress irrelevant activations and focuses on
salient features relevant to a specific task (tumor region).

Reference:
  Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas."
  MIDL 2018. arXiv:1804.03999
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ─── Attention Gate ────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Soft attention gate.

    Args:
        F_g: Feature channels from decoder (gating signal)
        F_l: Feature channels from encoder (skip connection)
        F_int: Intermediate channels
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from decoder  (B, F_g, H, W)
            x: Skip connection from encoder (B, F_l, H, W)
        Returns:
            x_hat: Attention-weighted skip  (B, F_l, H, W)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Align spatial size (g may be smaller)
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ─── Reusable Blocks ───────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch, dropout),
            ConvBnRelu(out_ch, out_ch, dropout),
        )

    def forward(self, x):
        return self.block(x)


# ─── Attention U-Net ───────────────────────────────────────

class AttentionUNet2D(nn.Module):
    """
    2D Attention U-Net.

    Args:
        in_channels:  Input channels (4 for BraTS)
        num_classes:  Output classes (4 for BraTS)
        features:     Encoder feature sizes [f1, f2, f3, f4]
        dropout:      Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        features: List[int] = (64, 128, 256, 512),
        dropout: float = 0.1,
    ):
        super().__init__()
        f = features
        self.in_channels = in_channels
        self.num_classes = num_classes

        # ── Encoder ──────────────────────────────────────────
        self.enc1 = DoubleConv(in_channels, f[0], dropout)
        self.enc2 = DoubleConv(f[0], f[1], dropout)
        self.enc3 = DoubleConv(f[1], f[2], dropout)
        self.enc4 = DoubleConv(f[2], f[3], dropout)

        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ───────────────────────────────────────
        self.bottleneck = DoubleConv(f[3], f[3] * 2, dropout)

        # ── Attention Gates ──────────────────────────────────
        self.attn4 = AttentionGate(F_g=f[3] * 2, F_l=f[3], F_int=f[3] // 2)
        self.attn3 = AttentionGate(F_g=f[2],     F_l=f[2], F_int=f[2] // 2)
        self.attn2 = AttentionGate(F_g=f[1],     F_l=f[1], F_int=f[1] // 2)
        self.attn1 = AttentionGate(F_g=f[0],     F_l=f[0], F_int=f[0] // 2)

        # ── Decoder ──────────────────────────────────────────
        self.up4   = nn.ConvTranspose2d(f[3] * 2, f[3], 2, stride=2)
        self.dec4  = DoubleConv(f[3] * 2, f[3], dropout)

        self.up3   = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3  = DoubleConv(f[2] * 2, f[2], dropout)

        self.up2   = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2  = DoubleConv(f[1] * 2, f[1], dropout)

        self.up1   = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1  = DoubleConv(f[0] * 2, f[0], dropout)

        # ── Output ───────────────────────────────────────────
        self.out_conv = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with attention gates
        d4 = self.up4(b)
        e4_att = self.attn4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([e4_att, d4], dim=1))

        d3 = self.up3(d4)
        e3_att = self.attn3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([e3_att, d3], dim=1))

        d2 = self.up2(d3)
        e2_att = self.attn2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([e2_att, d2], dim=1))

        d1 = self.up1(d2)
        e1_att = self.attn1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([e1_att, d1], dim=1))

        return self.out_conv(d1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
