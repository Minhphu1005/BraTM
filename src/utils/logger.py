"""
utils/logger.py
================
Unified logging utility that writes to:
  1. Python logging (console + rotating file)
  2. TensorBoard SummaryWriter
  3. Weights & Biases (optional)

Usage:
    logger = ExperimentLogger(cfg, run_dir="./logs/run_001")
    logger.log_scalars({"loss": 0.23, "dice": 0.71}, step=100, phase="train")
    logger.log_images({"pred": img_tensor}, step=100)
"""

import os
import logging
import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """
    Centralized logger for training experiments.

    Args:
        cfg:     Full config dict (for hyperparameter logging)
        run_dir: Directory to save logs (tensorboard events, text logs)
    """

    def __init__(self, cfg: dict, run_dir: str):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # ── Text Logger ──────────────────────────────────────
        self._setup_text_logger()

        # ── TensorBoard ──────────────────────────────────────
        self.tb_writer: Optional[SummaryWriter] = None
        if cfg.get("logging", {}).get("tensorboard", True):
            tb_dir = self.run_dir / "tensorboard"
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            self.info(f"TensorBoard log dir: {tb_dir}")
            self.info(f"  Run: tensorboard --logdir {tb_dir.parent}")

        # ── Weights & Biases ─────────────────────────────────
        self.wandb_run = None
        wandb_cfg = cfg.get("logging", {}).get("wandb", {})
        if wandb_cfg.get("enabled", False):
            self._setup_wandb(wandb_cfg, cfg)

    # ── Text Logger Setup ─────────────────────────────────────

    def _setup_text_logger(self):
        log_file = self.run_dir / "train.log"
        fmt = "%(asctime)s [%(levelname)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ]
        logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                            handlers=handlers, force=True)
        self._logger = logging.getLogger("BrainTM")

    # ── WandB Setup ───────────────────────────────────────────

    def _setup_wandb(self, wandb_cfg: dict, cfg: dict):
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=wandb_cfg.get("project", "brain-tumor-seg"),
                entity=wandb_cfg.get("entity"),
                config=cfg,
                name=wandb_cfg.get("run_name", None),
                dir=str(self.run_dir),
            )
            self.info("Weights & Biases initialized.")
        except ImportError:
            self.warning("wandb not installed. Skipping W&B logging.")

    # ── Public API ────────────────────────────────────────────

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def log_scalars(
        self,
        metrics: Dict[str, float],
        step: int,
        phase: str = "train",
    ):
        """Log scalar metrics to TensorBoard and W&B."""
        for k, v in metrics.items():
            tag = f"{phase}/{k}"
            if self.tb_writer:
                self.tb_writer.add_scalar(tag, v, global_step=step)
            if self.wandb_run:
                import wandb
                wandb.log({tag: v}, step=step)

    def log_images(
        self,
        images: Dict[str, torch.Tensor],
        step: int,
        phase: str = "val",
        max_images: int = 4,
    ):
        """
        Log image tensors to TensorBoard.

        Args:
            images: Dict of name → tensor (B,C,H,W) or (B,H,W)
        """
        if self.tb_writer is None:
            return
        for name, img in images.items():
            tag = f"{phase}/{name}"
            if img.dim() == 3:
                # Grayscale batch → (B, 1, H, W)
                img = img[:max_images].unsqueeze(1).float()
                # Normalize to [0,1]
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            elif img.dim() == 4:
                img = img[:max_images].float()
            self.tb_writer.add_images(tag, img, global_step=step)

    def log_hyperparams(self, hparams: dict, metrics: dict):
        """Log hyperparameters and final metrics."""
        if self.tb_writer:
            self.tb_writer.add_hparams(hparams, metrics)

    def log_model_graph(self, model: torch.nn.Module, sample_input: torch.Tensor):
        """Log model architecture graph to TensorBoard."""
        if self.tb_writer:
            try:
                self.tb_writer.add_graph(model, sample_input)
            except Exception as e:
                self.warning(f"Could not log model graph: {e}")

    def close(self):
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
