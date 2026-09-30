"""
trainer/trainer.py
===================
Main training engine for Brain Tumor Segmentation.

Handles:
  - Training loop with gradient accumulation
  - Validation with full metrics
  - Mixed precision (AMP) for T4 GPU efficiency
  - Checkpoint saving and early stopping
  - TensorBoard logging (images + scalars)
  - Progress bars via tqdm
"""

import os
import time
import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast      # ← new API (PyTorch 2.x)
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.losses import DiceFocalLoss, build_loss
from src.metrics import dice_score, MetricTracker
from src.utils import ExperimentLogger, CheckpointManager, EarlyStopping
from src.utils.visualization import visualize_batch, plot_training_curves


class Trainer:
    """
    Full training loop for 2D brain tumor segmentation.

    Args:
        model:        PyTorch model (UNet2D or AttentionUNet2D)
        train_loader: DataLoader for training set
        val_loader:   DataLoader for validation set
        cfg:          Full config dict (see base_config.yaml)
        run_dir:      Directory for this run's outputs
        device:       'cuda' or 'cpu'
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        run_dir: str = "./runs/exp_001",
        device: str = "cuda",
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.run_dir      = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        tcfg = cfg["training"]
        self.epochs        = tcfg["epochs"]
        self.accum_steps   = tcfg.get("accumulate_grad_batches", 1)
        self.num_classes   = cfg["model"]["num_classes"]
        self.use_amp       = tcfg.get("amp", True) and device == "cuda"
        self.log_every     = cfg.get("logging", {}).get("log_every_n_steps", 10)
        self.n_vis_samples = cfg.get("logging", {}).get("num_vis_samples", 4)
        self.save_vis      = cfg.get("logging", {}).get("save_prediction_images", True)

        # ── Loss ─────────────────────────────────────────────
        self.criterion = build_loss(tcfg["loss"])

        # ── Optimizer ────────────────────────────────────────
        self.optimizer = self._build_optimizer(tcfg)

        # ── Scheduler ────────────────────────────────────────
        self.scheduler = self._build_scheduler(tcfg)

        # ── AMP Scaler ───────────────────────────────────────
        self.scaler = GradScaler("cuda", enabled=self.use_amp)  # new API

        # ── Logger ───────────────────────────────────────────
        self.logger = ExperimentLogger(cfg, run_dir=str(self.run_dir / "logs"))

        # ── Checkpoint Manager ───────────────────────────────
        ckpt_cfg = cfg["checkpoint"]
        self.ckpt_manager = CheckpointManager(
            save_dir=str(self.run_dir / "checkpoints"),
            save_top_k=ckpt_cfg.get("save_top_k", 3),
            monitor=ckpt_cfg.get("monitor", "val_dice_mean"),
            mode=ckpt_cfg.get("mode", "max"),
        )

        # ── Early Stopping ───────────────────────────────────
        es_cfg = tcfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 20),
            monitor=es_cfg.get("monitor", "val_dice_mean"),
            mode=es_cfg.get("mode", "max"),
        ) if es_cfg.get("enabled", True) else None

        # ── History ──────────────────────────────────────────
        self.history: Dict[str, list] = {}
        self.global_step = 0
        self.best_val_dice = 0.0

        # Print summary
        self.logger.info("=" * 60)
        self.logger.info(f"  Run directory: {self.run_dir}")
        self.logger.info(f"  Device:        {device}")
        self.logger.info(f"  AMP:           {self.use_amp}")
        self.logger.info(f"  Epochs:        {self.epochs}")
        self.logger.info(f"  Accum steps:   {self.accum_steps}")
        self.logger.info(f"  Train batches: {len(train_loader)}")
        self.logger.info(f"  Val batches:   {len(val_loader)}")
        self.logger.info("=" * 60)

    # ── Optimizer & Scheduler Builders ────────────────────────

    def _build_optimizer(self, tcfg: dict):
        name = tcfg.get("optimizer", "AdamW")
        lr   = tcfg["lr"]
        wd   = tcfg.get("weight_decay", 1e-5)
        if name == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif name == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif name == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,
                                   weight_decay=wd, nesterov=True)
        raise ValueError(f"Unknown optimizer: {name}")

    def _build_scheduler(self, tcfg: dict):
        name   = tcfg.get("scheduler", "CosineAnnealingWarmRestarts")
        params = tcfg.get("scheduler_params", {})
        if name == "CosineAnnealingWarmRestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=params.get("T_0", 30),
                T_mult=params.get("T_mult", 2),
                eta_min=params.get("eta_min", 1e-6),
            )
        elif name == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=params.get("eta_min", 1e-6),
            )
        elif name == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=10, factor=0.5
            )
        return None

    # ── Resume from Checkpoint ────────────────────────────────

    def resume_from_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model, optimizer, and scheduler state từ checkpoint.

        Returns:
            start_epoch (int): epoch tiếp theo cần train
                               (ví dụ: checkpoint epoch=50 → trả về 50 → train từ epoch 51)
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint không tồn tại: {checkpoint_path}\n"
                f"Kiểm tra lại đường dẫn hoặc Drive đã mount chưa."
            )

        self.logger.info(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(str(path), map_location=self.device)

        # ── Load model weights ────────────────────────────────
        self.model.load_state_dict(ckpt["model_state"])
        self.logger.info(f"  ✅ Model weights loaded")

        # ── Load optimizer state ──────────────────────────────
        if "optim_state" in ckpt and ckpt["optim_state"] is not None:
            self.optimizer.load_state_dict(ckpt["optim_state"])
            self.logger.info(f"  ✅ Optimizer state loaded")

        # ── Load scheduler state ──────────────────────────────
        if self.scheduler and "sched_state" in ckpt and ckpt["sched_state"] is not None:
            self.scheduler.load_state_dict(ckpt["sched_state"])
            self.logger.info(f"  ✅ Scheduler state loaded")

        # ── Restore best metric for checkpoint manager ────────
        if "metrics" in ckpt:
            val_dice = ckpt["metrics"].get("val_dice_mean", None)
            if val_dice is not None:
                self.ckpt_manager.best_value = val_dice
                self.logger.info(f"  ✅ Best val_dice restored: {val_dice:.4f}")

        start_epoch = ckpt.get("epoch", 0)
        self.logger.info(
            f"  📍 Resuming from epoch {start_epoch + 1}/{self.epochs}"
        )
        return start_epoch

    # ── Training Loop ─────────────────────────────────────────

    def fit(self, start_epoch: int = 0) -> Dict[str, list]:
        """
        Run full training loop.

        Args:
            start_epoch: Resume from this epoch (for checkpoint resume)
        Returns:
            Training history dict
        """
        self.logger.info(f"Starting training from epoch {start_epoch + 1}")
        start_time = time.time()

        for epoch in range(start_epoch, self.epochs):
            epoch_num = epoch + 1
            self.logger.info(f"\n{'─'*50}")
            self.logger.info(f"Epoch {epoch_num}/{self.epochs}  |  "
                             f"LR: {self._get_lr():.2e}")

            # ── Train ────────────────────────────────────────
            train_metrics = self._train_epoch(epoch_num)
            self._update_history(train_metrics, prefix="train")

            # ── Validate ─────────────────────────────────────
            val_metrics = self._val_epoch(epoch_num)
            self._update_history(val_metrics, prefix="val")

            # ── Scheduler step ───────────────────────────────
            self._step_scheduler(val_metrics)

            # ── Log epoch summary ────────────────────────────
            self._log_epoch(epoch_num, train_metrics, val_metrics)

            # ── Save checkpoint ──────────────────────────────
            # Prefix val_metrics with 'val_' so monitor key 'val_dice_mean' is found
            val_metrics_prefixed = {f"val_{k}": v for k, v in val_metrics.items()}
            self.ckpt_manager.save(
                epoch=epoch_num,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=val_metrics_prefixed,
                extra={"train_metrics": train_metrics},
            )

            # ── Early stopping ───────────────────────────────
            if self.early_stopping and self.early_stopping.step(val_metrics_prefixed):
                self.logger.info(
                    f"\nEarly stopping triggered at epoch {epoch_num} "
                    f"(patience={self.early_stopping.patience})"
                )
                break

        # ── Final Summary ─────────────────────────────────────
        elapsed = time.time() - start_time
        self.logger.info(f"\nTraining complete in {str(datetime.timedelta(seconds=int(elapsed)))}")
        self.logger.info(f"Best val dice: {self.ckpt_manager.best_value:.4f}")
        self.logger.info(f"Best checkpoint: {self.ckpt_manager.best_path}")

        # Plot and save training curves
        fig = plot_training_curves(self.history,
                                   save_path=str(self.run_dir / "training_curves.png"))

        self.logger.close()
        return self.history

    # ── Train One Epoch ───────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        tracker = MetricTracker()
        total_loss = 0.0
        n_batches  = len(self.train_loader)

        pbar = tqdm(
            enumerate(self.train_loader),
            total=n_batches,
            desc=f"  Train",
            leave=False,
            ncols=100,
        )
        self.optimizer.zero_grad()

        for batch_idx, batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)   # (B, 4, H, W)
            labels = batch["label"].to(self.device, non_blocking=True)   # (B, H, W)

            # Forward pass with AMP
            with autocast("cuda", enabled=self.use_amp):  # new API
                logits = self.model(images)                               # (B, C, H, W)
                loss, loss_dict = self.criterion(logits, labels)
                loss = loss / self.accum_steps

            # Backward
            self.scaler.scale(loss).backward()

            # Optimizer step every `accum_steps`
            if (batch_idx + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Metrics (detach to save memory)
            with torch.no_grad():
                batch_metrics = dice_score(
                    logits.detach(), labels,
                    num_classes=self.num_classes,
                    include_background=False,
                )
                batch_metrics["loss"] = loss_dict["total"]
                batch_metrics["loss_dice"]  = loss_dict["dice"]
                batch_metrics["loss_focal"] = loss_dict["focal"]

            total_loss += loss_dict["total"]
            tracker.update(batch_metrics, n=images.size(0))

            # Log to TensorBoard every N steps
            if self.global_step % self.log_every == 0:
                self.logger.log_scalars(batch_metrics, self.global_step, phase="train_step")

            pbar.set_postfix({
                "loss": f"{loss_dict['total']:.4f}",
                "dice": f"{batch_metrics.get('dice_mean', 0):.4f}",
            })
            self.global_step += 1

        return tracker.compute()

    # ── Validation One Epoch ──────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()

        vis_images, vis_gts, vis_preds = [], [], []

        pbar = tqdm(
            self.val_loader,
            desc=f"  Val  ",
            leave=False,
            ncols=100,
        )

        for i, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):  # new API
                logits = self.model(images)
                loss, loss_dict = self.criterion(logits, labels)

            # Dice metrics
            batch_metrics = dice_score(
                logits, labels,
                num_classes=self.num_classes,
                include_background=False,
            )
            batch_metrics["loss"]       = loss_dict["total"]
            batch_metrics["loss_dice"]  = loss_dict["dice"]
            batch_metrics["loss_focal"] = loss_dict["focal"]
            tracker.update(batch_metrics, n=images.size(0))

            # Collect samples for visualization
            if i == 0 and self.save_vis:
                preds = logits.argmax(dim=1)  # (B, H, W)
                n = min(self.n_vis_samples, images.size(0))
                vis_images = images[:n].cpu()
                vis_gts    = labels[:n].cpu()
                vis_preds  = preds[:n].cpu()

            pbar.set_postfix({"dice": f"{batch_metrics.get('dice_mean', 0):.4f}"})

        # Save prediction images to TensorBoard
        if self.save_vis and len(vis_images):
            self._log_prediction_images(
                vis_images, vis_gts, vis_preds, epoch
            )

        return tracker.compute()

    # ── Helpers ───────────────────────────────────────────────

    def _log_prediction_images(
        self,
        images: torch.Tensor,
        gts: torch.Tensor,
        preds: torch.Tensor,
        epoch: int,
    ):
        """Save visual predictions to TensorBoard and disk."""
        vis_dir = self.run_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        save_path = str(vis_dir / f"epoch{epoch:03d}.png")

        fig = visualize_batch(images, gts, preds,
                              n_samples=self.n_vis_samples,
                              save_path=save_path)

        # Log raw prediction mask to TensorBoard
        if self.logger.tb_writer:
            # Stack for TensorBoard: normalize pred as heatmap
            pred_norm = preds.float().unsqueeze(1) / max(self.num_classes - 1, 1)
            gt_norm   = gts.float().unsqueeze(1)   / max(self.num_classes - 1, 1)
            t1ce      = images[:, 1:2]  # T1ce channel
            t1ce_norm = (t1ce - t1ce.min()) / (t1ce.max() - t1ce.min() + 1e-8)

            self.logger.log_images(
                {"t1ce": t1ce_norm, "gt": gt_norm, "pred": pred_norm},
                step=epoch,
                phase="val",
            )

        import matplotlib.pyplot as plt
        plt.close(fig)

    def _log_epoch(
        self,
        epoch: int,
        train_m: Dict[str, float],
        val_m: Dict[str, float],
    ):
        """Print and log epoch-level metrics."""
        combined = {f"train_{k}": v for k, v in train_m.items()}
        combined.update({f"val_{k}": v for k, v in val_m.items()})

        self.logger.log_scalars(combined, step=epoch, phase="epoch")
        self.logger.log_scalars({"lr": self._get_lr()}, step=epoch, phase="epoch")

        self.logger.info(
            f"  [Train] loss={train_m.get('loss', 0):.4f}  "
            f"dice={train_m.get('dice_mean', 0):.4f}  "
            f"ncr={train_m.get('dice_cls1', 0):.4f}  "
            f"ed={train_m.get('dice_cls2', 0):.4f}  "
            f"et={train_m.get('dice_cls3', 0):.4f}"
        )
        self.logger.info(
            f"  [Val]   loss={val_m.get('loss', 0):.4f}  "
            f"dice={val_m.get('dice_mean', 0):.4f}  "
            f"ncr={val_m.get('dice_cls1', 0):.4f}  "
            f"ed={val_m.get('dice_cls2', 0):.4f}  "
            f"et={val_m.get('dice_cls3', 0):.4f}"
        )

        if val_m.get("dice_mean", 0) > self.best_val_dice:
            self.best_val_dice = val_m["dice_mean"]
            self.logger.info(f"  ★ New best val_dice: {self.best_val_dice:.4f}")

    def _step_scheduler(self, val_metrics: dict):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_metrics.get("val_dice_mean", 0))
        else:
            self.scheduler.step()

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _update_history(self, metrics: Dict[str, float], prefix: str):
        for k, v in metrics.items():
            key = f"{prefix}_{k}"
            self.history.setdefault(key, []).append(v)

    # ── Resume from Checkpoint ────────────────────────────────

    def resume_from_checkpoint(self, ckpt_path: str):
        """Load model + optimizer + scheduler state from checkpoint."""
        self.logger.info(f"Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        if self.scheduler and ckpt.get("sched_state"):
            self.scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = ckpt["epoch"]
        self.logger.info(f"  Resumed from epoch {start_epoch}")
        return start_epoch
