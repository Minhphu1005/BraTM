"""
utils/checkpoint.py
====================
Model checkpointing utilities.

Features:
  - Save best N checkpoints (tracked by monitored metric)
  - Auto-cleanup of old checkpoints
  - Resume training from checkpoint
  - Export best checkpoint for inference
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, List
import torch


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Args:
        save_dir:   Directory to store checkpoints
        save_top_k: Keep only top-k checkpoints by monitored metric
        monitor:    Metric name to monitor (e.g. 'val_dice_mean')
        mode:       'max' or 'min' — direction of the monitored metric
    """

    def __init__(
        self,
        save_dir: str,
        save_top_k: int = 3,
        monitor: str = "val_dice_mean",
        mode: str = "max",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self._checkpoints: List[Dict] = []   # [{path, value, epoch}]
        self.best_path: Optional[str] = None

    def _is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        metrics: Dict[str, float],
        extra: dict = None,
    ) -> Optional[str]:
        """
        Save checkpoint. Tracks best by monitored metric.

        Returns:
            Path to saved checkpoint file.
        """
        value = metrics.get(self.monitor, None)
        if value is None:
            # Monitor key not found — print available keys to help debug
            import warnings
            warnings.warn(
                f"[CheckpointManager] Monitor key '{self.monitor}' not found in metrics.\n"
                f"  Available keys: {list(metrics.keys())}\n"
                f"  Checkpoint will be saved but best tracking is disabled until key is found.",
                stacklevel=2,
            )
            # Save anyway (without best tracking)
            fname = f"epoch{epoch:03d}.pt"
            path  = str(self.save_dir / fname)
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict() if scheduler else None,
                "metrics":     metrics,
                "extra":       extra or {},
            }, path)
            return path

        ckpt = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict() if scheduler else None,
            "metrics":     metrics,
            "extra":       extra or {},
        }

        # Save checkpoint
        fname = f"epoch{epoch:03d}_{self.monitor.replace('/', '_')}{value:.4f}.pt"
        path = str(self.save_dir / fname)
        torch.save(ckpt, path)

        # Track checkpoints sorted by value
        self._checkpoints.append({"path": path, "value": value, "epoch": epoch})
        self._checkpoints.sort(
            key=lambda x: x["value"],
            reverse=(self.mode == "max"),
        )

        # Remove old checkpoints beyond top-k
        while len(self._checkpoints) > self.save_top_k:
            old = self._checkpoints.pop()
            if os.path.exists(old["path"]):
                os.remove(old["path"])

        # Track best
        if self._is_better(value):
            self.best_value = value
            self.best_path = path
            # Copy as "best.pt" for easy inference
            shutil.copy(path, str(self.save_dir / "best.pt"))

        # Save metadata
        self._save_metadata()
        return path

    def _save_metadata(self):
        meta = {
            "best_value": self.best_value,
            "best_path":  self.best_path,
            "monitor":    self.monitor,
            "mode":       self.mode,
            "checkpoints": self._checkpoints,
        }
        with open(self.save_dir / "checkpoint_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str, device: str = "cpu") -> dict:
        """Load checkpoint dict from path."""
        return torch.load(path, map_location=device)

    def load_best(self, device: str = "cpu") -> Optional[dict]:
        """Load the best checkpoint."""
        best = self.save_dir / "best.pt"
        if best.exists():
            return torch.load(str(best), map_location=device)
        return None

    @classmethod
    def restore(cls, save_dir: str, device: str = "cuda") -> dict:
        """
        Restore training state from best checkpoint.

        Returns:
            {'epoch': int, 'model_state': ..., 'optim_state': ..., ...}
        """
        best = Path(save_dir) / "best.pt"
        if not best.exists():
            raise FileNotFoundError(f"No checkpoint found at {best}")
        return torch.load(str(best), map_location=device)


# ─── Early Stopping ────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when monitored metric does not improve for `patience` epochs.

    Args:
        patience: Number of epochs to wait
        monitor:  Metric name
        mode:     'max' or 'min'
        delta:    Minimum change to qualify as improvement
    """

    def __init__(
        self,
        patience: int = 20,
        monitor: str = "val_dice_mean",
        mode: str = "max",
        delta: float = 1e-4,
    ):
        self.patience = patience
        self.monitor  = monitor
        self.mode     = mode
        self.delta    = delta
        self.best     = float("-inf") if mode == "max" else float("inf")
        self.counter  = 0
        self.should_stop = False

    def step(self, metrics: Dict[str, float]) -> bool:
        """
        Call after each validation epoch.

        Returns:
            True if training should stop.
        """
        value = metrics.get(self.monitor, None)
        if value is None:
            return False

        improved = (
            value > self.best + self.delta
            if self.mode == "max"
            else value < self.best - self.delta
        )

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
