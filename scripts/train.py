#!/usr/bin/env python3
"""
scripts/train.py
=================
Main training entry point for Brain Tumor Segmentation.

Usage:
    # Basic training
    python scripts/train.py

    # With custom config
    python scripts/train.py --config configs/base_config.yaml

    # Override specific params
    python scripts/train.py --epochs 50 --batch_size 16 --model UNet2D

    # Resume from checkpoint
    python scripts/train.py --resume ./runs/exp_001/checkpoints/best.pt

    # Dry run (smoke test with tiny dataset)
    python scripts/train.py --dry_run

On Colab:
    !python scripts/train.py --config configs/base_config.yaml
"""

import os
import sys
import argparse
import random
import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Path Setup ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (
    BraTS2DDataset,
    scan_brats_patients,
    split_patients,
    get_train_transforms,
    get_val_transforms,
)
from src.models import build_model
from src.trainer import Trainer
from src.utils.checkpoint import CheckpointManager


# ─── Config Loader ─────────────────────────────────────────

def load_config(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        raise ImportError("PyYAML not installed. Run: pip install pyyaml")


def merge_args_into_config(cfg: dict, args: argparse.Namespace) -> dict:
    """Allow CLI args to override config values."""
    tcfg = cfg["training"]
    if args.epochs     is not None: tcfg["epochs"]     = args.epochs
    if args.batch_size is not None: tcfg["batch_size"] = args.batch_size
    if args.lr         is not None: tcfg["lr"]         = args.lr
    if args.model      is not None: cfg["model"]["name"] = args.model
    if args.amp is not None:        tcfg["amp"]        = args.amp
    return cfg


# ─── Reproducibility ───────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Dry Run Dataset ───────────────────────────────────────

class TinyFakeDataset(torch.utils.data.Dataset):
    """
    Fake dataset for smoke-testing without actual BraTS data.
    Generates random 4-channel 240×240 images with random labels.
    """
    def __init__(self, n_samples: int = 64, num_classes: int = 4):
        self.n    = n_samples
        self.nc   = num_classes

    def __len__(self): return self.n

    def __getitem__(self, idx):
        img   = torch.randn(4, 240, 240)
        label = torch.randint(0, self.nc, (240, 240))
        return {
            "image":      img,
            "label":      label,
            "patient_id": f"fake_{idx:04d}",
            "slice_idx":  idx,
            "has_tumor":  True,
        }


# ─── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Brain Tumor Segmentation — Training Script"
    )
    parser.add_argument("--config",     type=str, default="configs/base_config.yaml")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--model",      type=str, default=None,
                        choices=["UNet2D", "AttentionUNet2D"])
    parser.add_argument("--amp",        action="store_true", default=None)
    parser.add_argument("--no_amp",     action="store_false", dest="amp")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--run_name",   type=str, default=None,
                        help="Custom run name (default: timestamp)")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Smoke test with fake data — no real dataset needed")
    parser.add_argument("--data_root",  type=str, default=None,
                        help="Override data root dir")
    args = parser.parse_args()

    # ── Load Config ──────────────────────────────────────────
    cfg_path = ROOT / args.config
    print(f"\n Loading config: {cfg_path}")
    cfg = load_config(str(cfg_path))
    cfg = merge_args_into_config(cfg, args)

    if args.data_root:
        cfg["data"]["root_dir"] = args.data_root

    # ── Setup ────────────────────────────────────────────────
    seed   = cfg["project"].get("seed", 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Run Directory → Google Drive ─────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"{cfg['model']['name']}_{timestamp}"

    if args.dry_run:
        # Dry run: lưu local (Drive sẽ chậm với fake data)
        run_dir = ROOT / "runs" / run_name
    else:
        # Real training: lưu thẳng vào Drive
        output_base = cfg.get("output", {}).get("base_dir", None)
        if output_base:
            run_dir = Path(output_base) / "runs" / run_name
        else:
            # Fallback: lưu local (nếu không cấu hình Drive)
            run_dir = ROOT / "runs" / run_name
            print("  ⚠️  output.base_dir không được cấu hình — lưu local (sẽ mất khi Colab disconnect!)")
            print("     Thêm vào config: output.base_dir: /content/drive/MyDrive/BraTM/outputs")

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f" Run directory: {run_dir}")

    # ── Datasets ─────────────────────────────────────────────
    tcfg  = cfg["training"]
    dcfg  = cfg["data"]
    img_size = tuple(dcfg["image_size"])

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE — using fake data")
        train_ds = TinyFakeDataset(n_samples=128, num_classes=dcfg["num_classes"])
        val_ds   = TinyFakeDataset(n_samples=32,  num_classes=dcfg["num_classes"])
    else:
        print(f"\n Scanning dataset: {dcfg['root_dir']}")
        patients = scan_brats_patients(dcfg["root_dir"], dcfg["modalities"])
        print(f"  Found {len(patients)} patients")

        if len(patients) == 0:
            print("\n❌ No patients found! Check data.root_dir in config.")
            print("   Expected structure: <root>/<patient_id>/<patient_id>_t1.nii.gz etc.")
            print("   TIP: Use --dry_run to test the pipeline without data.")
            sys.exit(1)

        # Load or create splits
        # Ưu tiên: Drive → local → tạo mới
        output_base  = cfg.get("output", {}).get("base_dir", None)
        drive_splits = Path(output_base) / "splits.json" if output_base else None
        local_splits = ROOT / "data" / "splits.json"

        splits_path = None
        if drive_splits and drive_splits.exists():
            splits_path = drive_splits
            print(f"  Loaded splits from Drive: {splits_path}")
        elif local_splits.exists():
            splits_path = local_splits
            print(f"  Loaded splits from local: {splits_path}")

        if splits_path:
            from src.data.dataset import load_splits
            splits = load_splits(str(splits_path))
            pid_map = {p["id"]: p for p in patients}
            train_p = [pid_map[pid] for pid in splits["train"] if pid in pid_map]
            val_p   = [pid_map[pid] for pid in splits["val"]   if pid in pid_map]
            test_p  = [pid_map[pid] for pid in splits["test"]  if pid in pid_map]
            print(f"  Splits: {len(train_p)} train / {len(val_p)} val / {len(test_p)} test")
        else:
            train_p, val_p, test_p = split_patients(
                patients,
                train_ratio=dcfg["train_split"],
                val_ratio=dcfg["val_split"],
                seed=seed,
            )
            from src.data.dataset import save_splits
            split_data = {
                "train": [p["id"] for p in train_p],
                "val":   [p["id"] for p in val_p],
                "test":  [p["id"] for p in test_p],
            }
            # Lưu lên Drive (ưu tiên) + local backup
            if drive_splits:
                drive_splits.parent.mkdir(parents=True, exist_ok=True)
                save_splits(split_data, str(drive_splits))
            save_splits(split_data, str(local_splits))
            print(f"  New splits: {len(train_p)} train / {len(val_p)} val / {len(test_p)} test")

        train_transforms = get_train_transforms(img_size)
        val_transforms   = get_val_transforms(img_size)

        print("\n Building train dataset...")
        train_ds = BraTS2DDataset(
            patient_list=train_p,
            modalities=dcfg["modalities"],
            slice_axis=dcfg["slice_axis"],
            transform=train_transforms,
            skip_empty_ratio=dcfg["skip_empty_ratio"],
            min_tumor_pixels=dcfg["min_tumor_pixels"],
        )

        print("\n Building val dataset...")
        val_ds = BraTS2DDataset(
            patient_list=val_p,
            modalities=dcfg["modalities"],
            slice_axis=dcfg["slice_axis"],
            transform=val_transforms,
            skip_empty_ratio=0.0,    # Keep all slices for validation
            min_tumor_pixels=0,
        )

    # ── DataLoaders ──────────────────────────────────────────
    bs = tcfg["batch_size"]
    nw = dcfg.get("num_workers", 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device == "cuda"),
    )
    print(f"\n DataLoaders ready:")
    print(f"  Train: {len(train_ds)} samples → {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} samples   → {len(val_loader)} batches")

    # ── Model ────────────────────────────────────────────────
    print(f"\n Building model: {cfg['model']['name']}")
    model = build_model(cfg["model"])

    # ── Trainer ──────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        run_dir=str(run_dir),
        device=device,
    )

    # ── Resume ───────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.resume_from_checkpoint(args.resume)

    # ── Train ────────────────────────────────────────────────
    print(f"\n🚀 Starting training...\n")
    history = trainer.fit(start_epoch=start_epoch)

    print(f"\n Training complete!")
    print(f" Best val dice: {trainer.ckpt_manager.best_value:.4f}")
    print(f" Best checkpoint: {trainer.ckpt_manager.best_path}")
    print(f" TensorBoard: tensorboard --logdir {run_dir / 'logs' / 'tensorboard'}")


if __name__ == "__main__":
    main()
