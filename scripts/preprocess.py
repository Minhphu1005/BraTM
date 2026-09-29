#!/usr/bin/env python3
"""
scripts/preprocess.py
======================
Optional preprocessing script.

For BraTS 2020/2021, data is already:
  ✅ Co-registered to T1ce
  ✅ Skull-stripped
  ✅ Resampled to 1mm³ isotropic

This script handles:
  1. Dataset validation (check all files exist)
  2. Statistics computation (mean/std per modality for logging)
  3. Label distribution analysis

Usage:
    # Tự đọc data_root từ configs/base_config.yaml
    python scripts/preprocess.py --validate
    python scripts/preprocess.py --validate --label_dist

    # Hoặc truyền tay
    python scripts/preprocess.py --data_root /path/to/BraTS --validate
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import scan_brats_patients, load_nifti


def validate_dataset(patients: list, modalities: list) -> bool:
    """Check all expected files exist and are readable."""
    print(f"\n Validating {len(patients)} patients...")
    errors = []

    for p in tqdm(patients, desc="  Checking", ncols=80):
        for mod in modalities:
            path = Path(p[mod])
            if not path.exists():
                errors.append(f"MISSING: {path}")
                continue
            try:
                arr = load_nifti(str(path))
                if arr.ndim != 3:
                    errors.append(f"WRONG_SHAPE: {path} → {arr.shape}")
            except Exception as e:
                errors.append(f"READ_ERROR: {path} → {e}")

        seg_path = Path(p["seg"])
        if not seg_path.exists():
            errors.append(f"MISSING SEG: {seg_path}")
        else:
            seg = load_nifti(str(seg_path))
            unique_labels = set(np.unique(seg).astype(int).tolist())
            valid_labels  = {0, 1, 2, 4}
            extra = unique_labels - valid_labels
            if extra:
                errors.append(f"UNEXPECTED LABELS in {seg_path}: {extra}")

    if errors:
        print(f"\n❌ Found {len(errors)} error(s):")
        for e in errors[:20]:
            print(f"  {e}")
        return False
    else:
        print(f"\n✅ All {len(patients)} patients validated successfully!")
        return True


def compute_dataset_stats(patients: list, modalities: list):
    """
    Compute per-modality mean and std across all patients.
    Useful for understanding data distribution.
    """
    print(f"\n Computing dataset statistics...")
    stats = {mod: {"means": [], "stds": [], "mins": [], "maxs": []} for mod in modalities}

    for p in tqdm(patients, desc="  Processing", ncols=80):
        for mod in modalities:
            vol = load_nifti(p[mod])
            mask = vol > 0   # Brain region only
            if mask.sum() == 0:
                continue
            stats[mod]["means"].append(float(vol[mask].mean()))
            stats[mod]["stds"].append(float(vol[mask].std()))
            stats[mod]["mins"].append(float(vol[mask].min()))
            stats[mod]["maxs"].append(float(vol[mask].max()))

    print(f"\n{'='*55}")
    print(f"  DATASET STATISTICS (brain region, non-zero voxels)")
    print(f"{'='*55}")
    print(f"  {'Modality':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*44}")
    for mod, s in stats.items():
        if s["means"]:
            print(f"  {mod:<10} "
                  f"{np.mean(s['means']):>8.2f} "
                  f"{np.mean(s['stds']):>8.2f} "
                  f"{np.mean(s['mins']):>8.2f} "
                  f"{np.mean(s['maxs']):>8.2f}")
    return stats


def count_label_distribution(patients: list):
    """Count voxels per class across all segmentation files."""
    print(f"\n Computing label distribution...")
    counts = {0: 0, 1: 0, 2: 0, 4: 0}

    for p in tqdm(patients, desc="  Processing", ncols=80):
        seg = load_nifti(p["seg"]).astype(int)
        for label in [0, 1, 2, 4]:
            counts[label] += int((seg == label).sum())

    total = sum(counts.values())
    label_names = {0: "Background", 1: "NCR/NET", 2: "Edema", 4: "Enhancing"}
    print(f"\n{'='*55}")
    print(f"  LABEL DISTRIBUTION")
    print(f"{'='*55}")
    print(f"  {'Class':<20} {'Voxels':>12} {'Percent':>8}")
    print(f"  {'-'*42}")
    for label, name in label_names.items():
        c = counts.get(label, 0)
        pct = 100 * c / total if total > 0 else 0
        print(f"  {name:<20} {c:>12,} {pct:>7.2f}%")
    print(f"\n  ℹ️  Class imbalance confirms need for Dice Loss!")


def load_config(config_path: str) -> dict:
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="BraTS Dataset Preprocessing & Analysis")
    parser.add_argument("--data_root",  type=str, default=None,
                        help="Path to BraTS data root. If not set, reads from configs/base_config.yaml")
    parser.add_argument("--config",     type=str, default="configs/base_config.yaml",
                        help="Config file to read data_root from (fallback)")
    parser.add_argument("--modalities", type=str, nargs="+",
                        default=["t1", "t1ce", "t2", "flair"])
    parser.add_argument("--validate",   action="store_true",
                        help="Validate all patient files exist and are readable")
    parser.add_argument("--stats",      action="store_true",
                        help="Compute per-modality intensity statistics")
    parser.add_argument("--label_dist", action="store_true",
                        help="Compute voxel-level class distribution")
    args = parser.parse_args()

    # ── Resolve data_root ────────────────────────────────────
    data_root = args.data_root
    if data_root is None:
        # Try to load from config
        cfg_path = ROOT / args.config
        cfg = load_config(str(cfg_path))
        data_root = cfg.get("data", {}).get("root_dir", None)
        if data_root:
            print(f"  data_root loaded from config: {data_root}")
        else:
            print("❌ data_root not provided and not found in config.")
            print("   Pass it explicitly: --data_root /path/to/BraTS/data")
            sys.exit(1)

    print(f"\n Scanning: {data_root}")
    patients = scan_brats_patients(data_root, args.modalities)
    print(f"  Found {len(patients)} patients")

    if not patients:
        print("\n❌ No patients found! Check directory structure.")
        print(f"   Expected folders like: BraTS20_Training_001/ or BraTS2021_00001/")
        sys.exit(1)

    if args.validate or (not args.stats and not args.label_dist):
        validate_dataset(patients, args.modalities)

    if args.stats:
        compute_dataset_stats(patients, args.modalities)

    if args.label_dist:
        count_label_distribution(patients)

    print("\n✅ Preprocessing analysis complete!")


if __name__ == "__main__":
    main()
