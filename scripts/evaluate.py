#!/usr/bin/env python3
"""
scripts/evaluate.py
====================
Evaluation script — runs full validation/test set evaluation.

Computes:
  - Per-class Dice (NCR, Edema, ET)
  - BraTS compound region Dice (WT, TC, ET)
  - Hausdorff Distance 95% (HD95) per region
  - Per-patient results saved to CSV

Usage:
    python scripts/evaluate.py --checkpoint ./runs/exp_001/checkpoints/best.pt
    python scripts/evaluate.py --checkpoint ./runs/exp_001/checkpoints/best.pt --split test
"""

import os
import sys
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (
    BraTS2DDataset,
    scan_brats_patients,
    get_val_transforms,
)
from src.data.dataset import load_splits
from src.models import build_model
from src.metrics import (
    dice_score,
    compute_brats_regions,
    hausdorff_distance_95,
    MetricTracker,
    CLASS_NAMES,
)


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Segmentation — Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--config",     type=str, default="configs/base_config.yaml")
    parser.add_argument("--split",      type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # ── Load Config & Checkpoint ─────────────────────────────
    cfg  = load_config(ROOT / args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    print(f"\n Checkpoint loaded from: {args.checkpoint}")
    print(f"  Trained epoch: {ckpt.get('epoch', '?')}")
    print(f"  Metrics:       {ckpt.get('metrics', {})}")

    # ── Output Dir ───────────────────────────────────────────
    out_dir = Path(args.output_dir or Path(args.checkpoint).parent.parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f" Output directory: {out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Model ────────────────────────────────────────────────
    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f" Model loaded: {cfg['model']['name']}")

    # ── Dataset ──────────────────────────────────────────────
    dcfg = cfg["data"]
    img_size = tuple(dcfg["image_size"])

    patients  = scan_brats_patients(dcfg["root_dir"], dcfg["modalities"])
    splits    = load_splits(str(ROOT / "data" / "splits.json"))
    pid_map   = {p["id"]: p for p in patients}
    split_pids = splits[args.split]
    split_pts  = [pid_map[pid] for pid in split_pids if pid in pid_map]

    print(f" Evaluating on {args.split} split: {len(split_pts)} patients")

    ds = BraTS2DDataset(
        patient_list=split_pts,
        modalities=dcfg["modalities"],
        slice_axis=dcfg["slice_axis"],
        transform=get_val_transforms(img_size),
        skip_empty_ratio=0.0,
        min_tumor_pixels=0,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Evaluate ─────────────────────────────────────────────
    tracker       = MetricTracker()
    per_patient   = {}   # patient_id → accumulated pred/gt

    print(f"\n Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader, ncols=90):
            images  = batch["image"].to(device)
            labels  = batch["label"].to(device)
            pids    = batch["patient_id"]
            s_idxs  = batch["slice_idx"]

            logits = model(images)
            preds  = logits.argmax(dim=1)  # (B, H, W)

            # Dice
            metrics = dice_score(logits, labels,
                                  num_classes=cfg["model"]["num_classes"],
                                  include_background=False)
            tracker.update(metrics, n=images.size(0))

            # Accumulate per-patient slices for BraTS region metrics
            for pid, s_idx, pred_slice, gt_slice in zip(
                pids, s_idxs.tolist(),
                preds.cpu().numpy(), labels.cpu().numpy()
            ):
                if pid not in per_patient:
                    per_patient[pid] = {"preds": {}, "gts": {}}
                per_patient[pid]["preds"][s_idx] = pred_slice
                per_patient[pid]["gts"][s_idx]   = gt_slice

    # ── Global Metrics ───────────────────────────────────────
    global_metrics = tracker.compute()
    print(f"\n{'='*55}")
    print(f" GLOBAL METRICS ({args.split} split)")
    print(f"{'='*55}")
    for k, v in sorted(global_metrics.items()):
        print(f"  {k:30s}: {v:.4f}")

    # ── Per-Patient BraTS Region Metrics ─────────────────────
    print(f"\n Computing BraTS region metrics per patient...")
    patient_rows = []
    region_tracker = MetricTracker()

    for pid, data in tqdm(per_patient.items(), ncols=90):
        if not data["preds"]:
            continue
        # Stack slices into 3D volume
        max_s  = max(data["preds"].keys()) + 1
        h, w   = next(iter(data["preds"].values())).shape
        pred_vol = np.zeros((h, w, max_s), dtype=np.int64)
        gt_vol   = np.zeros((h, w, max_s), dtype=np.int64)
        for s, arr in data["preds"].items():
            pred_vol[:, :, s] = arr
        for s, arr in data["gts"].items():
            gt_vol[:, :, s]   = arr

        region_m = compute_brats_regions(pred_vol, gt_vol)
        region_tracker.update(region_m)

        row = {"patient_id": pid}
        row.update(region_m)

        # HD95 per region
        for region, pred_labels, gt_labels in [
            ("WT", [1,2,3], [1,2,3]),
            ("TC", [1,3],   [1,3]),
            ("ET", [3],     [3]),
        ]:
            p_bin = np.isin(pred_vol, pred_labels)
            g_bin = np.isin(gt_vol,   gt_labels)
            hd95 = hausdorff_distance_95(p_bin, g_bin)
            row[f"hd95_{region}"] = hd95 if hd95 != float("inf") else -1

        patient_rows.append(row)

    # ── BraTS Summary ────────────────────────────────────────
    region_summary = region_tracker.compute()
    print(f"\n{'='*55}")
    print(f" BraTS REGION METRICS")
    print(f"{'='*55}")
    print(f"  {'Metric':<25} {'Value':>8}")
    print(f"  {'-'*33}")
    for k, v in sorted(region_summary.items()):
        print(f"  {k:<25} {v:>8.4f}")

    # ── Save CSV ─────────────────────────────────────────────
    csv_path = out_dir / f"results_{args.split}.csv"
    if patient_rows:
        fieldnames = list(patient_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(patient_rows)
        print(f"\n Per-patient results saved → {csv_path}")

    # ── Save Summary ─────────────────────────────────────────
    summary_path = out_dir / f"summary_{args.split}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Split: {args.split}\n\n")
        f.write("=== Global Dice ===\n")
        for k, v in sorted(global_metrics.items()):
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\n=== BraTS Region Dice ===\n")
        for k, v in sorted(region_summary.items()):
            f.write(f"  {k}: {v:.4f}\n")
    print(f" Summary saved       → {summary_path}")


if __name__ == "__main__":
    main()
