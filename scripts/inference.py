#!/usr/bin/env python3
"""
scripts/inference.py
=====================
Run inference on a single patient's MRI volume.

Produces:
  - 2D prediction visualization for each axial slice
  - Optional: 3D NIfTI output file
  - BraTS region Dice (if ground truth available)

Usage:
    # Single patient
    python scripts/inference.py \\
        --checkpoint ./runs/exp_001/checkpoints/best.pt \\
        --patient_dir ./data/raw/BraTS2021_00001

    # With ground truth comparison
    python scripts/inference.py \\
        --checkpoint ./runs/exp_001/checkpoints/best.pt \\
        --patient_dir ./data/raw/BraTS2021_00001 \\
        --save_nifti
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_nifti, normalize_volume, remap_labels
from src.data.transforms import get_val_transforms
from src.models import build_model
from src.metrics import compute_brats_regions
from src.utils.visualization import visualize_prediction


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    patient_dir: Path,
    modalities: list,
    img_size: tuple,
    device: str,
    slice_axis: int = 2,
    tta: bool = False,
) -> np.ndarray:
    """
    Run 2D slice-wise inference on a full 3D volume.

    Returns:
        pred_vol: (H, W, D) predicted label map
    """
    transform = get_val_transforms(img_size)
    pid = patient_dir.name

    # Load all modalities
    vols = []
    for mod in modalities:
        p = patient_dir / f"{pid}_{mod}.nii.gz"
        v = load_nifti(str(p))
        v = normalize_volume(v)
        vols.append(v)
    imgs = np.stack(vols, axis=0)   # (C, H, W, D)

    H, W, D = imgs.shape[1], imgs.shape[2], imgs.shape[3]
    pred_vol = np.zeros((H, W, D), dtype=np.int64)

    for s in tqdm(range(D), desc="  Slices", leave=False):
        img_slice = imgs[:, :, :, s]           # (C, H, W)
        img_hwc   = np.transpose(img_slice, (1, 2, 0))

        aug = transform(image=img_hwc, mask=np.zeros((H, W), dtype=np.int32))
        img_t = torch.from_numpy(
            np.transpose(aug["image"], (2, 0, 1)).copy()
        ).float().unsqueeze(0).to(device)      # (1, C, H, W)

        logits = model(img_t)

        if tta:
            # Horizontal flip TTA
            logits_hf = model(torch.flip(img_t, dims=[3]))
            logits += torch.flip(logits_hf, dims=[3])
            # Vertical flip TTA
            logits_vf = model(torch.flip(img_t, dims=[2]))
            logits += torch.flip(logits_vf, dims=[2])
            logits /= 3

        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()   # (H, W)
        pred_vol[:, :, s] = pred

    return pred_vol


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Segmentation — Inference")
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--patient_dir",  type=str, required=True,
                        help="Path to patient directory (contains *_t1.nii.gz etc.)")
    parser.add_argument("--config",       type=str, default="configs/base_config.yaml")
    parser.add_argument("--output_dir",   type=str, default=None)
    parser.add_argument("--save_nifti",   action="store_true",
                        help="Save prediction as NIfTI file")
    parser.add_argument("--tta",          action="store_true",
                        help="Use Test-Time Augmentation (flip ensemble)")
    parser.add_argument("--vis_slices",   type=int, default=5,
                        help="Number of slices to visualize")
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────
    cfg    = load_config(ROOT / args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Device: {device}")

    patient_dir = Path(args.patient_dir)
    pid         = patient_dir.name

    out_dir = Path(args.output_dir or ROOT / "predictions" / pid)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f" Patient:   {pid}")
    print(f" Output:    {out_dir}")

    # ── Model ────────────────────────────────────────────────
    print(f"\n Loading model from: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    # ── Predict ──────────────────────────────────────────────
    dcfg     = cfg["data"]
    img_size = tuple(dcfg["image_size"])

    print(f"\n Running inference (TTA={args.tta})...")
    pred_vol = predict_volume(
        model=model,
        patient_dir=patient_dir,
        modalities=dcfg["modalities"],
        img_size=img_size,
        device=device,
        slice_axis=dcfg["slice_axis"],
        tta=args.tta,
    )
    print(f"  Prediction shape: {pred_vol.shape}")
    classes, counts = np.unique(pred_vol, return_counts=True)
    print(f"  Class distribution: { {int(c): int(n) for c, n in zip(classes, counts)} }")

    # ── Ground Truth Comparison ──────────────────────────────
    seg_path = patient_dir / f"{pid}_seg.nii.gz"
    gt_vol   = None
    if seg_path.exists():
        gt_vol = remap_labels(load_nifti(str(seg_path)).astype(np.int64))
        region_m = compute_brats_regions(pred_vol, gt_vol)
        print(f"\n BraTS Region Dice:")
        for k, v in sorted(region_m.items()):
            print(f"  {k}: {v:.4f}")

    # ── Load original volume for visualization ───────────────
    vols = []
    for mod in dcfg["modalities"]:
        p = patient_dir / f"{pid}_{mod}.nii.gz"
        v = load_nifti(str(p))
        v = normalize_volume(v)
        vols.append(v)
    imgs_chwd = np.stack(vols, axis=0)  # (C, H, W, D)

    # ── Visualize key slices ──────────────────────────────────
    D = pred_vol.shape[2]
    # Choose slices with most tumor
    tumor_px = [np.sum(pred_vol[:, :, s] > 0) for s in range(D)]
    top_slices = sorted(range(D), key=lambda s: tumor_px[s], reverse=True)
    vis_slices = top_slices[:args.vis_slices]

    print(f"\n Saving visualizations for {len(vis_slices)} slices...")
    for s in vis_slices:
        img_slice  = imgs_chwd[:, :, :, s]    # (C, H, W)
        pred_slice = pred_vol[:, :, s]         # (H, W)
        gt_slice   = gt_vol[:, :, s] if gt_vol is not None else None

        fig = visualize_prediction(
            image=img_slice,
            gt=gt_slice if gt_slice is not None else pred_slice,
            pred=pred_slice,
            modality_idx=1,   # T1ce
            save_path=str(out_dir / f"slice_{s:03d}.png"),
            title=f"{pid} — Axial Slice {s}",
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    # ── Save NIfTI ───────────────────────────────────────────
    if args.save_nifti:
        # Use original affine
        orig_nii   = nib.load(str(patient_dir / f"{pid}_{dcfg['modalities'][0]}.nii.gz"))
        pred_nii   = nib.Nifti1Image(pred_vol.astype(np.int16), orig_nii.affine)
        nifti_path = out_dir / f"{pid}_pred.nii.gz"
        nib.save(pred_nii, str(nifti_path))
        print(f"\n Prediction NIfTI saved → {nifti_path}")

    print(f"\n Done! Visualizations → {out_dir}")


if __name__ == "__main__":
    main()
