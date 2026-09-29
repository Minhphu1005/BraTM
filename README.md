# 🧠 BrainTM — Brain Tumor Segmentation

A complete, production-ready pipeline for **2D Brain Tumor Segmentation** on the BraTS 2021 dataset.

---

## 📁 Project Structure

```
BrainTM/
├── configs/
│   └── base_config.yaml        ← All hyperparameters in one place
│
├── src/
│   ├── data/
│   │   ├── dataset.py          ← BraTS2DDataset, slice indexing, label remapping
│   │   └── transforms.py       ← Albumentations augmentation pipelines
│   │
│   ├── models/
│   │   ├── unet2d.py           ← Classic 2D U-Net
│   │   └── attention_unet2d.py ← Attention U-Net (better for small tumors)
│   │
│   ├── losses/
│   │   └── combined_loss.py    ← DiceLoss + FocalLoss + combined
│   │
│   ├── metrics/
│   │   └── segmentation_metrics.py  ← Dice, BraTS regions (WT/TC/ET), HD95
│   │
│   ├── trainer/
│   │   └── trainer.py          ← Full training loop (AMP, grad accum, logging)
│   │
│   └── utils/
│       ├── logger.py           ← TensorBoard + optional W&B logging
│       ├── checkpoint.py       ← Top-k checkpoint management + early stopping
│       └── visualization.py   ← Dark-themed prediction visualizer
│
├── scripts/
│   ├── train.py                ← 🚀 Main training entry point
│   ├── evaluate.py             ← Full evaluation with per-patient CSV
│   ├── inference.py            ← Single patient inference + NIfTI export
│   └── preprocess.py           ← Dataset validation & statistics
│
├── notebooks/
│   └── train_colab.py          ← Step-by-step Colab training guide
│
├── data/                       ← (gitignored) Place BraTS data here
│   └── raw/
│       └── BraTS2021_XXXXX/
│           ├── BraTS2021_XXXXX_t1.nii.gz
│           ├── BraTS2021_XXXXX_t1ce.nii.gz
│           ├── BraTS2021_XXXXX_t2.nii.gz
│           ├── BraTS2021_XXXXX_flair.nii.gz
│           └── BraTS2021_XXXXX_seg.nii.gz
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Smoke Test (No Data Needed!)
```bash
python scripts/train.py --dry_run --epochs 3
```

### 3. Validate Your Dataset
```bash
python scripts/preprocess.py --data_root ./data/raw --validate --label_dist
```

### 4. Train
```bash
# Default config (UNet2D, 100 epochs)
python scripts/train.py

# Custom settings
python scripts/train.py \
    --model AttentionUNet2D \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --amp

# Resume from checkpoint
python scripts/train.py --resume ./runs/exp_001/checkpoints/best.pt
```

### 5. Monitor Training
```bash
tensorboard --logdir ./runs
```

### 6. Evaluate
```bash
python scripts/evaluate.py \
    --checkpoint ./runs/exp_001/checkpoints/best.pt \
    --split val
```

### 7. Inference on One Patient
```bash
python scripts/inference.py \
    --checkpoint ./runs/exp_001/checkpoints/best.pt \
    --patient_dir ./data/raw/BraTS2021_00001 \
    --save_nifti \
    --tta
```

---

## 🔬 Models

| Model | Parameters | Notes |
|-------|-----------|-------|
| `UNet2D` | ~7.7M | Baseline, fast, solid |
| `AttentionUNet2D` | ~34.9M | Attention gates, better for small ET |

Configure in `configs/base_config.yaml`:
```yaml
model:
  name: "UNet2D"       # or "AttentionUNet2D"
  in_channels: 4       # 4 MRI modalities
  num_classes: 4       # BG + NCR + Edema + ET
```

---

## 📊 BraTS Label Convention

| Label | Class | Color |
|-------|-------|-------|
| 0 | Background | — |
| 1 | NCR/NET (Necrotic core) | 🔴 Red |
| 2 | Edema (peritumoral) | 🟢 Green |
| 3 | Enhancing Tumor | 🟡 Yellow |

> **Note:** BraTS uses labels `{0, 1, 2, 4}`. We remap `4 → 3` internally.

### BraTS Evaluation Regions:
- **WT** (Whole Tumor) = labels {1, 2, 3}
- **TC** (Tumor Core) = labels {1, 3}
- **ET** (Enhancing Tumor) = labels {3}

---

## ⚙️ Configuration

All settings are in [`configs/base_config.yaml`](configs/base_config.yaml).
Key sections:

```yaml
training:
  epochs: 100
  batch_size: 8         # T4 GPU safe
  lr: 1.0e-4
  amp: true             # Mixed precision
  loss:
    name: DiceFocalLoss
    dice_weight: 1.0
    focal_weight: 1.0
```

---

## 🎯 Expected Results (BraTS 2021)

| Model | WT Dice | TC Dice | ET Dice |
|-------|---------|---------|---------|
| U-Net 2D (baseline) | ~0.87 | ~0.78 | ~0.73 |
| Attention U-Net 2D | ~0.89 | ~0.81 | ~0.76 |

> Results vary by training duration and augmentation. These are approximate targets.

---

## 💡 Tips for Colab T4

1. **Use `--amp`** always — 2x speedup, saves VRAM
2. **Batch size 8** is safe; try 12-16 if using `AttentionUNet2D` smaller features
3. **Save checkpoints to Drive** — Colab sessions disconnect!
4. **Use `cache_data=True`** in dataset if RAM allows (speeds up training)
5. **Monitor GPU**: `!nvidia-smi` in Colab
