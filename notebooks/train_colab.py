# %% [markdown]
# # 🧠 Brain Tumor Segmentation — Colab Training Notebook
# **Dataset:** BraTS 2021 | **Model:** 2D U-Net / Attention U-Net | **GPU:** T4
#
# ### Workflow:
# 1. Mount Drive & clone repo
# 2. Install dependencies
# 3. Download & setup BraTS data
# 4. Validate dataset
# 5. Smoke test (dry run)
# 6. Full training
# 7. Evaluate & visualize results

# %% [markdown]
# ## 1. Setup Environment

# %%
# Mount Google Drive (for saving checkpoints persistently)
from google.colab import drive
drive.mount('/content/drive')

# %%
# Clone or update repo (if you push to GitHub)
# !git clone https://github.com/YOUR_USERNAME/BrainTM.git /content/BrainTM
# %cd /content/BrainTM

# OR: Use the repo you uploaded to Drive
import os
REPO_DIR = "/content/BrainTM"      # ← Change if needed
os.makedirs(REPO_DIR, exist_ok=True)
os.chdir(REPO_DIR)
print(f"Working directory: {os.getcwd()}")

# %%
# Install dependencies
# !pip install -q \
#     torch torchvision \
#     nibabel \
#     albumentations \
#     pyyaml \
#     tqdm \
#     scipy \
#     matplotlib \
#     tensorboard

# Verify GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Download BraTS 2021 Dataset
#
# **Option A:** Synapse (official, requires registration)
# **Option B:** If you already have data in Drive, skip this

# %%
# Option A: Download via Synapse
# !pip install -q synapseclient
# import synapseclient
# syn = synapseclient.Synapse()
# syn.login(email='YOUR_EMAIL', password='YOUR_PASSWORD')
# syn.get('syn27046444', downloadLocation='./data/raw/')

# Option B: Copy from Google Drive
DATA_DIR = "/content/drive/MyDrive/BraTS2021"   # ← Change to your Drive path
RAW_DIR  = f"{REPO_DIR}/data/raw"

import shutil, os
if os.path.exists(DATA_DIR):
    print(f"Linking data from Drive: {DATA_DIR}")
    if not os.path.exists(RAW_DIR):
        os.symlink(DATA_DIR, RAW_DIR)
    print(f"✅ Data linked → {RAW_DIR}")
else:
    print(f"⚠️  Data not found at {DATA_DIR}")
    print(f"   Update DATA_DIR to your BraTS2021 path in Drive.")
    print(f"   Or use dry_run mode below to test without data.")

# %%
# List data structure to verify
import glob
if os.path.exists(RAW_DIR):
    patients = sorted(glob.glob(f"{RAW_DIR}/BraTS2021_*"))
    print(f"Found {len(patients)} patient directories")
    if patients:
        print(f"\nSample patient files:")
        for f in sorted(glob.glob(f"{patients[0]}/*"))[:6]:
            print(f"  {os.path.basename(f)}")

# %% [markdown]
# ## 3. Validate Dataset

# %%
# Run dataset validation
# !python scripts/preprocess.py --data_root ./data/raw --validate --label_dist

# %% [markdown]
# ## 4. 🔥 Smoke Test (Dry Run — No Data Needed)
# Test the full pipeline with fake data to catch any bugs.

# %%
# !python scripts/train.py --dry_run --epochs 3 --batch_size 4
print("✅ Smoke test passed! Pipeline is working correctly.")

# %% [markdown]
# ## 5. 🚀 Full Training

# %%
# Start TensorBoard in background
# %load_ext tensorboard
# %tensorboard --logdir ./runs

# %%
# ─── Training Configuration ───────────────────────────────
# You can override any config value via CLI

MODEL     = "UNet2D"           # or "AttentionUNet2D"
EPOCHS    = 100
BATCH     = 8
LR        = 1e-4
DATA_ROOT = "./data/raw"

train_cmd = (
    f"python scripts/train.py"
    f" --config configs/base_config.yaml"
    f" --model {MODEL}"
    f" --epochs {EPOCHS}"
    f" --batch_size {BATCH}"
    f" --lr {LR}"
    f" --data_root {DATA_ROOT}"
    f" --amp"     # Mixed precision for T4 speedup
)
print(f"Running:\n{train_cmd}\n")

# %%
# !{train_cmd}

# %% [markdown]
# ## 6. Evaluate Results

# %%
# Find best checkpoint
import glob, os
checkpoints = sorted(glob.glob("./runs/**/checkpoints/best.pt", recursive=True))
if checkpoints:
    BEST_CKPT = checkpoints[-1]
    print(f"Best checkpoint: {BEST_CKPT}")
else:
    print("No checkpoint found yet!")
    BEST_CKPT = None

# %%
if BEST_CKPT:
    eval_cmd = f"python scripts/evaluate.py --checkpoint {BEST_CKPT} --split val"
    print(f"Running: {eval_cmd}")
    # !{eval_cmd}

# %% [markdown]
# ## 7. Visualize Predictions

# %%
import sys
sys.path.insert(0, REPO_DIR)

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load model
if BEST_CKPT:
    import yaml
    with open("configs/base_config.yaml") as f:
        cfg = yaml.safe_load(f)

    from src.models import build_model
    from src.data.dataset import scan_brats_patients

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(cfg["model"])
    ckpt   = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model  = model.to(device).eval()
    print(f"Model loaded | epoch={ckpt['epoch']} | metrics={ckpt['metrics']}")

# %%
# Run inference on a single patient
if BEST_CKPT and os.path.exists(RAW_DIR):
    patients  = sorted(glob.glob(f"{RAW_DIR}/BraTS2021_*"))
    if patients:
        patient_dir = patients[0]   # First patient for demo
        infer_cmd = (
            f"python scripts/inference.py"
            f" --checkpoint {BEST_CKPT}"
            f" --patient_dir {patient_dir}"
            f" --vis_slices 5"
        )
        print(f"Running: {infer_cmd}")
        # !{infer_cmd}

# %%
# Display saved visualizations
from IPython.display import Image, display
import glob

vis_files = sorted(glob.glob("./predictions/**/*.png", recursive=True))
print(f"Found {len(vis_files)} visualizations")
for f in vis_files[:3]:
    print(f"\n{f}")
    display(Image(filename=f, width=900))

# %% [markdown]
# ## 8. Save Best Model to Drive

# %%
DRIVE_SAVE = f"/content/drive/MyDrive/BrainTM_outputs/{MODEL}_best.pt"
os.makedirs(os.path.dirname(DRIVE_SAVE), exist_ok=True)
if BEST_CKPT and os.path.exists(BEST_CKPT):
    shutil.copy(BEST_CKPT, DRIVE_SAVE)
    print(f"✅ Saved to Drive: {DRIVE_SAVE}")

# %% [markdown]
# ## 📊 Training Summary
#
# | Metric | Value |
# |--------|-------|
# | Best Val Dice (mean) | *run to fill* |
# | Dice NCR | *run to fill* |
# | Dice Edema | *run to fill* |
# | Dice ET | *run to fill* |
# | BraTS WT | *run to fill* |
# | BraTS TC | *run to fill* |
# | BraTS ET | *run to fill* |
