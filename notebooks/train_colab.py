# %% [markdown]
# # 🧠 Brain Tumor Segmentation — Colab Training Notebook
# **Dataset:** BraTS 2020 | **Model:** 2D U-Net / Attention U-Net | **GPU:** T4
#
# ### Workflow:
# 1. Mount Drive & setup repo
# 2. Install dependencies
# 3. Setup data paths (BraTS2020)
# 4. Validate dataset
# 5. Smoke test (dry run)
# 6. Full training
# 7. Evaluate & visualize results

# %% [markdown]
# ## 1. Setup Environment

# %%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# %%
# ─── Paths — CHỈNH SỬA NẾU CẦN ──────────────────────────────────────────────

# Thư mục chứa repo BrainTM trên Colab
REPO_DIR = "/content/BrainTM"

# BraTS2020 Training data (có nhãn _seg.nii.gz) — dùng để train/val/test
TRAIN_DATA_DIR = "/content/drive/MyDrive/BraTM/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# BraTS2020 Validation data (KHÔNG có nhãn) — chỉ dùng để inference, không dùng để tính Dice
VAL_DATA_DIR   = "/content/drive/MyDrive/BraTM/dataset/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"

# Thư mục lưu checkpoint lên Drive
CHECKPOINT_SAVE_DIR = "/content/drive/MyDrive/BraTM/checkpoints"

# ─────────────────────────────────────────────────────────────────────────────

import os, shutil, glob
os.makedirs(REPO_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_SAVE_DIR, exist_ok=True)
os.chdir(REPO_DIR)
print(f"Working directory : {os.getcwd()}")
print(f"Train data        : {TRAIN_DATA_DIR}")
print(f"Val data (no GT)  : {VAL_DATA_DIR}")

# %%
# Kiểm tra data tồn tại
assert os.path.exists(TRAIN_DATA_DIR), f"❌ Không tìm thấy: {TRAIN_DATA_DIR}"

patients = sorted(glob.glob(f"{TRAIN_DATA_DIR}/BraTS20_*"))
print(f"\n✅ Tìm thấy {len(patients)} bệnh nhân trong training set")
print(f"\nVí dụ cấu trúc file:")
for f in sorted(glob.glob(f"{patients[0]}/*"))[:6]:
    print(f"  {os.path.basename(f)}")

# %% [markdown]
# ---
# ## ℹ️ Lưu Ý Về Tập Test
#
# | Thư mục | Số patients | Có nhãn? | Dùng để |
# |---------|------------|----------|---------|
# | `MICCAI_BraTS2020_TrainingData` | 369 | ✅ Có `_seg.nii.gz` | **Train + Val + Test** |
# | `MICCAI_BraTS2020_ValidationData` | 125 | ❌ Không (ban tổ chức giữ) | Inference cho leaderboard |
#
# **→ Code sẽ tự động chia 369 patients thành:**
# - **80% (295 patients)** → Training
# - **10% (37 patients)** → Validation (monitor trong lúc train)
# - **10% (37 patients)** → Test (evaluate sau khi train xong)
#
# Tỷ lệ chia được lưu vào `data/splits.json` để **tái sử dụng và reproducible**.

# %% [markdown]
# ## 2. Install Dependencies

# %%
!pip install -q nibabel albumentations pyyaml scipy tensorboard

# Verify GPU
import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 3. Link Data vào Repo (Symlink — không copy, tiết kiệm dung lượng)

# %%
import os

RAW_DIR = f"{REPO_DIR}/data/raw"
os.makedirs(f"{REPO_DIR}/data", exist_ok=True)

if not os.path.exists(RAW_DIR):
    os.symlink(TRAIN_DATA_DIR, RAW_DIR)
    print(f"✅ Symlink tạo: {RAW_DIR} → {TRAIN_DATA_DIR}")
else:
    print(f"✅ data/raw đã tồn tại: {RAW_DIR}")

# Verify
test_files = glob.glob(f"{RAW_DIR}/BraTS20_*")
print(f"   Nhìn thấy {len(test_files)} patients qua symlink")

# %% [markdown]
# ## 4. Validate Dataset

# %%
# Kiểm tra tất cả file có đầy đủ không và xem phân bố nhãn
!python scripts/preprocess.py \
    --data_root ./data/raw \
    --validate \
    --label_dist

# %% [markdown]
# ## 5. 🔥 Smoke Test (Không cần data)
# Chạy thử toàn bộ pipeline với data giả để kiểm tra code không có lỗi.

# %%
!python scripts/train.py --dry_run --epochs 2 --batch_size 4
print("\n✅ Smoke test passed! Pipeline hoạt động bình thường.")

# %% [markdown]
# ## 6. 🚀 Full Training

# %%
# Start TensorBoard (mở tab bên cạnh để monitor)
%load_ext tensorboard
%tensorboard --logdir ./runs

# %%
# ─── Cấu hình Training ────────────────────────────────────────────────────────
MODEL     = "UNet2D"          # hoặc "AttentionUNet2D" (chậm hơn nhưng tốt hơn)
EPOCHS    = 100
BATCH     = 8                 # T4 15GB: an toàn với batch=8, thử 12 nếu muốn
LR        = 1e-4
DATA_ROOT = "./data/raw"      # trỏ vào symlink đã tạo ở trên
# ──────────────────────────────────────────────────────────────────────────────

train_cmd = (
    f"python scripts/train.py"
    f" --config configs/base_config.yaml"
    f" --model {MODEL}"
    f" --epochs {EPOCHS}"
    f" --batch_size {BATCH}"
    f" --lr {LR}"
    f" --data_root {DATA_ROOT}"
    f" --amp"
)
print(f"Lệnh sẽ chạy:\n{train_cmd}")

# %%
# Chạy training (bỏ dấu # ở dòng dưới)
!{train_cmd}

# %% [markdown]
# ## 7. Tự Động Lưu Checkpoint Lên Drive
# **Quan trọng**: Colab disconnect sau ~12h, checkpoint cần được lưu lên Drive!

# %%
import glob, shutil

checkpoints = sorted(glob.glob("./runs/**/checkpoints/best.pt", recursive=True))
if checkpoints:
    BEST_CKPT = checkpoints[-1]
    DRIVE_SAVE = f"{CHECKPOINT_SAVE_DIR}/{MODEL}_best.pt"
    shutil.copy(BEST_CKPT, DRIVE_SAVE)
    print(f"✅ Checkpoint đã lưu lên Drive:")
    print(f"   {BEST_CKPT}")
    print(f"   → {DRIVE_SAVE}")
else:
    print("⚠️  Chưa có checkpoint nào. Hãy chạy training trước.")
    BEST_CKPT = None

# %% [markdown]
# ## 8. Evaluate — Tính Dice Score trên Tập Test

# %%
# Lấy checkpoint (từ Drive nếu đã lưu)
if BEST_CKPT is None:
    BEST_CKPT = f"{CHECKPOINT_SAVE_DIR}/{MODEL}_best.pt"

if os.path.exists(BEST_CKPT):
    # Evaluate trên validation set
    !python scripts/evaluate.py \
        --checkpoint {BEST_CKPT} \
        --split val

    print("\n" + "="*50)

    # Evaluate trên test set (10% chưa từng thấy trong training)
    !python scripts/evaluate.py \
        --checkpoint {BEST_CKPT} \
        --split test
else:
    print(f"❌ Không tìm thấy checkpoint: {BEST_CKPT}")

# %% [markdown]
# ## 9. Visualize Predictions

# %%
import sys
sys.path.insert(0, REPO_DIR)

import torch, yaml, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

if BEST_CKPT and os.path.exists(BEST_CKPT):
    with open("configs/base_config.yaml") as f:
        cfg = yaml.safe_load(f)

    from src.models import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(cfg["model"])
    ckpt   = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model  = model.to(device).eval()
    print(f"✅ Model loaded")
    print(f"   Epoch    : {ckpt.get('epoch', '?')}")
    print(f"   Metrics  : {ckpt.get('metrics', {})}")

# %%
# Inference trên 1 patient từ training set
if BEST_CKPT and os.path.exists(BEST_CKPT) and os.path.exists(RAW_DIR):
    sample_patients = sorted(glob.glob(f"{RAW_DIR}/BraTS20_*"))

    if sample_patients:
        patient_dir = sample_patients[10]  # Lấy patient thứ 10 làm demo
        print(f"Demo patient: {os.path.basename(patient_dir)}")

        !python scripts/inference.py \
            --checkpoint {BEST_CKPT} \
            --patient_dir {patient_dir} \
            --vis_slices 5

# %%
# Hiển thị kết quả visual
from IPython.display import Image, display

vis_files = sorted(glob.glob("./predictions/**/*.png", recursive=True))
print(f"Tìm thấy {len(vis_files)} visualization files")
for f in vis_files[:3]:
    print(f"\n📸 {os.path.basename(f)}")
    display(Image(filename=f, width=900))

# %% [markdown]
# ## 📊 Training Summary Template
#
# Sau khi training xong, điền kết quả vào đây:
#
# | Metric | Val Set | Test Set |
# |--------|---------|----------|
# | Dice Mean | | |
# | Dice NCR (class 1) | | |
# | Dice Edema (class 2) | | |
# | Dice ET (class 3) | | |
# | **BraTS WT** | | |
# | **BraTS TC** | | |
# | **BraTS ET** | | |
#
# **Model:** UNet2D | **Epochs:** 100 | **Batch:** 8 | **LR:** 1e-4
