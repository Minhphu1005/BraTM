"""
data/dataset.py
================
BraTS Dataset classes for 2D slice-based segmentation.

Supports BraTS2020 and BraTS2021 (auto-detected from folder names):

  BraTS2020:
    dataset_root/
      BraTS20_Training_001/
        BraTS20_Training_001_t1.nii.gz
        BraTS20_Training_001_t1ce.nii.gz
        BraTS20_Training_001_t2.nii.gz
        BraTS20_Training_001_flair.nii.gz
        BraTS20_Training_001_seg.nii.gz

  BraTS2021:
    dataset_root/
      BraTS2021_00001/
        BraTS2021_00001_t1.nii.gz  (same structure)

Label Convention (identical for both versions):
  Original:  {0, 1, 2, 4}  →  Remapped: {0, 1, 2, 3}
    0 = Background
    1 = NCR/NET (Necrotic core)
    2 = ED  (Peritumoral Edema)
    4 → 3 = ET (GD-Enhancing Tumor)
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# ─── Label Remapping ───────────────────────────────────────
BRATS_LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}  # remap label 4 → 3


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """Remap BraTS original labels {0,1,2,4} → {0,1,2,3}."""
    out = np.zeros_like(seg)
    for orig, new in BRATS_LABEL_MAP.items():
        out[seg == orig] = new
    return out


# ─── NIfTI Loading Utilities ───────────────────────────────
def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI file and return its numpy array (H, W, D)."""
    return nib.load(path).get_fdata(dtype=np.float32)


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Z-score normalization on non-zero voxels (brain region only)."""
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    mean = vol[mask].mean()
    std  = vol[mask].std() + 1e-8
    out = vol.copy()
    out[mask] = (vol[mask] - mean) / std
    return out


# ─── BraTS Version Detection ───────────────────────────────
def detect_brats_version(root_dir: str) -> str:
    """
    Auto-detect BraTS dataset version from folder naming.

    Returns '2020', '2021', or 'unknown'.
    """
    root = Path(root_dir)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("BraTS20_"):    # BraTS20_Training_001
            return "2020"
        if d.name.startswith("BraTS2021_"): # BraTS2021_00001
            return "2021"
    return "unknown"


def _find_modality_file(patient_dir: Path, pid: str, modality: str) -> Optional[Path]:
    """
    Locate a modality NIfTI file using several naming patterns.

    Handles:
      BraTS2021: BraTS2021_00001_t1.nii.gz
      BraTS2020: BraTS20_Training_001_t1.nii.gz
    """
    candidates = [
        patient_dir / f"{pid}_{modality}.nii.gz",
        patient_dir / f"{pid}_{modality}_stripped.nii.gz",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback glob
    matches = list(patient_dir.glob(f"*_{modality}.nii.gz"))
    return matches[0] if matches else None


# ─── Patient Scanner ───────────────────────────────────────
def scan_brats_patients(root_dir: str, modalities: List[str]) -> List[Dict]:
    """
    Walk a BraTS directory and collect patient file paths.

    Auto-detects BraTS2020 ('BraTS20_Training_XXX') and
    BraTS2021 ('BraTS2021_XXXXX') naming conventions.

    Returns:
        List of dicts: {'id': str, 't1': path, 't1ce': path,
                        't2': path, 'flair': path, 'seg': path,
                        'version': '2020'|'2021'}
    """
    root = Path(root_dir)
    patients = []

    version = detect_brats_version(root_dir)
    print(f"  Detected BraTS version : {version}")

    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue

        pid = patient_dir.name
        entry = {"id": pid, "version": version}
        valid = True

        for mod in modalities:
            fpath = _find_modality_file(patient_dir, pid, mod)
            if fpath is None:
                valid = False
                break
            entry[mod] = str(fpath)

        seg_path = _find_modality_file(patient_dir, pid, "seg")
        if seg_path is None:
            valid = False

        if valid:
            entry["seg"] = str(seg_path)
            patients.append(entry)

    return patients


# ─── 2D Slice Dataset ──────────────────────────────────────
class BraTS2DDataset(Dataset):
    """
    Extracts 2D axial slices from BraTS 3D volumes.

    Each item is a (C, H, W) tensor of 4 modalities stacked as channels,
    and a (H, W) long tensor label map.

    Args:
        patient_list:     List of patient dicts from scan_brats_patients()
        modalities:       Which MRI modalities to use (in order)
        slice_axis:       Axis along which to slice (2 = axial)
        transform:        Albumentations transform pipeline
        skip_empty_ratio: Probability to skip a fully-background slice
        min_tumor_pixels: Min # tumor pixels to consider slice as "tumor"
        cache_data:       If True, pre-load all volumes to RAM (faster on Colab)
    """

    def __init__(
        self,
        patient_list: List[Dict],
        modalities: List[str] = ("t1", "t1ce", "t2", "flair"),
        slice_axis: int = 2,
        transform=None,
        skip_empty_ratio: float = 0.90,
        min_tumor_pixels: int = 50,
        cache_data: bool = False,
    ):
        self.modalities = list(modalities)
        self.slice_axis = slice_axis
        self.transform = transform
        self.skip_empty_ratio = skip_empty_ratio
        self.min_tumor_pixels = min_tumor_pixels
        self.cache_data = cache_data

        # Build slice index: list of (patient_dict, slice_idx, has_tumor)
        self.slice_index: List[Tuple[Dict, int, bool]] = []
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        print(f"  Indexing {len(patient_list)} patients...")
        self._build_index(patient_list)
        print(f"  Total slices in dataset: {len(self.slice_index)}")

    def _build_index(self, patient_list: List[Dict]):
        """Pre-scan volumes to build slice index with tumor-aware sampling."""
        for patient in tqdm(patient_list, desc="  Building index", leave=False):
            seg = load_nifti(patient["seg"])
            seg = remap_labels(seg.astype(np.int64))

            n_slices = seg.shape[self.slice_axis]

            for s in range(n_slices):
                sl = np.take(seg, s, axis=self.slice_axis)
                tumor_px = (sl > 0).sum()
                has_tumor = tumor_px >= self.min_tumor_pixels

                self.slice_index.append((patient, s, has_tumor))

            # Cache volume data if requested
            if self.cache_data:
                imgs = self._load_modalities(patient)
                self._cache[patient["id"]] = (imgs, seg)

    def _load_modalities(self, patient: Dict) -> np.ndarray:
        """Load & normalize all modalities, return (C, H, W, D) array."""
        vols = []
        for mod in self.modalities:
            v = load_nifti(patient[mod])
            v = normalize_volume(v)
            vols.append(v)
        return np.stack(vols, axis=0)  # (C, H, W, D)

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient, s_idx, has_tumor = self.slice_index[idx]

        # Optionally skip empty slices at training time
        if not has_tumor and random.random() < self.skip_empty_ratio:
            # Return a random tumor slice instead
            tumor_slices = [i for i, (_, _, ht) in enumerate(self.slice_index) if ht]
            idx = random.choice(tumor_slices)
            patient, s_idx, has_tumor = self.slice_index[idx]

        # Load data (from cache or disk)
        pid = patient["id"]
        if pid in self._cache:
            imgs, seg = self._cache[pid]
        else:
            imgs = self._load_modalities(patient)
            seg_vol = load_nifti(patient["seg"])
            seg = remap_labels(seg_vol.astype(np.int64))

        # Extract 2D slice → (C, H, W)
        image = np.take(imgs, s_idx, axis=-1)   # (C, H, W)
        label = np.take(seg,  s_idx, axis=-1)   # (H, W)

        # HWC format for albumentations
        image_hwc = np.transpose(image, (1, 2, 0))  # (H, W, C)

        if self.transform is not None:
            augmented = self.transform(image=image_hwc, mask=label.astype(np.int32))
            image_hwc = augmented["image"]
            label = augmented["mask"]

        # Back to CHW tensor
        image_t = torch.from_numpy(
            np.transpose(image_hwc, (2, 0, 1)).copy()
        ).float()
        label_t = torch.from_numpy(label.copy()).long()

        return {
            "image":     image_t,   # (4, H, W)
            "label":     label_t,   # (H, W) — class indices
            "patient_id": pid,
            "slice_idx":  s_idx,
            "has_tumor":  has_tumor,
        }


# ─── Split Utilities ───────────────────────────────────────
def split_patients(
    patients: List[Dict],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Reproducible random split into train / val / test."""
    rng = random.Random(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]

    return train, val, test


def save_splits(splits: Dict, path: str):
    """Persist patient splits to JSON for reproducibility."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"  Splits saved → {path}")


def load_splits(path: str) -> Dict:
    """Load previously saved patient splits."""
    with open(path) as f:
        return json.load(f)
