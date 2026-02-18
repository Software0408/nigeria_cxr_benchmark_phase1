# src/data/preprocess_data.py
"""
Preprocessing pipeline for Nigerian Chest X-ray Benchmark (Phase 1 Pilot at FMC Ebute-Metta).
Handles DICOM → normalized/resized PNG + NPY arrays, with chest-view filtering and idempotency.
"""

import os
import shutil
import numpy as np
import pydicom
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
from typing import Optional


# Configuration
TARGET_SIZE = (512, 512)
LOW_PCT = 1
HIGH_PCT = 99


def percentile_clip_and_normalize(img: np.ndarray) -> np.ndarray:
    """Clip outliers and normalize to [0, 1]."""
    p_low, p_high = np.percentile(img, (LOW_PCT, HIGH_PCT))
    img = np.clip(img, p_low, p_high)
    img = (img - p_low) / (p_high - p_low + 1e-6)
    return img.astype(np.float32)


def resize_with_padding(img: np.ndarray, target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """Aspect-preserving resize + zero-padding."""
    if img.ndim != 2:
        raise ValueError("Expected 2D grayscale image")
    
    h, w = img.shape
    target_h, target_w = target_size
    
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros(target_size, dtype=resized.dtype)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def is_likely_chest_view(ds: pydicom.dataset.Dataset) -> bool:
    """Heuristic to identify chest X-ray views using common DICOM tags."""
    # View Position (0018,5101): PA, AP, LAT, etc.
    view_pos = ds.get((0x0018, 0x5101), None)
    if view_pos and any(v in str(view_pos.value).upper() for v in ['PA', 'AP', 'FRONT']):
        return True
    
    # Body Part Examined (0018,0015)
    body_part = ds.get((0x0018, 0x0015), None)
    if body_part and 'CHEST' in str(body_part.value).upper():
        return True
    
    # Study Description (0008,1030) or Series Description (0008,103E)
    study_desc = ds.get((0x0008, 0x1030), None)
    series_desc = ds.get((0x0008, 0x103E), None)
    desc = (study_desc.value if study_desc else "") + (series_desc.value if series_desc else "")
    if any(kw in desc.upper() for kw in ['CHEST', 'THORAX', 'LUNG', 'CXR']):
        return True
    
    return False  # Default: assume not chest if tags don't match


def preprocess_dicom(
    input_path: Path,
    output_npy_path: Path,
    output_png_path: Optional[Path] = None
) -> None:
    """Process one DICOM file → normalized array + PNG."""
    output_npy_path = Path(output_npy_path)
    if output_npy_path.exists():
        print(f"  Skipping (already exists): {output_npy_path}")
        return

    try:
        ds = pydicom.dcmread(input_path)
        img = ds.pixel_array.astype(np.float32)

        if img.ndim != 2:
            print(f"  Skipping non-2D image: {input_path}")
            return

        # Filter: only process likely chest views
        if not is_likely_chest_view(ds):
            print(f"  Skipping non-chest view: {input_path}")
            return

        img_norm = percentile_clip_and_normalize(img)
        img_final = resize_with_padding(img_norm)

        os.makedirs(output_npy_path.parent, exist_ok=True)
        np.save(output_npy_path, img_final)

        if output_png_path:
            img_uint16 = (img_final * 65535).astype(np.uint16)
            os.makedirs(output_png_path.parent, exist_ok=True)
            Image.fromarray(img_uint16).save(output_png_path)

        print(f"  Processed: {input_path} → {output_npy_path}")

    except Exception as e:
        print(f"  Error processing {input_path}: {e}")


def preprocess_study(input_study_path: Path, output_study_path: Path):
    """Process one study folder → keep only one primary chest image."""
    output_study_path.mkdir(parents=True, exist_ok=True)

    # Copy report once
    report_src = input_study_path / "report.txt"
    if report_src.exists():
        shutil.copy(report_src, output_study_path / "report.txt")
    else:
        print(f"  No report.txt found in {input_study_path}")

    dcm_files = list(input_study_path.glob("*.dcm"))
    if not dcm_files:
        print(f"  No DICOM files in {input_study_path}")
        return

    # Sort by Instance Number (0020,0013) if present, else by filename
    def sort_key(p: Path):
        try:
            ds = pydicom.dcmread(p)
            return int(ds.get((0x0020, 0x0013), 999999).value)
        except:
            return p.name

    dcm_files.sort(key=sort_key)

    processed = False
    for dcm_file in dcm_files:
        npy_path = output_study_path / "primary_image.npy"
        png_path = output_study_path / "primary_image.png"

        preprocess_dicom(
            dcm_file,
            output_npy_path=npy_path,
            output_png_path=png_path,
        )

        # Stop after first valid chest image (for now)
        if (npy_path.exists()):
            processed = True
            break

    if not processed:
        print(f"  No valid chest view found in {input_study_path}")


def preprocess_dataset(input_root: Path, output_root: Path):
    """Process all study folders in input_root."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    studies = [s for s in input_root.iterdir() if s.is_dir()]
    print(f"Found {len(studies)} study folders to process.")

    for study in tqdm(studies, desc="Preprocessing studies"):
        preprocess_study(study, output_root / study.name)


if __name__ == "__main__":
    import os
    input_root = os.getenv("INPUT_ROOT", "Z:/anonymized_dataset")
    output_root = os.getenv("OUTPUT_ROOT", "Z:/preprocessed_dataset")
    print(f"Using INPUT_ROOT = {input_root}")
    print(f"Using OUTPUT_ROOT = {output_root}")
    preprocess_dataset(input_root, output_root)