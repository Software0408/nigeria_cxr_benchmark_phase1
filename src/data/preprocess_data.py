# workspace/src/data/preprocess_data.py
# Preprocessing for Nigerian Chest X-ray Benchmark
# Preprocessing: Percentile clipping, aspect-preserving resize, and normalization for CXR

import os
import shutil
import numpy as np
import pydicom
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm


TARGET_SIZE = (512, 512)

def percentile_clip_and_normalize(img, low_percentile=1, high_percentile=99):
    p_low, p_high = np.percentile(img, (low_percentile, high_percentile))
    img = np.clip(img, p_low, p_high)
    img = (img - p_low) / (p_high - p_low + 1e-6)
    return img.astype(np.float32)

def resize_with_padding(img, target_size=TARGET_SIZE):
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

def preprocess_dicom(input_path, output_npy_path, output_png_path=None):
    ds = pydicom.dcmread(input_path)
    img = ds.pixel_array.astype(np.float32)

    if img.ndim != 2:
        raise ValueError(f"Non-2D image in {input_path}")

    img_norm = percentile_clip_and_normalize(img)
    img_final = resize_with_padding(img_norm)

    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    np.save(output_npy_path, img_final)

    if output_png_path:
        img_uint16 = (img_final * 65535).astype(np.uint16)
        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        Image.fromarray(img_uint16).save(output_png_path)

def preprocess_study(input_study_path, output_study_path):
    input_path = Path(input_study_path)
    output_path = Path(output_study_path)
    output_path.mkdir(parents=True, exist_ok=True)

    report = input_path / "report.txt"
    if report.exists():
        shutil.copy(report, output_path / "report.txt")

    for dcm_file in input_path.glob("*.dcm"):
        preprocess_dicom(
            input_path=dcm_file,
            output_npy_path=output_path / f"{dcm_file.stem}.npy",
            output_png_path=output_path / f"{dcm_file.stem}.png",
        )

def preprocess_dataset(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    studies = [s for s in input_root.iterdir() if s.is_dir()]
    for study in tqdm(studies, desc="Preprocessing studies"):
        preprocess_study(study, output_root / study.name)


if __name__ == "__main__":
    preprocess_dataset("Z:/anonymized_dataset", "Z:/preprocessed_dataset")
