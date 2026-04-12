# workspace/tests/test_preprocess_data.py
# pytest suite for preprocess_data.py
# Tests percentile normalization, aspect-preserving resize, and file outputs

import sys
import pytest
import numpy as np
from pathlib import Path
import shutil
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from PIL import Image

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocess_data import (
    percentile_clip_and_normalize,
    resize_with_padding,
    preprocess_dicom,
    preprocess_study,
    preprocess_dataset,
    TARGET_SIZE,
)

# -----------------------------
# Helpers: synthetic DICOM
# -----------------------------
def create_mock_dicom(tmp_path, filename, shape=(512, 1024)):
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(
        filename,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )

    img = np.linspace(0, 1000, np.prod(shape)).reshape(shape).astype(np.uint16)

    ds.Rows, ds.Columns = img.shape
    ds.ViewPosition = "PA"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = img.tobytes()

    path = tmp_path / filename
    ds.save_as(path)
    return path


# -----------------------------
# Unit tests
# -----------------------------
def test_percentile_clip_and_normalize():
    img = np.linspace(0, 1000, 512 * 512).reshape(512, 512).astype(np.float32)
    out = percentile_clip_and_normalize(img)

    assert out.dtype == np.float32
    assert out.min() >= 0
    assert out.max() <= 1


def test_resize_with_padding_wide():
    img = np.ones((512, 1024), dtype=np.float32)
    out = resize_with_padding(img)

    assert out.shape == TARGET_SIZE
    assert np.any(out > 0)  # data exists
    assert np.all(out[0, :] == 0)  # padded top
    assert np.all(out[-1, :] == 0)  # padded bottom


def test_resize_with_padding_tall():
    img = np.ones((1024, 512), dtype=np.float32)
    out = resize_with_padding(img)

    assert out.shape == TARGET_SIZE
    assert np.all(out[:, 0] == 0)  # padded left
    assert np.all(out[:, -1] == 0)  # padded right


# -----------------------------
# Integration tests
# -----------------------------
def test_preprocess_dicom(tmp_path):
    dcm = create_mock_dicom(tmp_path, "image.dcm", shape=(600, 1200))

    out_npy = tmp_path / "image.npy"
    out_png = tmp_path / "image.png"

    preprocess_dicom(dcm, out_npy, out_png)

    arr = np.load(out_npy)
    assert arr.shape == TARGET_SIZE
    assert arr.dtype == np.float32
    assert 0 <= arr.min() <= arr.max() <= 1

    png = np.array(Image.open(out_png))
    assert png.shape == TARGET_SIZE
    assert png.dtype == np.uint16


def test_preprocess_study(tmp_path):
    study_in = tmp_path / "study"
    study_in.mkdir()
    create_mock_dicom(study_in, "img1.dcm")
    (study_in / "report.txt").write_text("Mock report")

    study_out = tmp_path / "out"
    preprocess_study(study_in, study_out)

    assert (study_out / "primary_image.npy").exists()
    assert (study_out / "primary_image.png").exists()
    assert (study_out / "report.txt").exists()


def test_preprocess_dataset(tmp_path):
    dataset_in = tmp_path / "dataset"
    dataset_in.mkdir()

    study = dataset_in / "study1"
    study.mkdir()
    create_mock_dicom(study, "img.dcm")
    (study / "report.txt").write_text("Report")

    dataset_out = tmp_path / "processed"
    preprocess_dataset(dataset_in, dataset_out)

    assert (dataset_out / "study1" / "primary_image.npy").exists()
