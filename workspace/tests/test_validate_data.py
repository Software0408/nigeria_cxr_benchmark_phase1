# tests/test_validate_data.py
# Automated tests for validate_data.py
# Uses pytest with in-memory/mock DICOMs for reproducibility and CI compatibility

import pytest
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pathlib import Path
import shutil

from workspace.src.data.validate_data import (
    is_chest_xray,
    validate_dicom_file,
    validate_study_folder,
    validate_dataset,
)

# -----------------------------
# Fixtures for mock DICOMs
# -----------------------------
@pytest.fixture
def mock_chest_dicom(tmp_path):
    """Create a mock chest X-ray DICOM in memory and save to temp file."""
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    ds = FileDataset("mock_chest.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "CR"  # Computed Radiography - common for CXR
    ds.BodyPartExamined = "CHEST"
    ds.ViewPosition = "PA"
    ds.PixelData = b""  # Empty but present (simulates image)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    file_path = tmp_path / "mock_chest.dcm"
    ds.save_as(file_path)
    return str(file_path)


@pytest.fixture
def mock_non_chest_dicom(tmp_path):
    """Mock non-chest DICOM (e.g., abdominal)."""
    ds = Dataset()
    ds.Modality = "CT"
    ds.BodyPartExamined = "ABDOMEN"
    ds.PixelData = b""

    file_path = tmp_path / "mock_non_chest.dcm"
    pydicom.dcmwrite(file_path, ds)
    return str(file_path)


@pytest.fixture
def mock_invalid_dicom(tmp_path):
    """Mock invalid DICOM (corrupted header)."""
    file_path = tmp_path / "invalid.dcm"
    file_path.write_bytes(b"not a DICOM")  # Invalid content
    return str(file_path)


# -----------------------------
# Unit Tests
# -----------------------------
def test_is_chest_xray_true(mock_chest_dicom):
    ds = pydicom.dcmread(mock_chest_dicom)
    assert is_chest_xray(ds) is True


def test_is_chest_xray_false(mock_non_chest_dicom):
    ds = pydicom.dcmread(mock_non_chest_dicom)
    assert is_chest_xray(ds) is False


def test_validate_dicom_file_valid(mock_chest_dicom):
    result = validate_dicom_file(mock_chest_dicom)
    assert result["is_valid"] is True
    assert result["has_pixel_data"] is True
    assert result["is_chest_candidate"] is True
    assert result["error"] is None


def test_validate_dicom_file_invalid(mock_invalid_dicom):
    result = validate_dicom_file(mock_invalid_dicom)
    assert result["is_valid"] is False
    assert "Invalid DICOM" in result["error"]


# -----------------------------
# Integration Tests
# -----------------------------
def test_validate_study_folder_valid(tmp_path, mock_chest_dicom):
    study_path = tmp_path / "study_valid"
    study_path.mkdir()
    # Copy mock file into study folder
    shutil.copy(mock_chest_dicom, study_path / "image.dcm")
    (study_path / "report.txt").write_text("Mock report")

    result = validate_study_folder(str(study_path))
    assert result["status"] == "valid"
    assert result["chest_dicoms"] >= 1
    assert result["report_present"] is True


def test_validate_study_folder_invalid_no_report(tmp_path, mock_chest_dicom):
    study_path = tmp_path / "study_no_report"
    study_path.mkdir()
    shutil.copy(mock_chest_dicom, study_path / "image.dcm")

    result = validate_study_folder(str(study_path))
    assert result["status"] == "invalid"
    assert "Missing report.txt" in result["reasons"]


# -----------------------------
# Dataset-level Test (lightweight)
# -----------------------------
def test_validate_dataset(tmp_path, mock_chest_dicom):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    study = dataset_root / "study1"
    study.mkdir()
    shutil.copy(mock_chest_dicom, study / "image.dcm")
    (study / "report.txt").write_text("Mock")
    validate_dataset(str(dataset_root))