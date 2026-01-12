# tests/test_validate_data.py
# Automated tests for validate_data.py
# Uses pytest with in-memory/mock DICOMs for reproducibility and CI compatibility

import os
import sys
import pytest
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pathlib import Path
import shutil

# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from workspace.src.data.validate_data import (
    is_chest_xray,
    validate_dicom_file,
    validate_study_folder,
    validate_dataset,
)

# -----------------------------
# Standardized Mock Fixtures (valid image structure)
# -----------------------------
def _create_mock_dicom(tmp_path, filename, modality="CR", body_part="CHEST", view_position="PA"):
    """Helper to create valid mock DICOM with minimal image tags."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  # Fixes deprecation

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Required for valid image
    ds.Rows = 1
    ds.Columns = 1
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Custom metadata
    ds.Modality = modality
    ds.BodyPartExamined = body_part
    ds.ViewPosition = view_position

    # Valid pixel data (1 black pixel)
    ds.PixelData = b"\x00"

    file_path = tmp_path / filename
    ds.save_as(file_path)
    return str(file_path)


@pytest.fixture
def mock_chest_dicom(tmp_path):
    return _create_mock_dicom(tmp_path, "mock_chest.dcm", modality="CR", body_part="CHEST", view_position="PA")


@pytest.fixture
def mock_non_chest_dicom(tmp_path):
    return _create_mock_dicom(tmp_path, "mock_non_chest.dcm", modality="CT", body_part="ABDOMEN")


@pytest.fixture
def mock_invalid_dicom(tmp_path):
    file_path = tmp_path / "invalid.dcm"
    file_path.write_bytes(b"invalid DICOM content")
    return str(file_path)


# -----------------------------
# Unit & Integration Tests
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


def test_validate_study_folder_valid(tmp_path, mock_chest_dicom):
    study_path = tmp_path / "study_valid"
    study_path.mkdir()
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


def test_validate_dataset(tmp_path, mock_chest_dicom):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    study = dataset_root / "study1"
    study.mkdir()
    shutil.copy(mock_chest_dicom, study / "image.dcm")
    (study / "report.txt").write_text("Mock")

    validate_dataset(str(dataset_root))


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

if __name__ == "__main__":
    pytest.main([__file__])