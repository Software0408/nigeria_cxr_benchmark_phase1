# tests/test_validate_data.py
# Phase 1 pytest suite for validate_data.py
# Aligned with benchmark rules:
# - Mixed body parts allowed
# - At least one chest DICOM with pixel data required
# - report.txt REQUIRED
# - PHI presence flagged (presence only)

import sys
import shutil
from pathlib import Path

import pytest
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset

# --------------------------------------------------
# Ensure project root is on sys.path (for root-level src/)
# --------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.validate_data import (
    is_chest_xray,
    validate_dicom_file,
    validate_study_folder,
    validate_dataset,
)

# --------------------------------------------------
# Helper: create mock DICOM files
# --------------------------------------------------
def _create_mock_dicom(
    tmp_path,
    filename,
    modality="CR",
    body_part="CHEST",
    view_position="PA",
    has_pixel=True,
    patient_id=None,
):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    ds = FileDataset(
        filename,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )

    ds.Modality = modality
    ds.BodyPartExamined = body_part
    ds.ViewPosition = view_position

    ds.Rows = 1
    ds.Columns = 1
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    if patient_id is not None:
        ds.PatientID = patient_id

    if has_pixel:
        ds.PixelData = b"\x00"

    file_path = tmp_path / filename
    ds.save_as(file_path)
    return str(file_path)


# --------------------------------------------------
# Fixtures
# --------------------------------------------------
@pytest.fixture
def mock_chest_dicom(tmp_path):
    return _create_mock_dicom(tmp_path, "chest.dcm")


@pytest.fixture
def mock_non_chest_dicom(tmp_path):
    return _create_mock_dicom(
        tmp_path,
        "abdomen.dcm",
        modality="CT",
        body_part="ABDOMEN",
        view_position="AP",
    )


@pytest.fixture
def mock_no_pixel_dicom(tmp_path):
    return _create_mock_dicom(tmp_path, "no_pixel.dcm", has_pixel=False)


@pytest.fixture
def mock_phi_dicom(tmp_path):
    return _create_mock_dicom(tmp_path, "phi.dcm", patient_id="REAL_PATIENT_ID")


@pytest.fixture
def mock_invalid_dicom(tmp_path):
    file_path = tmp_path / "invalid.dcm"
    file_path.write_bytes(b"not a dicom")
    return str(file_path)


# --------------------------------------------------
# Unit tests: metadata logic
# --------------------------------------------------
def test_is_chest_xray_true(mock_chest_dicom):
    ds = pydicom.dcmread(mock_chest_dicom)
    assert is_chest_xray(ds) is True 


def test_is_chest_xray_false(mock_non_chest_dicom):
    ds = pydicom.dcmread(mock_non_chest_dicom)
    assert is_chest_xray(ds) is False


# --------------------------------------------------
# Unit tests: file-level validation
# --------------------------------------------------
def test_validate_dicom_file_valid(mock_chest_dicom):
    result = validate_dicom_file(mock_chest_dicom)
    assert result["is_valid"] is True
    assert result["has_pixel_data"] is True
    assert result["is_chest_candidate"] is True
    assert result["error"] is None


def test_validate_dicom_file_no_pixel(mock_no_pixel_dicom):
    result = validate_dicom_file(mock_no_pixel_dicom)
    assert result["is_valid"] is True
    assert result["has_pixel_data"] is False


def test_validate_dicom_file_phi_flag(mock_phi_dicom):
    result = validate_dicom_file(mock_phi_dicom)
    assert result["patient_id_leak"] is True


def test_validate_dicom_file_invalid(mock_invalid_dicom):
    result = validate_dicom_file(mock_invalid_dicom)
    assert result["is_valid"] is False
    assert "Invalid DICOM" in result["error"]


# --------------------------------------------------
# Integration tests: study-level validation
# --------------------------------------------------
def test_validate_study_valid(tmp_path, mock_chest_dicom):
    study = tmp_path / "study_valid"
    study.mkdir()

    shutil.copy(mock_chest_dicom, study / "image.dcm")
    (study / "report.txt").write_text("Report")

    result = validate_study_folder(str(study))

    assert result["status"] == "valid"
    assert result["chest_dicoms"] >= 1
    assert result["pixel_dicoms"] >= 1
    assert result["report_present"] is True
    assert result["reasons"] == []


def test_validate_study_invalid_no_report(tmp_path, mock_chest_dicom):
    study = tmp_path / "study_no_report"
    study.mkdir()

    shutil.copy(mock_chest_dicom, study / "image.dcm")

    result = validate_study_folder(str(study))

    assert result["status"] == "invalid"
    assert "Missing report.txt" in result["reasons"]


def test_validate_study_mixed_pixel(tmp_path, mock_chest_dicom, mock_no_pixel_dicom):
    study = tmp_path / "study_mixed"
    study.mkdir()

    shutil.copy(mock_chest_dicom, study / "chest.dcm")
    shutil.copy(mock_no_pixel_dicom, study / "no_pixel.dcm")
    (study / "report.txt").write_text("Report")

    result = validate_study_folder(str(study))

    assert result["status"] == "valid"  # At least one pixel
    assert result["pixel_dicoms"] == 1
    assert "No DICOM with pixel data" not in result["reasons"]


def test_validate_study_all_no_pixel(tmp_path, mock_no_pixel_dicom):
    study = tmp_path / "study_no_pixel"
    study.mkdir()

    shutil.copy(mock_no_pixel_dicom, study / "a.dcm")
    shutil.copy(mock_no_pixel_dicom, study / "b.dcm")
    (study / "report.txt").write_text("Report")

    result = validate_study_folder(str(study))

    assert result["status"] == "invalid"
    assert "No DICOM with pixel data" in result["reasons"]


# --------------------------------------------------
# Dataset-level smoke test
# --------------------------------------------------
def test_validate_dataset_runs(tmp_path, mock_chest_dicom):
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    study = dataset / "study1"
    study.mkdir()
    shutil.copy(mock_chest_dicom, study / "image.dcm")
    (study / "report.txt").write_text("Report")

    # Should run without raising
    validate_dataset(str(dataset))


if __name__ == "__main__":
    pytest.main([__file__])
