# workspace/tests/test_anonymize_data.py
# Phase 1 pytest suite for anonymize_data.py
# Tests PHI removal, hashing, date shifting, and study-level anonymization
# Uses mock DICOMs for self-contained, ethical testing

import os
import sys
import pytest
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pathlib import Path
import shutil
import datetime

# Add project root to sys.path for reliable imports in nested structure
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from workspace.src.data.anonymize_data import (
    hash_id,
    shift_date,
    anonymize_dicom_file,
    anonymize_study,
    anonymize_dataset,
)

# Mock secret key for testing (bypasses env check)
@pytest.fixture(autouse=True)
def mock_secret_key(monkeypatch):
    monkeypatch.setenv("CXR_ANON_SECRET", "test_key_2026")

# -----------------------------
# Helper for mock DICOM creation
# -----------------------------
def _create_mock_dicom(tmp_path, filename, patient_id="REAL_ID", study_date="20240101"):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.PatientID = patient_id
    ds.PatientName = "REAL_NAME"
    ds.StudyDate = study_date
    ds.PixelData = b"\x00"  # Placeholder pixel data

    file_path = tmp_path / filename
    ds.save_as(file_path)
    return str(file_path)


# -----------------------------
# Unit Tests
# -----------------------------
def test_hash_id():
    hashed = hash_id("REAL_ID")
    assert len(hashed) == 16
    assert hashed != "REAL_ID"
    assert hash_id("REAL_ID") == hashed  # Deterministic


def test_shift_date():
    shifted = shift_date("20240101", 5)
    assert shifted == "20240106"
    assert shift_date("", 5) == ""
    assert shift_date("invalid", 5) == "invalid"  # Fallback to original


# -----------------------------
# Integration Tests
# -----------------------------
def test_anonymize_dicom_file(tmp_path):
    input_path = _create_mock_dicom(tmp_path, "input.dcm", patient_id="REAL_ID", study_date="20240101")
    output_path = str(tmp_path / "output.dcm")

    anonymize_dicom_file(input_path, output_path, shift_days=10)

    ds = pydicom.dcmread(output_path)
    assert ds.PatientID.startswith("ANON_")
    assert ds.PatientName == "ANON"
    assert "PatientBirthDate" not in ds  # Removed
    assert ds.StudyDate == "20240111"  # Shifted by 10 days
    assert ds.DeidentificationMethod == "NG-CXR Phase 1 anonymization"


def test_anonymize_study(tmp_path):
    study_in = tmp_path / "study_in"
    study_in.mkdir()
    dcm_path = _create_mock_dicom(study_in, "image.dcm")
    (study_in / "report.txt").write_text("Report content")

    study_out = tmp_path / "study_out"

    anonymize_study(str(study_in), str(study_out))

    # Check anonymized DICOM
    ds = pydicom.dcmread(study_out / "image.dcm")
    assert ds.PatientID.startswith("ANON_")

    # Check report copied unchanged
    assert (study_out / "report.txt").read_text() == "Report content"


# -----------------------------
# Dataset-level Smoke Test
# -----------------------------
def test_anonymize_dataset_runs(tmp_path):
    dataset_in = tmp_path / "dataset_in"
    dataset_in.mkdir()
    study = dataset_in / "study1"
    study.mkdir()
    _create_mock_dicom(study, "image.dcm")
    (study / "report.txt").write_text("Report")

    dataset_out = tmp_path / "dataset_out"

    anonymize_dataset(str(dataset_in), str(dataset_out))  # Runs without error

if __name__ == "__module__":
    pytest.main([__file__])