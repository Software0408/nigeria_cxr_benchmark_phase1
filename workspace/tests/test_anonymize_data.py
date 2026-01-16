# workspace/tests/test_anonymize_data.py
# Phase 1 pytest suite for anonymize_data.py
# Tests PHI removal, hashing, date shifting, UID regeneration, report cleaning, preserved Age/Sex
# Uses mock DICOMs with required tags for PixelData

import os
import sys
import pytest
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
from pathlib import Path
import shutil

# Add project root to sys.path for reliable imports in nested structure
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from workspace.src.data.anonymize_data import (
    hash_id,
    shift_date,
    clean_report_text,
    anonymize_dicom_file,
    anonymize_study,
    anonymize_dataset,
)

# Mock secret key for testing
@pytest.fixture(autouse=True)
def mock_secret_key(monkeypatch):
    monkeypatch.setenv("CXR_ANON_SECRET", "test_key_2026")

# -----------------------------
# Helper for mock DICOM creation (with required PixelData tags)
# -----------------------------
def _create_mock_dicom(tmp_path, filename, patient_id="REAL_ID", study_date="20240101", patient_age="050Y", patient_sex="M"):
    ds = Dataset()

    # Required for PixelData
    ds.Rows = 512
    ds.Columns = 512
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Test tags
    ds.PatientID = patient_id
    ds.PatientName = "REAL_NAME"
    ds.StudyDate = study_date
    ds.PatientAge = patient_age
    ds.PatientSex = patient_sex
    ds.SOPInstanceUID = generate_uid()  # Original UID
    ds.PixelData = b"\x00" * (512 * 512 * 2)  # 16-bit placeholder

    # File meta
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    file_path = tmp_path / filename
    pydicom.dcmwrite(str(file_path), ds)
    return str(file_path), ds.SOPInstanceUID

# -----------------------------
# Unit Tests
# -----------------------------
def test_hash_id():
    hashed = hash_id("REAL_ID")
    assert len(hashed) == 16
    assert hashed != "REAL_ID"
    assert hash_id("REAL_ID") == hashed

def test_shift_date():
    shifted = shift_date("20240101", 5)
    assert shifted == "20240106"
    assert shift_date("", 5) == ""
    assert shift_date("invalid", 5) == "invalid"

def test_clean_report_text():
    raw = "Findings...\nImpression: Cardiomegaly\nDrs Owoo / Umeh"
    cleaned = clean_report_text(raw)
    assert "Drs Owoo / Umeh" not in cleaned
    assert "Cardiomegaly" in cleaned

# -----------------------------
# Integration Tests
# -----------------------------
def test_anonymize_dicom_file(tmp_path):
    input_path, original_uid = _create_mock_dicom(tmp_path, "input.dcm", patient_id="REAL_ID", study_date="20240101", patient_age="050Y", patient_sex="M")
    output_path = str(tmp_path / "output.dcm")

    success, error = anonymize_dicom_file(input_path, output_path, shift_days=10)
    assert success is True
    assert error is None

    ds = pydicom.dcmread(output_path)
    assert ds.PatientID.startswith("ANON_")
    assert ds.PatientName == "ANON"
    assert ds.PatientAge == "050Y"  # Preserved
    assert ds.PatientSex == "M"  # Preserved
    assert ds.SOPInstanceUID != original_uid  # Regenerated
    assert ds.StudyDate == "20240111"  # Shifted
    assert ds.PatientIdentityRemoved == "YES"
    assert ds.DeidentificationMethod == "NG-CXR Phase 1 anonymization"

def test_anonymize_study(tmp_path):
    study_in = tmp_path / "study_in"
    study_in.mkdir()
    _create_mock_dicom(study_in, "image.dcm")
    (study_in / "report.txt").write_text("Findings...\nDrs Owoo / Umeh")

    study_out = tmp_path / "study_out"

    anonymize_study(str(study_in), str(study_out))
    assert (study_out / "image.dcm").exists()

# -----------------------------
# Dataset-level Smoke Test
# -----------------------------
def test_anonymize_dataset_runs(tmp_path):
    dataset_in = tmp_path / "dataset_in"
    dataset_in.mkdir()
    study = dataset_in / "study1"
    study.mkdir()
    _create_mock_dicom(study, "image.dcm")
    (study / "report.txt").write_text("Report\nDrs Name")

    dataset_out = tmp_path / "dataset_out"

    anonymize_dataset(str(dataset_in), str(dataset_out))

if __name__ == "__main__":
    pytest.main([__file__])