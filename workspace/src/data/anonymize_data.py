# workspace/src/data/anonymize_data.py
# Phase 3 – Anonymization Pipeline for Chest X-ray Benchmark
# Scope:
# - Deterministic pseudonymization of PatientID
# - Removal of all direct & indirect PHI
# - Study-level date shifting (relative chronology preserved)
# - Removal of all private DICOM tags
# - NO pixel modification
# - NO report modification
# - Originals preserved; anonymized copies written separately

import os
import hashlib
import datetime
import random
import shutil
from pathlib import Path
import pydicom


# ------------------------------------------------------------------
# Secret key (must be set in encrypted environment)
# ------------------------------------------------------------------
SECRET_KEY = os.environ.get("CXR_ANON_SECRET")
if not SECRET_KEY:
    raise RuntimeError(
        "CXR_ANON_SECRET not set. "
        "Anonymization aborted to prevent unsafe pseudonymization."
    )


# ------------------------------------------------------------------
# PHI tags to REMOVE completely
# ------------------------------------------------------------------
PHI_TAGS = [
    "PatientBirthDate",
    "PatientAddress",
    "OtherPatientIDs",
    "OtherPatientNames",
    "AccessionNumber",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "StationName",
    "DeviceSerialNumber",
]


# ------------------------------------------------------------------
# Date tags to SHIFT (not delete)
# ------------------------------------------------------------------
DATE_TAGS = [
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def hash_id(value: str) -> str:
    """Deterministically hash a PatientID using a secret salt."""
    if not value:
        return "ANON"
    return hashlib.sha256(
        (str(value) + SECRET_KEY).encode("utf-8")
    ).hexdigest()[:16]


def shift_date(date_str, shift_days):
    if not date_str:
        return ""
    try:
        date = datetime.datetime.strptime(date_str, "%Y%m%d")
        new_date = date + datetime.timedelta(days=shift_days)
        return new_date.strftime("%Y%m%d")
    except ValueError:
        return date_str  # Return original on invalid format


# ------------------------------------------------------------------
# Core anonymization (single DICOM)
# ------------------------------------------------------------------
def anonymize_dicom_file(input_path, output_path, shift_days):
    """Anonymize a single DICOM file."""
    try:
        ds = pydicom.dcmread(input_path)

        # Special handling for PatientID (hash) and PatientName (ANON)
        original_id = ds.get("PatientID", "")
        ds.PatientID = f"ANON_{hash_id(original_id)}"
        ds.PatientName = "ANON"

        for tag in PHI_TAGS:
            if tag in ds:
                del ds[tag]

        for tag in DATE_TAGS:
            if tag in ds:
                ds[tag].value = shift_date(ds[tag].value, shift_days)

        ds.remove_private_tags()

        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "NG-CXR Phase 1 anonymization"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ds.save_as(output_path)

        return True, None
    except Exception as e:
        return False, str(e)


# ------------------------------------------------------------------
# Study-level anonymization
# ------------------------------------------------------------------
def anonymize_study(input_study_path: str, output_study_path: str):
    input_path = Path(input_study_path)
    output_path = Path(output_study_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy report unchanged
    report_in = input_path / "report.txt"
    if report_in.exists():
        shutil.copy(report_in, output_path / "report.txt")

    # One random date shift per study
    shift_days = random.randint(-365, 365)

    success_count = 0
    error_count = 0

    for dcm_file in input_path.glob("*.dcm"):
        output_file = output_path / dcm_file.name
        success, error = anonymize_dicom_file(
            str(dcm_file),
            str(output_file),
            shift_days
        )

        if success:
            success_count += 1
        else:
            error_count += 1
            print(f"[ERROR] {dcm_file.name}: {error}")

    print(
        f"Study {input_path.name}: "
        f"{success_count} anonymized, {error_count} failed"
    )


# ------------------------------------------------------------------
# Dataset-level anonymization
# ------------------------------------------------------------------
def anonymize_dataset(input_root: str, output_root: str):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for study in input_root.iterdir():
        if study.is_dir():
            anonymize_study(
                str(study),
                str(output_root / study.name)
            )


# ------------------------------------------------------------------
# Manual run
# ------------------------------------------------------------------
if __name__ == "__main__":
    anonymize_dataset(
        "Z:/valid_dataset",
        "Z:/anonymized_dataset"
    )
