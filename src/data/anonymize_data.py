# workspace/src/data/anonymize_data.py
# Phase 1 – Anonymization Pipeline for Chest X-ray Benchmark
# Scope:
# - Deterministic pseudonymization of PatientID
# - Removal of all direct & indirect PHI
# - Study-level date shifting (relative chronology preserved)
# - SOPInstanceUID regeneration
# - Removal of all private DICOM tags
# - Report signature cleaning
# - NO pixel modification
# - NO clinical content modification
#
# NOTE:
# PatientAge and PatientSex are intentionally preserved
# for downstream demographic and bias evaluation.

import os
import hashlib
import datetime
import random
from pathlib import Path
from tqdm import tqdm
import pydicom
from pydicom.uid import generate_uid
import re

# ------------------------------------------------------------------
# Secret key (must be set in environment)
# ------------------------------------------------------------------
SECRET_KEY = os.environ.get("CXR_ANON_SECRET")
if not SECRET_KEY:
    raise RuntimeError(
        "CXR_ANON_SECRET not set. "
        "Anonymization aborted to prevent unsafe pseudonymization."
    )

# ------------------------------------------------------------------
# PHI tags to REMOVE completely
# (PatientAge and PatientSex intentionally excluded)
# ------------------------------------------------------------------
PHI_TAGS = [
    "PatientBirthDate",
    "PatientAddress",
    "OtherPatientIDs",
    "OtherPatientNames",
    "IssuerOfPatientID",
    "AccessionNumber",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "StationName",
    "DeviceSerialNumber",
    "StudyID",
    "RequestingPhysician",
    "RequestingService",
    "AdmissionID",
    "PerformedProcedureStepID",
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
        return date_str

def clean_report_text(report_text: str) -> str:
    """Remove radiologist signatures and identifiers from reports."""
    lines = report_text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if re.match(r"^\s*(Dr|Drs|Radiologist)[\s:]?", stripped, re.IGNORECASE):
            continue
        if stripped.lower().startswith(
            ("signed", "reported by", "verified by", "dictated by")
        ):
            continue
        if re.search(r"\w+\s*/\s*\w+", stripped):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)

# ------------------------------------------------------------------
# Core anonymization (single DICOM)
# ------------------------------------------------------------------
def anonymize_dicom_file(input_path: str, output_path: str, shift_days: int):
    """Anonymize a single DICOM file."""
    try:
        ds = pydicom.dcmread(input_path, force=True)

        # Deterministic PatientID hashing
        original_id = ds.get("PatientID", "")
        ds.PatientID = f"ANON_{hash_id(original_id)}"

        # Remove identifiable name
        ds.PatientName = "ANON"

        # Regenerate SOPInstanceUID
        ds.SOPInstanceUID = generate_uid()

        # Remove PHI tags
        for tag in PHI_TAGS:
            if tag in ds:
                del ds[tag]

        # Shift dates
        for tag in DATE_TAGS:
            if tag in ds and ds[tag].value:
                ds[tag].value = shift_date(ds[tag].value, shift_days)

        # Remove private tags
        ds.remove_private_tags()

        # De-identification declaration
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "NG-CXR Phase 1 anonymization"

        # Ensure minimum pixel metadata if PixelData present
        if "PixelData" in ds:
            ds.setdefault("Rows", 1)
            ds.setdefault("Columns", 1)
            ds.setdefault("BitsAllocated", 8)
            ds.setdefault("BitsStored", 8)
            ds.setdefault("HighBit", 7)
            ds.setdefault("SamplesPerPixel", 1)
            ds.setdefault("PhotometricInterpretation", "MONOCHROME2")

        # Ensure file meta exists and is valid
        if not hasattr(ds, "file_meta") or ds.file_meta is None:
            ds.file_meta = pydicom.dataset.FileMetaDataset()

        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ds.save_as(output_path, enforce_file_format=True)

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

    # Clean and copy report
    report_in = input_path / "report.txt"
    if report_in.exists():
        raw_text = report_in.read_text(encoding="utf-8", errors="ignore")
        cleaned_text = clean_report_text(raw_text)
        (output_path / "report.txt").write_text(cleaned_text, encoding="utf-8")

    # One date shift per study
    shift_days = random.randint(-365, 365)

    success = 0
    failed = 0

    for dcm_file in input_path.glob("*.dcm"):
        out_file = output_path / dcm_file.name
        ok, err = anonymize_dicom_file(
            str(dcm_file),
            str(out_file),
            shift_days
        )
        if ok:
            success += 1
        else:
            failed += 1
            print(f"[ERROR] {dcm_file.name}: {err}")

    print(
        f"Study {input_path.name}: "
        f"{success} anonymized, {failed} failed "
        f"(date shift = {shift_days} days)"
    )

# ------------------------------------------------------------------
# Dataset-level anonymization
# ------------------------------------------------------------------
def anonymize_dataset(input_root: str, output_root: str):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    studies = [s for s in input_root.iterdir() if s.is_dir()]
    for study in tqdm(studies, desc="Anonymizing studies"):
        anonymize_study(str(study), str(output_root / study.name))

# ------------------------------------------------------------------
# Manual run
# ------------------------------------------------------------------
if __name__ == "__main__":
    anonymize_dataset(
        "Z:/valid_dataset",
        "Z:/anonymized_dataset"
    )