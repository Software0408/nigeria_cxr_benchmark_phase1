# src/data/validate_data.py
# Phase 1 – Dataset Validation for Chest X-ray Benchmark
# Enhanced with detailed invalid study logging

import os
import pydicom
from pydicom.errors import InvalidDicomError
import hashlib
from tqdm import tqdm


# -----------------------------
# Helper: Chest identification
# -----------------------------
def is_chest_xray(ds):
    """Determine if DICOM is likely a chest X-ray using metadata."""
    chest_keywords = ["CHEST", "CXR", "THORAX", "LUNG", "PA", "AP", "THORACIC", "CARDIO", "MEDIASTIN", "PNEUMO"]
    modality = str(ds.get("Modality", "")).upper()
    if modality not in ["CR", "DX", "PX"]:
        return False
    fields = [
        ds.get("BodyPartExamined", ""),
        ds.get("StudyDescription", ""),
        ds.get("SeriesDescription", ""),
        ds.get("ProtocolName", ""),
        ds.get("ViewPosition", "")
    ]
    combined = " ".join(str(f).upper() for f in fields)
    return any(keyword in combined for keyword in chest_keywords)


# -----------------------------
# File-level validation
# -----------------------------
def validate_dicom_file(file_path):
    result = {
        "file": file_path,
        "is_valid": False,
        "has_pixel_data": False,
        "is_chest_candidate": False,
        "patient_id_leak": False,  # New: Flag potential PHI leakage
        "file_hash": None,        # New: For integrity/duplicates
        "error": None,
    }

    try:
        ds = pydicom.dcmread(file_path, force=False)
        result["is_valid"] = True
        result["is_chest_candidate"] = is_chest_xray(ds)
        result["has_pixel_data"] = 'PixelData' in ds and ds.PixelData is not None

        # PatientID leakage check (flag if non-empty and not anonymized-looking)
        patient_id = ds.get("PatientID", "")
        if patient_id and not patient_id.startswith("ANON") and len(patient_id) > 4:  # Customize threshold
            result["patient_id_leak"] = True

        # File integrity hash (MD5 for quick check)
        with open(file_path, "rb") as f:
            result["file_hash"] = hashlib.md5(f.read()).hexdigest()

    except InvalidDicomError:
        result["error"] = "Invalid DICOM file"
    except Exception as e:
        result["error"] = str(e)

    return result


# -----------------------------
# Study-level validation
# -----------------------------
def validate_study_folder(study_path):
    """Validate study folder and return detailed results."""
    dcm_files = [os.path.join(study_path, f) for f in os.listdir(study_path) if f.lower().endswith('.dcm')]
    report_present = os.path.exists(os.path.join(study_path, "report.txt"))

    if not dcm_files:
        return {
            "study": study_path,
            "status": "invalid",
            "reasons": ["No DICOM files found"],
            "dicom_count": 0,
            "valid_dicoms": 0,
            "pixel_dicoms": 0,
            "chest_dicoms": 0,
            "phi_leak_count": 0,  # New
            "report_present": report_present,
        }

    valid_dicom = pixel_dicom = chest_dicom = phi_leak_count = 0
    reasons = []

    for dcm in dcm_files:
        res = validate_dicom_file(dcm)  # Single call
        if res["is_valid"]:
            valid_dicom += 1
        if res["has_pixel_data"]:
            pixel_dicom += 1
        if res["is_chest_candidate"]:
            chest_dicom += 1
        if res["patient_id_leak"]:
            phi_leak_count += 1

    if valid_dicom == 0:
        reasons.append("No valid DICOM files")
    if pixel_dicom == 0:
        reasons.append("No DICOM with pixel data")
    if chest_dicom == 0:
        reasons.append("No chest-candidate images")
    if not report_present:
        reasons.append("Missing report.txt")
    if phi_leak_count > 0:
        reasons.append(f"Potential PHI leakage in PatientID ({phi_leak_count} files)")

    status = "valid" if not reasons else "invalid"

    return {
        "study": study_path,
        "status": status,
        "reasons": reasons,
        "dicom_count": len(dcm_files),
        "valid_dicoms": valid_dicom,
        "pixel_dicoms": pixel_dicom,
        "chest_dicoms": chest_dicom,
        "phi_leak_count": phi_leak_count,
        "report_present": report_present,
    }


# -----------------------------
# Dataset-level validation with invalid logging
# -----------------------------
def validate_dataset(dataset_root):
    """Validate full dataset and log invalid studies."""
    total = valid = invalid = 0
    invalid_studies = []

    study_ids = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]

    for study_id in tqdm(study_ids, desc="Validating studies"):
        study_path = os.path.join(dataset_root, study_id)
        total += 1
        result = validate_study_folder(study_path)
        if result["status"] == "valid":
            valid += 1
        else:
            invalid += 1
            invalid_studies.append(result)

    print("\n=== DATASET VALIDATION SUMMARY ===")
    print(f"Total studies: {total}")
    print(f"Valid studies: {valid}")
    print(f"Invalid studies: {invalid}")

    if invalid_studies:
        print("\n=== DETAILED INVALID STUDIES REPORT ===")
        for inv in invalid_studies:
            print(f"\nStudy: {inv['study']}")
            print(f"Reasons: {', '.join(inv['reasons'])}")
            print(f"DICOM count: {inv['dicom_count']} (Valid: {inv['valid_dicoms']}, Pixel: {inv['pixel_dicoms']}, Chest: {inv['chest_dicoms']})")
            print(f"Report present: {inv['report_present']}")

if __name__ == "__main__":
    validate_dataset("Z:/chest_dataset")
    