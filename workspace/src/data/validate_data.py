# src/data/validate_data.py
# Phase 1 – Dataset Validation for Chest X-ray Benchmark
# Enhanced with valid study moving, detailed invalid logging, PHI leakage, and integrity

import os
import hashlib
import shutil
from tqdm import tqdm
import pydicom
from pathlib import Path
from pydicom.errors import InvalidDicomError


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
        "patient_id_leak": False,
        "file_hash": None,
        "error": None,
    }

    try:
        ds = pydicom.dcmread(file_path, force=False)
        result["is_valid"] = True
        result["is_chest_candidate"] = is_chest_xray(ds)
        result["has_pixel_data"] = 'PixelData' in ds and ds.PixelData is not None

        # PatientID leakage check
        patient_id = ds.get("PatientID", "")
        if patient_id and not patient_id.startswith("ANON") and len(patient_id) > 4:
            result["patient_id_leak"] = True

        # File integrity hash
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
            "report_present": report_present,
        }

    valid_dicom = pixel_dicom = chest_dicom = 0
    reasons = []

    for dcm in dcm_files:
        res = validate_dicom_file(dcm)
        if res["is_valid"]:
            valid_dicom += 1
        if res["has_pixel_data"]:
            pixel_dicom += 1
        if res["is_chest_candidate"]:
            chest_dicom += 1

    if valid_dicom == 0:
        reasons.append("No valid DICOM files")
    if pixel_dicom == 0:
        reasons.append("No DICOM with pixel data")
    if chest_dicom == 0:
        reasons.append("No chest-candidate images")
    if not report_present:
        reasons.append("Missing report.txt")

    status = "valid" if not reasons else "invalid"

    return {
        "study": study_path,
        "status": status,
        "reasons": reasons,
        "dicom_count": len(dcm_files),
        "valid_dicoms": valid_dicom,
        "pixel_dicoms": pixel_dicom,
        "chest_dicoms": chest_dicom,
        "report_present": report_present,
    }


# -----------------------------
# Dataset-level validation
# -----------------------------
def validate_dataset(dataset_root, valid_output_dir=None, clean_chest_output_dir=None):
    """Validate full dataset and optionally copy/move valid studies or chest-only clean subset."""
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dataset_root}")

    total = valid = invalid = 0
    invalid_studies = []

    study_ids = [d for d in dataset_path.iterdir() if d.is_dir()]

    for study_path in tqdm(study_ids, desc="Validating studies"):
        total += 1
        result = validate_study_folder(str(study_path))
        if result["status"] == "valid":
            valid += 1
            # Full valid study copy (optional)
            if valid_output_dir:
                output_study_path = Path(valid_output_dir) / study_path.name
                output_study_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(study_path, output_study_path)
                print(f"Copied full valid study {study_path.name} to {output_study_path}")

            # Chest-only clean subset copy (optional)
            if clean_chest_output_dir:
                clean_study_path = Path(clean_chest_output_dir) / study_path.name
                clean_study_path.mkdir(parents=True, exist_ok=True)

                # Copy report
                report_in = study_path / "report.txt"
                if report_in.exists():
                    shutil.copy(report_in, clean_study_path / "report.txt")

                # Copy only chest-candidate DICOMs
                chest_count = 0
                for dcm_file in study_path.glob("*.dcm"):
                    res = validate_dicom_file(str(dcm_file))
                    if res["is_chest_candidate"]:
                        shutil.copy(dcm_file, clean_study_path / dcm_file.name)
                        chest_count += 1

                print(f"Copied {chest_count} chest DICOMs for clean study {study_path.name} to {clean_study_path}")
        else:
            invalid += 1
            invalid_studies.append(result)

    print("\n=== DATASET VALIDATION SUMMARY ===")
    print(f"Total studies: {total}")
    print(f"Valid studies (usable for chest benchmarking): {valid}")
    print(f"Invalid / incomplete studies: {invalid}")

    if invalid_studies:
        print("\n=== DETAILED INVALID STUDIES REPORT ===")
        for inv in invalid_studies:
            print(f"\nStudy: {inv['study']}")
            print(f"Reasons: {', '.join(inv['reasons'])}")
            print(f"DICOM count: {inv['dicom_count']} (Valid: {inv['valid_dicoms']}, Pixel: {inv['pixel_dicoms']}, Chest: {inv['chest_dicoms']})")
            print(f"Report present: {inv['report_present']}")

if __name__ == "__main__":
    validate_dataset(
        "Z:/chest_dataset",
        valid_output_dir="Z:/valid_dataset",
        clean_chest_output_dir="Z:/valid_dataset"
    )