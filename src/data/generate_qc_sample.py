# workspace/src/data/generate_qc_sample.py
# Generate random sample of 322 preprocessed images for manual QC
# Outputs folder for cloud upload + CSV reference list

import random
from pathlib import Path
import shutil
import pandas as pd

random.seed(42)  # Reproducible sample

preprocessed_dir = Path("Z:/preprocessed_dataset")
qc_dir = Path("Z:/manual_qc_sample_322")
qc_dir.mkdir(exist_ok=True)

# Collect all PNGs
all_png = list(preprocessed_dir.rglob("*.png"))
if len(all_png) < 322:
    raise ValueError(f"Only {len(all_png)} images found — need at least 322")

sample_png = random.sample(all_png, 322)

# Reference list for reviewer/form
reference_list = []

for idx, png_path in enumerate(sample_png, 1):
    study_id = png_path.parent.name
    image_name = png_path.name
    qc_study_dir = qc_dir / f"{idx:03d}_{study_id}"
    qc_study_dir.mkdir(parents=True, exist_ok=True)

    # Copy image
    shutil.copy(png_path, qc_study_dir / image_name)

    # Copy report
    report_in = png_path.parent / "report.txt"
    if report_in.exists():
        shutil.copy(report_in, qc_study_dir / "report.txt")

    # Add to reference
    reference_list.append({
        "QC_ID": idx,
        "Study_ID": study_id,
        "Image_File": image_name,
        "Folder_Name": qc_study_dir.name
    })

# Save reference CSV
ref_df = pd.DataFrame(reference_list)
ref_csv = qc_dir / "qc_reference_list.csv"
ref_df.to_csv(ref_csv, index=False)

print(f"QC sample generated: {qc_dir}")
print(f"{len(sample_png)} images in numbered folders")
print(f"Reference list: {ref_csv}")