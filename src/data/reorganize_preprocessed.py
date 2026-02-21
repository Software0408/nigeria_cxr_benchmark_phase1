# src/data/reorganize_preprocessed.py
"""
Enhanced migration script handling PNG images, report.txt, and .npy arrays.
Outputs flat structure: images/, reports/, arrays/ + metadata CSV.
"""

from pathlib import Path
import shutil
import pandas as pd

def reorganize(
    source_dir: Path = Path("Z:/preprocessed_dataset"),
    target_dir: Path = Path("workspace/data/preprocessed_data")
) -> None:
    if not source_dir.exists():
        raise ValueError(f"Source directory not found: {source_dir}")
    
    # Create target  
    images_dir = target_dir / "images"
    reports_dir = target_dir / "reports"
    arrays_dir = target_dir / "arrays"
    for d in [images_dir, reports_dir, arrays_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    records = []
    study_id_fmt = "anon_study_{:05d}"
    
    subfolders = sorted([f for f in source_dir.iterdir() if f.is_dir()])
    print(f"Found {len(subfolders)} study subfolders. Processing...")
    
    for idx, subfolder in enumerate(subfolders, start=1):
        study_id = study_id_fmt.format(idx)
        
        # Flexible file matching
        image_files = list(subfolder.glob("*.png"))
        report_files = list(subfolder.glob("report.txt"))
        npy_files = list(subfolder.glob("*.npy"))
        
        if len(image_files) != 1 or len(report_files) != 1 or len(npy_files) != 1:
            print(f"Warning: Skipping {subfolder.name} "
                  f"(images: {len(image_files)}, reports: {len(report_files)}, npy: {len(npy_files)})")
            continue
        
        image_src = image_files[0]
        report_src = report_files[0]
        npy_src = npy_files[0]
        
        # Destinations
        image_dest = images_dir / f"{study_id}.png"
        report_dest = reports_dir / f"{study_id}.txt"
        npy_dest = arrays_dir / f"{study_id}.npy"
        
        shutil.move(image_src, image_dest)
        shutil.move(report_src, report_dest)
        shutil.move(npy_src, npy_dest)
        
        records.append({
            "study_id": study_id,
            "image_path": f"images/{study_id}.png",
            "report_path": f"reports/{study_id}.txt",
            "array_path": f"arrays/{study_id}.npy",
            "original_folder": subfolder.name,
            "qc_flag": "good"  # Update later via validate_data.py
        })
    
    # Generate metadata CSV
    metadata_dir = target_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    metadata_path = metadata_dir / "dataset_metadata.csv"
    
    pd.DataFrame(records).to_csv(metadata_path, index=False)
    
    print(f"Reorganization complete: {len(records)} studies in {target_dir}")
    print(f"Images: {images_dir} | Reports: {reports_dir} | Arrays: {arrays_dir}")
    print(f"Metadata CSV: {metadata_path}")

    # Optional cleanup
    shutil.rmtree(source_dir)
    print(f"Cleaned up source directory: {source_dir}")

if __name__ == "__main__":
    reorganize()