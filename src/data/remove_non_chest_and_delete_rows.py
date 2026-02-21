# src/data/remove_non_chest_and_delete_rows.py
"""
Script to completely remove specified non-chest studies from the flat preprocessed_data structure.

- Deletes the matching PNG, TXT, and NPY files.
- Completely removes the corresponding rows from dataset_metadata.csv.
- Creates a timestamped log file in qc_logs/ for traceability and protocol documentation.

Usage:
    python src/data/remove_non_chest_and_delete_rows.py
"""

from pathlib import Path
import pandas as pd
from datetime import datetime

# Configuration
PREPROCESSED_DIR = Path("workspace/data/preprocessed_data")
METADATA_CSV = PREPROCESSED_DIR / "metadata" / "dataset_metadata.csv"
QC_LOG_DIR = PREPROCESSED_DIR / "qc_logs"
QC_LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = QC_LOG_DIR / f"excluded_non_chest_deleted_rows_{datetime.now():%Y%m%d_%H%M}.txt"

# List of study IDs to exclude (add more as needed)
EXCLUDE_STUDY_IDS = [
    "anon_study_01263", "anon_study_01448", "anon_study_01455", "anon_study_01467",
    "anon_study_01479", "anon_study_01525", "anon_study_01578", "anon_study_01592",
    "anon_study_01613", "anon_study_01671", "anon_study_01714",
    "anon_study_01839", "anon_study_01840", "anon_study_01844",
    "anon_study_01835", "anon_study_01836", "anon_study_01837", "anon_study_01838",
    "anon_study_02146", "anon_study_02170", "anon_study_02205", "anon_study_02230",
    "anon_study_02296", "anon_study_02318", "anon_study_02347",
    "anon_study_02558", "anon_study_02560", "anon_study_02569", 
    "anon_study_02593", "anon_study_02622", "anon_study_02626", "anon_study_02631", 
    "anon_study_02639", "anon_study_02697", "anon_study_02719", "anon_study_02832", 
    "anon_study_02849", "anon_study_02858", "anon_study_02933", 
    "anon_study_02867", "anon_study_03007", "anon_study_03036", 
    "anon_study_03136", "anon_study_03179", "anon_study_03200", "anon_study_03230", 
    "anon_study_03267", "anon_study_03354", "anon_study_03474", "anon_study_03476", 
    "anon_study_03505", "anon_study_03531", "anon_study_03554", 
    "anon_study_03632", "anon_study_03650", "anon_study_03691", 
    "anon_study_03895", "anon_study_03970", "anon_study_03931", "anon_study_03940", 
    "anon_study_03939", "anon_study_03948", "anon_study_03950", "anon_study_04004", 
    "anon_study_04005", "anon_study_04022", "anon_study_04060", 
    "anon_study_04076", "anon_study_04157", "anon_study_04121", 
    "anon_study_04193", "anon_study_04222", "anon_study_04314", "anon_study_04331", 
    "anon_study_04350", "anon_study_04352", "anon_study_04369", "anon_study_04382", 
    "anon_study_04410", "anon_study_04417", "anon_study_04455", 
    "anon_study_04491", "anon_study_04520", "anon_study_04560", 
    "anon_study_04588", "anon_study_04589", "anon_study_04711", "anon_study_04713", 
    "anon_study_04708", "anon_study_04728", "anon_study_04851", "anon_study_04893", 
    "anon_study_04901", "anon_study_04933", "anon_study_04938", 
    "anon_study_04940", "anon_study_04961", "anon_study_04976", "anon_study_04979", 
    "anon_study_05004", "anon_study_05072", "anon_study_05078", "anon_study_05038", 
    "anon_study_05193", "anon_study_05261", "anon_study_05267", 
    "anon_study_05268", "anon_study_05325", "anon_study_05361", 
    "anon_study_05366", "anon_study_05454", "anon_study_05561", "anon_study_05555", "anon_study_05590"
]

def main():
    if not METADATA_CSV.exists():
        print(f"Error: Metadata CSV not found → {METADATA_CSV}")
        return

    df = pd.read_csv(METADATA_CSV)
    print(f"Original CSV contains {len(df)} studies.")

    excluded = []
    log_lines = [
        f"Non-chest exclusions - rows fully deleted\n",
        f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n",
        "=" * 60 + "\n"
    ]

    for study_id in EXCLUDE_STUDY_IDS:
        if study_id not in df["study_id"].values:
            print(f"→ {study_id} not found in CSV → skipping")
            continue

        row = df[df["study_id"] == study_id].iloc[0]

        # Get file paths from CSV
        img_path = PREPROCESSED_DIR / row["image_path"]
        report_path = PREPROCESSED_DIR / row["report_path"]
        array_path = PREPROCESSED_DIR / row["array_path"]

        # Delete files if they exist
        for p in [img_path, report_path, array_path]:
            if p.exists():
                p.unlink()
                print(f"  Deleted: {p}")
            else:
                print(f"  File already missing: {p}")

        # Remove row from DataFrame
        df = df[df["study_id"] != study_id]
        excluded.append(study_id)

        # Log
        log_lines.append(f"{study_id}")
        log_lines.append(f"  Deleted files:")
        log_lines.append(f"    • {row['image_path']}")
        log_lines.append(f"    • {row['report_path']}")
        log_lines.append(f"    • {row['array_path']}")
        log_lines.append(f"  Reason: non-chest imaging (manual confirmation)")
        log_lines.append("-" * 50 + "\n")

    # Save updated (reduced) CSV
    df.to_csv(METADATA_CSV, index=False)
    print(f"\nUpdated CSV saved with {len(df)} remaining studies (removed {len(excluded)}).")

    # Save exclusion log
    with open(LOG_FILE, "w") as f:
        f.writelines(log_lines)

    print(f"Exclusion log written to: {LOG_FILE}")
    print("Cleanup complete. The dataset now contains only retained chest studies.")

if __name__ == "__main__":
    main()