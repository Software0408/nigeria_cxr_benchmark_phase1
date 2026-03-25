import pandas as pd
import numpy as np
from pathlib import Path

COMPILED_XLSX = r"results\compiled_reports_readable_20260226_2220.xlsx"
LABELED_XLSX  = r"results\labeled_reports_20260226_2220.xlsx"
OLD_CSV       = r"results\annotated_reports.csv"   # used only to copy target size (optional)
OUT_CSV       = r"results\annotated_reports.csv"   # overwrite (change if you want a new name)

RANDOM_SEED = 42

# -----------------------------
# Load
# -----------------------------
compiled = pd.read_excel(COMPILED_XLSX)
labeled  = pd.read_excel(LABELED_XLSX)

# Safety checks
required_compiled = {"study_id", "report_text"}
required_labeled  = {"study_id", "review_priority"}

missing_c = required_compiled - set(compiled.columns)
missing_l = required_labeled  - set(labeled.columns)

if missing_c:
    raise ValueError(f"Compiled file missing columns: {missing_c}")
if missing_l:
    raise ValueError(f"Labeled file missing columns: {missing_l}")

# -----------------------------
# Merge: report_text + review_priority
# -----------------------------
df = compiled.merge(
    labeled[["study_id", "review_priority"]],
    on="study_id",
    how="inner"
)

# Drop empties
df["report_text"] = df["report_text"].astype(str)
df = df[df["report_text"].str.strip().ne("")]
df = df.dropna(subset=["review_priority"])

# -----------------------------
# Choose how many to sample
# -----------------------------
target_n = None
old_path = Path(OLD_CSV)
if old_path.exists():
    old = pd.read_csv(old_path)
    target_n = len(old)
else:
    # fallback: pick a default
    target_n = min(500, len(df))

target_n = min(target_n, len(df))

# -----------------------------
# Stratified sample by review_priority (proportional)
# -----------------------------
rng = np.random.default_rng(RANDOM_SEED)

# Normalize review_priority values
# (helps if some are numeric and some strings)
df["review_priority"] = df["review_priority"].astype(str).str.strip()

proportions = df["review_priority"].value_counts(normalize=True)

parts = []
for priority, prop in proportions.items():
    n = int(round(prop * target_n))
    subset = df[df["review_priority"] == priority]
    n = min(n, len(subset))
    if n > 0:
        parts.append(subset.sample(n=n, random_state=RANDOM_SEED))

sampled = pd.concat(parts, ignore_index=True) if parts else df.sample(n=target_n, random_state=RANDOM_SEED)

# Fix rounding mismatch
if len(sampled) > target_n:
    sampled = sampled.sample(n=target_n, random_state=RANDOM_SEED)
elif len(sampled) < target_n:
    need = target_n - len(sampled)
    remaining = df[~df["study_id"].isin(sampled["study_id"])]
    if need > 0 and len(remaining) > 0:
        extra = remaining.sample(n=min(need, len(remaining)), random_state=RANDOM_SEED)
        sampled = pd.concat([sampled, extra], ignore_index=True)

# Shuffle final
sampled = sampled.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# -----------------------------
# Build annotated_reports.csv in the format your fine-tune expects
# -----------------------------
annotated = pd.DataFrame({
    "report_text": sampled["report_text"],
    # blank labels ready for clinicians (your preprocess treats empty as Normal → all zeros)
    "gold_labels": [""] * len(sampled),
    # optional helpful columns:
    "study_id": sampled["study_id"],
    "review_priority": sampled["review_priority"],
})

annotated.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"Saved: {OUT_CSV}  (rows={len(annotated)})")
print("review_priority distribution:")
print(annotated["review_priority"].value_counts())