"""
Manual Quality Control
To ensure the integrity of the preprocessed dataset, manual review was performed on a random sample of 322 images (~5.7% of the 5,637 valid studies). 
This sample size was determined using the finite population proportion formula:

                    $n = \frac{N \cdot Z^2 \cdot p \cdot (1-p)}{E^2 \cdot (N-1) + Z^2 \cdot p \cdot (1-p)}$

with N=5,637 (population), Z=1.96 (95% confidence), p=0.5 (conservative), and E=0.03 (3% margin of error). 
Random selection (seeded for reproducibility) minimizes bias and provides high confidence in detecting rare preprocessing artifacts while representing the dataset's clinical diversity.
"""


import pandas as pd
from urllib.parse import urlencode

CSV_PATH = "Z:/manual_qc_sample_322/qc_reference_list.csv"
OUTPUT_XLSX = "Z:/manual_qc_sample_322/qc_prefill_links.xlsx"

FORM_BASE_URL = "https://docs.google.com/forms/d/e/" \
                "1FAIpQLScurCCBfLe-_K1R4-_ORcC0Pu02dUUu-YDkFJlEUWP4fk1Y5w/viewform"

DRIVE_BASE_URL = "https://drive.google.com/drive/folders/1dNNoxSCt4jpkRknTT7Fwf8aduKOHlSh4?usp=sharing"


FIELD_QC_ID = "entry.1941674469"
FIELD_STUDY_ID = "entry.1588621296"


df = pd.read_csv(CSV_PATH)

rows = []

for _, row in df.iterrows():
    qc_id = str(row["QC_ID"])
    study_folder = row["Folder_Name"]

    form_params = {
        FIELD_QC_ID: qc_id,
        FIELD_STUDY_ID: study_folder
    }
    form_link = FORM_BASE_URL + "?" + urlencode(form_params)

    study_drive_link = f"{DRIVE_BASE_URL}/{study_folder}"

    rows.append({
        "QC_ID": qc_id,
        "Study_ID": study_folder,
        "QC_Form_Link": form_link,
        "Study_Folder_Link": study_drive_link,
        "Review_Status": "Not completed"
    })

out_df = pd.DataFrame(rows)
out_df.to_excel(OUTPUT_XLSX, index=False)

print("SUCCESS")
print(f"QC links written to: {OUTPUT_XLSX}")
