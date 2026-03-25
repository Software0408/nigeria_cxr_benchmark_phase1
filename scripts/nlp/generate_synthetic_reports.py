#!/usr/bin/env python3
# scripts/nlp/generate_synthetic_reports.py
"""
Generate synthetic radiology reports for rare pathology labels.

Week 8 — March 2026.

Purpose:
    Augment the training set for BioClinicalBERT by generating synthetic reports
    for under-represented labels. Uses template-based generation with real FMC EB
    report components — no LLM API required.

Method:
    1. Generate a consistent anatomical context per report (side, zone, size, CTR)
       so findings and impressions always reference the same anatomy
    2. Build report from label-specific findings + normal boilerplate + matching impression
    3. Optionally add realistic co-occurring labels (30-35% chance each)
    4. Introduce controlled variation (sentence order, optional sentences)

Usage:
    python scripts/nlp/generate_synthetic_reports.py --review     # review 5 samples first
    python scripts/nlp/generate_synthetic_reports.py              # generate full set
    python scripts/nlp/generate_synthetic_reports.py --target 50  # custom target per label

Output:
    results/nlp/synthetic_reports.csv           — synthetic reports with labels
    results/nlp/training_reports_augmented.csv   — real + synthetic combined for BERT training
    results/nlp/synthetic_review_samples.txt    — human-readable samples (--review mode)

IMPORTANT:
    - Synthetic reports are for BERT TRAINING ONLY — never for evaluation
    - The 497 gold-labelled reports remain the clean evaluation set, untouched
    - All synthetic IDs start with "synth_" and have is_synthetic=1
"""

import sys
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42

DEFAULT_TARGETS = {
    "Post-Surgical Changes": 40,
    "Lung Mass / Nodule": 40,
    "Pulmonary Edema": 40,
    "Suspected Malignancy": 40,
    "Tuberculosis": 40,
    "Atelectasis": 40,
}


# ---- Anatomical context (generated once per report, used everywhere) ----

@dataclass
class AnatomicalContext:
    """Consistent anatomical references for one synthetic report."""
    side: str       # "right" or "left"
    zone: str       # "upper", "middle", or "lower"
    size: float     # mass/nodule size in cm
    ctr: float      # cardiothoracic ratio
    opp_side: str   # opposite side (for bilateral/contralateral references)

    @classmethod
    def random(cls) -> "AnatomicalContext":
        side = random.choice(["right", "left"])
        return cls(
            side=side,
            zone=random.choice(["upper", "middle", "lower"]),
            size=round(random.uniform(1.5, 6.0), 1),
            ctr=round(random.uniform(0.53, 0.72), 2),
            opp_side="left" if side == "right" else "right",
        )

    def fill(self, template: str) -> str:
        """Fill all placeholders using this context's consistent values."""
        return (
            template
            .replace("{side}", self.side)
            .replace("{opp_side}", self.opp_side)
            .replace("{zone}", self.zone)
            .replace("{size}", str(self.size))
            .replace("{ctr}", str(self.ctr))
        )


# ---- Report structure templates ----

REPORT_HEADERS = [
    "CHEST RADIOGRAPH REPORT",
    "CHEST RADIOGRAPHY REPORT",
    "CHEST RADIOGRAPH REPORT",
    "CHEST RADIOGRAPH REPORT",
]

SECTION_HEADERS = {
    "findings": ["Findings:", "Comments", "Comments:"],
    "impression": ["Impression:", "Conclusion:", "CONCLUSION", "Impression"],
}

NORMAL_BOILERPLATE = [
    "The mediastinum, hila and pulmonary vasculature are normal.",
    "The mediastinum, hila and pulmonary vasculature are also normal.",
    "The mediastinal shadow is within normal limits.",
    "Normal pulmonary vasculature is seen.",
    "The costophrenic sulci and hemidiaphragms are preserved.",
    "Both costophrenic recesses and hemidiaphragms are normal.",
    "The bony rib cage and peripheral soft tissues are normal.",
    "The bones of the thorax appear normal.",
    "Extra pulmonary soft tissues are normal.",
    "Trachea is central.",
    "The hila appear normal.",
    "Both hilar angles are preserved.",
]

SIGNOFF_PHRASES = [
    "Kindly correlate with other clinical parameters.",
    "Clinical correlation advised.",
    "ECG advised.",
    "Further evaluation recommended.",
    "Suggest clinical correlation.",
    "",
]


# ---- Per-label finding and impression templates ----
# All use {side}, {zone}, {size}, {ctr} — filled from the SAME AnatomicalContext

LABEL_TEMPLATES = {
    "Post-Surgical Changes": {
        "findings": [
            "Sternotomy wires are noted in situ.",
            "Median sternotomy wires are noted.",
            "Post thoracotomy changes are seen on the {side}.",
            "There are post-operative changes noted in the {side} hemithorax.",
            "Surgical clips are noted in the {side} hemithorax.",
            "Post-operative changes with sternotomy wires in situ.",
            "There is evidence of previous thoracic surgery with metallic sutures noted.",
            "Sternal wires are seen consistent with previous cardiac surgery.",
            "Post lobectomy changes are noted on the {side} with volume loss.",
            "CABG changes with sternotomy wires are seen.",
            "Metallic clips are projected over the {side} hilum, in keeping with post-surgical changes.",
        ],
        "impressions": [
            "Post thoracotomy changes.",
            "Post-surgical changes.",
            "Post-operative changes noted.",
            "Evidence of previous thoracic surgery.",
            "Post sternotomy changes.",
            "Post-operative changes.",
        ],
    },
    "Lung Mass / Nodule": {
        "findings": [
            "A solitary well-defined round soft tissue density mass is seen in the {side} {zone} lung zone.",
            "There is a well-defined rounded opacity in the {side} {zone} lung zone measuring approximately {size}cm.",
            "A rounded opacity is seen in the {side} {zone} lung field.",
            "A solitary pulmonary nodule is noted in the {side} {zone} lung zone.",
            "There is a {size}cm rounded soft tissue density lesion in the {side} {zone} lung zone.",
            "An ovoid mass is seen in the {side} {zone} lung zone.",
            "A spiculated opacity is seen in the {side} {zone} lung field.",
            "A focal mass is noted in the {side} {zone} lung zone.",
            "Multiple rounded opacities of different sizes are seen in both lung fields.",
            "A well-defined nodular opacity is seen in the {side} {zone} lung zone.",
            "A coin lesion is noted in the {side} {zone} lung zone.",
        ],
        "impressions": [
            "Solitary pulmonary nodule in the {side} lung. Further evaluation recommended.",
            "{side} lung mass. CT scan advised for further characterisation.",
            "Pulmonary nodule in the {side} {zone} zone. Recommend CT evaluation.",
            "{side} lung mass for further evaluation.",
            "Rounded opacity in the {side} lung — differential includes granuloma, neoplasm.",
            "Solitary pulmonary nodule. CT chest recommended.",
        ],
    },
    "Pulmonary Edema": {
        "findings": [
            "There is vascular congestion with upper lobe blood diversion.",
            "There is pulmonary vascular congestion with cephalization of vessels.",
            "Diffuse ground-glass opacification of both lung fields is noted.",
            "There are bilateral perihilar haze and prominent bronchovascular markings.",
            "Kerley B lines are noted at the lung bases bilaterally.",
            "There is interstitial edema with bilateral perihilar opacities.",
            "There is alveolar edema with bilateral air space opacities in a bat-wing distribution.",
            "Prominent bronchovascular markings with upper lobe blood diversion are noted.",
            "There is bilateral pulmonary congestion with peribronchial cuffing.",
            "There are bilateral perihilar haze with associated pulmonary vascular congestion.",
            "Fluid overload changes with bilateral pleural effusion and pulmonary vascular congestion.",
        ],
        "impressions": [
            "Features of pulmonary edema. Suggest clinical correlation.",
            "Pulmonary edema. Heart failure changes.",
            "Pulmonary vascular congestion. Features in keeping with fluid overload.",
            "Interstitial pulmonary edema.",
            "Pulmonary edema with vascular congestion.",
            "Features suggestive of congestive cardiac failure with pulmonary edema.",
        ],
    },
    "Suspected Malignancy": {
        "findings": [
            "There is a large spiculated mass in the {side} {zone} lung zone.",
            "A large irregular opacity is seen in the {side} {zone} lung zone, suspicious for malignancy.",
            "There is a mass lesion in the {side} {zone} lung zone with destruction of the adjacent rib.",
            "Multiple rounded opacities of varying sizes in both lung fields, suggestive of metastatic deposits.",
            "Few round soft tissue density lesions are seen in the {side} {zone} lung zone, raising the suspicion of metastatic deposit.",
            "There is a large mass in the {side} hemithorax with associated pleural effusion.",
            "A lobulated mass is seen in the {side} hilum with mediastinal widening.",
            "There is extensive homogenous opacity in the {side} {zone} and lower lung zones with associated pleural effusion.",
            "There is a soft tissue density mass in the {side} {zone} lung zone with irregular margins.",
            "A mass is seen abutting the mediastinum on the {side} with possible chest wall invasion.",
        ],
        "impressions": [
            "Suspicious for malignancy. CT scan and tissue biopsy recommended.",
            "Features suspicious for bronchogenic carcinoma. CT chest advised.",
            "Metastatic deposits. Primary to be determined.",
            "{side} lung mass suspicious for malignancy. Recommend further evaluation.",
            "Features raising suspicion of malignancy.",
            "Suspicious mass in the {side} lung. Tissue diagnosis recommended.",
        ],
    },
    "Tuberculosis": {
        "findings": [
            "There are reticulonodular opacities in the {side} upper lung zone.",
            "There is a cavitating lesion in the {side} upper lobe.",
            "Diffuse perivascular nodular opacities are noted in the {side} lung field.",
            "There are apical fibrotic changes bilaterally.",
            "Miliary shadows are seen diffusely in both lung fields.",
            "There is apical fibrosis with cavitation in the {side} upper zone.",
            "There are fibrocalcific changes in the {side} upper lung zone.",
            "Bilateral upper zone infiltrates with cavitation are noted.",
            "There is a thick-walled cavity in the {side} upper lobe.",
            "There are bilateral apical infiltrates consistent with active pulmonary tuberculosis.",
            "There is hilar lymphadenopathy with {side} upper lobe consolidation.",
            "Reticulonodular opacities noted in both upper zones with bilateral hilar lymphadenopathy.",
        ],
        "impressions": [
            "Features suggestive of pulmonary tuberculosis. Sputum AFB and GeneXpert recommended.",
            "Active pulmonary tuberculosis. Clinical correlation and microbiological confirmation advised.",
            "? TB. Recommend sputum for AFB.",
            "Features consistent with pulmonary tuberculosis.",
            "Old TB changes with fibrocalcific lesions.",
            "Reactivation tuberculosis. Sputum evaluation recommended.",
            "Miliary TB. Urgent clinical review.",
        ],
    },
    "Atelectasis": {
        "findings": [
            "There is volume loss in the {side} lower lobe with elevation of the hemidiaphragm.",
            "There is segmental collapse of the {side} lower lobe.",
            "There is {side} lower lobe collapse with mediastinal shift.",
            "Elevation of the {side} hemidiaphragm with streaky opacities at the base.",
            "There is plate atelectasis in the {side} lower zone.",
            "There is discoid atelectasis noted in the {side} lower lung zone.",
            "Rib crowding is noted in the {side} lower lung with elevation of the diaphragm, consistent with lower lobe collapse.",
            "There is compressive atelectasis of the {side} lower lobe.",
            "There is subsegmental atelectasis in the {side} {zone} lung zone.",
            "There is partial collapse of the {side} lung with loss of volume.",
            "Linear opacities in the {side} lower zone in keeping with subsegmental atelectasis.",
        ],
        "impressions": [
            "{side} lower lobe collapse.",
            "Atelectasis of the {side} lower lobe.",
            "{side} lower lobe atelectasis.",
            "Volume loss in the {side} hemithorax — atelectasis.",
            "Segmental atelectasis {side} lung.",
            "{side} subsegmental atelectasis.",
        ],
    },
}

# Co-occurring labels with their finding templates (also use the same AnatomicalContext)
COMMON_CO_OCCURRENCES = {
    "Post-Surgical Changes": ["Device Present", "Cardiomegaly", "Pleural Effusion"],
    "Lung Mass / Nodule": ["Cardiomegaly", "Pleural Effusion", "Atelectasis"],
    "Pulmonary Edema": ["Cardiomegaly", "Pleural Effusion", "Reticulonodular / Interstitial Pattern"],
    "Suspected Malignancy": ["Pleural Effusion", "Lung Mass / Nodule", "Atelectasis"],
    "Tuberculosis": ["Reticulonodular / Interstitial Pattern", "Pleural Effusion", "Atelectasis"],
    "Atelectasis": ["Pleural Effusion", "Device Present", "Pneumonia / Consolidation"],
}

CO_OCCURRENCE_FINDINGS = {
    "Device Present": [
        "A central venous catheter is noted with its tip projected over the SVC.",
        "An endotracheal tube is seen in situ.",
        "A chest tube is noted in the {side} hemithorax.",
        "A nasogastric tube is seen in situ.",
    ],
    "Cardiomegaly": [
        "The cardiac silhouette is enlarged (CTR - {ctr}).",
        "There is cardiomegaly (CTR {ctr}).",
        "The heart is enlarged (CTR - {ctr}).",
    ],
    "Pleural Effusion": [
        "There is blunting of the {side} costophrenic angle.",
        "There is {side} pleural effusion.",
        "Both costophrenic sulci are blunted.",
    ],
    "Reticulonodular / Interstitial Pattern": [
        "Reticulonodular opacities are seen in both lung fields.",
        "There are prominent bronchovascular markings bilaterally.",
    ],
    "Pneumonia / Consolidation": [
        "There is consolidation in the {side} lower zone.",
        "There are patchy opacities in the {side} {zone} lung zone.",
    ],
    "Lung Mass / Nodule": [
        "A rounded opacity is seen in the {side} {zone} lung field.",
    ],
    "Atelectasis": [
        "There is volume loss in the {side} lower lobe.",
    ],
}

CO_OCCURRENCE_IMPRESSIONS = {
    "Cardiomegaly": "Cardiomegaly.",
    "Pleural Effusion": "{side} pleural effusion.",
    "Device Present": "Line in situ.",
    "Reticulonodular / Interstitial Pattern": "Reticulonodular pattern.",
    "Pneumonia / Consolidation": "{side} lower zone consolidation.",
    "Lung Mass / Nodule": "{side} lung opacity for further evaluation.",
    "Atelectasis": "{side} lower lobe atelectasis.",
}


# ---- Generator ----

def generate_one_report(
    primary_label: str,
    include_co_occurrence: bool = True,
) -> Tuple[str, List[str]]:
    """
    Generate one synthetic report for a primary label.
    All anatomy references (side, zone, size, CTR) are consistent within the report.
    Returns (report_text, list_of_labels).
    """
    # One context per report — ensures no side/zone mismatches
    ctx = AnatomicalContext.random()

    templates = LABEL_TEMPLATES[primary_label]
    labels = [primary_label]

    # ---- Build findings ----
    findings = []

    # Primary label findings (1-3 sentences)
    n_primary = random.randint(1, min(3, len(templates["findings"])))
    primary_findings = random.sample(templates["findings"], n_primary)
    findings.extend([ctx.fill(f) for f in primary_findings])

    # Optional co-occurrence (30-35% chance per co-label)
    co_impressions = []
    if include_co_occurrence and primary_label in COMMON_CO_OCCURRENCES:
        for co_label in COMMON_CO_OCCURRENCES[primary_label]:
            if random.random() < 0.35 and co_label in CO_OCCURRENCE_FINDINGS:
                co_finding = random.choice(CO_OCCURRENCE_FINDINGS[co_label])
                findings.append(ctx.fill(co_finding))
                labels.append(co_label)
                if co_label in CO_OCCURRENCE_IMPRESSIONS:
                    co_impressions.append(ctx.fill(CO_OCCURRENCE_IMPRESSIONS[co_label]))

    # Normal boilerplate (2-4 sentences)
    n_boilerplate = random.randint(2, 4)
    boilerplate = random.sample(NORMAL_BOILERPLATE, n_boilerplate)

    # Shuffle findings and boilerplate
    all_findings = findings + boilerplate
    random.shuffle(all_findings)

    # ---- Build impression (same ctx) ----
    impression = ctx.fill(random.choice(templates["impressions"]))
    if co_impressions:
        impression += " " + " ".join(co_impressions)

    # ---- Assemble report ----
    header = random.choice(REPORT_HEADERS)
    section_header = random.choice(SECTION_HEADERS["findings"]) if random.random() < 0.5 else ""
    impression_header = random.choice(SECTION_HEADERS["impression"])
    signoff = random.choice(SIGNOFF_PHRASES)

    parts = [header]
    if section_header:
        parts.append(section_header)
    parts.extend(all_findings)
    parts.append(impression_header)
    parts.append(impression)
    if signoff:
        parts.append(signoff)

    report = "\n".join(parts)
    return report, sorted(set(labels))


def generate_synthetic_dataset(
    real_df: pd.DataFrame,
    targets: Dict[str, int],
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate synthetic reports to bring rare labels up to target counts."""
    random.seed(seed)

    records = []
    synth_id = 0

    for label, target_count in targets.items():
        real_count = real_df["gold_labels"].apply(lambda x: label in [l.strip() for l in x.split(",")]).sum()
        n_to_generate = max(0, target_count - real_count)

        if n_to_generate == 0:
            logger.info(f"  {label}: already at {real_count} >= {target_count}, skipping")
            continue

        logger.info(f"  {label}: {real_count} real + {n_to_generate} synthetic = {target_count} target")

        for _ in range(n_to_generate):
            report_text, labels = generate_one_report(label)
            records.append({
                "study_id": f"synth_{synth_id:05d}",
                "report_text": report_text,
                "gold_labels": ",".join(labels),
                "is_synthetic": 1,
                "primary_augmented_label": label,
            })
            synth_id += 1

    df = pd.DataFrame(records)
    logger.info(f"\nGenerated {len(df)} synthetic reports total")
    return df


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic reports for rare labels")
    parser.add_argument("--target", type=int, default=40, help="Target count per rare label (default: 40)")
    parser.add_argument("--review", action="store_true", help="Generate 5 samples per label for review only")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    args = parser.parse_args()

    real_path = PROJECT_ROOT / "results" / "annotated_reports.csv"
    real_df = pd.read_csv(str(real_path))
    logger.info(f"Loaded {len(real_df)} real annotated reports")

    targets = {label: args.target for label in DEFAULT_TARGETS}

    if args.review:
        logger.info("\n=== REVIEW MODE: 5 samples per label ===\n")
        random.seed(args.seed)

        review_lines = []
        for label in DEFAULT_TARGETS:
            review_lines.append(f"\n{'=' * 70}")
            review_lines.append(f"  {label}")
            review_lines.append(f"{'=' * 70}")
            for i in range(5):
                report, labels = generate_one_report(label)
                review_lines.append(f"\n--- Sample {i + 1} | Labels: {labels} ---")
                review_lines.append(report)

        review_text = "\n".join(review_lines)
        print(review_text)

        results_dir = PROJECT_ROOT / "results" / "nlp"
        results_dir.mkdir(parents=True, exist_ok=True)
        review_path = results_dir / "synthetic_review_samples.txt"
        with open(review_path, "w") as f:
            f.write(review_text)
        logger.info(f"\nReview samples saved to {review_path}")
        return

    # Full generation
    logger.info(f"\nGenerating synthetic reports (target: {args.target} per label)...")
    synth_df = generate_synthetic_dataset(real_df, targets, seed=args.seed)

    results_dir = PROJECT_ROOT / "results" / "nlp"
    results_dir.mkdir(parents=True, exist_ok=True)

    synth_path = results_dir / "synthetic_reports.csv"
    synth_df.to_csv(synth_path, index=False)
    logger.info(f"Synthetic reports saved to {synth_path}")

    # Combined training set
    real_for_training = real_df[["study_id", "report_text", "gold_labels"]].copy()
    real_for_training["is_synthetic"] = 0
    real_for_training["primary_augmented_label"] = ""

    combined = pd.concat([real_for_training, synth_df], ignore_index=True)
    combined_path = results_dir / "training_reports_augmented.csv"
    combined.to_csv(combined_path, index=False)
    logger.info(f"Combined training set saved to {combined_path} ({len(combined)} reports)")

    # Distribution summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"AUGMENTED TRAINING SET — {len(combined)} reports")
    logger.info(f"  Real: {len(real_df)} | Synthetic: {len(synth_df)}")
    logger.info(f"{'=' * 60}")

    all_labels = set()
    for gl in combined["gold_labels"]:
        for l in gl.split(","):
            l = l.strip()
            if l and l != "Normal":
                all_labels.add(l)

    for label in sorted(all_labels):
        count = combined["gold_labels"].str.contains(label, na=False).sum()
        real_count = real_df["gold_labels"].str.contains(label, na=False).sum()
        synth_count = count - real_count
        logger.info(f"  {label:<42} {count:>5} (real={real_count}, synth={synth_count})")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
