#!/usr/bin/env python3
# scripts/nlp/generate_hard_negatives.py
"""
Generate hard negative synthetic reports to reduce BERT over-prediction.

Week 9 — March 2026.

Problem: BERT over-predicts because it hasn't seen enough examples where
confusing anatomy phrases appear in NORMAL context. When it sees
"costophrenic sulci" it partially activates Pleural Effusion, because
it learned the association but not the negation.

Solution: Generate reports that contain confusing phrases for each label
but are explicitly NOT that label. These teach BERT the difference between
"costophrenic sulci are preserved" (Normal) and "blunting of costophrenic
sulci" (Pleural Effusion).

Also generates positive Normal reports with varied phrasing to teach BERT
a strong "this is normal" representation.

Output:
    results/nlp/hard_negatives.csv — synthetic hard negative reports
    Columns: study_id, report_text, gold_labels, is_synthetic, neg_target

Usage:
    python scripts/nlp/generate_hard_negatives.py

    Then merge with existing augmented training set:
    python scripts/nlp/merge_training_data.py  (or manual pd.concat)
"""

import random
import logging
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---- Anatomical building blocks ----

@dataclass
class ReportTemplate:
    """A synthetic report with known gold labels and the label it's a hard negative FOR."""
    text: str
    gold_labels: List[str]  # What this report IS (can be empty = Normal)
    neg_target: str          # The label this is a hard negative FOR


# Normal anatomy phrases that BERT confuses with pathology
# Organized by the label they get falsely associated with

HARD_NEGATIVE_TEMPLATES: Dict[str, List[ReportTemplate]] = {}

# ============================================================================
# DEGENERATIVE / STRUCTURAL CHANGES (542 disagreements, biggest cluster)
# Problem: BERT fires on "bony rib cage", "thorax", "aorta" even when normal
# ============================================================================
HARD_NEGATIVE_TEMPLATES["Degenerative / Structural Changes"] = [
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nThe heart is normal in size and contour.\nThe mediastinum, hila and pulmonary vasculature are normal.\nNo focal lung lesion is seen.\nThe costophrenic sulci and hemidiaphragms are preserved.\nThe bony rib cage and peripheral soft tissues are normal.\nImpression:\nNormal chest radiograph.",
        gold_labels=[], neg_target="Degenerative / Structural Changes"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nComments\nThe lung fields are clear.\nThe cardiac and mediastinal shadows are within normal limits.\nNormal pulmonary vasculature is seen.\nBoth costophrenic recesses and hemidiaphragms are normal.\nThe bones of the thorax and peripheral soft tissue shadows are normal.\nImpression:\nNormal chest radiograph.",
        gold_labels=[], neg_target="Degenerative / Structural Changes"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nThe heart is normal in size (CTR - 0.48) and contour.\nThe aorta is normal in caliber.\nNo focal lung lesion is seen.\nThe costophrenic sulci and hemidiaphragms are preserved.\nThe bony thorax and overlying soft tissues are unremarkable.\nImpression:\nNormal study.",
        gold_labels=[], neg_target="Degenerative / Structural Changes"
    ),
    ReportTemplate(
        text="AP CHEST RADIOGRAPH REPORT\nFindings:\nThe cardiac shadow is within normal limits.\nThe mediastinum, hila and pulmonary vasculature are also normal.\nBoth lung fields are clear.\nBoth costophrenic sulci and hemidiaphragms are preserved.\nThe bony rib cage and peripheral soft tissues are intact.\nImpression:\nNormal chest radiograph.",
        gold_labels=[], neg_target="Degenerative / Structural Changes"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe heart is normal in size and contour.\nThe mediastinum and hila are unremarkable.\nNo focal lung lesion is seen.\nThe costophrenic angles and hemidiaphragms are preserved.\nThe visualized bony thorax is normal.\nConclusion:\nNo significant radiographic abnormality.",
        gold_labels=[], neg_target="Degenerative / Structural Changes"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nThe lung fields are clear with normal bronchovascular markings.\nThe cardiac size is within normal limits (CTR - 0.46).\nThe aortic arch is normal.\nThe bony structures of the thorax appear normal.\nBoth costophrenic angles are clear.\nImpression:\nNormal chest radiograph.",
        gold_labels=[], neg_target="Degenerative / Structural Changes"
    ),
    # Reports with aortic mention but NOT degenerative
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nThe cardiac shadow is enlarged (CTR - 0.62).\nThe aorta is normal.\nBilateral pleural effusion is noted with meniscus sign.\nThe bony rib cage is intact.\nImpression:\nCardiomegaly with bilateral pleural effusion.",
        gold_labels=["Cardiomegaly", "Pleural Effusion"], neg_target="Degenerative / Structural Changes"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nHomogenous opacity is seen in the right lower lung zone.\nThe aortic arch is normal.\nThe heart is normal in size.\nThe bony thorax is unremarkable.\nImpression:\nRight lower lobe pneumonia.",
        gold_labels=["Pneumonia / Consolidation"], neg_target="Degenerative / Structural Changes"
    ),
]

# ============================================================================
# PLEURAL EFFUSION (237 disagreements)
# Problem: BERT fires on "costophrenic" even when "preserved" / "clear"
# ============================================================================
HARD_NEGATIVE_TEMPLATES["Pleural Effusion"] = [
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nThe heart is normal in size and contour.\nThe mediastinum, hila and pulmonary vasculature are normal.\nNo focal lung lesion is seen.\nBoth costophrenic sulci and hemidiaphragms are preserved.\nThe bony rib cage and peripheral soft tissues are normal.\nImpression:\nNormal chest radiograph.",
        gold_labels=[], neg_target="Pleural Effusion"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe cardiac shadow is enlarged (CTR - 0.57).\nThe costophrenic sulci and hemidiaphragms are preserved bilaterally.\nNo pleural effusion is seen.\nThe lung fields are clear.\nImpression:\nCardiomegaly. No pleural effusion.",
        gold_labels=["Cardiomegaly"], neg_target="Pleural Effusion"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nReticular opacities are noted in both lung fields.\nBoth costophrenic angles are clear.\nThe hemidiaphragms are preserved.\nCardiac size is within normal limits.\nImpression:\nBilateral reticulonodular pattern. No effusion.",
        gold_labels=["Reticulonodular / Interstitial Pattern"], neg_target="Pleural Effusion"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe heart is normal in size.\nThe costophrenic recesses are normal bilaterally.\nThe hemidiaphragms are well visualized and normal.\nThe lung fields are clear.\nImpression:\nNormal study.",
        gold_labels=[], neg_target="Pleural Effusion"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nPatchy opacity is noted in the right middle lung zone.\nBoth costophrenic sulci are preserved.\nThe hemidiaphragms are intact.\nThe cardiac shadow is within normal limits.\nImpression:\nRight middle lobe pneumonia. No pleural effusion.",
        gold_labels=["Pneumonia / Consolidation"], neg_target="Pleural Effusion"
    ),
    ReportTemplate(
        text="AP CHEST RADIOGRAPH REPORT\nThe cardiac shadow is normal.\nThe costophrenic sulci and hemidiaphragms are preserved.\nClear lung fields bilaterally.\nNormal pulmonary vasculature.\nImpression:\nNo acute cardiopulmonary disease.",
        gold_labels=[], neg_target="Pleural Effusion"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe heart is enlarged (CTR - 0.59) with left ventricular configuration.\nBoth costophrenic angles are clear and well-preserved.\nNo pleural fluid is seen.\nImpression:\nCardiomegaly with left ventricular preponderance. Clear costophrenic angles.",
        gold_labels=["Cardiomegaly"], neg_target="Pleural Effusion"
    ),
]

# ============================================================================
# PNEUMONIA / CONSOLIDATION (300 disagreements)
# Problem: BERT fires on generic "opacity" / "haziness" without pneumonia context
# ============================================================================
HARD_NEGATIVE_TEMPLATES["Pneumonia / Consolidation"] = [
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nHomogenous opacity is noted in the left lower lung zone silhouetting the left hemidiaphragm with meniscus sign.\nThe cardiac shadow is within normal limits.\nImpression:\nModerate left pleural effusion.",
        gold_labels=["Pleural Effusion"], neg_target="Pneumonia / Consolidation"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nThere is haziness of the right hemithorax with associated volume loss.\nMediastinal shift to the right is noted.\nImpression:\nRight lung collapse with associated volume loss.",
        gold_labels=["Atelectasis"], neg_target="Pneumonia / Consolidation"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nDiffuse reticular opacities are seen bilaterally.\nNo focal consolidation is seen.\nBoth costophrenic angles are preserved.\nCardiac size is normal.\nImpression:\nBilateral interstitial lung disease.",
        gold_labels=["Reticulonodular / Interstitial Pattern"], neg_target="Pneumonia / Consolidation"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nHomogenous opacification of the left hemithorax with associated mediastinal shift to the right.\nThe right lung field is clear.\nImpression:\nMassive left pleural effusion.",
        gold_labels=["Pleural Effusion"], neg_target="Pneumonia / Consolidation"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nA round opacity is noted in the right upper lung zone.\nNo air bronchogram is seen.\nThe cardiac shadow is normal.\nBoth costophrenic sulci are preserved.\nImpression:\nRight upper lobe lung mass for further evaluation.",
        gold_labels=["Lung Mass / Nodule"], neg_target="Pneumonia / Consolidation"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nThere is near-total opacification of the right hemithorax.\nThe heart is displaced to the left.\nA chest tube artifact is seen on the right.\nImpression:\nRight hydropneumothorax with chest tube in situ.",
        gold_labels=["Device Present"], neg_target="Pneumonia / Consolidation"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nDense fibrotic opacities are noted bilaterally predominantly in the upper lung zones.\nVolume loss with upward retraction of both hila.\nImpression:\nBilateral pulmonary fibrosis with volume loss, likely sequelae of old TB.",
        gold_labels=["Reticulonodular / Interstitial Pattern", "Tuberculosis"], neg_target="Pneumonia / Consolidation"
    ),
]

# ============================================================================
# CARDIOMEGALY (304 disagreements)
# Problem: BERT fires on "cardiac" / "heart" mentions even when normal
# ============================================================================
HARD_NEGATIVE_TEMPLATES["Cardiomegaly"] = [
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nThe heart is normal in size and contour.\nThe mediastinum, hila and pulmonary vasculature are normal.\nNo focal lung lesion is seen.\nImpression:\nNormal chest radiograph.",
        gold_labels=[], neg_target="Cardiomegaly"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe cardiac shadow is within normal limits (CTR - 0.48).\nThe lung fields are clear.\nBoth costophrenic sulci are preserved.\nImpression:\nNormal study.",
        gold_labels=[], neg_target="Cardiomegaly"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe heart is normal in size (CTR - 0.45) and contour.\nThere is blunting of the right costophrenic sulcus.\nImpression:\nRight pleural effusion. Normal cardiac size.",
        gold_labels=["Pleural Effusion"], neg_target="Cardiomegaly"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe cardiac size is indeterminate due to massive pleural effusion.\nHomogenous opacity is noted in the left hemithorax.\nImpression:\nMassive left pleural effusion. Cardiac size indeterminate.",
        gold_labels=["Pleural Effusion"], neg_target="Cardiomegaly"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe heart is normal in size.\nReticular opacities are noted in both lower lung zones.\nThe costophrenic sulci are preserved.\nImpression:\nInterstitial lung disease. Normal cardiac size.",
        gold_labels=["Reticulonodular / Interstitial Pattern"], neg_target="Cardiomegaly"
    ),
    ReportTemplate(
        text="AP CHEST RADIOGRAPH REPORT\nRotated image.\nThe cardiac shadow appears prominent but this is likely due to rotation and AP projection.\nThe lung fields are clear.\nImpression:\nNormal study. Apparent cardiomegaly likely due to rotation and AP projection.",
        gold_labels=[], neg_target="Cardiomegaly"
    ),
]

# ============================================================================
# POST-SURGICAL CHANGES (111 disagreements, 89% BERT over-prediction)
# Problem: BERT hallucinates post-surgical from any structural mention
# ============================================================================
HARD_NEGATIVE_TEMPLATES["Post-Surgical Changes"] = [
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nThe heart is normal in size.\nCentral venous catheter artifact is noted with tip in the superior vena cava.\nThe lung fields are clear.\nImpression:\nCentral venous catheter in situ. No other abnormality.",
        gold_labels=["Device Present"], neg_target="Post-Surgical Changes"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nA chest tube is seen in the right hemithorax.\nResidual right pneumothorax is noted.\nNo pleural effusion.\nImpression:\nChest tube in situ with residual right pneumothorax.",
        gold_labels=["Device Present"], neg_target="Post-Surgical Changes"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nThe bony rib cage shows old healed rib fractures on the left (4th-6th ribs).\nThe lung fields are clear.\nCardiac size is normal.\nImpression:\nOld healed left rib fractures. No acute lung pathology.",
        gold_labels=["Degenerative / Structural Changes"], neg_target="Post-Surgical Changes"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nNG tube artifact is noted with tip projected over the gastric region.\nThe heart is normal in size.\nBoth lung fields are clear.\nImpression:\nNG tube in situ. Normal chest radiograph otherwise.",
        gold_labels=["Device Present"], neg_target="Post-Surgical Changes"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nRight subclavian central venous catheter noted with tip in the SVC.\nThe cardiac shadow is enlarged (CTR - 0.60).\nBilateral pleural effusion is seen.\nImpression:\nCardiomegaly. Bilateral pleural effusion. Central line in situ.",
        gold_labels=["Cardiomegaly", "Pleural Effusion", "Device Present"], neg_target="Post-Surgical Changes"
    ),
]

# ============================================================================
# PULMONARY EDEMA (98 disagreements, 69% BERT over-prediction)
# Problem: BERT fires on "vascular" / "haze" / "bilateral opacity" in non-PE context
# ============================================================================
HARD_NEGATIVE_TEMPLATES["Pulmonary Edema"] = [
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nThe heart is enlarged (CTR - 0.63).\nThe pulmonary vasculature is normal.\nBoth lung fields are clear.\nBoth costophrenic sulci are preserved.\nImpression:\nCardiomegaly. No pulmonary edema.",
        gold_labels=["Cardiomegaly"], neg_target="Pulmonary Edema"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nBilateral patchy opacities are noted predominantly in the lower lung zones.\nThe cardiac size is normal.\nImpression:\nBilateral pneumonia.",
        gold_labels=["Pneumonia / Consolidation"], neg_target="Pulmonary Edema"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nThe cardiac shadow is enlarged.\nNormal pulmonary vasculature. No vascular congestion.\nThe lung fields are clear.\nImpression:\nCardiomegaly. No features of pulmonary edema.",
        gold_labels=["Cardiomegaly"], neg_target="Pulmonary Edema"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nDiffuse ground glass opacification noted bilaterally.\nThe cardiac size is normal.\nNo Kerley B lines. No cephalization.\nImpression:\nBilateral ground glass opacities - likely infective etiology. No features of pulmonary edema.",
        gold_labels=["Pneumonia / Consolidation"], neg_target="Pulmonary Edema"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe heart is normal in size.\nThe mediastinum, hila and pulmonary vasculature are unremarkable.\nBoth lung fields are clear.\nImpression:\nNormal study.",
        gold_labels=[], neg_target="Pulmonary Edema"
    ),
]

# ============================================================================
# DEVICE PRESENT (207 disagreements)
# Problem: BERT fires on artifact mentions even when described as absent
# ============================================================================
HARD_NEGATIVE_TEMPLATES["Device Present"] = [
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nThe heart is normal in size and contour.\nNo tubes or lines are seen.\nBoth lung fields are clear.\nImpression:\nNormal chest radiograph. No devices.",
        gold_labels=[], neg_target="Device Present"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe previously noted central venous catheter has been removed.\nThe lung fields are clear.\nNo residual pneumothorax.\nImpression:\nPost-line removal. No pneumothorax.",
        gold_labels=[], neg_target="Device Present"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe cardiac shadow is enlarged (CTR - 0.58).\nAortic unfolding is noted.\nNo lines, tubes or surgical artifacts seen.\nImpression:\nCardiomegaly with aortic unfolding.",
        gold_labels=["Cardiomegaly", "Degenerative / Structural Changes"], neg_target="Device Present"
    ),
]

# ============================================================================
# ATELECTASIS (173 disagreements)
# Problem: BERT fires on "volume loss" / "opacity" descriptions that aren't atelectasis
# ============================================================================
HARD_NEGATIVE_TEMPLATES["Atelectasis"] = [
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nHomogenous opacity in the right lower lung zone with meniscus sign.\nNo mediastinal shift. No volume loss.\nCardiac size is normal.\nImpression:\nRight pleural effusion.",
        gold_labels=["Pleural Effusion"], neg_target="Atelectasis"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nBilateral patchy opacities noted.\nNo loss of lung volume.\nBoth hemidiaphragms are at normal level.\nImpression:\nBilateral pneumonic changes.",
        gold_labels=["Pneumonia / Consolidation"], neg_target="Atelectasis"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe lung fields are clear with normal lung volumes bilaterally.\nThe heart is normal in size.\nBoth hemidiaphragms are at normal level.\nImpression:\nNormal study.",
        gold_labels=[], neg_target="Atelectasis"
    ),
]

# ============================================================================
# RETICULONODULAR / INTERSTITIAL PATTERN (202 disagreements)
# Problem: BERT fires on "markings" and "pattern" in normal context
# ============================================================================
HARD_NEGATIVE_TEMPLATES["Reticulonodular / Interstitial Pattern"] = [
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe heart is normal in size.\nNormal bronchovascular markings bilaterally.\nNo focal lung lesion is seen.\nImpression:\nNormal chest radiograph.",
        gold_labels=[], neg_target="Reticulonodular / Interstitial Pattern"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nPatchy air-space opacity noted in the right middle lung zone.\nBronchovascular markings are otherwise normal.\nImpression:\nRight middle lobe pneumonia.",
        gold_labels=["Pneumonia / Consolidation"], neg_target="Reticulonodular / Interstitial Pattern"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe lung fields show normal bronchovascular pattern.\nThe heart is enlarged (CTR - 0.56).\nImpression:\nCardiomegaly. Lungs are clear.",
        gold_labels=["Cardiomegaly"], neg_target="Reticulonodular / Interstitial Pattern"
    ),
]

# ============================================================================
# TUBERCULOSIS (102 disagreements)
# Problem: BERT fires on "upper lobe" / "fibrosis" / "cavitation" without TB context
# ============================================================================
HARD_NEGATIVE_TEMPLATES["Tuberculosis"] = [
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFindings:\nA round well-defined mass lesion is noted in the right upper lung zone.\nNo cavitation. No satellite nodules.\nImpression:\nRight upper lobe mass. Neoplastic process suspected.",
        gold_labels=["Lung Mass / Nodule", "Suspected Malignancy"], neg_target="Tuberculosis"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nFibrotic changes noted in both upper lung zones.\nTraction bronchiectasis seen bilaterally.\nThe cardiac shadow is normal.\nImpression:\nBilateral upper lobe fibrosis - likely post-radiation changes.",
        gold_labels=["Reticulonodular / Interstitial Pattern"], neg_target="Tuberculosis"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nThe lung fields are clear.\nNo active parenchymal disease.\nNo cavitary lesion seen.\nImpression:\nNormal chest radiograph.",
        gold_labels=[], neg_target="Tuberculosis"
    ),
]

# ============================================================================
# LUNG MASS / NODULE (71 disagreements)
# Problem: BERT fires on "opacity" / "lesion" in non-mass context
# ============================================================================
HARD_NEGATIVE_TEMPLATES["Lung Mass / Nodule"] = [
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nPatchy opacity is noted in the left lower lung zone.\nAir bronchogram is seen within the opacity.\nImpression:\nLeft lower lobe consolidation, likely pneumonia.",
        gold_labels=["Pneumonia / Consolidation"], neg_target="Lung Mass / Nodule"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nNo focal lung lesion is seen.\nThe lung fields are clear bilaterally.\nImpression:\nNormal chest radiograph.",
        gold_labels=[], neg_target="Lung Mass / Nodule"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nDiffuse reticulonodular opacities bilaterally.\nNo discrete mass or nodule identified.\nImpression:\nInterstitial lung disease.",
        gold_labels=["Reticulonodular / Interstitial Pattern"], neg_target="Lung Mass / Nodule"
    ),
]

# ============================================================================
# SUSPECTED MALIGNANCY (68 disagreements)
# Problem: BERT fires on "mass" mentions even in benign context
# ============================================================================
HARD_NEGATIVE_TEMPLATES["Suspected Malignancy"] = [
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nA well-defined rounded opacity is noted in the right lower lung zone.\nSmooth margins. No spiculation.\nImpression:\nBenign-appearing right lung nodule. Likely granuloma.",
        gold_labels=["Lung Mass / Nodule"], neg_target="Suspected Malignancy"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nCalcified granuloma noted in the left upper lung zone.\nNo active parenchymal disease.\nImpression:\nOld granulomatous disease. No active pathology.",
        gold_labels=[], neg_target="Suspected Malignancy"
    ),
    ReportTemplate(
        text="CHEST RADIOGRAPH REPORT\nLung fields are clear.\nNo mass, nodule or cavitary lesion seen.\nImpression:\nNormal study.",
        gold_labels=[], neg_target="Suspected Malignancy"
    ),
]

# ============================================================================
# OTHER SIGNIFICANT ABNORMALITY — no hard negatives needed (BERT-primary by design)
# ============================================================================


def generate_hard_negatives(
    n_per_label: int = 15,
    seed: int = 42,
) -> list:
    """
    Generate hard negative synthetic reports.

    For each label, templates are repeated/sampled to reach n_per_label.
    Small random variations are applied (header style, whitespace, minor phrasing)
    to prevent BERT from memorizing exact templates.

    Args:
        n_per_label: target number of hard negatives per label
        seed: random seed for reproducibility

    Returns:
        list of dicts with keys: study_id, report_text, gold_labels, is_synthetic, neg_target
    """
    random.seed(seed)

    # Header variations
    headers = [
        "CHEST RADIOGRAPH REPORT",
        "CHEST RADIOGRAPHY REPORT",
        "AP CHEST RADIOGRAPH REPORT",
        "CHEST RADIOGRAPH  REPORT",
        "CHEST XRAY REPORT",
    ]

    # Minor phrasing swaps (original -> alternatives)
    swaps = [
        ("normal in size and contour", "within normal limits"),
        ("normal in size", "within normal limits"),
        ("is enlarged", "appears enlarged"),
        ("are preserved", "are intact"),
        ("are normal", "appear normal"),
        ("is seen", "is noted"),
        ("is noted", "is observed"),
        ("No focal lung lesion is seen", "No focal lung lesion is identified"),
        ("Normal chest radiograph", "No significant radiographic abnormality"),
        ("Normal study", "No gross abnormality seen"),
        ("Both lung fields are clear", "The lung fields are clear"),
        ("The lung fields are clear", "Both lung fields are clear bilaterally"),
    ]

    all_reports = []
    study_counter = 90001  # start after generate_synthetic_reports.py range

    for label, templates in HARD_NEGATIVE_TEMPLATES.items():
        if not templates:
            continue

        logger.info(f"Generating {n_per_label} hard negatives for: {label} (from {len(templates)} templates)")

        for i in range(n_per_label):
            # Pick a template (cycle through available ones)
            template = templates[i % len(templates)]
            text = template.text

            # Apply random header variation
            for h in headers:
                if h in text:
                    text = text.replace(h, random.choice(headers), 1)
                    break

            # Apply 0-2 random phrase swaps
            n_swaps = random.randint(0, 2)
            swap_candidates = [(old, new) for old, new in swaps if old in text]
            for old, new in random.sample(swap_candidates, min(n_swaps, len(swap_candidates))):
                text = text.replace(old, new, 1)

            # Random minor whitespace variation
            if random.random() < 0.3:
                text = text.replace("\n", "\n\n", 1)

            gold_str = ",".join(template.gold_labels) if template.gold_labels else "Normal"

            all_reports.append({
                "study_id": f"hard_neg_{study_counter:05d}",
                "report_text": text,
                "gold_labels": gold_str,
                "is_synthetic": True,
                "neg_target": template.neg_target,
            })
            study_counter += 1

    return all_reports


def main():
    reports = generate_hard_negatives(n_per_label=15, seed=42)

    logger.info(f"\nGenerated {len(reports)} hard negative reports")

    # Count by target label
    from collections import Counter
    target_counts = Counter(r["neg_target"] for r in reports)
    for label, count in sorted(target_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {label}: {count} hard negatives")

    # Count by gold label
    gold_counts = Counter()
    for r in reports:
        for lbl in r["gold_labels"].split(","):
            gold_counts[lbl.strip()] += 1
    logger.info("\nGold label distribution in hard negatives:")
    for label, count in sorted(gold_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {label}: {count}")

    # Save
    output_dir = PROJECT_ROOT / "results" / "nlp"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hard_negatives.csv"

    import pandas as pd
    df = pd.DataFrame(reports)
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved to {output_path}")

    # Also create merged training set
    # Load existing augmented data
    augmented_path = output_dir / "training_reports_augmented.csv"
    if augmented_path.exists():
        aug_df = pd.read_csv(augmented_path)
        if "neg_target" not in aug_df.columns:
            aug_df["neg_target"] = ""
        merged = pd.concat([aug_df, df], ignore_index=True)
        merged_path = output_dir / "training_reports_v2.csv"
        merged.to_csv(merged_path, index=False)

        n_real = int((~merged["is_synthetic"].astype(bool)).sum())
        n_synth_pos = int(merged["is_synthetic"].astype(bool).sum()) - len(reports)
        n_hard_neg = len(reports)

        logger.info(f"\nMerged training set saved to {merged_path}")
        logger.info(f"  Real: {n_real}")
        logger.info(f"  Synthetic positive (rare labels): {n_synth_pos}")
        logger.info(f"  Hard negatives: {n_hard_neg}")
        logger.info(f"  Total: {len(merged)}")
    else:
        logger.info(f"\nNo existing augmented data found at {augmented_path}")
        logger.info("Hard negatives saved standalone. Merge manually before training.")


if __name__ == "__main__":
    main()
