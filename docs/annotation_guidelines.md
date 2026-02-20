# Annotation Guidelines for Nigerian Chest X-ray Benchmark Labels

## Purpose
To ensure consistent, reproducible labeling of de-identified reports using the 3-tier schema (Appendix A). Targets: ≥95% agreement on Tier-3 labels; Cohen’s κ ≥0.85 on 10% dual-annotated subset. For use in NLP prototyping (February milestone) and clinician review.

## General Rules
- **Source**: Report text only (Findings + Impression).
- **Multi-label**: Allowed (e.g., Pneumonia + Pleural Effusion).
- **Uncertainty**: If "possible" or ambiguous, flag for clinician (Tier-1 = Abnormal + note).
- **Negation**: Do not assign if negated (e.g., "no effusion").
- **Nigerian-Specific**: Use defined terms (docs/nigerian_specific_terms.md).
- **Hierarchy**: Derive Tier-1/2 from Tier-3 using `labels.py`.
- **Tools**: Bilingual dictionary + BioClinicalBERT for proposals; clinician adjudication monthly.

## Label-Specific Rules

| Tier-3 Label                        | Positive Triggers (Phrases/Synonyms) | Negative Triggers | Notes / Nigerian Examples |
|-------------------------------------|--------------------------------------|-------------------|---------------------------|
| Normal                              | "normal study", "no active lung lesion", "unremarkable", "clear lung fields" | Any Tier-3 positive | Only if no abnormality; common in Nigerian "no gross abnormality". |
| Pneumonia / Consolidation           | "consolidation", "pneumonic changes", "airspace opacity", "basal pneumonic" | "clear lungs" | Often co-occurs with TB; e.g., "homogeneous opacity" in reports. |
| Tuberculosis                        | "tb", "koch's", "apical fibrosis", "cavitating lesion", "miliary shadows" | "no tb features" | High prevalence — err on suspicion; e.g., "fibrocalcific changes". |
| Pleural Effusion                    | "pleural effusion", "cp angle blunting", "meniscus sign" | "preserved sulci" | Unilateral/bilateral; e.g., "layering effusion" in local phrasing. |
| Cardiomegaly                        | "cardiomegaly", "increased ctr", "enlarged heart" | "normal ctr" | Common in hypertension; e.g., "ctr > 0.5". |
| Pulmonary Edema                     | "pulmonary edema", "bat-wing opacity", "kerley b lines" | "normal vasculature" | Heart failure-related; e.g., "upper lobe diversion". |
| Atelectasis                         | "atelectasis", "collapse", "volume loss" | "no collapse" | Post-surgical common; e.g., "segmental collapse". |
| Reticulonodular/Interstitial Pattrn | "reticulonodular", "interstitial markings", "fibrotic changes" | "clear interstitium" | Chronic/TB-related; e.g., "reticulo-nodular opacities". |
| Lung Mass / Nodule                  | "mass", "nodule", "coin lesion" | "no mass" | Suspicion trigger; e.g., "rounded opacity". |
| Suspected Malignancy                | "suspicious for malignancy", "?ca", "neoplastic lesion" | "benign" | Rare but critical; e.g., "mass ?carcinoma". |
| Degenerative / Structural Changes   | "degenerative changes", "spondylosis", "kyphosis" | "normal spine" | Age-related; e.g., "dorsal spondylosis". |
| Device Present                      | "et tube", "ng tube", "pacemaker" | "no devices" | ICU common; e.g., "central line". |
| Post-Surgical Changes               | "post thoracotomy", "sternotomy wires" | "no surgical changes" | CABG frequent; e.g., "surgical clips". |
| Other Significant Abnormality       | "atypical opacity", "incidental finding" | N/A | Catch-all; e.g., "aortic unfolding".

## Quality Control
- **10% Dual-Annotation**: Random subset → compute κ ≥0.85.
- **Clinician Adjudication**: Monthly meetings for flagged cases.
- **Disagreement Resolution**: Senior radiologist (Dr. Wahab/Dr. Latifat) decides.
- **Tools**: Use extractors.py for proposals; validate_data.py for integrity.

Last updated: February 2026