# Nigerian-Specific Terms for Chest X-ray Report Extraction

## Overview
This document defines terms, phrases, and abbreviations commonly found in Nigerian radiology reports but less emphasized in global datasets (e.g., CheXpert, MIMIC-CXR). These are influenced by local disease prevalence (e.g., TB), institutional styles (e.g., FMC Ebute-Metta), and linguistic variations (e.g., "CP angle" for costophrenic). Terms are grouped by Tier-3 labels (from `configs/labels/hierarchy.yaml`) for easy integration into `bilingual_dict.json` and extractors.py.

**Rationale**: Enhances NLP recall for domain-specific shifts (Research Dashboard: NLP & Labeling). Source: Compiled reports analysis + protocol literature (Section 2.3).

## Terms by Label

### Normal
- "No active lung lesion" (common negative phrasing for clear chests).
- "Unremarkable chest" (shorthand for no abnormality).
- "Normal study" (brief conclusion in busy clinics).

### Pneumonia / Consolidation
- "Basal pneumonic changes" (frequent in lower lobe infections).
- "Air bronchogram sign" (descriptive for consolidation in Nigerian reports).
- "Homogeneous opacity" (common for airspace filling).

### Tuberculosis
- "Koch's" or "koch's disease" (local abbreviation for TB, from Robert Koch).
- "Apical fibrosis with cavitation" (post-TB sequelae, high prevalence).
- "Fibrocalcific changes" (chronic TB indicator in African settings).
- "Upper zone fibrosis" (zonal description for TB suspicion).

### Pleural Effusion
- "CP angle blunting" (costophrenic angle, abbreviated in reports).
- "Meniscus sign" (classic for effusion in local teaching).
- "Layering effusion" (descriptive for supine views).

### Cardiomegaly
- "Increased CTR" (cardiothoracic ratio, common metric in Nigerian cardiology).
- "Left ventricular preponderance" (hypertensive heart disease indicator).

### Pulmonary Edema
- "Bat-wing opacity" (perihilar pattern in heart failure).
- "Upper lobe diversion" (vascular redistribution in local heart cases).

### Atelectasis
- "Segmental collapse" (location-specific, e.g., basal in bedridden patients).
- "Elevated hemidiaphragm" (post-surgical or phrenic nerve issue).

### Reticulonodular / Interstitial Pattern
- "Reticulo-nodular opacities" (TB or HIV-related interstitial changes).
- "Interstitial markings" (chronic lung disease descriptor).

### Lung Mass / Nodule
- "Coin lesion" (solitary nodule suspicion in smokers/TB).

### Suspected Malignancy
- "Mass lesion ?ca" (?carcinoma, shorthand in reports).

### Degenerative / Structural Changes
- "Dorsal spondylosis" (thoracic spine degeneration, common in elderly).

### Device Present
- "ET tube" (endotracheal, ICU common).

### Post-Surgical Changes
- "Sternotomy wires" (post-CABG in cardiac cases).

### Other Significant Abnormality
- "Aortic unfolding" (atherosclerotic change in hypertension).

## Integration Notes
- Add to `bilingual_dict.json` for fuzzy matching (e.g., lower-case, partial).
- Use in annotation guidelines for clinician training.
- Update based on monthly QC reviews (Protocol Section 3.11).

Last updated: February 2026