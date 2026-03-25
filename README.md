# Nigerian Chest X-ray AI Benchmark — Phase 1

> **Evaluation-only benchmark** for assessing commercial AI chest X-ray models against a Nigerian/African patient population.
> Solo research project at Federal Medical Centre (FMC) Ebute-Metta, Lagos.
> Target venue: MICCAI / MIDL 2027.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Ethics & Data Governance](#2-ethics--data-governance)
3. [Compute Environment](#3-compute-environment)
4. [Label System](#4-label-system)
5. [NLP Pipeline — Phase A (Weeks 6–8)](#5-nlp-pipeline--phase-a-weeks-68)
   - 5.1 [Rule-Based Extractor (Week 6)](#51-rule-based-extractor-week-6)
   - 5.2 [BioClinicalBERT Extractor (Week 7)](#52-bioclinicalbert-extractor-week-7)
   - 5.3 [Synthetic Augmentation & Ensemble (Week 8)](#53-synthetic-augmentation--ensemble-week-8)
6. [Results Summary](#6-results-summary)
7. [Full Dataset Statistics (5,517 Reports)](#7-full-dataset-statistics-5517-reports)
8. [Directory Structure](#8-directory-structure)
9. [Installation & Setup](#9-installation--setup)
10. [Running the Pipeline](#10-running-the-pipeline)
11. [Testing](#11-testing)
12. [CI/CD](#12-cicd)
13. [Project Roadmap](#13-project-roadmap)

---

## 1. Project Overview

This project constructs the first publicly documented Nigerian chest X-ray benchmark for evaluating AI diagnostic models. Phase 1 covers NLP-based label extraction from free-text radiology reports; subsequent phases will cover model evaluation, cross-dataset experiments, and interpretability.

**Scope (Phase 1):**

| Component | Detail |
|---|---|
| Dataset | 5,517 anonymised Nigerian CXR radiology reports |
| Models to evaluate (Phase D) | DenseNet-121, CheXNet, EfficientNet-B4 |
| Comparison datasets (Phase E) | CheXpert, MIMIC-CXR, VinDr-CXR |
| Interpretability (Phase F) | Grad-CAM |
| Domain adaptation | Deferred to Phase 2 |

---

## 2. Ethics & Data Governance

- **Ethics approval:** FMC Ebute-Metta Research Ethics Committee — approved **March 3, 2026**
- **Anonymisation:** Patient identifiers are replaced with deterministic hashed IDs using HMAC-SHA256. The anonymisation salt (`CXR_ANON_SECRET`) is stored as an encrypted environment variable under PI control and is **never committed to this repository**.
- **Storage:** Raw patient data lives in a VeraCrypt-encrypted volume (`workspace/`) tracked with DVC. Only derived, non-identifiable artefacts are committed to Git.

---

## 3. Compute Environment

| Item | Spec |
|---|---|
| GPU | NVIDIA GTX 1070 — 8 GB VRAM (Pascal / CUDA 6.1) |
| CUDA | 13.0 (driver 581.57) |
| PyTorch | 2.10.0+cu126 |
| Python | 3.10+ |
| OS | Windows 11 Pro (development); Ubuntu (CI via GitHub Actions) |

**Thermal constraint:** Sustained GPU load causes thermal shutdown. All training scripts implement epoch + fold level checkpoint/resume so any interrupted run can continue from the last saved state.

---

## 4. Label System

Reports are labelled with a **3-tier hierarchy** (defined in `configs/labels/hierarchy.yaml`):

| Tier | Purpose | Values |
|---|---|---|
| **1** | Clinical state | Normal / Abnormal / Indeterminate |
| **2** | Pathology category | Infectious, Cardiac, Pleural, Parenchymal, Neoplastic, Iatrogenic, Other |
| **3** | Specific finding (multi-label) | 13 pathologies — see below |

**Tier 3 labels (13 pathologies):**

| Label | Ensemble method | Notes |
|---|---|---|
| Cardiomegaly | Rule-primary | Most prevalent (30.3%) |
| Degenerative / Structural Changes | Rule-primary | |
| Pleural Effusion | Rule-primary | |
| Pneumonia / Consolidation | Rule-primary | |
| Atelectasis | Rule-primary | |
| Reticulonodular / Interstitial Pattern | Rule-primary | |
| Tuberculosis | Rule-primary | |
| Device Present | Rule-primary | |
| Pulmonary Edema | Rare-unstable (threshold 0.85) | BERT over-predicts; ensemble filters |
| Lung Mass / Nodule | Rare-unstable (threshold 0.85) | |
| Post-Surgical Changes | Rare-unstable (threshold 0.85) | Only 3 real training cases |
| Other Significant Abnormality | BERT-primary | Rules get F1=0.000 on this |
| Suspected Malignancy | BERT-primary | |

---

## 5. NLP Pipeline — Phase A (Weeks 6–8)

Three sequential milestones build toward the final ensemble label extraction.

### 5.1 Rule-Based Extractor (Week 6)

**File:** `src/nlp/labels.py`

A deterministic, pattern-matching extractor using the bilingual (English/Yoruba/Pidgin) phrase dictionary in `src/nlp/dictionary/bilingual_dict.json`.

**Key design decisions:**
- Section-aware parsing: separates Findings, Impression, and Clinical History sections before applying rules, preventing cross-section scope leakage.
- Negation scope: sentence-boundary negation with carve-outs for double negatives.
- Uncertainty handling: `?` patterns and hedging phrases map to an `indeterminate` flag rather than a positive label.
- Cardiothoracic ratio (CTR) parsing: numeric extraction with fallback to qualitative descriptors.
- Pulmonary embolism scope: disambiguates "PE" in historical vs. active contexts.

**Dictionary:** 429 bilingual phrases (expanded from 343 during Week 6 bug-fixing).

**Evaluation (497 gold-labelled reports):**

| Metric | Score |
|---|---|
| Weighted F1 | 0.797 |
| Macro F1 | 0.674 |
| Exact match | 75.9% |

**Tests:** 56 unit tests in `tests/test_rule_based_extractor.py` — all passing.

---

### 5.2 BioClinicalBERT Extractor (Week 7)

**Files:** `src/nlp/extractors/bioclinicalbert_extractor.py`, `scripts/nlp/train_bert_extractor.py`

Fine-tuned `emilyalsentzer/Bio_ClinicalBERT` (110M parameters) for multi-label classification.

**Architecture:**
```
Bio_ClinicalBERT encoder
    └── [CLS] token representation (768-dim)
        └── Dropout(p=0.1)
            └── Linear(768 → 13)
                └── Sigmoid activation (per-label probabilities)
```

**Training protocol:**
- 5-fold `MultilabelStratifiedKFold` cross-validation on 497 gold reports
- Loss: `BCEWithLogitsLoss` with per-fold `pos_weight` (capped at 50×) for class imbalance
- Optimiser: AdamW, lr=2e-5, linear warmup (10% of steps)
- Early stopping: patience=3 on macro F1 (validation fold)
- Mixed precision: `torch.amp` (FP16 forward pass, FP32 optimiser)
- Checkpoint/resume: saves at epoch + fold level

**Out-of-fold (OOF) evaluation (497 real reports):**

| Metric | Score |
|---|---|
| Weighted F1 | 0.740 |
| Macro F1 | 0.516 |
| Macro AUROC | 0.937 |
| Macro PR-AUC | 0.559 |
| Exact match | 67.2% |

BERT outperformed rules on Other Significant Abnormality (0.000 → 0.579) and Suspected Malignancy (0.000 → 0.174). Rules won on 11/13 labels.

---

### 5.3 Synthetic Augmentation & Ensemble (Week 8)

#### Synthetic Report Generation

**File:** `scripts/nlp/generate_synthetic_reports.py`

Template-based generation targeting 6 rare labels that had fewer than 20 real training cases:

| Label | Real cases | After augmentation |
|---|---|---|
| Post-Surgical Changes | 3 | 40 |
| Lung Mass / Nodule | 4 | 40 |
| Pulmonary Edema | 5 | 40 |
| Suspected Malignancy | 5 | 40 |
| Tuberculosis | 13 | 40 |
| Atelectasis | 20 | 40 |

190 synthetic reports were generated. Co-occurrence patterns were preserved (30–35% Cardiomegaly/PE/Device comorbidity) for clinical realism. Synthetic reports were injected into **training folds only**; all evaluation used real reports exclusively.

#### Augmented BERT Retrain

Retraining with 687 examples (497 real + 190 synthetic) per fold (validation remains real-only):

| Metric | Week 7 | Week 8 | Delta |
|---|---|---|---|
| Weighted F1 | 0.740 | **0.800** | +0.060 |
| Macro F1 | 0.516 | **0.634** | +0.118 |
| Macro AUROC | 0.937 | **0.971** | +0.034 |
| Macro PR-AUC | 0.559 | **0.693** | +0.134 |
| Exact match | 67.2% | **70.2%** | +3.0% |

Per-label gains (augmented labels marked ★):

| Label | Wk7 F1 | Wk8 F1 | Delta |
|---|---|---|---|
| ★ Atelectasis | 0.154 | 0.727 | +0.573 |
| ★ Pulmonary Edema | 0.000 | 0.533 | +0.533 |
| ★ Suspected Malignancy | 0.174 | 0.667 | +0.493 |
| ★ Lung Mass / Nodule | 0.000 | 0.333 | +0.333 |
| ★ Tuberculosis | 0.421 | 0.649 | +0.228 |
| Pneumonia / Consolidation | 0.513 | 0.720 | +0.207 |
| Reticulonodular / Interstitial | 0.390 | 0.644 | +0.254 |
| Cardiomegaly | 0.869 | 0.908 | +0.039 |
| ★ Post-Surgical Changes | 0.000 | 0.000 | 0.000 (3 real cases) |

#### Ensemble Design

**File:** `src/nlp/extractors/ensemble.py`

A method-priority-aware ensemble combining rule-based and augmented BERT predictions:

```
Label type          → Decision logic
────────────────────────────────────────────────────────────────────
RULE_PRIMARY        → Trust rule output; BERT used to detect misses
BERT_PRIMARY        → Trust BERT; rules used as filter (low FP rate)
RARE_UNSTABLE       → Require BERT probability ≥ 0.85 for inclusion
                      (conservative threshold; BERT over-predicts these)
```

Multi-fold BERT averaging: probabilities averaged across all 5 fold models before thresholding.

---

## 6. Results Summary

Final method comparison on 497 gold-labelled reports:

| Method | Macro F1 | Weighted F1 | Best on |
|---|---|---|---|
| Rule-based (Week 6) | 0.630 | 0.795 | 11 / 13 labels |
| BERT original (Week 7) | 0.516 | 0.740 | 2 / 13 labels |
| BERT augmented (Week 8) | 0.634 | 0.800 | Matches rules aggregate |
| Ensemble (final) | — | — | Used for full dataset |

---

## 7. Full Dataset Statistics (5,517 Reports)

**File:** `results/nlp/ensemble_predictions.csv`

| Split | Count | % |
|---|---|---|
| Normal | 2,359 | 42.8% |
| Abnormal | 3,158 | 57.2% |

Labels per report: mean=1.05, median=1.0, max=6.

**Label prevalence:**

| Label | Ensemble count | % |
|---|---|---|
| Cardiomegaly | 1,670 | 30.3% |
| Degenerative / Structural Changes | 1,322 | 24.0% |
| Pleural Effusion | 739 | 13.4% |
| Pneumonia / Consolidation | 449 | 8.1% |
| Device Present | 434 | 7.9% |
| Other Significant Abnormality | 361 | 6.5% |
| Reticulonodular / Interstitial | 294 | 5.3% |
| Atelectasis | 200 | 3.6% |
| Tuberculosis | 166 | 3.0% |
| Suspected Malignancy | 70 | 1.3% |
| Lung Mass / Nodule | 39 | 0.7% |
| Pulmonary Edema | 32 | 0.6% |
| Post-Surgical Changes | 12 | 0.2% |

**Disagreement analysis:**

- 2,776 disagreements across 1,611 reports (29.2% of dataset)
- 82.9% BERT-only (BERT fires, rules do not); 17.1% rules-only
- BERT over-prediction filtered by ensemble: Post-Surgical 89%, Pulmonary Edema 69%, Lung Mass 51%
- Confidence: high=96.1%, medium=1.2%, low=2.7%

---

## 8. Directory Structure

```
nigeria_cxr_benchmark_phase1/
│
├── .github/
│   └── workflows/
│       └── python_ci.yml               # CI: lint, test (host + Docker)
│
├── configs/
│   ├── defaults.yaml
│   └── labels/
│       └── hierarchy.yaml              # 3-tier label hierarchy definition
│
├── docs/
│   ├── annotation_guidelines.md        # Gold label annotation protocol
│   ├── nigerian_specific_terms.md      # Local clinical vocabulary
│   └── week6_evaluation_report.md      # Rule-based extractor evaluation report
│
├── notebooks/
│   └── 02_label_extraction.ipynb       # Exploratory analysis notebook
│
├── results/
│   ├── annotated_reports.csv           # 497 gold-labelled reports (ground truth)
│   └── nlp/
│       ├── bert_evaluation.json        # Per-fold BERT metrics + augmentation metadata
│       ├── bert_oof_predictions.csv    # Out-of-fold BERT predictions (497 reports)
│       ├── bert_thresholds.json        # Per-label optimal thresholds from CV
│       ├── synthetic_reports.csv       # 190 template-generated rare-label reports
│       ├── training_reports_augmented.csv  # 687 reports (497 real + 190 synthetic)
│       ├── ensemble_predictions.csv    # Final labels for 5,517 reports
│       ├── disagreement_log.csv        # 2,776 RB vs BERT disagreement records
│       └── label_distribution.json    # Prevalence + co-occurrence statistics
│
├── scripts/
│   └── nlp/
│       ├── generate_synthetic_reports.py   # Rare label template augmentation
│       ├── train_bert_extractor.py         # 5-fold CV BERT training + augmentation
│       └── run_extraction.py              # Full dataset ensemble extraction (checkpoint/resume)
│
├── src/
│   ├── data/
│   │   ├── anonymize_data.py           # HMAC-SHA256 patient ID hashing
│   │   ├── preprocess_data.py
│   │   ├── validate_data.py
│   │   └── .gitignore                  # Excludes raw patient data files
│   │
│   ├── nlp/
│   │   ├── labels.py                   # Rule-based extractor (section-aware, bilingual)
│   │   ├── dictionary/
│   │   │   └── bilingual_dict.json     # 429 English/Yoruba/Pidgin phrases
│   │   ├── extractors/
│   │   │   ├── __init__.py
│   │   │   ├── bioclinicalbert_extractor.py  # BERT classifier + inference
│   │   │   ├── ensemble.py             # Method-priority-aware ensemble logic
│   │   │   └── spacy_ner_extractor.py
│   │   └── utils/
│   │       ├── aggregate_reports.py
│   │       └── label_aggregate_reports.py
│   │
│   ├── evaluation/                     # Stub — Phase D (model evaluation)
│   └── interpretability/               # Stub — Phase F (Grad-CAM)
│
├── tests/
│   ├── test_rule_based_extractor.py    # 56 unit tests (all passing)
│   ├── test_anonymize_data.py
│   ├── test_preprocess_data.py
│   └── test_validate_data.py
│
├── workspace/                          # DVC-tracked, VeraCrypt-encrypted patient data
│   └── data/preprocessed_data/
│       ├── arrays/                     # Preprocessed image arrays
│       ├── images/                     # Raw CXR images
│       ├── metadata/                   # Patient metadata (anonymised)
│       ├── reports/                    # Raw radiology reports (anonymised)
│       └── qc_logs/                    # Quality control logs
│
├── annotated_reports_gen.py            # Gold label annotation helper script
├── generate_anon_key.py               # One-time secret key generator
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

> **Not committed:** `models/` (fine-tuned BERT weights, ~2 GB), `CLAUDE.md` (internal project context), `pytest.ini`, `run_pipeline.py`.
> Models are stored locally and re-trainable from `scripts/nlp/train_bert_extractor.py`.

---

## 9. Installation & Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (GTX 1070 or better recommended; CPU fallback available but slow)
- Docker (for CI parity)

### Environment

```bash
python -m venv cxr_env
source cxr_env/bin/activate          # Linux/macOS
# or: cxr_env\Scripts\activate       # Windows

pip install -r requirements.txt
```

### Environment Variables

The anonymisation secret must be set before running data preprocessing:

```bash
# Linux/macOS
export CXR_ANON_SECRET="your_64_char_hex_key"

# Windows PowerShell
$env:CXR_ANON_SECRET = "your_64_char_hex_key"
```

Generate a new key with:

```bash
python generate_anon_key.py
```

> Store the key in your password manager. It is required to reproduce any anonymised IDs. Do not commit it.

---

## 10. Running the Pipeline

### Rule-based extraction only

```python
from src.nlp.labels import rule_based_extractor, extract_with_scores

labels = rule_based_extractor(report_text)           # returns List[str]
result = extract_with_scores(report_text)            # returns Dict with evidence
```

### Train BioClinicalBERT (5-fold CV)

```bash
python scripts/nlp/train_bert_extractor.py \
    --data-path results/annotated_reports.csv \
    --output-dir models/ \
    --epochs 10 \
    --folds 5
```

Resume after interruption: re-run the same command; the script detects existing checkpoints and skips completed folds/epochs.

### Generate synthetic reports

```bash
python scripts/nlp/generate_synthetic_reports.py \
    --output results/nlp/synthetic_reports.csv
```

### Run full dataset extraction (ensemble)

```bash
python scripts/nlp/run_extraction.py \
    --reports-dir workspace/data/preprocessed_data/reports/ \
    --model-dir models/ \
    --output results/nlp/ensemble_predictions.csv
```

Checkpoint/resume is built in — interrupted runs continue from the last processed report.

---

## 11. Testing

```bash
pytest tests/ -v
```

Tests cover the rule-based extractor (56 tests), anonymisation, preprocessing, and validation. All tests are designed to run without GPU and without patient data.

The CI Docker image runs the same test suite in an isolated Ubuntu environment.

---

## 12. CI/CD

GitHub Actions workflow: `.github/workflows/python_ci.yml`

| Step | What it does |
|---|---|
| Free disk space | Removes unused toolchains to make room for Docker |
| Cache pip | Speeds up host-level installs |
| Install host deps | `pytest`, `pytest-cov`, `flake8`, `requirements.txt` |
| Build Docker image | Builds `ncb-phase1:latest` from `Dockerfile` |
| Run tests in Docker | `pytest -v tests/` with `CXR_ANON_SECRET` forwarded |
| Lint with flake8 | Strict on syntax/critical errors; relaxed on style |
| Run tests with coverage | Host-level pytest + coverage XML artefact |

The `CXR_ANON_SECRET` is stored as a **GitHub repository secret** (Settings → Secrets and variables → Actions) and injected at runtime — it is never stored in the workflow file or repository.

---

## 13. Project Roadmap

| Phase | Description | Weeks | Period |
|---|---|---|---|
| **A** | NLP Label Extraction | 6–12 | Mar–Apr 2026 |
| **B** | Clinician Validation | 13–18 | Apr–May 2026 |
| **C** | Dataset Card + Quality Gate 2 | 19–21 | Jun 2026 |
| **D** | Model Evaluation | 22–28 | Jul–Aug 2026 |
| **E** | Cross-Dataset Experiments | 29–34 | Aug–Oct 2026 |
| **F** | Interpretability (Grad-CAM) | 35–38 | Oct 2026 |
| **G** | Paper Writing | 39–43 | Nov–Dec 2026 |
| **H** | Submission & Revision | 44–48 | Dec 2026 |

**Current status:** Week 9 (March 2026) — Phase A. Quality Gate 1 passed. Full dataset labelled (5,517 reports). Next: label quality dashboard and clinician sample review.

---

## Known Issues (for clinician adjudication — Phase B)

- **Post-Surgical Changes:** Only 3 real training cases; BERT F1=0.000; BERT hallucinates 109 positives (ensemble filters to 12).
- **Pulmonary Edema:** 5 real cases; BERT over-predicts 102 (ensemble filters to 32).
- **Other Significant Abnormality:** Rules get F1=0.000; 361 full-dataset detections are BERT-primary only.
- ~5 rotation-qualified cardiomegaly cases misclassified.
- "Prominent bronchovascular markings" triggers Reticulonodular false positives.
- TB over-triggers on cavitation when radiologist suspects malignancy or abscess.
- 1,353 reports contain "normal" phrasing but are classified abnormal — require clinical review.
