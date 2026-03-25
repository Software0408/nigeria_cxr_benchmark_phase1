# src/nlp/utils/label_aggregate_reports.py
"""
Applies multi-model label extraction, ensemble voting, hierarchy derivation, and quality dashboard to aggregated reports.
- Input: Compiled Excel from aggregate_reports.py (e.g., 'results/compiled_reports_readable.xlsx')
- Output: Labeled Excel and dashboard plots
- Integrates with NLP & Labeling dashboard section

Usage:
    python src/nlp/utils/label_aggregate_reports.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from src.nlp.labels import LabelHierarchy
from src.nlp.extractors.bioclinicalbert_extractor import bioclinicalbert_extractor
from src.nlp.extractors.spacy_ner_extractor import spacy_ner_extractor
from datetime import datetime
from pathlib import Path
from tqdm import tqdm  # For progress bars
import logging  # For warnings
from src.nlp.extractors.spacy_ner_extractor import _post_filter_labels

logging.basicConfig(level=logging.INFO)  # Configure logging

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# Ensemble voting function (unchanged)
def ensemble_voting(rule_out: list, bert_out: list, spacy_out: list) -> list:
    # Evidence-first: keep high-precision sources
    evidence = set(rule_out) | set(spacy_out)

    # For now, block BERT-only labels (we’ll re-enable with confidence in Step 2)
    detected = sorted(evidence)
    return detected

# Batch labeling function
def label_aggregate_reports(input_path: str = f'results/compiled_reports_readable_{TIMESTAMP}.xlsx', output_path: str = f'results/labeled_reports_{TIMESTAMP}.xlsx'):
    lh = LabelHierarchy()
    df = pd.read_excel(input_path)  # Assumes 'study_id' and 'report_text' columns from aggregate_reports.py
    
    print(f"Labeling {len(df)} reports using multi-model extractors...")
    
    # Enable progress bars for Pandas apply operations
    tqdm.pandas(desc="Applying rule-based extractor")
    df['rule_based'] = df['report_text'].progress_apply(lh.rule_based_extractor)
    df['rule_based'] = df.apply(
    lambda r: _post_filter_labels(r['report_text'], r['rule_based']) if isinstance(r['report_text'], str) else r['rule_based'],
    axis=1)
    
    tqdm.pandas(desc="Applying BERT extractor")
    df['bert'] = df['report_text'].progress_apply(bioclinicalbert_extractor)
    
    tqdm.pandas(desc="Applying spaCy extractor")
    df['spacy'] = df['report_text'].progress_apply(spacy_ner_extractor)
    df['spacy'] = df.apply(lambda r: _post_filter_labels(r['report_text'], r['spacy']), axis=1)
    
    tqdm.pandas(desc="Applying ensemble voting")
    df['ensemble_pathologies'] = df.progress_apply(lambda row: ensemble_voting(row['rule_based'], row['bert'], row['spacy']), axis=1)
    
    tqdm.pandas(desc="Deriving hierarchy")
    df['hierarchy'] = df['ensemble_pathologies'].progress_apply(lambda p: lh.get_hierarchy(p))
    
    # ────────────────────────────────────────────────
    #  Compute pair-wise agreements HERE, BEFORE review columns
    # ────────────────────────────────────────────────
    def agreement(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if a | b else 1.0

    df['rule_bert_agree'] = df.apply(lambda row: agreement(row['rule_based'], row['bert']), axis=1)
    df['rule_spacy_agree'] = df.apply(lambda row: agreement(row['rule_based'], row['spacy']), axis=1)
    df['bert_spacy_agree'] = df.apply(lambda row: agreement(row['bert'], row['spacy']), axis=1)

    # Now compute review columns
    # Now compute review columns
# Keep agreement_score for logging/plots, but DO NOT use it to trigger reviews
    df['agreement_score'] = df[['rule_bert_agree', 'rule_spacy_agree', 'bert_spacy_agree']].mean(axis=1)

    # Evidence-first review logic (rule vs spaCy is what we trust now)
    HIGH_RISK = {
        "Tuberculosis",
        "Lung Mass / Nodule",
        "Suspected Malignancy",
    }

    df['has_high_risk'] = df['ensemble_pathologies'].apply(lambda xs: any(x in HIGH_RISK for x in xs))

    # Flag reports needing review (tight + meaningful)
    df['needs_review'] = (
        (df['rule_spacy_agree'] < 0.5) |
        (df['ensemble_pathologies'].apply(len) >= 3) |
        (df['has_high_risk']) |
        (df['word_count'] > 120)
    )

    # Review priority (3 = urgent)
    df['review_priority'] = 0

    # Priority 3: High-risk OR major rule-vs-spacy conflict
    df.loc[df['has_high_risk'], 'review_priority'] = 3
    df.loc[(df['rule_spacy_agree'] < 0.3), 'review_priority'] = 3

    # Priority 2: moderate conflict or lots of labels
    df.loc[(df['rule_spacy_agree'] >= 0.3) & (df['rule_spacy_agree'] < 0.5), 'review_priority'] = 2
    df.loc[(df['ensemble_pathologies'].apply(len) >= 3) & (df['review_priority'] < 2), 'review_priority'] = 2

    # Priority 1: everything else that still got flagged
    df.loc[df['needs_review'] & (df['review_priority'] == 0), 'review_priority'] = 1

    # Sort by priority descending
    df = df.sort_values('review_priority', ascending=False)

    # Save
    df.to_excel(output_path, index=False)
    print(f"\nLabeled XLSX created → {output_path}")
    print("Columns added: rule_based, bert, spacy, ensemble_pathologies, hierarchy")
    print("Review columns added: agreement_score, needs_review, review_priority")
    print("Use this for benchmarking, interpretability, and targeted manual review.")
    
    return df

# Label quality dashboard function (unchanged)
def generate_quality_dashboard(df: pd.DataFrame, output_dir: str = 'results/'):
    print("Generating label quality dashboard...")
    
    # Agreement (Jaccard similarity for set overlap)
    def agreement(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if a | b else 1.0
    
    df['rule_bert_agree'] = df.apply(lambda row: agreement(row['rule_based'], row['bert']), axis=1)
    df['rule_spacy_agree'] = df.apply(lambda row: agreement(row['rule_based'], row['spacy']), axis=1)
    df['bert_spacy_agree'] = df.apply(lambda row: agreement(row['bert'], row['spacy']), axis=1)
    
    avg_agree = df[['rule_bert_agree', 'rule_spacy_agree', 'bert_spacy_agree']].mean()
    print("Average Agreements:")
    print(avg_agree)
    
    # Bar plot
    avg_agree.plot(kind='bar')
    plt.title('Average Agreement Across Extractors (Jaccard Similarity)')
    plt.ylabel('Similarity Score')
    plt.xlabel('Extractor Pairs')
    plt.savefig(f'{output_dir}label_quality_agreement_{TIMESTAMP}.png')
    plt.close()
    print(f"Dashboard plot saved → {output_dir}label_quality_agreement_{TIMESTAMP}.png")
    print("Low agreement scores may indicate cases for manual review in active learning (Week 7).")

# Main execution
if __name__ == "__main__":
    df = label_aggregate_reports()
    generate_quality_dashboard(df)