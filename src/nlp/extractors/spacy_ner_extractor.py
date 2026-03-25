"""
src/nlp/extractors/spacy_ner_extractor.py

Custom spaCy NER extractor + trainer for the Nigerian Chest X-ray Benchmark (Phase 1).

What this module does
- Train a custom spaCy NER model (weak supervision) using:
  1) annotated_reports.csv (document-level gold labels)
  2) bilingual_dict.json (phrase library per label)
  We bootstrap entity spans by matching dictionary phrases inside each report, then
  (optionally) filtering those spans to ONLY labels present in that report's gold_labels.
- Run inference using the trained model, returning Tier-3 pathology labels.

Why "weak supervision"?
YTheour CSV currently has document-level labels, not span annotations. spaCy NER needs
(start, end, label) spans. This script generates those spans automatically via
phrase matching so it can train a first NER model quickly, then later improve it
with clinician-corrected spans if needed.
"""

from __future__ import annotations

import json
import random
import re
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import spacy
from spacy.pipeline import EntityRuler
from spacy.training import Example
from spacy.util import filter_spans

from src.nlp.labels import LabelHierarchy

# -----------------------------
# Resource loading (repo-aware)
# -----------------------------
lh = LabelHierarchy()

# Cache spaCy model for fast repeated inference
_NLP_CACHE: Dict[str, "spacy.language.Language"] = {}


def _repo_root() -> Path:
    """
    Best-effort repo root resolution:
    .../src/nlp/extractors/spacy_ner_extractor.py → .../
    """
    try:
        return Path(__file__).resolve().parents[3]
    except Exception:
        return Path.cwd()


def _resolve_path(candidate_paths: Sequence[Path]) -> Path:
    for p in candidate_paths:
        if p.exists():
            return p
    return candidate_paths[0]


def load_bilingual_dict(dict_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Loads the bilingual dictionary from the specified path or common repo locations.
    The dictionary is used for bootstrapping entity spans during training.
    """
    root = _repo_root()
    if dict_path:
        path = Path(dict_path)
    else:
        path = _resolve_path(
            [
                root / "src" / "nlp" / "dictionary" / "bilingual_dict.json",
                Path("src/nlp/dictionary/bilingual_dict.json"),
            ]
        )
    if not path.exists():
        raise FileNotFoundError(
            f"bilingual_dict.json not found at: {path}. "
            f"Provide dict_path=... or place it under src/nlp/dictionary/."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------
# CSV label parsing / cleanup
# -----------------------------
def normalize_label(label: str) -> str:
    """Fix known label typos from CSV so they match your canonical Tier-3 names."""
    label = label.strip()
    if label.lower() == "other significantt abnormality":
        return "Other Significant Abnormality"
    return label


def parse_gold_labels(s: str) -> List[str]:
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [normalize_label(x) for x in str(s).split(",") if x.strip()]


# -----------------------------
# Span bootstrapping from dict
# -----------------------------
def _compile_patterns(bilingual_dict: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    """
    Compile case-insensitive regex patterns for phrase matching.
    - underscores can match underscore OR whitespace (dict often has underscores)
    - prefer longer phrases first
    """
    patterns: Dict[str, List[re.Pattern]] = {}
    for label, phrases in bilingual_dict.items():
        uniq = sorted({str(p).strip() for p in phrases if str(p).strip()}, key=len, reverse=True)
        compiled: List[re.Pattern] = []
        for p in uniq:
            esc = re.escape(p)
            esc = esc.replace(r"\_", r"[_\s]+")
            compiled.append(
                re.compile(rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])", flags=re.IGNORECASE)
            )
        patterns[label] = compiled
    return patterns


def _find_spans(
    text: str,
    allowed_labels: Optional[Set[str]],
    patterns: Dict[str, List[re.Pattern]],
) -> List[Tuple[int, int, str]]:
    """
    Find (start, end, label) spans via regex patterns.
    If allowed_labels is provided, only search those labels (recommended).
    """
    spans: List[Tuple[int, int, str]] = []
    labels_to_search = allowed_labels if allowed_labels is not None else set(patterns.keys())

    for label in labels_to_search:
        if label not in patterns:
            continue
        for pat in patterns[label]:
            for m in pat.finditer(text):
                start, end = m.start(), m.end()
                if start < end:
                    spans.append((start, end, label))
    return spans


# -----------------------------
# Training
# -----------------------------
def train_spacy_ner(
    annotated_csv_path: str,
    output_path: str = "results/spacy_ner_model",
    dict_path: Optional[str] = None,
    iterations: int = 20,
    dev_split: float = 0.2,
    seed: int = 42,
    max_examples: Optional[int] = None,
    require_span_match: bool = True,
) -> Path:
    """
    Train a custom spaCy NER model from annotated_reports.csv + bilingual_dict.json.

    This bootstraps entity spans by phrase-matching the dictionary inside each report
    and restricting matches to the row's gold_labels
    """
    random.seed(seed)

    # Resolve CSV path against repo root if needed
    csv_path = Path(annotated_csv_path)
    if not csv_path.exists():
        root = _repo_root()
        csv_path = _resolve_path(
            [
                root / "results" / "annotated_reports.csv",
                root / "annotated_reports.csv",
                Path("results/annotated_reports.csv"),
            ]
        )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"annotated_reports.csv not found at: {annotated_csv_path} (or common repo locations). "
            "Provide annotated_csv_path=... with the correct path."
        )

    df = pd.read_csv(csv_path)
    if "report_text" not in df.columns:
        raise ValueError("annotated_reports.csv must contain a 'report_text' column.")
    if "gold_labels" not in df.columns:
        raise ValueError("annotated_reports.csv must contain a 'gold_labels' column.")

    bilingual_dict = load_bilingual_dict(dict_path=dict_path)
    patterns = _compile_patterns(bilingual_dict)

    train_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]] = []

    for _, row in df.iterrows():
        text = str(row["report_text"]) if not pd.isna(row["report_text"]) else ""
        if not text.strip():
            continue

        gold = set(parse_gold_labels(row.get("gold_labels", "")))

        # Skip pure Normal rows for NER (no spans to learn)
        if gold == {"Normal"} or len(gold) == 0:
            continue

        # Restrict to canonical Tier-3 labels
        gold = {g for g in gold if g in lh.tier3_labels}
        if len(gold) == 0:
            continue

        spans = _find_spans(text, allowed_labels=gold, patterns=patterns)

        if require_span_match and len(spans) == 0 and len(gold) > 0:
            continue

        train_data.append((text, {"entities": spans}))

        if max_examples and len(train_data) >= max_examples:
            break

    if len(train_data) < 10:
        raise ValueError(
            "Too few training examples after span bootstrapping. "
            "This usually means your dictionary phrases don't match the report text format. "
            "Try expanding bilingual_dict.json or set require_span_match=False."
        )

    # spaCy pipeline
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    # Register labels (use Tier-3 labels to keep the component stable)
    for pathology in lh.tier3_labels:
        ner.add_label(pathology)

    # Split train/dev
    random.shuffle(train_data)
    split = int((1.0 - dev_split) * len(train_data))
    train_split = train_data[:split]

    optimizer = nlp.initialize()

    for epoch in tqdm(range(iterations), desc="Training epochs", unit="epoch"):
        random.shuffle(train_split)
        losses = {}

        for text, annotations in tqdm(
            train_split,
            desc=f"Processing examples (epoch {epoch+1})",
            unit="example",
            leave=False,          # clean inner bar
        ):
            doc = nlp.make_doc(text)

            ents = []
            for start, end, label in annotations.get("entities", []):
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is not None:
                    ents.append(span)

            ents = filter_spans(ents)
            doc.ents = ents

            example = Example.from_dict(
                doc,
                {"entities": [(e.start_char, e.end_char, e.label_) for e in doc.ents]},
            )
            nlp.update([example], sgd=optimizer, losses=losses)

    out_dir = Path(output_path)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(out_dir)
    return out_dir

# -----------------------------
# Inference Tightening (Step 4)
# -----------------------------

def _ensure_entity_ruler(nlp, bilingual_dict: dict) -> None:
    """
    Add EntityRuler before NER so explicit dictionary phrases
    are always captured as entities.
    """
    if "entity_ruler" in nlp.pipe_names:
        return

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = []

    for label, phrases in bilingual_dict.items():
        for phrase in phrases:
            phrase = str(phrase).strip()
            if phrase:
                patterns.append({"label": label, "pattern": phrase})

    ruler.add_patterns(patterns)


def _post_filter_labels(report_text: str, labels: list) -> list:
    """
    Remove false positives using strong evidence rules.
    """
    t = "" if not isinstance(report_text, str) else report_text.lower()
    out = set(labels or [])

    # --- Pleural Effusion guard ---
    if "Pleural Effusion" in out:
        effusion_cues = [
            "effusion", "meniscus", "costophrenic", "blunting",
            "lamellar", "tracking", "pleural fluid",
            "pleural collection", "hydropneumothorax"
        ]
        if not any(cue in t for cue in effusion_cues):
            out.remove("Pleural Effusion")

    # --- Tuberculosis guard ---
    if "Tuberculosis" in out:
        tb_cues = [
            "tb", "ptb", "koch", "kochs",
            "gene expert", "cavity", "cavitary",
            "apical cavitation"
        ]
        if not any(cue in t for cue in tb_cues):
            out.remove("Tuberculosis")

    # --- Lung Mass guard ---
    if "Lung Mass / Nodule" in out:
        mass_cues = [
            "mass", "nodule", "nodular",
            "lesion", "tumor", "neoplasm",
            "mediastinal mass", "hilar mass"
        ]
        if not any(cue in t for cue in mass_cues):
            out.remove("Lung Mass / Nodule")

    return sorted(out)

# -----------------------------
# Inference
# -----------------------------
def spacy_ner_extractor(report: str, model_path: str = "results/spacy_ner_model", dict_path: Optional[str] = None,) -> List[str]:
    """
    Tight spaCy inference: Loads trained model, injects EntityRuler evidence layer, extracts entities and applies post-filter guardrails
    """
    if not isinstance(report, str) or pd.isna(report) or not report.strip():
        return []

    cache_key = f"{model_path}::{dict_path}"

    if cache_key not in _NLP_CACHE:
        nlp = spacy.load(model_path)

        bilingual_dict = load_bilingual_dict(dict_path=dict_path)
        _ensure_entity_ruler(nlp, bilingual_dict)

        _NLP_CACHE[cache_key] = nlp

    nlp = _NLP_CACHE[cache_key]
    doc = nlp(report)

    detected = {ent.label_ for ent in doc.ents if ent.label_ in lh.tier3_labels}
    detected = _post_filter_labels(report, list(detected))

    return sorted(detected)


if __name__ == "__main__":
    # Train using annotated reports (adjust path if needed)
    model_dir = train_spacy_ner(
        annotated_csv_path="results/annotated_reports.csv",
        output_path="results/spacy_ner_model",
        dict_path=None,
        iterations=20,
    )
    print("Saved model to:", model_dir)

    # Quick test
    sample_report = "Right basal consolidation suggestive of pneumonia."
    pathologies = spacy_ner_extractor(sample_report, model_path=str(model_dir))
    print("Detected:", pathologies)
    print(lh.get_hierarchy(pathologies))