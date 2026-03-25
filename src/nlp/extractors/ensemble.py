# src/nlp/extractors/ensemble.py
"""
Ensemble label extractor combining rule-based and BioClinicalBERT methods.

Weeks 8–9 — March 2026.

Strategy (informed by Week 7-8 head-to-head comparison):
    - Rules primary for 11 labels where RB outperforms BERT
    - BERT primary for OSA and Suspected Malignancy (rules get F1=0.000)
    - Rare/unstable labels have stricter BERT-only inclusion threshold
    - BERT probability used as confidence signal for all labels
    - ALL disagreements logged (both directions) for Phase B review

Week 9 changes:
    - Normal probability suppression: when BERT's Normal output is high (>= 0.7),
      BERT-only pathology detections are more aggressively filtered. This reduces
      false positives on reports that BERT recognizes as predominantly normal.
    - Backward-compatible: works with both 13-label and 14-label BERT models.
      When 13-label model is used, normal_prob is None and suppression is skipped.

BERT inference:
    Accepts a single BioClinicalBERTExtractor OR a list of them (multi-fold).
    When multiple models are provided, their sigmoid probabilities are averaged
    before thresholding.

Confidence logic (method-priority-aware):
    HIGH    — both methods agree (both detect or both miss)
    MEDIUM  — primary method detects, secondary does not
    LOW     — non-primary method detects, primary does not

Label policy groups:
    BERT_PRIMARY    — OSA, Suspected Malignancy
    RARE_UNSTABLE   — Lung Mass/Nodule, Post-Surgical, Pulmonary Edema
    RULE_PRIMARY    — all other 11 labels
"""

import logging
import numpy as np
from typing import Dict, List, Any, Set, Union, Optional

logger = logging.getLogger(__name__)

# ---- Label policy groups ----

BERT_PRIMARY_LABELS: Set[str] = {
    "Other Significant Abnormality",
    "Suspected Malignancy",
}

RARE_UNSTABLE_LABELS: Set[str] = {
    "Lung Mass / Nodule",
    "Post-Surgical Changes",
    "Pulmonary Edema",
}

ALL_PATHOLOGY_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Degenerative / Structural Changes",
    "Device Present",
    "Lung Mass / Nodule",
    "Other Significant Abnormality",
    "Pleural Effusion",
    "Pneumonia / Consolidation",
    "Post-Surgical Changes",
    "Pulmonary Edema",
    "Reticulonodular / Interstitial Pattern",
    "Suspected Malignancy",
    "Tuberculosis",
]


class EnsembleExtractor:
    """
    Combines rule-based (LabelHierarchy) and one or more BioClinicalBERT
    extractors using a method-priority-aware voting system.
    """

    def __init__(
        self,
        rule_based_extractor,
        bert_extractors: Union[Any, List[Any]],
        bert_thresholds: Dict[str, float],
        bert_confidence_threshold: float = 0.3,
        bert_high_confidence: float = 0.8,
        rare_bert_only_threshold: float = 0.85,
        normal_suppression_threshold: float = 0.7,
    ):
        """
        Args:
            rule_based_extractor: LabelHierarchy instance
            bert_extractors: single BioClinicalBERTExtractor OR list (multi-fold)
            bert_thresholds: per-label thresholds for BERT (averaged across folds)
            bert_confidence_threshold: BERT prob above which secondary detection
                is considered a weak signal (default 0.3)
            bert_high_confidence: BERT prob above which BERT-only detection on a
                rule-primary label is included (default 0.8)
            rare_bert_only_threshold: stricter threshold for BERT-only detection
                on rare/unstable labels (default 0.85)
            normal_suppression_threshold: when averaged Normal prob >= this value,
                BERT-only pathology detections on rule-primary labels are suppressed
                (default 0.7). Only active when 14-label BERT models are used.
        """
        self.rb = rule_based_extractor

        if isinstance(bert_extractors, list):
            self.bert_models = bert_extractors
        else:
            self.bert_models = [bert_extractors]

        self.bert_thresholds = bert_thresholds
        self.bert_confidence_threshold = bert_confidence_threshold
        self.bert_high_confidence = bert_high_confidence
        self.rare_bert_only_threshold = rare_bert_only_threshold
        self.normal_suppression_threshold = normal_suppression_threshold

        # Check if any model has Normal label
        self.has_normal_label = any(
            getattr(m, 'has_normal_label', False) for m in self.bert_models
        )

        logger.info(
            f"Ensemble initialised: {len(self.bert_models)} BERT fold(s), "
            f"thresholds for {len(self.bert_thresholds)} labels, "
            f"normal_label={'yes' if self.has_normal_label else 'no'}"
        )

    def _get_bert_predictions(self, report: str) -> tuple:
        """
        Run all BERT fold models and average their sigmoid probabilities.

        Returns:
            bert_probs: dict of label -> averaged probability (13 pathology labels)
            normal_prob: averaged Normal probability (float or None if 13-label model)
        """
        all_probs = []
        normal_probs = []

        for model in self.bert_models:
            result = model.predict_with_scores(report)
            fold_probs = {}
            for label in ALL_PATHOLOGY_LABELS:
                if label in result["labels"]:
                    fold_probs[label] = result["labels"][label]["score"]
                else:
                    fold_probs[label] = 0.0
            all_probs.append(fold_probs)

            # Collect Normal probability if available
            meta_normal = result.get("meta", {}).get("normal_prob")
            if meta_normal is not None:
                normal_probs.append(meta_normal)

        # Average pathology probs across folds
        averaged = {}
        for label in ALL_PATHOLOGY_LABELS:
            probs = [fp[label] for fp in all_probs]
            averaged[label] = float(np.mean(probs))

        # Average Normal prob across folds (None if no model has it)
        normal_prob = float(np.mean(normal_probs)) if normal_probs else None

        return averaged, normal_prob

    def extract(self, report: str) -> Dict[str, Any]:
        """
        Run both extractors and combine using the ensemble strategy.

        Returns dict with: labels, confidence, rb_labels, bert_labels,
        bert_scores, rb_scores, disagreements, meta
        """
        if not isinstance(report, str) or not report.strip():
            return self._empty_result()

        # ---- Rule-based ----
        rb_labels_raw = set(self.rb.rule_based_extractor(report))
        rb_detailed = self.rb.extract_with_scores(report)

        rb_labels = rb_labels_raw - {"Normal"}

        rb_scores = {}
        for label in ALL_PATHOLOGY_LABELS:
            if label in rb_detailed["labels"]:
                rb_scores[label] = rb_detailed["labels"][label]
            else:
                rb_scores[label] = {"score": 0.0}

        # ---- BERT (multi-fold averaged, with Normal prob) ----
        bert_probs, normal_prob = self._get_bert_predictions(report)

        # Apply per-label thresholds to averaged probabilities
        bert_labels = set()
        for label in ALL_PATHOLOGY_LABELS:
            threshold = self.bert_thresholds.get(label, 0.5)
            if bert_probs[label] >= threshold:
                bert_labels.add(label)

        # ---- Normal suppression logic (Week 9) ----
        # When BERT is highly confident this is a normal report,
        # raise the bar for BERT-only pathology detections.
        normal_suppression_active = False
        if normal_prob is not None and normal_prob >= self.normal_suppression_threshold:
            normal_suppression_active = True

        # ---- Ensemble voting ----
        final_labels = []
        confidence = {}
        disagreements = []

        for label in ALL_PATHOLOGY_LABELS:
            in_rb = label in rb_labels
            in_bert = label in bert_labels
            bert_prob = bert_probs.get(label, 0.0)
            rb_score = rb_scores.get(label, {}).get("score", 0.0)

            if label in BERT_PRIMARY_LABELS:
                policy = "bert_primary"
            elif label in RARE_UNSTABLE_LABELS:
                policy = "rare_unstable"
            else:
                policy = "rule_primary"

            def make_disagreement(decision: str, reason: str):
                return {
                    "label": label,
                    "policy": policy,
                    "rb_detected": in_rb,
                    "bert_detected": in_bert,
                    "bert_prob": round(bert_prob, 4),
                    "rb_score": round(rb_score, 4),
                    "decision": decision,
                    "reason": reason,
                }

            # ---- Both agree ----
            if in_rb and in_bert:
                final_labels.append(label)
                confidence[label] = "high"
                continue

            if not in_rb and not in_bert:
                confidence[label] = "high"
                continue

            # ---- Disagreement ----

            if policy == "bert_primary":
                if in_bert and not in_rb:
                    final_labels.append(label)
                    confidence[label] = "medium"
                    disagreements.append(make_disagreement(
                        "included (BERT primary)",
                        "BERT-primary label: BERT detected, rules missed",
                    ))
                elif in_rb and not in_bert:
                    final_labels.append(label)
                    confidence[label] = "low"
                    disagreements.append(make_disagreement(
                        "included (RB backup)",
                        f"BERT-primary label: rules detected, BERT missed (prob={bert_prob:.3f})",
                    ))

            elif policy == "rare_unstable":
                if in_rb and not in_bert:
                    final_labels.append(label)
                    if bert_prob >= self.bert_confidence_threshold:
                        confidence[label] = "medium"
                        disagreements.append(make_disagreement(
                            "included (RB detected, BERT weak signal, rare)",
                            f"Rare label: rules detected, BERT weak signal (prob={bert_prob:.3f} >= {self.bert_confidence_threshold})",
                        ))
                    else:
                        confidence[label] = "low"
                        disagreements.append(make_disagreement(
                            "included (RB detected, no BERT signal, rare)",
                            f"Rare label: rules detected, BERT no signal (prob={bert_prob:.3f} < {self.bert_confidence_threshold})",
                        ))
                elif in_bert and not in_rb:
                    effective_threshold = self.rare_bert_only_threshold
                    if normal_suppression_active:
                        effective_threshold = 0.95  # Near-certain required when Normal is high
                    if bert_prob >= effective_threshold:
                        final_labels.append(label)
                        confidence[label] = "low"
                        disagreements.append(make_disagreement(
                            "included (high BERT prob, rare)",
                            f"Rare label: BERT-only, high prob ({bert_prob:.3f} >= {effective_threshold})"
                            + (f" [normal_suppression active, normal_prob={normal_prob:.3f}]" if normal_suppression_active else ""),
                        ))
                    else:
                        confidence[label] = "low"
                        disagreements.append(make_disagreement(
                            "excluded (low BERT prob, rare)",
                            f"Rare label: BERT-only, prob too low ({bert_prob:.3f} < {effective_threshold})"
                            + (f" [normal_suppression active, normal_prob={normal_prob:.3f}]" if normal_suppression_active else ""),
                        ))

            else:  # rule_primary
                if in_rb and not in_bert:
                    final_labels.append(label)
                    if bert_prob >= self.bert_confidence_threshold:
                        confidence[label] = "medium"
                        disagreements.append(make_disagreement(
                            "included (RB primary, BERT weak signal)",
                            f"Rule-primary: rules detected, BERT weak signal (prob={bert_prob:.3f} >= {self.bert_confidence_threshold})",
                        ))
                    else:
                        confidence[label] = "medium"
                        disagreements.append(make_disagreement(
                            "included (RB primary, no BERT signal)",
                            f"Rule-primary: rules detected, BERT no signal (prob={bert_prob:.3f} < {self.bert_confidence_threshold})",
                        ))
                elif in_bert and not in_rb:
                    effective_threshold = self.bert_high_confidence
                    if normal_suppression_active:
                        effective_threshold = 0.90  # Stricter when Normal is high
                    if bert_prob >= effective_threshold:
                        final_labels.append(label)
                        confidence[label] = "low"
                        disagreements.append(make_disagreement(
                            "included (high BERT prob)",
                            f"Rule-primary: BERT-only, high prob ({bert_prob:.3f} >= {effective_threshold})"
                            + (f" [normal_suppression active, normal_prob={normal_prob:.3f}]" if normal_suppression_active else ""),
                        ))
                    else:
                        confidence[label] = "low"
                        disagreements.append(make_disagreement(
                            "excluded (low BERT prob)",
                            f"Rule-primary: BERT-only, prob too low ({bert_prob:.3f} < {effective_threshold})"
                            + (f" [normal_suppression active, normal_prob={normal_prob:.3f}]" if normal_suppression_active else ""),
                        ))

        meta = {
            "rb_count": len(rb_labels),
            "bert_count": len(bert_labels),
            "final_count": len(final_labels),
            "disagreement_count": len(disagreements),
            "is_normal": len(final_labels) == 0,
            "has_normal_phrase": rb_detailed.get("meta", {}).get("has_normal_phrase", False),
            "n_bert_folds": len(self.bert_models),
            "normal_prob": round(normal_prob, 4) if normal_prob is not None else None,
            "normal_suppression_active": normal_suppression_active,
        }

        return {
            "labels": sorted(final_labels),
            "confidence": confidence,
            "rb_labels": sorted(rb_labels),
            "bert_labels": sorted(bert_labels),
            "bert_scores": {k: round(v, 4) for k, v in bert_probs.items()},
            "rb_scores": rb_scores,
            "disagreements": disagreements,
            "meta": meta,
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "labels": [],
            "confidence": {},
            "rb_labels": [],
            "bert_labels": [],
            "bert_scores": {},
            "rb_scores": {},
            "disagreements": [],
            "meta": {
                "rb_count": 0, "bert_count": 0, "final_count": 0,
                "disagreement_count": 0, "is_normal": True,
                "has_normal_phrase": False, "n_bert_folds": 0,
                "normal_prob": None, "normal_suppression_active": False,
            },
        }
