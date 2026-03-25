# src/nlp/extractors/bioclinicalbert_extractor.py
"""
BioClinicalBERT-based multi-label extractor for Nigerian chest X-ray reports.

Weeks 7–9 — March 2026.

Architecture:
    Bio_ClinicalBERT -> [CLS] -> dropout(0.1) -> linear head -> sigmoid outputs
    Supports 13 pathology labels (original) or 14 labels (with explicit Normal).

Week 9 changes:
    - Normal as optional 14th label (explicit normality representation)
    - Dynamic num_labels detection from saved model weights
    - Normal probability available as suppression signal for ensemble
    - Backward-compatible: 13-label models load and run unchanged

Usage:
    extractor = BioClinicalBERTExtractor.load("models/bert_fold0/")
    labels = extractor.predict("The cardiac silhouette is enlarged.")
    scores = extractor.predict_with_scores("The cardiac silhouette is enlarged.")
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Canonical label order — must match training
PATHOLOGY_LABELS = [
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

NUM_LABELS = len(PATHOLOGY_LABELS)  # 13

# Extended label list with Normal as 14th
ALL_LABELS = PATHOLOGY_LABELS + ["Normal"]
NUM_ALL_LABELS = len(ALL_LABELS)  # 14


class CXRBertClassifier(nn.Module):
    """BERT-base with a multi-label classification head."""

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", num_labels: int = NUM_LABELS):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for BERT-base
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


class BioClinicalBERTExtractor:
    """
    Production wrapper for inference. Matches the rule-based extractor interface.

    Supports both 13-label (pathology-only) and 14-label (pathology + Normal) models.
    When a 14-label model is loaded, the Normal probability is available as a
    suppression signal — high Normal prob means the model is confident this is
    a normal report, which can reduce false positive pathology predictions.
    """

    def __init__(
        self,
        model: CXRBertClassifier,
        tokenizer: AutoTokenizer,
        thresholds: Dict[str, float],
        device: str = "cpu",
        max_length: int = 512,
        has_normal_label: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.thresholds = thresholds
        self.device = device
        self.max_length = max_length
        self.has_normal_label = has_normal_label
        self.model.to(device)
        self.model.eval()

        # Determine label list based on model configuration
        if has_normal_label:
            self.label_names = list(ALL_LABELS)
            self.num_labels = NUM_ALL_LABELS
        else:
            self.label_names = list(PATHOLOGY_LABELS)
            self.num_labels = NUM_LABELS

    @classmethod
    def load(cls, model_dir: str, device: str = None) -> "BioClinicalBERTExtractor":
        """
        Load a trained model from a directory.

        Automatically detects whether the model has 13 or 14 output labels
        by inspecting the classifier weight shape.
        """
        model_dir = Path(model_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(model_dir / "thresholds.json", "r") as f:
            thresholds = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")

        # Detect num_labels from saved weights
        state_dict = torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
        classifier_weight = state_dict.get("classifier.weight")
        if classifier_weight is not None:
            detected_num_labels = classifier_weight.shape[0]
        else:
            detected_num_labels = NUM_LABELS

        has_normal_label = detected_num_labels == NUM_ALL_LABELS

        if has_normal_label:
            logger.info(f"Loaded 14-label model (with Normal) from {model_dir}")
        else:
            logger.info(f"Loaded 13-label model from {model_dir}")

        model = CXRBertClassifier(num_labels=detected_num_labels)
        model.load_state_dict(state_dict)

        return cls(
            model=model,
            tokenizer=tokenizer,
            thresholds=thresholds,
            device=device,
            has_normal_label=has_normal_label,
        )

    def _run_inference(self, report: str) -> np.ndarray:
        """Run inference and return raw sigmoid probabilities for all labels."""
        encoding = self.tokenizer(
            report,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, token_type_ids)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        return probs

    def predict(self, report: str) -> List[str]:
        """
        Predict labels for a single report.
        Returns List[str] — same interface as rule_based_extractor().
        Empty list means Normal. Normal is never returned as a label name
        (to maintain backward compatibility).
        """
        scores = self.predict_with_scores(report)
        labels = [
            label for label, info in scores["labels"].items()
            if info["predicted"] and label != "Normal"
        ]
        return labels

    def predict_with_scores(self, report: str) -> Dict[str, Any]:
        """
        Predict with full scores — same interface as extract_with_scores().

        For 14-label models, includes Normal probability in meta.
        """
        if not isinstance(report, str) or not report.strip():
            return {"labels": {}, "meta": {"model": "BioClinicalBERT", "normal_prob": None}}

        probs = self._run_inference(report)

        labels_out = {}
        normal_prob = None

        for i, label in enumerate(self.label_names):
            score = float(probs[i])

            if label == "Normal":
                normal_prob = score
                # Normal is tracked in meta, not as a label prediction
                continue

            threshold = self.thresholds.get(label, 0.5)
            labels_out[label] = {
                "score": round(score, 4),
                "threshold": threshold,
                "predicted": score >= threshold,
                "flags": {"uncertain": 0.3 <= score < threshold},
            }

        meta = {
            "model": "BioClinicalBERT",
            "has_normal_label": self.has_normal_label,
            "normal_prob": round(normal_prob, 4) if normal_prob is not None else None,
        }

        return {"labels": labels_out, "meta": meta}

    def predict_batch(self, reports: List[str], batch_size: int = 16) -> List[List[str]]:
        """Predict labels for a batch of reports."""
        all_labels = []
        for i in range(0, len(reports), batch_size):
            batch = reports[i : i + batch_size]
            encoding = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            token_type_ids = encoding.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask, token_type_ids)
                probs = torch.sigmoid(logits).cpu().numpy()

            for prob_row in probs:
                labels = []
                for j in range(len(self.label_names)):
                    label = self.label_names[j]
                    if label == "Normal":
                        continue  # Normal not returned as a predicted label
                    if prob_row[j] >= self.thresholds.get(label, 0.5):
                        labels.append(label)
                all_labels.append(labels)

        return all_labels

    def get_normal_prob(self, report: str) -> Optional[float]:
        """
        Get the Normal probability for a report.
        Returns None if model doesn't have Normal label.
        Useful for ensemble suppression logic.
        """
        if not self.has_normal_label:
            return None

        probs = self._run_inference(report)
        normal_idx = self.label_names.index("Normal")
        return float(probs[normal_idx])

    def save(self, model_dir: str):
        """Save model, tokenizer, and thresholds."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), model_dir / "model.pt")
        self.tokenizer.save_pretrained(model_dir / "tokenizer")
        with open(model_dir / "thresholds.json", "w") as f:
            json.dump(self.thresholds, f, indent=2)

        # Save model config for easier loading
        config = {
            "num_labels": self.num_labels,
            "has_normal_label": self.has_normal_label,
            "label_names": self.label_names,
        }
        with open(model_dir / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {model_dir} ({self.num_labels} labels)")
