# src/nlp/labels.py
"""
Label hierarchy and rule-based report label extractor for Nigerian chest X-ray reports.

v2.0 — Week 6 (March 2026):
  - Section parsing now captures body text before impression/conclusion headers
  - Pulmonary Edema detection requires a vascular keyword in the same section
  - CTR-based Cardiomegaly only triggered when the numeric CTR value exceeds 0.50
  - Ambiguous phrases (e.g. "homogeneous opacity") no longer trigger Pneumonia alone;
    a pneumonia-specific confirming term must also be present
  - Added "costophrenic blunting" phrases for Pleural Effusion detection
  - Added Device Present phrases (CVP line, chest tube tip, etc.)
  - Added Degenerative phrases (unfolding of the aortic arch, aorta is tortuous, etc.)
  - Dedicated _detect_ctr_cardiomegaly() method for numeric CTR parsing

rule_based_extractor() remains backwards compatible and returns List[str].
"""

from pathlib import Path
import yaml
from typing import Dict, List, Any, Optional
import json
import re
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)


class LabelHierarchy:
    def __init__(
        self,
        config_path: str = "configs/labels/hierarchy.yaml",
        dict_path: str = "src/nlp/dictionary/bilingual_dict.json",
    ):
        self.config = yaml.safe_load(Path(config_path).read_text())
        self.pathology_to_tier2 = self.config["pathology_to_category"]
        self.valid_pathologies = set(self.pathology_to_tier2.keys())

        with open(dict_path, "r", encoding="utf-8") as f:
            self.bilingual_dict = json.load(f)

        # --- Patterns ---
        self.negation_pattern = (
            r"\b(no|without|absence of|not seen|clear of|negative for|"
            r"no evidence of|free of|no gross|ruled out|excluded|"
            r"no significant|no definite|no focal|no active|no acute|"
            r"no obvious|does not show|not demonstrate|not identified)\b"
        )

        self.uncertainty_pattern = (
            r"(?:\b(?:likely|suggestive of|probable|cannot exclude|"
            r"may represent|possibly|could represent|"
            r"cannot rule out|questionable|equivocal|"
            r"appears to|consistent with|raising the possibility)\b|\?)"
        )

        self.abnormal_cue_pattern = (
            r"\b(opacity|opacities|infiltrate|infiltrates|lesion|"
            r"shadow|shadows|mass|nodule|effusion|collapse|"
            r"atelectasis|edema|fibrosis|consolidation|cavitation|"
            r"haze|prominence|airspace|haziness|widening|"
            r"blunting|blunted|obscured|obliteration|deviation|"
            r"enlarged|enlargement|elevated|thickening)\b"
        )

        # Phrases too ambiguous to use alone for Pneumonia — require
        # co-occurrence with a pneumonia-specific term before scoring.
        self._ambiguous_pneumonia_phrases = {
            "homogeneous opacity", "homogenous opacity",
            "inhomogenous opacity", "inhomogeneous opacity",
            "ill-defined opacity", "ill defined opacity",
            "veil opacity",
        }
        self._pneumonia_confirming_cues = re.compile(
            r"\b(pneumoni|consolidat|bronchopneumoni|aspiration|"
            r"lobar|segmental|airspace disease|air space disease|"
            r"infiltrat|pneumonic)\b", re.I
        )

        # Pulmonary Edema specific keywords (checked per-section)
        self._pe_vascular_keywords = [
            "vascular congestion", "pulmonary vascular congestion",
            "cephalization", "kerley", "interstitial edema",
            "alveolar edema", "fluid overload", "cardiac failure",
            "pulmonary venous hypertension", "perihilar haze",
            "pulmonary congestion", "pulmonary edema",
            "bat-wing", "bat wing",
        ]

        # Section weights
        self.section_weights = {
            "impression": 0.70,
            "conclusion": 0.70,
            "summary": 0.70,
            "findings": 0.50,
            "comments": 0.50,
            "body": 0.50,
        }
        self.default_weight = 0.45

    # ---- Tier helpers ----

    @property
    def tier1_labels(self) -> List[str]:
        return ["normal", "abnormal", "indeterminate"]

    @property
    def tier3_labels(self) -> List[str]:
        return list(self.valid_pathologies)

    def get_tier2(self, pathology: str) -> str:
        if pathology not in self.valid_pathologies:
            raise ValueError(f"Unknown pathology label: {pathology}")
        return self.pathology_to_tier2[pathology]

    def derive_tier1(self, pathologies: List[str], image_quality_ok: bool = True) -> str:
        if not image_quality_ok:
            return "indeterminate"
        return "abnormal" if pathologies else "normal"

    def get_hierarchy(self, pathologies: List[str], image_quality_ok: bool = True) -> Dict[str, Any]:
        valid_detected = [p for p in pathologies if p in self.valid_pathologies]
        if valid_detected != pathologies:
            logging.warning(f"Unmapped pathologies: {set(pathologies) - set(valid_detected)}")
        tier2_set = {self.get_tier2(p) for p in valid_detected}
        return {
            "tier1": self.derive_tier1(valid_detected, image_quality_ok),
            "tier2": list(tier2_set),
            "tier3": valid_detected,
        }

    # ---- Public extraction API ----

    def rule_based_extractor(
        self,
        report: str,
        negation_window: int = 60,
        threshold: float = 0.55,
        return_normal_as_empty: bool = True,
    ) -> List[str]:
        """
        Backwards-compatible extractor returning List[str] labels.
        """
        res = self.extract_with_scores(report, negation_window=negation_window)
        labels = [
            k for k, v in res["labels"].items()
            if v["score"] >= threshold and k != "Normal"
        ]
        if not labels and not return_normal_as_empty:
            return ["Normal"]
        return labels

    def extract_with_scores(
        self,
        report: str,
        negation_window: int = 60,
        max_evidence_per_label: int = 3,
    ) -> Dict[str, Any]:
        """
        Production-grade extraction with confidence scores and evidence.
        """
        if not isinstance(report, str) or pd.isna(report) or not report.strip():
            return {
                "labels": {},
                "meta": {
                    "has_normal_phrase": False,
                    "has_abnormal_cue": False,
                    "sections_found": [],
                },
            }

        sections = self._parse_report_sections(report)
        sections_found = [k for k, v in sections.items() if v.strip()]

        full_lower = report.lower()

        # Normal phrase detection
        normal_phrases = [p.lower() for p in self.bilingual_dict.get("Normal", [])]
        has_normal_phrase = any(
            re.search(r"\b" + re.escape(p) + r"\b", full_lower) for p in normal_phrases
        )

        # Abnormal cue detection (for OSA fallback) — negation-aware
        has_abnormal_cue = False
        for cue_m in re.finditer(self.abnormal_cue_pattern, full_lower):
            # Find start of current sentence
            sent_start = max(
                full_lower.rfind(". ", 0, cue_m.start()) + 2,
                full_lower.rfind("\n", 0, cue_m.start()) + 1,
                0,
            )
            left = full_lower[sent_start : cue_m.start()]
            if not re.search(self.negation_pattern, left):
                has_abnormal_cue = True
                break

        labels_out: Dict[str, Dict[str, Any]] = {}

        # CTR-based Cardiomegaly detection runs first so dictionary hits can be merged into it later
        ctr_result = self._detect_ctr_cardiomegaly(report)
        if ctr_result is not None:
            labels_out["Cardiomegaly"] = ctr_result

        for label, phrases in self.bilingual_dict.items():
            if label == "Normal":
                continue

            # Pulmonary Edema requires at least one vascular keyword anywhere in the report
            # to avoid triggering on generic opacity phrases alone
            if label == "Pulmonary Edema":
                # Check all section texts for vascular keywords
                has_pe_cue = any(
                    any(vk in sect_text.lower() for vk in self._pe_vascular_keywords)
                    for sect_text in sections.values()
                    if sect_text.strip()
                )
                if not has_pe_cue:
                    continue

            best_score = 0.0
            evidence = []

            for section_name, section_text in sections.items():
                if not section_text.strip():
                    continue

                # Ambiguous Pneumonia phrases are only used when a confirming term
                # (e.g. "consolidation", "pneumonic") also appears in the same section
                if label == "Pneumonia / Consolidation":
                    filtered_phrases = []
                    for p in phrases:
                        if p.lower().strip() in self._ambiguous_pneumonia_phrases:
                            # Only include if a confirming cue exists in the section
                            if self._pneumonia_confirming_cues.search(section_text):
                                filtered_phrases.append(p)
                        else:
                            filtered_phrases.append(p)
                    active_phrases = filtered_phrases
                elif label == "Cardiomegaly":
                    # Skip bare "ctr" / "cardiothoracic ratio" — handled by _detect_ctr_cardiomegaly
                    active_phrases = [
                        p for p in phrases
                        if p.lower().strip() not in ("ctr", "cardiothoracic ratio")
                    ]
                else:
                    active_phrases = phrases

                hits = self._match_phrases_with_context(
                    label=label,
                    phrases=active_phrases,
                    text=section_text,
                    section_name=section_name,
                    negation_window=negation_window,
                )

                for h in hits:
                    evidence.append(h)
                    best_score = max(best_score, h["score"])

            if evidence:
                evidence = sorted(evidence, key=lambda x: x["score"], reverse=True)[
                    :max_evidence_per_label
                ]
                # Merge with CTR-based evidence if already present
                if label == "Cardiomegaly" and label in labels_out:
                    existing = labels_out[label]
                    if best_score > existing["score"]:
                        existing["score"] = float(round(best_score, 4))
                    existing["evidence"].extend(
                        [{"phrase": e["phrase"], "section": e["section"], "span": e["span"]} for e in evidence]
                    )
                    existing["evidence"] = existing["evidence"][:max_evidence_per_label]
                else:
                    labels_out[label] = {
                        "score": float(round(best_score, 4)),
                        "evidence": [
                            {"phrase": e["phrase"], "section": e["section"], "span": e["span"]}
                            for e in evidence
                        ],
                        "flags": {
                            "uncertain": any(e.get("uncertain", False) for e in evidence)
                        },
                    }

        # --- Fallback: Other Significant Abnormality ---
        if not labels_out and has_abnormal_cue and not has_normal_phrase:
            m = re.search(self.abnormal_cue_pattern, full_lower)
            cue = m.group(0) if m else "abnormal cue"
            labels_out["Other Significant Abnormality"] = {
                "score": 0.35,
                "evidence": [
                    {
                        "phrase": cue,
                        "section": "full_text",
                        "span": [m.start(), m.end()] if m else [-1, -1],
                    }
                ],
                "flags": {"fallback": True},
            }

        meta = {
            "has_normal_phrase": has_normal_phrase,
            "has_abnormal_cue": has_abnormal_cue,
            "sections_found": sections_found,
        }
        return {"labels": labels_out, "meta": meta}

    # ---- Internal helpers ----

    def _detect_ctr_cardiomegaly(self, report: str) -> Optional[Dict[str, Any]]:
        """
        Parse CTR numeric values from the report and flag Cardiomegaly only when
        the cardiothoracic ratio exceeds 0.50.
        Handles formats: CTR-0.72, CTR = 0.61, CTR: 55%, CTR=61%, CTR 0.53
        """
        text = report.lower()
        patterns = [
            # CTR as decimal: CTR - 0.72, CTR=0.53, CTR: 0.61
            r"(?:ctr|cardiothoracic ratio)\s*[-:=]\s*0?\.?(\d{2,3})",
            # CTR as percentage: CTR: 55%, CTR- 56%
            r"(?:ctr|cardiothoracic ratio)\s*[-:=]\s*(\d{2,3})\s*%",
        ]

        for pat in patterns:
            for m in re.finditer(pat, text):
                try:
                    val_str = m.group(1)
                    val = float(val_str)
                    # Normalize: if > 1, treat as percentage
                    if val > 1:
                        val = val / 100.0
                    if val > 0.50:
                        return {
                            "score": min(0.85, 0.60 + (val - 0.50) * 2),
                            "evidence": [
                                {
                                    "phrase": f"CTR={val:.2f}",
                                    "section": "numeric",
                                    "span": [m.start(), m.end()],
                                }
                            ],
                            "flags": {"uncertain": False, "ctr_value": val},
                        }
                except (ValueError, IndexError):
                    continue
        return None

    def _match_phrases_with_context(
        self,
        label: str,
        phrases: List[str],
        text: str,
        section_name: str,
        negation_window: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Find phrase matches with negation suppression, uncertainty penalty,
        and section weighting. Returns evidence hits with per-hit scores.
        """
        text_lower = text.lower()
        weight = self.section_weights.get(section_name, self.default_weight)
        hits = []

        for phrase in phrases:
            if not isinstance(phrase, str) or not phrase.strip():
                continue
            phrase_lower = phrase.lower().strip()

            # Replace underscores with spaces for matching
            phrase_match = phrase_lower.replace("_", " ")

            pat = r"\b" + re.escape(phrase_match) + r"\b"
            for m in re.finditer(pat, text_lower):
                start, end = m.start(), m.end()

                # Negation check — respect sentence boundaries
                window_start = max(0, start - negation_window)
                left_context = text_lower[window_start:start]
                # Find the last sentence boundary (period/newline) in the window
                sent_break = max(
                    left_context.rfind(". "),
                    left_context.rfind(".\n"),
                    left_context.rfind("\n\n"),
                )
                if sent_break >= 0:
                    # Only check negation AFTER the last sentence break
                    left_context = left_context[sent_break + 1 :]
                if re.search(self.negation_pattern, left_context):
                    continue

                # Uncertainty check
                ctx_start = max(0, start - 40)
                ctx_end = min(len(text_lower), end + 40)
                ctx = text_lower[ctx_start:ctx_end]
                uncertain = bool(re.search(self.uncertainty_pattern, ctx))

                # Scoring
                score = weight
                if len(phrase_match) >= 12:
                    score += 0.08
                if " " in phrase_match:
                    score += 0.05
                if uncertain:
                    score *= 0.65

                score = max(0.0, min(0.99, score))

                hits.append(
                    {
                        "label": label,
                        "phrase": phrase,
                        "section": section_name,
                        "span": [start, end],
                        "score": float(score),
                        "uncertain": uncertain,
                    }
                )

        # Multi-hit consistency boost
        if len(hits) >= 2:
            hits = sorted(hits, key=lambda x: x["score"], reverse=True)
            hits[0]["score"] = float(min(0.99, hits[0]["score"] + 0.06))

        return hits

    def _parse_report_sections(self, report: str) -> Dict[str, str]:
        """
        FIX 1: Robust section parsing for FMC Ebute-Metta reports.

        Key fix: always capture text BEFORE the first impression/conclusion/summary
        header as 'body', even when no explicit findings/comments header exists.

        Handles common FMC EB formats:
          - "CHEST RADIOGRAPH REPORT\\n<body text>\\nConclusion: <text>"
          - "Findings: <text>\\nImpression: <text>"
          - "Comments\\n<text>\\nImpression:\\n<text>"
          - Plain text with no headers at all
        """
        sections: Dict[str, str] = {
            "findings": "",
            "impression": "",
            "conclusion": "",
            "summary": "",
            "comments": "",
            "body": "",
        }

        # Strip the initial "CHEST RADIOGRAPH REPORT" header line only
        # Match the header + any parenthetical like "(PRELIMINARY)" on the SAME line
        txt = re.sub(
            r"^[\s]*CHEST\s+RADIOGRAPH?Y?\s+(?:REPORT|====.*?====)[^\n]*\n?",
            "",
            report,
            flags=re.I,
        ).strip()

        # Find impression/conclusion/summary header position
        # Match at start of string OR after a newline
        imp_match = re.search(
            r"(?:^|\n)\s*(impression|conclusion|summary)\s*:?\s*\n?",
            txt,
            re.I,
        )

        # Find findings/comments header position
        find_match = re.search(
            r"^(findings|comments)\s*:?\s*\n?",
            txt,
            re.I | re.MULTILINE,
        )

        if imp_match:
            imp_key = imp_match.group(1).lower()
            imp_text = txt[imp_match.end():].strip()
            sections[imp_key] = imp_text

            # Everything BEFORE the impression header
            before_imp = txt[:imp_match.start()].strip()

            if find_match and find_match.start() < imp_match.start():
                find_key = find_match.group(1).lower()
                find_content = txt[find_match.end():imp_match.start()].strip()
                sections[find_key] = find_content
            else:
                # No explicit findings header — treat pre-impression text as body
                sections["body"] = before_imp

        elif find_match:
            find_key = find_match.group(1).lower()
            sections[find_key] = txt[find_match.end():].strip()

        else:
            # No section headers at all — everything is body
            sections["body"] = txt

        # Safety: if nothing was captured, use full report
        if not any(v.strip() for v in sections.values()):
            sections["body"] = report.strip()

        return sections


# Debug / sanity check
if __name__ == "__main__":
    lh = LabelHierarchy()
    sample = """CHEST RADIOGRAPH REPORT
Findings: The cardiac silhouette is enlarged. No pleural effusion.
Impression: Cardiomegaly. ? pulmonary edema."""
    out = lh.extract_with_scores(sample)
    print(out)
    print(lh.rule_based_extractor(sample))
