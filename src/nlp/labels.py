# src/nlp/labels.py
from pathlib import Path
import yaml
from typing import Dict, List, Set
import json
import re


class LabelHierarchy:
    def __init__(self, config_path: str = "configs/labels/hierarchy.yaml", dict_path: str = "src/nlp/dictionary/bilingual_dict.json"):
        self.config = yaml.safe_load(Path(config_path).read_text())
        self.pathology_to_tier2 = self.config["pathology_to_category"]
        self.valid_pathologies = set(self.pathology_to_tier2.keys())
        
        # Load bilingual dictionary once
        with open(dict_path, 'r') as f:
            self.bilingual_dict = json.load(f)

    @property
    def tier1_labels(self) -> List[str]:
        return ["normal", "abnormal", "indeterminate"]

    @property
    def tier3_labels(self) -> List[str]:
        return list(self.valid_pathologies)

    # Tier helpers
    def get_tier2(self, pathology: str) -> str:
        if pathology not in self.valid_pathologies:
            raise ValueError(f"Unknown pathology label: {pathology}")
        return self.pathology_to_tier2[pathology]

    def derive_tier1(self, pathologies: List[str], image_quality_ok: bool = True) -> str:
        if not image_quality_ok:
            return "indeterminate"
        if pathologies:  # Any pathology → abnormal
            return "abnormal"
        return "normal"  # No pathologies + good quality

    def get_hierarchy(
        self, pathologies: List[str], 
        image_quality_ok: bool = True) -> Dict:
        """
        Full hierarchy derivation for a study.
        """
        tier2_set: Set[str] = {self.get_tier2(p) for p in pathologies if p in self.valid_pathologies}
        return {
            "tier1": self.derive_tier1(pathologies, image_quality_ok),
            "tier2": list(tier2_set),
            "tier3": pathologies
        }

    def rule_based_extractor(self, report: str, negation_window: int = 50) -> List[str]:
        """
        Extracts Tier-3 pathologies using regex matches from the bilingual dictionary.
        Handles basic negations and normal cases.
        
        Args:
            report: The radiology report text.
            negation_window: Character window to check for negations before the phrase.
        
        Returns:
            List of detected Tier-3 pathologies (e.g., ['Pneumonia / Consolidation', 'Cardiomegaly']).
        """
        detected_pathologies = []
        report_lower = report.lower()
        
        # Negation regex (expandable based on nigeria_specific_terms.md)
        negation_pattern = r'\b(no|without|absence of|not seen|clear of|negative for|no evidence of)\b'
        
        for pathology, phrases in self.bilingual_dict.items():
            if pathology == "Normal":
                continue  # Handle separately
            for phrase in phrases:
                phrase_lower = phrase.lower()
                # Full pattern for negation check
                neg_check = f'({negation_pattern}\\s{{0,{negation_window}}}?{re.escape(phrase_lower)})'
                if re.search(neg_check, report_lower):
                    break  # Negated; skip this pathology
                if re.search(r'\b' + re.escape(phrase_lower) + r'\b', report_lower):
                    detected_pathologies.append(pathology)
                    break  # Match found; move to next pathology
        
        # Check for "Normal" if no pathologies detected
        if not detected_pathologies:
            normal_phrases = self.bilingual_dict.get("Normal", [])
            if any(re.search(r'\b' + re.escape(p.lower()) + r'\b', report_lower) for p in normal_phrases):
                return []  # Normal study
        
        return list(set(detected_pathologies))  # Deduplicate

# Debug / sanity check
if __name__ == "__main__":
    lh = LabelHierarchy()
    print(lh.get_hierarchy([]))
    print(lh.get_hierarchy(["Tuberculosis"]))
    print(lh.get_hierarchy(["Pleural Effusion", "Cardiomegaly"]))
    print(lh.get_hierarchy([], image_quality_ok=False))