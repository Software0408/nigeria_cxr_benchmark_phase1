# src/nlp/labels.py
from pathlib import Path
import yaml
from typing import Dict, List, Set


class LabelHierarchy:
    def __init__(self, config_path: str = "configs/labels/hierarchy.yaml"):
        self.config = yaml.safe_load(Path(config_path).read_text())

        self.pathology_to_tier2 = self.config["pathology_to_category"]
        self.valid_pathologies = set(self.pathology_to_tier2.keys())

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

# Debug / sanity check
if __name__ == "__main__":
    lh = LabelHierarchy()
    print(lh.get_hierarchy([]))
    print(lh.get_hierarchy(["Tuberculosis"]))
    print(lh.get_hierarchy(["Pleural Effusion", "Cardiomegaly"]))
    print(lh.get_hierarchy([], image_quality_ok=False))
    # indeterminate
