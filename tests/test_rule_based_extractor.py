# tests/test_rule_based_extractor.py
"""
Unit tests for the rule-based label extractor (src/nlp/labels.py).
Week 6 — March 2026.

Run: pytest tests/test_rule_based_extractor.py -v
"""

import pytest
from nlp.labels import LabelHierarchy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lh():
    return LabelHierarchy(
        config_path="configs/labels/hierarchy.yaml",
        dict_path="src/nlp/dictionary/bilingual_dict.json",
    )


# ---------------------------------------------------------------------------
# 1. HIERARCHY TESTS
# ---------------------------------------------------------------------------

class TestLabelHierarchy:

    def test_tier1_labels(self, lh):
        assert set(lh.tier1_labels) == {"normal", "abnormal", "indeterminate"}

    def test_tier3_labels_count(self, lh):
        assert len(lh.tier3_labels) == 13

    def test_get_tier2_valid(self, lh):
        assert lh.get_tier2("Cardiomegaly") == "cardiac"
        assert lh.get_tier2("Tuberculosis") == "infectious"
        assert lh.get_tier2("Device Present") == "iatrogenic"

    def test_get_tier2_invalid(self, lh):
        with pytest.raises(ValueError, match="Unknown pathology"):
            lh.get_tier2("Made Up Disease")

    def test_derive_tier1_normal(self, lh):
        assert lh.derive_tier1([]) == "normal"

    def test_derive_tier1_abnormal(self, lh):
        assert lh.derive_tier1(["Cardiomegaly"]) == "abnormal"

    def test_derive_tier1_indeterminate(self, lh):
        assert lh.derive_tier1([], image_quality_ok=False) == "indeterminate"

    def test_get_hierarchy_multi_label(self, lh):
        result = lh.get_hierarchy(["Cardiomegaly", "Pleural Effusion"])
        assert result["tier1"] == "abnormal"
        assert set(result["tier2"]) == {"cardiac", "pleural"}
        assert set(result["tier3"]) == {"Cardiomegaly", "Pleural Effusion"}


# ---------------------------------------------------------------------------
# 2. NORMAL DETECTION
# ---------------------------------------------------------------------------

class TestNormalDetection:

    def test_clearly_normal_report(self, lh):
        report = """CHEST RADIOGRAPH REPORT
The heart is normal in size and contour.
The mediastinum, hila and pulmonary vasculature are also normal.
No focal lung lesion is seen.
The costophrenic sulci and hemidiaphragms are preserved.
CONCLUSION
No gross chest abnormality is seen."""
        labels = lh.rule_based_extractor(report)
        assert labels == [], f"Expected empty (Normal), got {labels}"

    def test_normal_with_meta(self, lh):
        """Negated abnormal cue words should NOT set has_abnormal_cue."""
        report = "Normal chest radiograph. No active lung lesion."
        result = lh.extract_with_scores(report)
        assert result["meta"]["has_normal_phrase"] is True
        assert result["meta"]["has_abnormal_cue"] is False

    def test_empty_report(self, lh):
        result = lh.extract_with_scores("")
        assert result["labels"] == {}

    def test_nan_report(self, lh):
        result = lh.extract_with_scores(float("nan"))
        assert result["labels"] == {}


# ---------------------------------------------------------------------------
# 3. SECTION PARSING
# ---------------------------------------------------------------------------

class TestSectionParsing:

    def test_body_captured_without_findings_header(self, lh):
        report = """CHEST RADIOGRAPH REPORT
The cardiac silhouette is enlarged.
Some degenerative changes noted.
CONCLUSION
Cardiomegaly."""
        result = lh.extract_with_scores(report)
        assert "Cardiomegaly" in result["labels"]
        assert "Degenerative / Structural Changes" in result["labels"]

    def test_findings_and_impression_both_parsed(self, lh):
        """Both Findings and Impression sections must be extracted."""
        report = """CHEST RADIOGRAPH REPORT
Findings:
The cardiac shadow is enlarged (CTR - 0.61).
There is unfolding of the aortic arch.
Impression:
Cardiomegaly."""
        result = lh.extract_with_scores(report)
        assert "Cardiomegaly" in result["labels"]
        assert "Degenerative / Structural Changes" in result["labels"]

    def test_comments_section_parsed(self, lh):
        report = """CHEST RADIOGRAPHY REPORT
Comments
There is cardiomegaly.
Impression
Cardiomegaly noted."""
        result = lh.extract_with_scores(report)
        assert "Cardiomegaly" in result["labels"]

    def test_no_headers_at_all(self, lh):
        report = "Cardiomegaly with pleural effusion."
        result = lh.extract_with_scores(report)
        assert "Cardiomegaly" in result["labels"]
        assert "Pleural Effusion" in result["labels"]


# ---------------------------------------------------------------------------
# 4. CTR-BASED CARDIOMEGALY
# ---------------------------------------------------------------------------

class TestCTRDetection:

    def test_ctr_above_threshold(self, lh):
        report = "The heart is enlarged (CTR - 0.61)."
        result = lh.extract_with_scores(report)
        assert "Cardiomegaly" in result["labels"]

    def test_ctr_at_normal_value(self, lh):
        report = "The heart is normal in size (CTR - 0.45). No focal lesion."
        labels = lh.rule_based_extractor(report)
        assert "Cardiomegaly" not in labels

    def test_ctr_borderline(self, lh):
        report = "The cardiothoracic ratio is 0.50."
        result = lh._detect_ctr_cardiomegaly(report)
        assert result is None

    def test_ctr_percentage_format(self, lh):
        report = "CTR: 55%"
        result = lh._detect_ctr_cardiomegaly(report)
        assert result is not None
        assert result["flags"]["ctr_value"] == 0.55

    def test_ctr_equals_format(self, lh):
        report = "The heart is enlarged (CTR=0.56)."
        result = lh._detect_ctr_cardiomegaly(report)
        assert result is not None

    def test_bare_ctr_mention_no_cardiomegaly(self, lh):
        report = """CHEST RADIOGRAPH REPORT
The heart is normal in size and contour. The cardiothoracic ratio is 0.50.
The costophrenic sulci and hemidiaphragms are preserved.
CONCLUSION
No gross chest abnormality is seen."""
        labels = lh.rule_based_extractor(report)
        assert "Cardiomegaly" not in labels


# ---------------------------------------------------------------------------
# 5. NEGATION DETECTION
# ---------------------------------------------------------------------------

class TestNegation:

    def test_simple_negation(self, lh):
        report = "No pleural effusion. No cardiomegaly."
        labels = lh.rule_based_extractor(report)
        assert "Pleural Effusion" not in labels
        assert "Cardiomegaly" not in labels

    def test_negation_respects_sentence_boundary(self, lh):
        report = "No focal lung lesion is seen. A central venous catheter is noted."
        labels = lh.rule_based_extractor(report)
        assert "Device Present" in labels

    def test_no_evidence_of(self, lh):
        report = "No evidence of consolidation. No evidence of pleural effusion."
        labels = lh.rule_based_extractor(report)
        assert "Pneumonia / Consolidation" not in labels
        assert "Pleural Effusion" not in labels

    def test_negation_does_not_cross_newline(self, lh):
        report = "No focal lung lesion is seen.\nThe cardiac silhouette is enlarged."
        labels = lh.rule_based_extractor(report)
        assert "Cardiomegaly" in labels


# ---------------------------------------------------------------------------
# 6. UNCERTAINTY HANDLING
# ---------------------------------------------------------------------------

class TestUncertainty:

    def test_uncertainty_reduces_score(self, lh):
        """Question mark should trigger uncertainty penalty."""
        certain = "Impression: Cardiomegaly."
        uncertain = "Impression: ? Cardiomegaly."
        res_c = lh.extract_with_scores(certain)
        res_u = lh.extract_with_scores(uncertain)
        assert res_u["labels"]["Cardiomegaly"]["score"] < res_c["labels"]["Cardiomegaly"]["score"]

    def test_uncertain_flag_set(self, lh):
        report = "Cannot exclude tuberculosis."
        result = lh.extract_with_scores(report)
        if "Tuberculosis" in result["labels"]:
            assert result["labels"]["Tuberculosis"]["flags"]["uncertain"] is True

    def test_suggestive_language(self, lh):
        report = "Features suggestive of pleural effusion."
        result = lh.extract_with_scores(report)
        assert "Pleural Effusion" in result["labels"]
        assert result["labels"]["Pleural Effusion"]["flags"]["uncertain"] is True


# ---------------------------------------------------------------------------
# 7. PULMONARY EDEMA
# ---------------------------------------------------------------------------

class TestPulmonaryEdema:

    def test_pe_with_vascular_cue(self, lh):
        report = "There is vascular congestion with perihilar haze. Impression: Pulmonary edema."
        labels = lh.rule_based_extractor(report)
        assert "Pulmonary Edema" in labels

    def test_pe_without_vascular_cue_not_triggered(self, lh):
        report = "There is patchy opacity in the right lower zone."
        labels = lh.rule_based_extractor(report)
        assert "Pulmonary Edema" not in labels


# ---------------------------------------------------------------------------
# 8. PNEUMONIA AMBIGUOUS PHRASE FILTERING
# ---------------------------------------------------------------------------

class TestPneumoniaFiltering:

    def test_homogeneous_opacity_alone_no_pneumonia(self, lh):
        report = "There is a homogeneous opacity in the left hemithorax. Impression: Left pleural effusion."
        labels = lh.rule_based_extractor(report)
        assert "Pneumonia / Consolidation" not in labels

    def test_homogeneous_opacity_with_pneumonic_cue(self, lh):
        report = "There is a homogeneous opacity with pneumonic changes in the right lower zone."
        labels = lh.rule_based_extractor(report)
        assert "Pneumonia / Consolidation" in labels

    def test_clear_pneumonia_phrase(self, lh):
        report = "There is consolidation in the right lower lobe. Impression: Right basal pneumonia."
        labels = lh.rule_based_extractor(report)
        assert "Pneumonia / Consolidation" in labels


# ---------------------------------------------------------------------------
# 9. SPECIFIC PATHOLOGY DETECTION
# ---------------------------------------------------------------------------

class TestPathologyDetection:

    def test_tuberculosis(self, lh):
        report = "There is apical fibrosis with cavitation. Impression: Pulmonary tuberculosis."
        labels = lh.rule_based_extractor(report)
        assert "Tuberculosis" in labels

    def test_pleural_effusion_costophrenic_blunting(self, lh):
        report = "Both costophrenic sulci are blunted. Impression: Bilateral pleural effusion."
        labels = lh.rule_based_extractor(report)
        assert "Pleural Effusion" in labels

    def test_atelectasis(self, lh):
        report = "There is volume loss in the left lower lobe with elevation of the hemidiaphragm."
        labels = lh.rule_based_extractor(report)
        assert "Atelectasis" in labels

    def test_device_present(self, lh):
        report = "An endotracheal tube is noted in situ. Central venous line tip projected over SVC."
        labels = lh.rule_based_extractor(report)
        assert "Device Present" in labels

    def test_degenerative_aortic_unfolding(self, lh):
        """Aortic unfolding phrases must trigger Degenerative / Structural Changes."""
        report = "There is unfolding of the aortic arch. The aorta is tortuous."
        labels = lh.rule_based_extractor(report)
        assert "Degenerative / Structural Changes" in labels

    def test_reticulonodular_pattern(self, lh):
        report = "There are prominent bronchovascular markings bilaterally with reticulonodular opacities."
        labels = lh.rule_based_extractor(report)
        assert "Reticulonodular / Interstitial Pattern" in labels

    def test_suspected_malignancy(self, lh):
        report = "There is a suspicious mass in the right upper lobe. Impression: Suspicious for malignancy."
        labels = lh.rule_based_extractor(report)
        assert "Suspected Malignancy" in labels

    def test_post_surgical(self, lh):
        report = "Sternotomy wires are noted. Post thoracotomy changes."
        labels = lh.rule_based_extractor(report)
        assert "Post-Surgical Changes" in labels

    def test_lung_mass(self, lh):
        report = "A solitary pulmonary nodule is seen in the right upper lobe."
        labels = lh.rule_based_extractor(report)
        assert "Lung Mass / Nodule" in labels


# ---------------------------------------------------------------------------
# 10. MULTI-LABEL REPORTS
# ---------------------------------------------------------------------------

class TestMultiLabel:

    def test_cardiomegaly_with_effusion(self, lh):
        report = """CHEST RADIOGRAPH REPORT
The cardiac silhouette is enlarged (CTR - 0.65).
There is blunting of the costophrenic angle on the right.
Impression: Cardiomegaly with right pleural effusion."""
        labels = lh.rule_based_extractor(report)
        assert "Cardiomegaly" in labels
        assert "Pleural Effusion" in labels

    def test_cardiomegaly_with_degenerative(self, lh):
        report = """CHEST RADIOGRAPH REPORT
There is cardiomegaly (CTR 0.57) with associated unfolding of the aorta.
Conclusion: Cardiomegaly. Aortic unfolding."""
        labels = lh.rule_based_extractor(report)
        assert "Cardiomegaly" in labels
        assert "Degenerative / Structural Changes" in labels

    def test_pneumonia_with_effusion(self, lh):
        report = "Consolidation in the right lower zone with associated right pleural effusion."
        labels = lh.rule_based_extractor(report)
        assert "Pneumonia / Consolidation" in labels
        assert "Pleural Effusion" in labels


# ---------------------------------------------------------------------------
# 11. FMC EBUTE-METTA SPECIFIC PATTERNS
# ---------------------------------------------------------------------------

class TestFMCSpecificPatterns:

    def test_preliminary_report_header(self, lh):
        report = """CHEST RADIOGRAPH REPORT (PRELIMINARY)
The heart is normal in size and contour.
CONCLUSION
No gross chest abnormality is seen."""
        labels = lh.rule_based_extractor(report)
        assert labels == []

    def test_updated_report_header(self, lh):
        report = """CHEST RADIOGRAPH REPORT (UPDATED)
The cardiac silhouette is enlarged.
Impression: Cardiomegaly."""
        labels = lh.rule_based_extractor(report)
        assert "Cardiomegaly" in labels

    def test_radiography_variant_header(self, lh):
        report = """CHEST RADIOGRAPHY REPORT
Comments
The lung fields are clear.
Impression
No radiographic abnormality is seen."""
        labels = lh.rule_based_extractor(report)
        assert labels == []

    def test_kindly_correlate_phrase(self, lh):
        report = """CHEST RADIOGRAPH REPORT
The heart is normal in size.
CONCLUSION
No gross chest abnormality is seen.
Kindly correlate with other clinical parameters."""
        labels = lh.rule_based_extractor(report)
        assert labels == []


# ---------------------------------------------------------------------------
# 12. EVIDENCE AND SCORING
# ---------------------------------------------------------------------------

class TestEvidenceAndScoring:

    def test_evidence_has_required_fields(self, lh):
        report = "Impression: Cardiomegaly."
        result = lh.extract_with_scores(report)
        evidence = result["labels"]["Cardiomegaly"]["evidence"]
        assert len(evidence) >= 1
        for ev in evidence:
            assert "phrase" in ev
            assert "section" in ev
            assert "span" in ev
            assert isinstance(ev["span"], list) and len(ev["span"]) == 2

    def test_impression_scores_higher_than_body(self, lh):
        """Impression section should receive a higher weight than body."""
        body_report = "Cardiomegaly noted."
        imp_report = "Impression: Cardiomegaly."
        body_score = lh.extract_with_scores(body_report)["labels"]["Cardiomegaly"]["score"]
        imp_score = lh.extract_with_scores(imp_report)["labels"]["Cardiomegaly"]["score"]
        assert imp_score > body_score, f"imp={imp_score} should be > body={body_score}"

    def test_threshold_filtering(self, lh):
        report = "Impression: Cardiomegaly."
        high_threshold = lh.rule_based_extractor(report, threshold=0.99)
        low_threshold = lh.rule_based_extractor(report, threshold=0.01)
        assert len(low_threshold) >= len(high_threshold)


# ---------------------------------------------------------------------------
# 13. BACKWARDS COMPATIBILITY
# ---------------------------------------------------------------------------

class TestBackwardsCompatibility:

    def test_returns_list_of_strings(self, lh):
        report = "Impression: Cardiomegaly with pleural effusion."
        result = lh.rule_based_extractor(report)
        assert isinstance(result, list)
        assert all(isinstance(l, str) for l in result)

    def test_return_normal_as_empty_default(self, lh):
        report = "No gross chest abnormality is seen."
        result = lh.rule_based_extractor(report, return_normal_as_empty=True)
        assert result == []

    def test_return_normal_explicit(self, lh):
        report = "No gross chest abnormality is seen."
        result = lh.rule_based_extractor(report, return_normal_as_empty=False)
        assert result == ["Normal"]
