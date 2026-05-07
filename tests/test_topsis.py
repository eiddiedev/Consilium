"""Tests for the TOPSIS healthcare scorer.

Run: .venv/bin/python -m pytest tests/test_topsis.py -v
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.topsis import Recommendation, Preferences, score_topsis


def _make_rec(specialty, evidence_score=0.5, patient_match=0.5,
              drug_interaction_risk=1.0, guideline_priority=0.5, **kwargs):
    return Recommendation(
        specialty=specialty,
        recommendation=f"Recommendation from {specialty}",
        confidence=0.8,
        evidence_score=evidence_score,
        patient_match=patient_match,
        drug_interaction_risk=drug_interaction_risk,
        guideline_priority=guideline_priority,
        **kwargs,
    )


class TestTOPSISBasic:
    """Basic TOPSIS functionality tests."""

    def test_two_recommendations_ranked(self):
        """Two recs should produce rank 1 and 2."""
        recs = [
            _make_rec("cardiology", evidence_score=0.9, patient_match=0.8, guideline_priority=0.9),
            _make_rec("nephrology", evidence_score=0.6, patient_match=0.5, guideline_priority=0.6),
        ]
        results = score_topsis(recs)
        assert len(results) == 2
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[0].total_score >= results[1].total_score

    def test_higher_evidence_wins(self):
        """The rec with higher evidence should rank #1 when other dims are equal."""
        recs = [
            _make_rec("cardiology", evidence_score=0.9),
            _make_rec("nephrology", evidence_score=0.3),
        ]
        results = score_topsis(recs)
        assert results[0].recommendation.specialty == "cardiology"

    def test_higher_drug_safety_wins(self):
        """The rec with higher medication safety score should rank higher."""
        recs = [
            _make_rec("cardiology", drug_interaction_risk=0.8),
            _make_rec("nephrology", drug_interaction_risk=0.1),
        ]
        results = score_topsis(recs)
        assert results[0].recommendation.specialty == "cardiology"

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert score_topsis([]) == []

    def test_single_recommendation(self):
        """Single rec gets rank 1 with score 0.5 (neutral)."""
        recs = [_make_rec("cardiology")]
        results = score_topsis(recs)
        assert len(results) == 1
        assert results[0].rank == 1

    def test_breakdown_populated(self):
        """Breakdown dict should have all 4 dimensions."""
        recs = [
            _make_rec("cardiology"),
            _make_rec("nephrology"),
        ]
        results = score_topsis(recs)
        for r in results:
            assert "evidence_level" in r.breakdown
            assert "patient_match" in r.breakdown
            assert "drug_interaction_risk" in r.breakdown
            assert "guideline_priority" in r.breakdown

    def test_reasoning_populated(self):
        """Reasoning string should not be empty."""
        recs = [
            _make_rec("cardiology"),
            _make_rec("nephrology"),
        ]
        results = score_topsis(recs)
        for r in results:
            assert len(r.reasoning) > 0


class TestTOPSISMedical:
    """Tests specific to the HF+T2DM+CKD clinical scenario."""

    def test_ckd_patient_penalizes_nephrotoxic_drugs(self):
        """When medication safety is low, that recommendation should rank lower."""
        # Scenario: eGFR=28 means nephrotoxic choices should have low safety scores.
        recs = [
            _make_rec("cardiology",
                      evidence_score=0.8,
                      patient_match=0.7,
                      drug_interaction_risk=0.1,  # NSAID-like, low safety
                      guideline_priority=0.8),
            _make_rec("nephrology",
                      evidence_score=0.8,
                      patient_match=0.9,
                      drug_interaction_risk=0.9,  # SGLT2i, high safety
                      guideline_priority=0.9),
            _make_rec("endocrinology",
                      evidence_score=0.7,
                      patient_match=0.6,
                      drug_interaction_risk=0.7,  # metformin dose adjust
                      guideline_priority=0.7),
        ]
        results = score_topsis(recs)
        # Nephrology should rank #1 (best safety + best patient match)
        assert results[0].recommendation.specialty == "nephrology"
        # Cardiology (lowest safety) should score lower than nephrology
        cardiology_score = next(r.total_score for r in results if r.recommendation.specialty == "cardiology")
        nephro_score = next(r.total_score for r in results if r.recommendation.specialty == "nephrology")
        assert cardiology_score < nephro_score

    def test_equal_recs_similar_scores(self):
        """When all dimensions are equal, scores should be very close."""
        recs = [
            _make_rec("cardiology", evidence_score=0.7, patient_match=0.7,
                      drug_interaction_risk=0.3, guideline_priority=0.7),
            _make_rec("nephrology", evidence_score=0.7, patient_match=0.7,
                      drug_interaction_risk=0.3, guideline_priority=0.7),
            _make_rec("endocrinology", evidence_score=0.7, patient_match=0.7,
                      drug_interaction_risk=0.3, guideline_priority=0.7),
        ]
        results = score_topsis(recs)
        scores = [r.total_score for r in results]
        # All scores should be very close (within 0.01)
        assert max(scores) - min(scores) < 0.01

    def test_custom_weights(self):
        """Custom weights should shift the ranking."""
        recs = [
            _make_rec("cardiology", evidence_score=0.9, patient_match=0.5),
            _make_rec("nephrology", evidence_score=0.5, patient_match=0.9),
        ]
        # Default weights: evidence=0.3, match=0.3 → cardiology wins on evidence
        default_results = score_topsis(recs)
        assert default_results[0].recommendation.specialty == "cardiology"

        # Heavy patient_match weight: nephrology should win
        custom_prefs = Preferences(
            evidence_level=0.10,
            patient_match=0.60,
            drug_interaction_risk=0.10,
            guideline_priority=0.20,
        )
        custom_results = score_topsis(recs, preferences=custom_prefs)
        assert custom_results[0].recommendation.specialty == "nephrology"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
