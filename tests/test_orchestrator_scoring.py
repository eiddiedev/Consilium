"""Regression tests for deterministic orchestrator scoring."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.agent import _parse_patient_values, run_orchestration


def test_negated_heart_failure_is_not_counted_as_hf():
    vals = _parse_patient_values(
        "55F with T2DM, CKD stage 3a eGFR 52, and no history of heart failure."
    )

    assert vals["has_hf"] is False
    assert vals["has_ckd"] is True
    assert vals["has_diabetes"] is True


def test_patient_b_scores_are_deterministic_and_non_extreme():
    msg = (
        "A 55-year-old female with type 2 diabetes (HbA1c 9.1%), chronic kidney "
        "disease stage 3a (eGFR 52), and no history of heart failure. Current "
        "medications: Metformin 1000mg twice daily, Glipizide 10mg daily, "
        "Lisinopril 20mg daily."
    )

    first = run_orchestration(msg)["ranked_recommendations"]
    second = run_orchestration(msg)["ranked_recommendations"]

    assert first == second
    assert [(r["specialty"], r["total_score"]) for r in first] == [
        ("endocrinology", 0.9),
        ("nephrology", 0.625),
        ("cardiology", 0.35),
    ]
    assert all(r["total_score"] not in (0.0, 1.0) for r in first)
