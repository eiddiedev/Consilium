"""Regression tests for orchestrator scoring and specialist parsing."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from orchestrator.agent import (
    SpecialistPayloadError,
    _has_minimum_patient_info,
    _normalize_evidence,
    _parse_patient_values,
    _preferences_for_patient,
    parse_specialist_json,
    run_orchestration,
    run_orchestration_async,
)


def test_negated_heart_failure_is_not_counted_as_hf():
    vals = _parse_patient_values(
        "55F with T2DM, CKD stage 3a eGFR 52, and no history of heart failure."
    )

    assert vals["has_hf"] is False
    assert vals["has_ckd"] is True
    assert vals["has_diabetes"] is True


def test_negated_diabetes_and_ckd_are_not_counted_from_keywords():
    vals = _parse_patient_values(
        "55-year-old female with HFpEF LVEF 58%, eGFR 82, no diabetes, "
        "and no kidney disease."
    )

    assert vals["has_hf"] is True
    assert vals["has_ckd"] is False
    assert vals["has_diabetes"] is False


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


def test_egfr_under_30_metformin_prioritizes_nephrology_safety():
    msg = (
        "68-year-old male with HFrEF LVEF 32%, type 2 diabetes HbA1c 8.2%, "
        "CKD eGFR 28. Current medications: Lisinopril, Metformin, "
        "Furosemide, Aspirin, Glipizide."
    )

    ranked = run_orchestration(msg)["ranked_recommendations"]

    assert [(r["specialty"], r["total_score"]) for r in ranked] == [
        ("nephrology", 0.9),
        ("endocrinology", 0.625),
        ("cardiology", 0.35),
    ]
    assert "STOP metformin" in ranked[0]["recommendation"]


def test_lvef_under_35_boosts_guideline_priority_when_not_ckd_safety_case():
    vals = _parse_patient_values(
        "55-year-old female with HFrEF LVEF 25%, eGFR 82, no diabetes, "
        "on Lisinopril and Furosemide."
    )

    prefs = _preferences_for_patient(vals)

    assert prefs.guideline_priority == 0.30
    assert prefs.drug_interaction_risk == 0.20


def test_no_hf_case_does_not_recommend_hf_beta_blocker():
    msg = (
        "55F with T2DM HbA1c 9.1%, CKD stage 3a eGFR 52, and no history of "
        "heart failure. Current medications: Metformin, Glipizide, Lisinopril."
    )

    result = run_orchestration(msg)
    text = result["formatted_output"].lower()

    assert result["ranked_recommendations"][0]["specialty"] == "endocrinology"
    assert "no hf beta-blocker is indicated solely for this presentation" in text
    assert "start carvedilol" not in text


def test_hfpef_case_does_not_trigger_ckd_dm_or_hfref_language():
    msg = (
        "55-year-old female with HFpEF LVEF 58%, eGFR 82, no diabetes, "
        "no kidney disease. Current medications: Lisinopril, Carvedilol."
    )

    text = run_orchestration(msg)["formatted_output"].lower()

    assert "hfpef phenotype" in text
    assert "given t2dm+ckd and no hf" not in text
    assert "stop metformin" not in text
    assert "no major conflicts detected" in text


def test_minimum_patient_info_gate_rejects_instruction_only_text():
    assert _has_minimum_patient_info("Run the full multi-specialty orchestration for this patient.") is False
    assert _has_minimum_patient_info("68M with HFrEF LVEF 32%, CKD eGFR 28, on metformin.") is True


def test_instruction_only_text_does_not_run_specialists_or_topsis():
    ctx = SimpleNamespace(run_config=SimpleNamespace(custom_metadata={}))

    result = asyncio.run(
        run_orchestration_async(
            ctx,
            "Run the full multi-specialty orchestration for this patient.",
        )
    )

    assert result["status"] == "needs_input"
    assert result["ranked_recommendations"] == []
    assert result["specialist_mode"] == "not run - insufficient patient context"
    assert "More Patient Information Needed" in result["formatted_output"]


def test_parse_specialist_json_accepts_plain_and_fenced_json():
    plain = (
        '{"specialty":"cardiology","recommendation":"Start carvedilol.",'
        '"risks":["hypotension"],"citation":"ACC/AHA 2022"}'
    )
    fenced = (
        "```json\n"
        '{"specialty":"nephrology","recommendation":"Stop metformin.",'
        '"risks":["monitor renal function"],"citation":"KDIGO 2024"}'
        "\n```"
    )

    assert parse_specialist_json(plain, "cardiology")["source"] == "llm"
    assert parse_specialist_json(fenced, "nephrology")["specialty"] == "nephrology"


def test_parse_specialist_json_prefers_last_valid_object():
    text = (
        '{"specialty":"cardiology","recommendation":"Draft.",'
        '"risks":["monitor"],"citation":"draft"}\n'
        '{"specialty":"cardiology","recommendation":"Final recommendation.",'
        '"risks":["hypotension"],"citation":"ACC/AHA 2022"}'
    )

    parsed = parse_specialist_json(text, "cardiology")

    assert parsed["recommendation"] == "Final recommendation."


def test_parse_specialist_json_rejects_missing_fields_and_wrong_specialty():
    with pytest.raises(SpecialistPayloadError):
        parse_specialist_json('{"specialty":"cardiology"}', "cardiology")

    with pytest.raises(SpecialistPayloadError):
        parse_specialist_json(
            '{"specialty":"nephrology","recommendation":"Stop metformin.",'
            '"risks":["monitor"],"citation":"KDIGO 2024"}',
            "cardiology",
        )


def test_evidence_mapping_is_deterministic_for_all_guideline_families():
    assert _normalize_evidence("Class I Level A") == 1.0
    assert _normalize_evidence("KDIGO Grade 1B recommendation") == 0.85
    assert _normalize_evidence("ADA Level A") == 1.0
