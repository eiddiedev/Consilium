"""asm_orchestrator - multi-specialty clinical orchestrator.

Primary path:
  1. Build a patient summary from SHARP/FHIR context when available.
  2. Ask cardiology, nephrology, and endocrinology ADK specialist agents for
     structured JSON recommendations.
  3. Validate each specialist response and fall back only for failed agents.
  4. Rank recommendations with deterministic TOPSIS scoring.

The deterministic recommendation builder remains available for tests and for
runtime reliability when a model call or JSON parse fails.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from cardiology_agent.agent import root_agent as cardiology_agent
from endocrinology_agent.agent import root_agent as endocrinology_agent
from nephrology_agent.agent import root_agent as nephrology_agent
from shared.fhir_hook import FHIR_CONTEXT_KEY
from shared.tools.fhir import (
    get_active_conditions,
    get_active_medications,
    get_patient_demographics,
    get_recent_observations,
)
from tools.topsis import Preferences, Recommendation, score_topsis

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
_model_name = os.getenv("ORCHESTRATOR_MODEL", DEFAULT_MODEL)

SPECIALIST_ORDER = ("cardiology", "nephrology", "endocrinology")
SPECIALIST_AGENTS = {
    "cardiology": cardiology_agent,
    "nephrology": nephrology_agent,
    "endocrinology": endocrinology_agent,
}


# Deterministic evidence score lookup table.
EVIDENCE_SCORE_TABLE = {
    # ACC/AHA grades
    "Class I Level A": 1.00,
    "Class I Level B-R": 0.90,
    "Class I Level B-NR": 0.85,
    "Class I Level B": 0.85,
    "Class I Level C": 0.75,
    "Class I": 0.80,
    "Class IIa Level A": 0.70,
    "Class IIa Level B": 0.60,
    "Class IIa Level C": 0.55,
    "Class IIa": 0.60,
    "Class IIb Level A": 0.45,
    "Class IIb Level B": 0.35,
    "Class IIb Level C": 0.25,
    "Class IIb": 0.35,
    "Class III": 0.05,
    # KDIGO grades
    "Grade 1A": 1.00,
    "Grade 1B": 0.85,
    "Grade 1C": 0.70,
    "Grade 1D": 0.55,
    "Grade 2A": 0.45,
    "Grade 2B": 0.35,
    "Grade 2C": 0.25,
    "Grade 2D": 0.15,
    # ADA levels
    "Level A": 1.00,
    "Level B": 0.75,
    "Level C": 0.50,
    "Level E": 0.30,
}


class SpecialistPayloadError(ValueError):
    """Raised when a specialist response cannot be used for ranking."""


def _normalize_evidence(evidence: str) -> float:
    """Return a deterministic evidence score using guideline grade text only."""
    if evidence in EVIDENCE_SCORE_TABLE:
        return EVIDENCE_SCORE_TABLE[evidence]

    e = (evidence or "").strip().lower()
    for key, score in sorted(EVIDENCE_SCORE_TABLE.items(), key=lambda item: len(item[0]), reverse=True):
        if key.lower() in e:
            return score
    return 0.55


def _compute_guideline_priority(evidence: str) -> float:
    """Use the deterministic evidence lookup as the guideline-strength proxy."""
    return _normalize_evidence(evidence)


def _parse_patient_values(ctx: str) -> dict:
    ctx_lower = ctx.lower()
    ef_m = re.search(r"(?:ef|lvef)\D*(\d+)", ctx, re.IGNORECASE)
    egfr_m = re.search(r"egfr\D*(\d+)", ctx, re.IGNORECASE)
    hba1c_m = re.search(r"hba1c\D*(\d+\.?\d*)", ctx, re.IGNORECASE)
    has_hf_positive = any(kw in ctx_lower for kw in ["hf", "heart failure", "hfref", "lvef"])
    has_hf_negated = any(
        re.search(pattern, ctx_lower)
        for pattern in [
            r"\bno\s+(?:history\s+of\s+)?(?:heart failure|hf|hfref|hfr?ef)\b",
            r"\bwithout\s+(?:history\s+of\s+)?(?:heart failure|hf|hfref|hfr?ef)\b",
            r"\bdenies\s+(?:history\s+of\s+)?(?:heart failure|hf|hfref|hfr?ef)\b",
        ]
    )
    egfr_val = float(egfr_m.group(1)) if egfr_m else None
    hba1c_val = float(hba1c_m.group(1)) if hba1c_m else None
    return {
        "ef": float(ef_m.group(1)) if ef_m else None,
        "egfr": egfr_val,
        "hba1c": hba1c_val,
        "has_hf": has_hf_positive and not has_hf_negated,
        # CKD: explicit keywords OR eGFR < 60 (normal eGFR doesn't mean CKD)
        "has_ckd": (
            any(kw in ctx_lower for kw in ["ckd", "kidney", "renal"])
            or (egfr_val is not None and egfr_val < 60)
        ),
        # Diabetes: explicit keywords OR HbA1c >= 6.5% (normal HbA1c doesn't mean DM)
        "has_diabetes": (
            any(kw in ctx_lower for kw in ["diabetes", "t2dm", "diabetic"])
            or (hba1c_val is not None and hba1c_val >= 6.5)
        ),
        "has_metformin": "metformin" in ctx_lower,
        "has_beta_blocker": any(kw in ctx_lower for kw in ["carvedilol", "metoprolol", "bisoprolol"]),
    }


def _parse_age_sex(ctx: str) -> str:
    age_match = re.search(r"\b(\d{1,3})\s*(?:/|-year-old\s+)(male|female|m|f)\b", ctx, re.IGNORECASE)
    if not age_match:
        return "patient"

    age = age_match.group(1)
    sex_text = age_match.group(2).lower()
    sex = "F" if sex_text.startswith("f") else "M"
    return f"{age}{sex}"


def _has_minimum_patient_info(ctx: str) -> bool:
    """Return True when free-text input has enough clinical signal to rank safely."""
    vals = _parse_patient_values(ctx)
    ctx_lower = ctx.lower()

    has_diagnosis = vals["has_hf"] or vals["has_ckd"] or vals["has_diabetes"]
    has_objective_value = any(vals[key] is not None for key in ("ef", "egfr", "hba1c"))
    has_age_sex = _parse_age_sex(ctx) != "patient"
    has_medication = any(
        med in ctx_lower
        for med in [
            "metformin",
            "lisinopril",
            "furosemide",
            "carvedilol",
            "metoprolol",
            "dapagliflozin",
            "empagliflozin",
            "glipizide",
            "aspirin",
            "arni",
            "sacubitril",
            "valsartan",
        ]
    )

    signal_count = sum([has_diagnosis, has_objective_value, has_age_sex, has_medication])
    return (has_diagnosis or has_objective_value) and signal_count >= 2


def _condition_summary(vals: dict) -> str:
    parts = []
    if vals["has_hf"]:
        label = "HF"
        if vals["ef"] is not None:
            label += f"(LVEF={vals['ef']}%)"
        parts.append(label)
    elif vals["ef"] is None:
        parts.append("no HF")

    if vals["has_diabetes"]:
        label = "T2DM"
        if vals["hba1c"] is not None:
            label += f"(HbA1c={vals['hba1c']}%)"
        parts.append(label)

    if vals["has_ckd"]:
        label = "CKD"
        if vals["egfr"] is not None:
            label += f"(eGFR={vals['egfr']})"
        parts.append(label)

    return ", ".join(parts) if parts else "See patient message"


def _compute_patient_match(
    specialty: str,
    has_hf: bool,
    has_ckd: bool,
    has_diabetes: bool,
    ef: float | None = None,
    egfr: float | None = None,
    hba1c: float | None = None,
) -> float:
    specialty = specialty.lower()
    score = 0.0

    if specialty == "cardiology":
        if has_hf:
            score += 0.6
            if ef is not None and ef < 40:
                score += 0.25
            if ef is not None and ef < 35:
                score += 0.05
        else:
            score += 0.2

    elif specialty == "nephrology":
        if has_ckd:
            score += 0.65
            if egfr is not None and egfr < 30:
                score += 0.35
            elif egfr is not None and egfr < 60:
                score += 0.10
        else:
            score += 0.1

    elif specialty == "endocrinology":
        if has_diabetes:
            score += 0.5
            if hba1c is not None and hba1c > 9.0:
                score += 0.35
            elif hba1c is not None and hba1c > 7.5:
                score += 0.15
        else:
            score += 0.1

    return min(score, 1.0)


def _compute_drug_risk(risks: list[str], recommendation: str = "") -> float:
    """Return a medication safety score: higher means fewer unresolved concerns."""
    if not risks:
        return 1.0

    rec_l = recommendation.lower()
    resolves_metformin_risk = (
        "metformin" in rec_l
        and any(action in rec_l for action in ["stop", "hold", "discontinue"])
    )

    high_risk_keywords = [
        "contraindicated",
        "absolute contraindication",
        "lactic acidosis",
        "hyperkalemia",
        "fatal",
    ]
    medium_risk_keywords = [
        "caution",
        "monitor",
        "reduce dose",
        "hypoglycemia",
        "hypotension",
        "volume depletion",
        "worsening renal",
    ]

    high_count = 0
    medium_count = 0
    for risk in risks:
        risk_l = risk.lower()
        if resolves_metformin_risk and (
            "metformin" in risk_l or "lactic acidosis" in risk_l or "contraindicat" in risk_l
        ):
            continue
        high_count += sum(1 for kw in high_risk_keywords if kw in risk_l)
        medium_count += sum(1 for kw in medium_risk_keywords if kw in risk_l)

    penalty = (high_count * 0.3) + (medium_count * 0.1)
    return max(0.25, 1.0 - penalty)


def _evidence_for_specialty(specialty: str, vals: dict) -> str:
    if specialty == "cardiology":
        return "Class I Level A" if vals["has_hf"] else "Class IIa Level B"
    if specialty == "nephrology":
        return "Grade 1A" if vals["egfr"] is not None and vals["egfr"] < 30 else "Grade 1B"
    if specialty == "endocrinology":
        return "Level A"
    return ""


def _preferences_for_patient(vals: dict) -> Preferences:
    """Shift ranking toward medication safety for eGFR<30 + metformin cases."""
    if vals["has_metformin"] and vals["egfr"] is not None and vals["egfr"] < 30:
        return Preferences(
            evidence_level=0.20,
            patient_match=0.25,
            drug_interaction_risk=0.35,
            guideline_priority=0.20,
        )
    return Preferences()


def _fallback_specialist_payloads(vals: dict) -> dict[str, dict[str, Any]]:
    cardio_rec = (
        "Given T2DM+CKD and no HF, consider dapagliflozin 10mg daily for "
        "cardiovascular and renal risk reduction. Continue lisinopril; no HF "
        "beta-blocker is indicated solely for this presentation."
    )
    if vals["has_hf"] and vals["ef"] is not None and vals["ef"] < 40:
        cardio_rec = (
            f"Start carvedilol 3.125mg BID for HFrEF (EF={vals['ef']}%) plus "
            "dapagliflozin 10mg daily. Continue lisinopril with K+/Cr monitoring."
        )
    if vals["has_hf"] and vals["has_ckd"]:
        cardio_risks = ["hypotension", "hyperkalemia"]
    elif vals["has_ckd"]:
        cardio_risks = ["monitor renal function"]
    else:
        cardio_risks = ["hypotension"]

    nephro_rec = "Stop metformin. Start dapagliflozin 10mg daily. Continue lisinopril."
    nephro_risks = ["monitor renal function"]
    if vals["has_metformin"] and vals["egfr"] is not None and vals["egfr"] < 30:
        nephro_rec = (
            f"STOP metformin immediately (eGFR={vals['egfr']}, below the <30 "
            "contraindication threshold). Start dapagliflozin 10mg daily if "
            "tolerated and continue lisinopril with K+/Cr monitoring."
        )
        nephro_risks = ["lactic acidosis risk if metformin continued", "monitor renal function"]
    elif vals["egfr"] is not None and vals["egfr"] >= 30:
        nephro_rec = (
            f"Continue metformin with monitoring (eGFR={vals['egfr']}>=30). "
            "Consider dapagliflozin 10mg daily for kidney protection and continue lisinopril."
        )

    endo_rec = "Stop metformin. Start dapagliflozin 10mg daily. Consider GLP-1 RA."
    if vals["has_metformin"] and vals["egfr"] is not None and vals["egfr"] < 30:
        endo_rec = (
            f"STOP metformin (eGFR={vals['egfr']}<30, contraindicated). Start "
            "dapagliflozin 10mg daily and consider semaglutide for additional CV benefit."
        )
    elif vals["hba1c"] is not None and vals["hba1c"] > 8:
        endo_rec = (
            f"HbA1c={vals['hba1c']}% above target. Start dapagliflozin 10mg daily, "
            "consider GLP-1 RA, and adjust current diabetes regimen."
        )
    endo_risks = ["volume depletion"] if vals["has_diabetes"] else []

    return {
        "cardiology": {
            "specialty": "cardiology",
            "recommendation": cardio_rec,
            "risks": cardio_risks,
            "citation": "ACC/AHA 2022 Sec 7.3.1",
            "source": "fallback",
        },
        "nephrology": {
            "specialty": "nephrology",
            "recommendation": nephro_rec,
            "risks": nephro_risks,
            "citation": "KDIGO 2024 Ch 3",
            "source": "fallback",
        },
        "endocrinology": {
            "specialty": "endocrinology",
            "recommendation": endo_rec,
            "risks": endo_risks,
            "citation": "ADA 2025 Sec 10",
            "source": "fallback",
        },
    }


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_specialist_json(text: str, expected_specialty: str) -> dict[str, Any]:
    """Parse and validate a specialist's JSON recommendation."""
    raw = _strip_markdown_fence(text)
    decoder = json.JSONDecoder()
    candidates = []
    for idx, char in enumerate(raw):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw[idx:])
            if isinstance(parsed, dict):
                candidates.append(parsed)
        except json.JSONDecodeError:
            continue

    if not candidates:
        raise SpecialistPayloadError("specialist response did not contain a JSON object")

    last_error: SpecialistPayloadError | None = None
    for parsed in reversed(candidates):
        try:
            return _validate_specialist_payload(parsed, expected_specialty)
        except SpecialistPayloadError as exc:
            last_error = exc
    raise last_error or SpecialistPayloadError("specialist response did not contain a usable JSON object")


def _validate_specialist_payload(parsed: dict[str, Any], expected_specialty: str) -> dict[str, Any]:
    required = ("specialty", "recommendation", "risks", "citation")
    missing = [field for field in required if field not in parsed]
    if missing:
        raise SpecialistPayloadError(f"specialist JSON missing required fields: {', '.join(missing)}")

    specialty = str(parsed["specialty"]).strip().lower()
    if specialty != expected_specialty:
        raise SpecialistPayloadError(
            f"expected specialty={expected_specialty}, got specialty={specialty or '[empty]'}"
        )

    recommendation = str(parsed["recommendation"]).strip()
    citation = str(parsed["citation"]).strip()
    risks = parsed["risks"]
    if not recommendation:
        raise SpecialistPayloadError("recommendation is empty")
    if not citation:
        raise SpecialistPayloadError("citation is empty")
    if not isinstance(risks, list) or not all(isinstance(r, str) and r.strip() for r in risks):
        raise SpecialistPayloadError("risks must be a list of non-empty strings")

    return {
        "specialty": specialty,
        "recommendation": recommendation,
        "risks": [r.strip() for r in risks],
        "citation": citation,
        "source": "llm",
    }


def _build_recommendations(
    patient_message: str,
    specialist_payloads: dict[str, dict[str, Any]] | None = None,
) -> list[Recommendation]:
    vals = _parse_patient_values(patient_message)
    match_args = {
        "has_hf": vals["has_hf"],
        "has_ckd": vals["has_ckd"],
        "has_diabetes": vals["has_diabetes"],
        "ef": vals["ef"],
        "egfr": vals["egfr"],
        "hba1c": vals["hba1c"],
    }
    fallback_payloads = _fallback_specialist_payloads(vals)
    payloads = {
        specialty: {
            **fallback_payloads[specialty],
            **((specialist_payloads or {}).get(specialty) or {}),
        }
        for specialty in SPECIALIST_ORDER
    }

    recommendations = []
    for specialty in SPECIALIST_ORDER:
        payload = payloads[specialty]
        evidence = _evidence_for_specialty(specialty, vals)
        recommendations.append(
            Recommendation(
                specialty=specialty,
                recommendation=payload["recommendation"],
                confidence=0.85 if payload.get("source") == "llm" else 0.80,
                evidence_level=evidence,
                evidence_score=_normalize_evidence(evidence),
                patient_match=_compute_patient_match(specialty, **match_args),
                drug_interaction_risk=_compute_drug_risk(payload["risks"], payload["recommendation"]),
                guideline_priority=_compute_guideline_priority(evidence),
                risk_flags=payload["risks"],
                citation=payload["citation"],
            )
        )
    return recommendations


def _format_orchestration_result(result: dict) -> str:
    patient = result["patient"]
    rows = []
    for rec in result["ranked_recommendations"]:
        rows.append(
            "| {rank} | {specialty} | {score:.3f} | {recommendation} |".format(
                rank=rec["rank"],
                specialty=rec["specialty"].title(),
                score=rec["total_score"],
                recommendation=rec["recommendation"],
            )
        )

    top = result["top_pick"] or {}
    consensus = result["consensus"]
    citations = ", ".join(
        dict.fromkeys(rec["citation"] for rec in result["ranked_recommendations"])
    )
    fallback_note = ""
    if result.get("specialist_errors"):
        failed = ", ".join(sorted(result["specialist_errors"]))
        fallback_note = f"\n**Fallback Used:** {failed} specialist recommendation(s) used deterministic backup.\n"
    fhir_note = ""
    if result.get("fhir_errors"):
        fhir_note = "\n**FHIR Note:** FHIR context was received, but some resources could not be loaded; text summary fallback was used where needed.\n"

    conflicts = result.get("conflicts", ["No major conflicts detected"])
    conflict_lines = "\n".join(f"- {c}" for c in conflicts)

    return (
        "---\n"
        "## TOPSIS Clinical Decision\n"
        f"**Patient:** {patient['age_sex']}, {patient['conditions']}\n"
        f"**Data Source:** {result['data_source']}\n"
        f"**Specialist Mode:** {result['specialist_mode']}\n"
        f"{fallback_note}{fhir_note}\n"
        "| Rank | Specialty | Score | Recommendation |\n"
        "|:----:|:---------:|:-----:|:---------------|\n"
        + "\n".join(rows)
        + "\n\n"
        f"**Top Pick:** {top.get('specialty', 'N/A').title()} - "
        f"{top.get('recommendation', 'No recommendation available')}\n\n"
        "### Action Plan\n"
        f"1. {consensus[0]}\n"
        f"2. {consensus[1]}\n"
        f"3. {consensus[2]}\n"
        "4. Monitor blood pressure, potassium, creatinine/eGFR, volume status, and glucose.\n"
        "5. Reassess tolerability and labs within 1-2 weeks after medication changes.\n\n"
        "### Key Conflicts Resolved\n"
        f"{conflict_lines}\n\n"
        f"**Citations:** {citations}\n"
        "**Disclaimer:** Advisory only. Final decisions rest with the physician.\n"
        "---"
    )


def _build_consensus(vals: dict) -> list[str]:
    """Build consensus action items from patient values — no hallucination."""
    items = []

    # Metformin: only mention if patient is actually on it
    if vals["has_metformin"] and vals["egfr"] is not None and vals["egfr"] < 30:
        items.append(f"STOP metformin immediately (eGFR={vals['egfr']}<30, contraindicated)")
    elif vals["has_metformin"] and vals["egfr"] is not None and vals["egfr"] < 45:
        items.append(f"REDUCE metformin dose (eGFR={vals['egfr']}, 30-45 range)")
    elif vals["has_metformin"]:
        items.append(f"Continue metformin with monitoring (eGFR={vals['egfr']})")

    # SGLT2i: describe benefit based on which conditions the patient actually has
    benefits = []
    if vals["has_hf"]:
        benefits.append("HF")
    if vals["has_ckd"] and vals["egfr"] is not None and vals["egfr"] < 60:
        benefits.append("CKD")
    if vals["has_diabetes"]:
        benefits.append("T2DM")
    if benefits:
        items.append(f"START SGLT2i (dapagliflozin 10mg) — benefit for {'+'.join(benefits)}")
    else:
        items.append("SGLT2i not strongly indicated (no HF/CKD/DM)")

    # Beta-blocker: only for HFrEF
    if vals["has_hf"] and vals["ef"] is not None and vals["ef"] < 40:
        items.append("START carvedilol for HFrEF (Class I)")
    elif vals["has_hf"] and vals["ef"] is not None and vals["ef"] >= 50:
        items.append("No HF-directed beta-blocker indicated (HFpEF)")
    elif vals["has_hf"]:
        items.append("Consider beta-blocker based on HF phenotype")

    # Ensure at least 3 items
    while len(items) < 3:
        items.append("Monitor clinical status and reassess")

    return items[:3]


def _build_conflicts(vals: dict) -> list[str]:
    """Build conflict resolution notes from patient values — no hallucination."""
    conflicts = []
    if vals["has_metformin"]:
        if vals["egfr"] is not None and vals["egfr"] < 30:
            conflicts.append("Metformin vs CKD: STOP metformin — eGFR <30 is absolute contraindication (KDIGO/ADA)")
        elif vals["egfr"] is not None and vals["egfr"] < 60:
            conflicts.append("Metformin vs CKD: dose adjustment may be needed (eGFR 30-60)")
    if vals["has_hf"] and vals["has_ckd"] and vals["egfr"] is not None and vals["egfr"] < 30:
        conflicts.append("ACEi/ARB vs CKD: continue with K+/Cr monitoring — renoprotective but hyperkalemia risk")
    if vals["has_hf"] and vals["has_diabetes"]:
        conflicts.append("HF vs T2DM: SGLT2i addresses both — preferred over agents that treat only one condition")
    if not conflicts:
        conflicts.append("No major conflicts detected for this patient profile")
    return conflicts


def run_orchestration(
    patient_message: str,
    specialist_payloads: dict[str, dict[str, Any]] | None = None,
    *,
    data_source: str = "text summary",
    specialist_errors: dict[str, str] | None = None,
    fhir_errors: list[str] | None = None,
) -> dict:
    """Rank specialist recommendations with deterministic TOPSIS scoring."""
    vals = _parse_patient_values(patient_message)
    recs = _build_recommendations(patient_message, specialist_payloads)
    scored = score_topsis(recs, preferences=_preferences_for_patient(vals))

    results = []
    for scored_rec in scored:
        rec = scored_rec.recommendation
        results.append({
            "rank": scored_rec.rank,
            "specialty": rec.specialty,
            "recommendation": rec.recommendation,
            "total_score": scored_rec.total_score,
            "breakdown": scored_rec.breakdown,
            "evidence_level": rec.evidence_level,
            "risk_flags": rec.risk_flags,
            "citation": rec.citation,
        })

    specialist_errors = specialist_errors or {}
    result = {
        "status": "success",
        "patient": {
            "age_sex": _parse_age_sex(patient_message),
            "conditions": _condition_summary(vals),
        },
        "data_source": data_source,
        "specialist_mode": (
            "llm specialists with deterministic fallback"
            if specialist_payloads
            else "deterministic fallback"
        ),
        "specialist_errors": specialist_errors,
        "fhir_errors": fhir_errors or [],
        "ranked_recommendations": results,
        "top_pick": results[0] if results else None,
        "consensus": _build_consensus(vals),
        "conflicts": _build_conflicts(vals),
    }
    result["formatted_output"] = _format_orchestration_result(result)
    return result


def _content_to_text(content) -> str:
    parts = getattr(content, "parts", None) or []
    text_parts = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            text_parts.append(text)
    return "\n".join(text_parts).strip()


def _coerce_fhir_data(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _extract_fhir_context_from_invocation(ctx: InvocationContext) -> dict[str, Any] | None:
    run_config = getattr(ctx, "run_config", None)
    custom_metadata = getattr(run_config, "custom_metadata", None) if run_config else None
    a2a_metadata = custom_metadata.get("a2a_metadata") if isinstance(custom_metadata, dict) else None
    if not isinstance(a2a_metadata, dict):
        return None
    for key, value in a2a_metadata.items():
        if FHIR_CONTEXT_KEY in str(key):
            return _coerce_fhir_data(value)
    return None


def _has_success(result: dict[str, Any]) -> bool:
    return isinstance(result, dict) and result.get("status") == "success"


def _insufficient_patient_info_response(
    *,
    data_source: str,
    fhir_errors: list[str] | None = None,
) -> dict[str, Any]:
    message = (
        "---\n"
        "## More Patient Information Needed\n"
        f"**Data Source:** {data_source}\n"
        "**Specialist Mode:** not run - insufficient patient context\n\n"
        "I do not have enough patient-specific clinical information to run a safe "
        "multi-specialty orchestration. Please provide at least a diagnosis or key "
        "objective metric plus supporting context.\n\n"
        "Useful details include: age/sex, HFrEF/LVEF, CKD/eGFR, diabetes/HbA1c, "
        "current medications, and the clinical question.\n\n"
        "**Example:** 68M with HFrEF LVEF 32%, CKD stage 4 eGFR 28, T2DM HbA1c 8.2%, "
        "on lisinopril, metformin, and furosemide. Run the full multi-specialty orchestration.\n"
        "---"
    )
    return {
        "status": "needs_input",
        "patient": {"age_sex": "patient", "conditions": "insufficient information"},
        "data_source": data_source,
        "specialist_mode": "not run - insufficient patient context",
        "specialist_errors": {"all": "insufficient patient context"},
        "fhir_errors": fhir_errors or [],
        "ranked_recommendations": [],
        "top_pick": None,
        "consensus": [],
        "formatted_output": message,
    }


def _summarize_fhir_payloads(
    original_message: str,
    demographics: dict,
    conditions: dict,
    medications: dict,
    observations: dict,
) -> str:
    lines = ["FHIR-derived patient summary:"]
    if _has_success(demographics):
        lines.append(
            "Patient: {name}, DOB {dob}, gender {gender}.".format(
                name=demographics.get("name", "Unknown"),
                dob=demographics.get("birth_date", "unknown"),
                gender=demographics.get("gender", "unknown"),
            )
        )
    if _has_success(conditions):
        names = [c.get("condition", "Unknown") for c in conditions.get("conditions", [])]
        lines.append("Active conditions: " + (", ".join(names[:10]) if names else "none returned") + ".")
    if _has_success(medications):
        meds = [
            f"{m.get('medication', 'Unknown')} ({m.get('dosage', 'dose not specified')})"
            for m in medications.get("medications", [])
        ]
        lines.append("Current medications: " + (", ".join(meds[:12]) if meds else "none returned") + ".")
    if _has_success(observations):
        obs_lines = []
        for obs in observations.get("observations", [])[:12]:
            value = obs.get("value")
            unit = obs.get("unit") or ""
            obs_lines.append(f"{obs.get('observation', 'Observation')}={value}{unit}")
        lines.append("Recent observations: " + (", ".join(obs_lines) if obs_lines else "none returned") + ".")
    if original_message:
        lines.append("Clinician request/context: " + original_message)
    return "\n".join(lines)


async def _patient_message_from_fhir_or_text(
    ctx: InvocationContext,
    original_message: str,
) -> tuple[str, str, list[str]]:
    fhir_data = _extract_fhir_context_from_invocation(ctx)
    fhir_url = fhir_data.get("fhirUrl", "") if fhir_data else ""
    fhir_token = fhir_data.get("fhirToken", "") if fhir_data else ""
    patient_id = fhir_data.get("patientId", "") if fhir_data else ""
    logger.warning("FHIR DEBUG - fhirUrl: %s", fhir_url)
    logger.warning("FHIR DEBUG - patientId: %s", patient_id)
    logger.warning("FHIR DEBUG - token present: %s", bool(fhir_token))
    logger.warning(
        "FHIR DEBUG - data_source will be: %s",
        "fhir" if (fhir_url and patient_id) else "text",
    )
    if not fhir_data:
        return original_message, "text summary", []

    required = ("fhirUrl", "fhirToken", "patientId")
    missing = [field for field in required if not fhir_data.get(field)]
    if missing:
        return original_message, "text summary (FHIR context incomplete)", [
            f"FHIR context missing: {', '.join(missing)}"
        ]

    tool_context = SimpleNamespace(state={
        "fhir_url": fhir_data["fhirUrl"],
        "fhir_token": fhir_data["fhirToken"],
        "patient_id": fhir_data["patientId"],
    })

    demographics, conditions, medications, observations = await asyncio.gather(
        asyncio.to_thread(get_patient_demographics, tool_context),
        asyncio.to_thread(get_active_conditions, tool_context),
        asyncio.to_thread(get_active_medications, tool_context),
        asyncio.to_thread(get_recent_observations, "laboratory", tool_context),
    )
    results = [demographics, conditions, medications, observations]
    logger.warning("FHIR TOOL - demographics: %s", demographics)
    logger.warning("FHIR TOOL - conditions: %s", conditions)
    logger.warning("FHIR TOOL - medications: %s", medications)
    logger.warning("FHIR TOOL - observations: %s", observations)
    errors = [
        str(result.get("error_message", "FHIR resource failed"))
        for result in results
        if isinstance(result, dict) and result.get("status") == "error"
    ]
    if errors and not any(_has_success(result) for result in results):
        return original_message, "text summary (FHIR unavailable)", errors

    summary = _summarize_fhir_payloads(original_message, demographics, conditions, medications, observations)
    return summary, "FHIR context", errors


def _model_api_key_available(model_name: str) -> bool:
    if model_name.startswith("deepseek/"):
        return bool(os.getenv("DEEPSEEK_API_KEY"))
    if model_name.startswith("openai/"):
        return bool(os.getenv("OPENAI_API_KEY"))
    if model_name.startswith("anthropic/"):
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if model_name.startswith("gemini/"):
        return bool(os.getenv("GOOGLE_API_KEY"))
    return True


def _specialist_model_name(specialty: str) -> str:
    env_name = f"{specialty.upper()}_AGENT_MODEL"
    return os.getenv(env_name, DEFAULT_MODEL)


def _specialist_llms_enabled() -> bool:
    raw = os.getenv("CONSILIUM_USE_SPECIALIST_LLMS", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


async def _run_adk_specialist(specialty: str, patient_message: str) -> str:
    agent = SPECIALIST_AGENTS[specialty]
    app_name = f"{specialty}_agent"
    user_id = "consilium_orchestrator"
    session_id = f"{specialty}-{uuid.uuid4()}"
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={},
    )
    runner = Runner(app_name=app_name, agent=agent, session_service=session_service)
    specialist_prompt = (
        "Return ONLY the final JSON object requested by your instruction. "
        "Do not include analysis, markdown, or any text before or after JSON.\n\n"
        f"Patient summary:\n{patient_message}"
    )
    message = types.Content(role="user", parts=[types.Part(text=specialist_prompt)])

    text_events = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        text = _content_to_text(getattr(event, "content", None))
        if text:
            text_events.append(text)
    if not text_events:
        raise SpecialistPayloadError("specialist returned no text content")
    return text_events[-1]


async def _collect_specialist_payloads(patient_message: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    payloads: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    if not _specialist_llms_enabled():
        return payloads, {"all": "CONSILIUM_USE_SPECIALIST_LLMS disabled"}

    runnable_specialties = []
    for specialty in SPECIALIST_ORDER:
        model_name = _specialist_model_name(specialty)
        if not _model_api_key_available(model_name):
            errors[specialty] = f"missing API key for configured model {model_name}"
        else:
            runnable_specialties.append(specialty)

    async def invoke(specialty: str):
        raw = await _run_adk_specialist(specialty, patient_message)
        return specialty, parse_specialist_json(raw, specialty)

    results = await asyncio.gather(
        *(invoke(specialty) for specialty in runnable_specialties),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, Exception):
            logger.warning("specialist_call_failed error=%s", result)
            continue
        specialty, payload = result
        payloads[specialty] = payload

    for specialty in runnable_specialties:
        if specialty not in payloads and specialty not in errors:
            errors[specialty] = "specialist call failed or returned invalid JSON"
    return payloads, errors


async def run_orchestration_async(ctx: InvocationContext, patient_message: str) -> dict:
    patient_summary, data_source, fhir_errors = await _patient_message_from_fhir_or_text(ctx, patient_message)
    if data_source.startswith("text summary") and not _has_minimum_patient_info(patient_summary):
        logger.warning(
            "patient_info_insufficient skipping_specialists data_source=%s text_preview=%s",
            data_source,
            (patient_summary or "")[:200],
        )
        return _insufficient_patient_info_response(
            data_source=data_source,
            fhir_errors=fhir_errors,
        )

    specialist_payloads, specialist_errors = await _collect_specialist_payloads(patient_summary)
    return run_orchestration(
        patient_summary or patient_message,
        specialist_payloads=specialist_payloads,
        data_source=data_source,
        specialist_errors=specialist_errors,
        fhir_errors=fhir_errors,
    )


class ConsiliumOrchestratorAgent(BaseAgent):
    """Runs specialist ADK agents and ranks their outputs with TOPSIS."""

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        patient_message = _content_to_text(ctx.user_content)
        result = await run_orchestration_async(ctx, patient_message)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(
                role="model",
                parts=[types.Part(text=result["formatted_output"])],
            ),
            turn_complete=True,
            custom_metadata={
                "ranked_recommendations": result["ranked_recommendations"],
                "data_source": result["data_source"],
                "specialist_mode": result["specialist_mode"],
                "specialist_errors": result["specialist_errors"],
                "fhir_errors": result["fhir_errors"],
            },
        )


root_agent = ConsiliumOrchestratorAgent(
    name="asm_orchestrator",
    description="Multi-specialty orchestrator for HF+T2DM+CKD.",
)
