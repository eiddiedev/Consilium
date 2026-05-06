"""asm_orchestrator — Multi-specialty clinical orchestrator.

Single LLM call generates all 3 specialist recommendations.
TOPSIS scores are computed DETERMINISTICALLY from structured clinical rules,
not from LLM-assessed scores — eliminating inter-call variability.

Scoring rules:
- evidence_score: from guideline class/grade string (deterministic)
- patient_match: computed from clinical thresholds (eGFR, LVEF, HbA1c)
- drug_interaction_risk: computed from risk flag keywords (deterministic)
- guideline_priority: from guideline class/grade string (deterministic)
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

from shared.fhir_hook import extract_fhir_context
from tools.topsis import Recommendation, score_topsis, _normalize_evidence

_model_name = os.getenv("ORCHESTRATOR_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)


# ──────────────────────────────────────────────
# Deterministic scoring functions
# ──────────────────────────────────────────────

def _compute_patient_match(specialty: str, has_hf: bool, has_ckd: bool, has_dm: bool,
                           ef: float | None, egfr: float | None, hba1c: float | None) -> float:
    """Compute patient match score from clinical thresholds.

    Higher score = more relevant to this patient's specific conditions.
    """
    score = 0.5  # baseline

    if specialty == "cardiology":
        if has_hf and ef is not None:
            if ef < 40:
                score += 0.4  # HFrEF — high relevance
            elif ef < 50:
                score += 0.2  # HFmrEF — moderate
            else:
                score += 0.1  # HFpEF — lower
        if has_ckd and egfr is not None and egfr < 30:
            score += 0.1  # CKD affects HF management

    elif specialty == "nephrology":
        if has_ckd and egfr is not None:
            if egfr < 15:
                score += 0.45  # CKD G5 — critical
            elif egfr < 30:
                score += 0.4   # CKD G4 — high relevance
            elif egfr < 45:
                score += 0.25  # CKD G3b
            elif egfr < 60:
                score += 0.15  # CKD G3a
        if has_dm:
            score += 0.05  # diabetes compounds CKD

    elif specialty == "endocrinology":
        if has_dm and hba1c is not None:
            if hba1c > 9:
                score += 0.35  # very poor control
            elif hba1c > 8:
                score += 0.25  # poor control
            elif hba1c > 7:
                score += 0.15  # above target
            else:
                score += 0.05  # at target
        if has_ckd and egfr is not None and egfr < 30:
            score += 0.1  # CKD constrains diabetes meds

    return min(score, 1.0)


def _compute_drug_risk(risk_flags: list[str]) -> float:
    """Compute drug interaction risk from risk flag keywords."""
    if not risk_flags:
        return 0.0
    high = {"contraindicated", "stop", "severe", "avoid", "critical", "immediate", "danger", "lactic acidosis"}
    moderate = {"caution", "monitor", "watch", "reduce", "adjust", "titrate"}
    score = 0.0
    for flag in risk_flags:
        flag_lower = flag.lower()
        if any(kw in flag_lower for kw in high):
            score += 0.2
        elif any(kw in flag_lower for kw in moderate):
            score += 0.05
        else:
            score += 0.08
    return min(score, 1.0)


def _compute_guideline_priority(evidence_level: str) -> float:
    """Map guideline class/grade to priority score (deterministic)."""
    level = evidence_level.strip().lower()
    mapping = {
        "class i": 1.0, "class ii": 0.75, "class iia": 0.75, "class iib": 0.5,
        "class iii": 0.25, "1a": 1.0, "1b": 0.85, "2a": 0.6, "2b": 0.4,
        "a": 1.0, "level a": 1.0, "b": 0.75, "level b": 0.75,
        "c": 0.5, "level c": 0.5, "e": 0.3, "level e": 0.3,
    }
    for key, val in mapping.items():
        if key in level:
            return val
    return 0.5


def generate_decision(
    # Patient context
    has_heart_failure: bool,
    has_ckd: bool,
    has_diabetes: bool,
    ejection_fraction: float,
    egfr: float,
    hba1c: float,
    # Cardiology
    cardiology_recommendation: str,
    cardiology_evidence: str,
    cardiology_risks: list[str],
    cardiology_citation: str,
    # Nephrology
    nephrology_recommendation: str,
    nephrology_evidence: str,
    nephrology_risks: list[str],
    nephrology_citation: str,
    # Endocrinology
    endocrinology_recommendation: str,
    endocrinology_evidence: str,
    endocrinology_risks: list[str],
    endocrinology_citation: str,
) -> dict:
    """Receive 3 specialist recommendations + patient context, compute TOPSIS deterministically.

    The LLM provides clinical analysis (recommendations, evidence levels, risks, citations).
    ALL scores are computed from deterministic rules — no LLM-assessed scores.
    """
    specs = [
        ("cardiology", cardiology_recommendation, cardiology_evidence,
         cardiology_risks, cardiology_citation),
        ("nephrology", nephrology_recommendation, nephrology_evidence,
         nephrology_risks, nephrology_citation),
        ("endocrinology", endocrinology_recommendation, endocrinology_evidence,
         endocrinology_risks, endocrinology_citation),
    ]

    recs = []
    for (specialty, rec, evidence, risks, citation) in specs:
        recs.append(Recommendation(
            specialty=specialty,
            recommendation=rec,
            confidence=0.85,  # uniform — not used in TOPSIS ranking
            evidence_level=evidence,
            evidence_score=_normalize_evidence(evidence),
            patient_match=_compute_patient_match(
                specialty, has_heart_failure, has_ckd, has_diabetes,
                ejection_fraction, egfr, hba1c,
            ),
            drug_interaction_risk=_compute_drug_risk(risks),
            guideline_priority=_compute_guideline_priority(evidence),
            risk_flags=risks,
            citation=citation,
        ))

    scored = score_topsis(recs)

    results = []
    for s in scored:
        results.append({
            "rank": s.rank,
            "specialty": s.recommendation.specialty,
            "recommendation": s.recommendation.recommendation,
            "total_score": s.total_score,
            "breakdown": s.breakdown,
            "evidence_level": s.recommendation.evidence_level,
            "risk_flags": s.recommendation.risk_flags,
            "citation": s.recommendation.citation,
            "reasoning": s.reasoning,
        })

    return {
        "status": "success",
        "ranked_recommendations": results,
        "top_pick": results[0] if results else None,
    }


root_agent = Agent(
    name="asm_orchestrator",
    model=_model,
    description=(
        "Multi-specialty clinical orchestrator for HF+T2DM+CKD patients. "
        "Generates cardiology, nephrology, and endocrinology recommendations, "
        "scores them with TOPSIS, and produces an explainable ranked decision."
    ),
    instruction=(
        "You are the ASM Clinical Orchestrator for complex chronic disease patients "
        "with heart failure (HF), type 2 diabetes (T2DM), and chronic kidney disease (CKD).\n\n"
        "You will receive a patient summary. Analyze it from THREE specialist perspectives "
        "and call generate_decision ONCE with all fields filled in.\n\n"
        "You MUST extract these patient values from the message and pass them to generate_decision:\n"
        "- has_heart_failure: true if patient has HF (heart failure, HFrEF, HFpEF, LVEF mentioned)\n"
        "- has_ckd: true if patient has CKD (chronic kidney disease, low eGFR, renal impairment)\n"
        "- has_diabetes: true if patient has T2DM or diabetes (high HbA1c, diabetes mellitus)\n"
        "- ejection_fraction: the LVEF number (e.g. 32 for 'LVEF 32%')\n"
        "- egfr: the eGFR number (e.g. 28 for 'eGFR 28')\n"
        "- hba1c: the HbA1c number (e.g. 8.2 for 'HbA1c 8.2%')\n\n"
        "IMPORTANT: If the patient has these conditions, set the booleans to TRUE. "
        "Do NOT set them to false if the patient clearly has the condition.\n\n"
        "For EACH specialist, provide:\n"
        "- recommendation: 2-3 sentences with specific drug names and doses\n"
        "- evidence: guideline class/grade (e.g. 'Class I', '1A', 'Level A')\n"
        "- risks: list of specific safety concerns (be conservative — only real risks)\n"
        "- citation: specific guideline section (e.g. 'ACC/AHA 2022 Sec 7.3.1')\n\n"
        "SPECIALIST GUIDELINES:\n\n"
        "1. CARDIOLOGY (ACC/AHA 2022 HF Guidelines):\n"
        "   - Beta-blocker for HFrEF: Class I, Sec 7.3.1 — MANDATORY if EF<40%\n"
        "   - ACEi/ARB: Class I, Sec 7.3.1\n"
        "   - SGLT2i: Class IIa, Sec 7.3.2\n"
        "   - ARNI preferred over ACEi: Class IIa, Sec 7.3.1\n"
        "   - NSAIDs: Class III (Harm), Sec 7.6 — AVOID\n"
        "   - MRA: Class IIa but caution if eGFR <30, Sec 7.3.3\n\n"
        "2. NEPHROLOGY (KDIGO 2024 CKD Guidelines):\n"
        "   - Metformin: CONTRAINDICATED if eGFR <30, Grade 1A, Ch 4\n"
        "   - SGLT2i: renoprotective if eGFR ≥20, Grade 1A, Ch 3\n"
        "   - NSAIDs: avoid, Grade 2A, Ch 4\n"
        "   - ACEi/ARB: renoprotective, Grade 1B, Ch 3\n"
        "   - Finerenone: consider for CKD+T2DM, Grade 2A, Ch 3\n\n"
        "3. ENDOCRINOLOGY (ADA 2025 Standards of Care):\n"
        "   - SGLT2i preferred if CKD or HF, Level A, Sec 10\n"
        "   - GLP-1 RA: CV benefit, Level A, Sec 10\n"
        "   - Metformin contraindicated if eGFR <30, Sec 9\n"
        "   - TZDs: contraindicated in HF, Level B, Sec 10\n"
        "   - HbA1c target: <7% for most, Level A, Sec 6\n\n"
        "Call generate_decision exactly ONCE as your ONLY tool call."
    ),
    tools=[
        FunctionTool(func=generate_decision),
    ],
    before_model_callback=extract_fhir_context,
)
