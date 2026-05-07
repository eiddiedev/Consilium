"""asm_orchestrator — Multi-specialty clinical orchestrator.

Sub-agents return JSON directly (no tools, single LLM call each).
Orchestrator parses results, runs deterministic TOPSIS, formats output.

ALL scoring is deterministic — no LLM-assessed scores.
"""
import json
import os
import re

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool

from cardiology_agent.agent import root_agent as cardiology_agent
from nephrology_agent.agent import root_agent as nephrology_agent
from endocrinology_agent.agent import root_agent as endocrinology_agent
from shared.fhir_hook import extract_fhir_context
from tools.topsis import Recommendation, score_topsis

_model_name = os.getenv("ORCHESTRATOR_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name, max_tokens=800)


# ════════════════════════════════════════════════════════════════
# Deterministic Evidence Score Lookup Table
# ════════════════════════════════════════════════════════════════

EVIDENCE_SCORE_TABLE = {
    # ACC/AHA classes
    "class i level a":   1.00,
    "class i level b-r": 0.90,
    "class i level b-nr": 0.85,
    "class i level c":   0.75,
    "class i":           0.85,  # generic Class I without sub-level
    "class iia level a":  0.70,
    "class iia level b-r": 0.65,
    "class iia level b-nr": 0.60,
    "class iia level c":  0.55,
    "class iia":          0.60,  # generic Class IIa
    "class iib level a":  0.45,
    "class iib level b":  0.35,
    "class iib level c":  0.25,
    "class iib":          0.35,  # generic Class IIb
    "class iii":          0.00,
    # KDIGO grades
    "grade 1a": 1.00,
    "grade 1b": 0.85,
    "grade 1c": 0.70,
    "grade 1d": 0.55,
    "grade 2a": 0.45,
    "grade 2b": 0.35,
    "grade 2c": 0.25,
    "grade 2d": 0.15,
    "1a":       1.00,  # shorthand
    "1b":       0.85,
    "2a":       0.45,
    # ADA levels
    "level a":  1.00,
    "level b":  0.75,
    "level c":  0.50,
    "level e":  0.30,
}

# Guideline priority: maps ACC/AHA class to priority score
GUIDELINE_PRIORITY_TABLE = {
    "class i":   1.0,
    "class iia":  0.7,
    "class iib":  0.4,
    "class iii":  0.0,
    # KDIGO
    "grade 1a": 1.0,
    "grade 1b": 0.85,
    "grade 1c": 0.70,
    "grade 2a": 0.45,
    "grade 2b": 0.35,
    # ADA
    "level a": 1.0,
    "level b": 0.75,
    "level c": 0.50,
    "level e": 0.30,
}


def _infer_evidence_from_recommendation(specialty: str, recommendation: str) -> str:
    """Deterministically infer evidence level from recommendation content.

    Instead of relying on LLM to provide evidence level (which varies),
    we infer it from the drugs/treatments mentioned and the specialty.
    This ensures the same recommendation always gets the same evidence level.
    """
    rec = recommendation.lower()

    if specialty == "cardiology":
        if "beta-blocker" in rec or "carvedilol" in rec or "metoprolol" in rec or "bisoprolol" in rec:
            return "Class I"  # Beta-blocker for HFrEF = Class I
        if "sacubitril" in rec or "arni" in rec:
            return "Class I"  # ARNI preferred over ACEi = Class I
        if "sglt2" in rec or "dapagliflozin" in rec or "empagliflozin" in rec:
            return "Class IIa"  # SGLT2i for HF = Class IIa
        if "acei" in rec or "lisinopril" in rec or "arb" in rec:
            return "Class I"  # ACEi/ARB = Class I
        return "Class IIa"  # default for cardiology

    elif specialty == "nephrology":
        if "stop metformin" in rec or "discontinue metformin" in rec or "contraindicated" in rec:
            return "Grade 1A"  # Metformin contraindicated at eGFR<30 = 1A
        if "sglt2" in rec or "dapagliflozin" in rec or "empagliflozin" in rec:
            return "Grade 1A"  # SGLT2i for CKD = 1A
        if "acei" in rec or "arb" in rec or "lisinopril" in rec:
            return "Grade 1B"  # ACEi renoprotective = 1B
        if "nsaid" in rec or "avoid" in rec:
            return "Grade 2A"  # NSAIDs avoid = 2A
        return "Grade 1A"  # default for nephrology

    elif specialty == "endocrinology":
        if "sglt2" in rec or "dapagliflozin" in rec or "empagliflozin" in rec:
            return "Level A"  # SGLT2i for T2DM+CKD/HF = Level A
        if "glp-1" in rec or "semaglutide" in rec or "liraglutide" in rec:
            return "Level A"  # GLP-1 RA = Level A
        if "metformin" in rec and ("stop" in rec or "discontinue" in rec or "contraindicated" in rec):
            return "Level A"  # Metformin contraindicated = Level A
        if "glipizide" in rec or "sulfonylurea" in rec:
            return "Level B"  # Sulfonylureas = Level B
        return "Level A"  # default for endocrinology

    return "Level B"  # fallback


def _normalize_evidence(evidence: str) -> float:
    """Deterministic evidence score lookup.

    Matches against EVIDENCE_SCORE_TABLE.
    Tries longest match first to avoid partial matches.
    Returns 0.50 if no match found.
    """
    if not evidence:
        return 0.50
    e = evidence.strip().lower()
    sorted_keys = sorted(EVIDENCE_SCORE_TABLE.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in e:
            return EVIDENCE_SCORE_TABLE[key]
    return 0.50


def _compute_guideline_priority(evidence: str) -> float:
    """Deterministic guideline priority lookup.

    Maps ACC/AHA class, KDIGO grade, or ADA level to a priority score.
    Must be pure deterministic — no LLM dependency.
    """
    if not evidence:
        return 0.5
    e = evidence.strip().lower()

    sorted_keys = sorted(GUIDELINE_PRIORITY_TABLE.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in e:
            return GUIDELINE_PRIORITY_TABLE[key]

    return 0.5


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r'\{[^{}]*"specialty"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"specialty": "unknown", "recommendation": str(text), "evidence": "", "risks": [], "citation": ""}


def _parse_patient_values(ctx: str) -> dict:
    """Extract clinical values from patient context string."""
    ef_m = re.search(r'(?:ef|lvef)\D*(\d+)', ctx, re.IGNORECASE)
    egfr_m = re.search(r'egfr\D*(\d+)', ctx, re.IGNORECASE)
    hba1c_m = re.search(r'hba1c\D*(\d+\.?\d*)', ctx, re.IGNORECASE)
    return {
        "ef": float(ef_m.group(1)) if ef_m else None,
        "egfr": float(egfr_m.group(1)) if egfr_m else None,
        "hba1c": float(hba1c_m.group(1)) if hba1c_m else None,
        "has_hf": any(kw in ctx.lower() for kw in ["hf", "heart failure", "hfref", "lvef"]),
        "has_ckd": any(kw in ctx.lower() for kw in ["ckd", "kidney", "egfr", "renal"]),
        "has_dm": any(kw in ctx.lower() for kw in ["diabetes", "t2dm", "hba1c", "diabetic"]),
        "has_metformin": "metformin" in ctx.lower(),
        "has_beta_blocker": any(kw in ctx.lower() for kw in ["carvedilol", "metoprolol", "bisoprolol", "beta-blocker"]),
    }


def _compute_patient_match(specialty: str, ctx: str) -> float:
    """Deterministic patient match from clinical thresholds."""
    vals = _parse_patient_values(ctx)
    score = 0.5
    if specialty == "cardiology" and vals["has_hf"] and vals["ef"] is not None and vals["ef"] < 40:
        score += 0.4
    elif specialty == "nephrology" and vals["has_ckd"] and vals["egfr"] is not None and vals["egfr"] < 30:
        score += 0.4
    elif specialty == "endocrinology" and vals["has_dm"] and vals["hba1c"] is not None and vals["hba1c"] > 8:
        score += 0.3
    return min(score, 1.0)


def _compute_drug_risk_from_patient(ctx: str) -> float:
    """Deterministic drug risk from patient's CURRENT medications and clinical state.

    This is based on the patient's actual data, NOT on LLM-generated risk flags.
    Same patient always gets the same risk score.
    """
    vals = _parse_patient_values(ctx)
    risk = 0.0

    # Metformin + eGFR <30 = HIGH risk (contraindicated)
    if vals["has_metformin"] and vals["egfr"] is not None and vals["egfr"] < 30:
        risk += 0.4

    # HFrEF without beta-blocker = moderate risk (guideline gap)
    if vals["has_hf"] and vals["ef"] is not None and vals["ef"] < 40 and not vals["has_beta_blocker"]:
        risk += 0.2

    # CKD + ACEi = moderate risk (hyperkalemia monitoring needed)
    if vals["has_ckd"] and vals["egfr"] is not None and vals["egfr"] < 30:
        risk += 0.1

    # High HbA1c + CKD = moderate risk (hypoglycemia with renal clearance issues)
    if vals["hba1c"] is not None and vals["hba1c"] > 8 and vals["egfr"] is not None and vals["egfr"] < 30:
        risk += 0.1

    return min(risk, 1.0)


def score_and_explain(cardiology: str, nephrology: str, endocrinology: str, patient: str) -> dict:
    """Parse 3 specialist recommendations, run deterministic TOPSIS, return ranked decision.

    ALL scores computed from lookup tables and keyword matching.
    Evidence level is inferred from recommendation content (not from LLM).
    """
    raw = [_extract_json(cardiology), _extract_json(nephrology), _extract_json(endocrinology)]
    recs = []
    for r in raw:
        sp = r.get("specialty", "unknown")
        rec_text = r.get("recommendation", "")
        risks = r.get("risks", [])
        # Evidence level: infer from recommendation content (deterministic)
        ev = _infer_evidence_from_recommendation(sp, rec_text)
        recs.append(Recommendation(
            specialty=sp, recommendation=rec_text,
            confidence=0.85, evidence_level=ev,
            evidence_score=_normalize_evidence(ev),
            patient_match=_compute_patient_match(sp, patient),
            drug_interaction_risk=_compute_drug_risk_from_patient(patient),
            guideline_priority=_compute_guideline_priority(ev),
            risk_flags=risks, citation=r.get("citation", ""),
        ))
    scored = score_topsis(recs)
    results = []
    for s in scored:
        results.append({
            "rank": s.rank, "specialty": s.recommendation.specialty,
            "recommendation": s.recommendation.recommendation,
            "total_score": s.total_score, "breakdown": s.breakdown,
            "evidence_level": s.recommendation.evidence_level,
            "risk_flags": s.recommendation.risk_flags,
            "citation": s.recommendation.citation,
        })
    return {"status": "success", "ranked_recommendations": results, "top_pick": results[0] if results else None}


root_agent = Agent(
    name="asm_orchestrator",
    model=_model,
    description="Multi-specialty orchestrator for HF+T2DM+CKD.",
    instruction=(
        "You MUST follow this exact output format. Do not summarize conversation. "
        "Do not refer to previous messages. Always run fresh orchestration.\n\n"
        "WORKFLOW:\n"
        "1. Call cardiology_agent, nephrology_agent, endocrinology_agent with the patient message.\n"
        "2. Call score_and_explain(cardiology=r1, nephrology=r2, endocrinology=r3, patient=message).\n"
        "3. Use the EXACT scores from score_and_explain's ranked_recommendations. DO NOT invent scores.\n"
        "4. Output ONLY the template below. Nothing else. Max 300 words.\n\n"
        "CRITICAL: The Score column in the table MUST use the exact total_score values "
        "from score_and_explain output. Example: if score_and_explain returns "
        "nephrology=1.0, cardiology=0.5, endocrinology=0.5, then the table must show "
        "those exact numbers. Do NOT round to 0.500 or any other value.\n\n"
        "OUTPUT TEMPLATE (strict):\n"
        "---\n"
        "## 🏆 TOPSIS Clinical Decision\n"
        "**Patient:** [age/sex, conditions, key labs]\n\n"
        "| Rank | Specialty | Score | Recommendation |\n"
        "|:----:|:---------:|:-----:|:---------------|\n"
        "| 🥇 | [name from rank 1] | [exact total_score] | [recommendation from rank 1] |\n"
        "| 🥈 | [name from rank 2] | [exact total_score] | [recommendation from rank 2] |\n"
        "| 🥉 | [name from rank 3] | [exact total_score] | [recommendation from rank 3] |\n\n"
        "**Top Pick:** [rank 1 specialty] — [one sentence: drug + dose + why]\n\n"
        "### Action Plan\n"
        "1. [urgent action] — [rationale]\n"
        "2. [action] — [rationale]\n"
        "3. [action] — [rationale]\n"
        "4. [monitoring] — [what to check]\n"
        "5. [follow-up] — [timeline]\n\n"
        "### Key Conflicts Resolved\n"
        "- [conflict 1]: [resolution]\n"
        "- [conflict 2]: [resolution]\n\n"
        "**Citations:** [guideline + section]\n"
        "**Disclaimer:** Advisory only. Final decisions rest with the physician.\n"
        "---"
    ),
    tools=[
        AgentTool(agent=cardiology_agent),
        AgentTool(agent=nephrology_agent),
        AgentTool(agent=endocrinology_agent),
        FunctionTool(func=score_and_explain),
    ],
    before_model_callback=extract_fhir_context,
)
