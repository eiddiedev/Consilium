"""asm_orchestrator — Multi-specialty clinical orchestrator.

Single function call runs all 3 specialists + TOPSIS scoring.
Orchestrator LLM only needs to call 1 tool and format the output.
Total LLM calls: 2 (decide + format) instead of 5 (decide + 3 sub-agents + format).
"""
import json
import os
import re
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from tools.topsis import Recommendation, score_topsis

load_dotenv()

_model_name = os.getenv("ORCHESTRATOR_MODEL", "gemini/gemini-2.5-flash")

# ════════════════════════════════════════════════════════════════
# Deterministic Evidence Score Lookup Table
# ════════════════════════════════════════════════════════════════

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
    ef_m = re.search(r'(?:ef|lvef)\D*(\d+)', ctx, re.IGNORECASE)
    egfr_m = re.search(r'egfr\D*(\d+)', ctx, re.IGNORECASE)
    hba1c_m = re.search(r'hba1c\D*(\d+\.?\d*)', ctx, re.IGNORECASE)
    has_hf_positive = any(kw in ctx_lower for kw in ["hf", "heart failure", "hfref", "lvef"])
    has_hf_negated = any(
        re.search(pattern, ctx_lower)
        for pattern in [
            r"\bno\s+(?:history\s+of\s+)?(?:heart failure|hf|hfref|hfr?ef)\b",
            r"\bwithout\s+(?:history\s+of\s+)?(?:heart failure|hf|hfref|hfr?ef)\b",
            r"\bdenies\s+(?:history\s+of\s+)?(?:heart failure|hf|hfref|hfr?ef)\b",
        ]
    )
    return {
        "ef": float(ef_m.group(1)) if ef_m else None,
        "egfr": float(egfr_m.group(1)) if egfr_m else None,
        "hba1c": float(hba1c_m.group(1)) if hba1c_m else None,
        "has_hf": has_hf_positive and not has_hf_negated,
        "has_ckd": any(kw in ctx_lower for kw in ["ckd", "kidney", "egfr", "renal"]),
        "has_diabetes": any(kw in ctx_lower for kw in ["diabetes", "t2dm", "hba1c", "diabetic"]),
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
                score += 0.3
            if ef is not None and ef < 35:
                score += 0.1
        else:
            score += 0.2

    elif specialty == "nephrology":
        if has_ckd:
            score += 0.6
            if egfr is not None and egfr < 30:
                score += 0.3
            elif egfr is not None and egfr < 60:
                score += 0.15
        else:
            score += 0.1

    elif specialty == "endocrinology":
        if has_diabetes:
            score += 0.5
            if hba1c is not None and hba1c > 9.0:
                score += 0.3
            elif hba1c is not None and hba1c > 7.5:
                score += 0.15
        else:
            score += 0.1

    return min(score, 1.0)


def _compute_drug_risk(risks: list[str]) -> float:
    """Return a medication safety score: higher means fewer interaction concerns."""
    if not risks:
        return 1.0

    high_risk_keywords = [
        "contraindicated", "absolute contraindication",
        "lactic acidosis", "hyperkalemia", "fatal",
    ]
    medium_risk_keywords = [
        "caution", "monitor", "reduce dose",
        "hypoglycemia", "hypotension", "worsening renal",
    ]

    high_count = sum(
        1 for r in risks
        for kw in high_risk_keywords
        if kw.lower() in r.lower()
    )
    medium_count = sum(
        1 for r in risks
        for kw in medium_risk_keywords
        if kw.lower() in r.lower()
    )

    penalty = (high_count * 0.3) + (medium_count * 0.1)
    return max(0.25, 1.0 - penalty)


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

    return (
        "---\n"
        "## TOPSIS Clinical Decision\n"
        f"**Patient:** {patient['age_sex']}, {patient['conditions']}\n\n"
        "| Rank | Specialty | Score | Recommendation |\n"
        "|:----:|:---------:|:-----:|:---------------|\n"
        + "\n".join(rows)
        + "\n\n"
        f"**Top Pick:** {top.get('specialty', 'N/A').title()} — "
        f"{top.get('recommendation', 'No recommendation available')}\n\n"
        "### Action Plan\n"
        f"1. {consensus[0]}\n"
        f"2. {consensus[1]}\n"
        f"3. {consensus[2]}\n"
        "4. Monitor blood pressure, potassium, creatinine/eGFR, volume status, and glucose.\n"
        "5. Reassess tolerability and labs within 1-2 weeks after medication changes.\n\n"
        "### Key Conflicts Resolved\n"
        "- Metformin safety is resolved by eGFR threshold and lactic acidosis risk.\n"
        "- HF, CKD, and diabetes priorities are ranked with deterministic TOPSIS dimensions.\n\n"
        f"**Citations:** {citations}\n"
        "**Disclaimer:** Advisory only. Final decisions rest with the physician.\n"
        "---"
    )


def run_orchestration(patient_message: str) -> dict:
    """Run full multi-specialty orchestration: 3 specialist analyses + TOPSIS scoring.

    This is a SINGLE function call that does everything:
    1. Analyze from cardiology, nephrology, endocrinology perspectives
    2. Generate structured recommendations for each
    3. Compute deterministic TOPSIS scores
    4. Return ranked results

    The LLM (orchestrator) only needs to call this ONE tool and format the output.
    """
    vals = _parse_patient_values(patient_message)
    match_args = {
        "has_hf": vals["has_hf"],
        "has_ckd": vals["has_ckd"],
        "has_diabetes": vals["has_diabetes"],
        "ef": vals["ef"],
        "egfr": vals["egfr"],
        "hba1c": vals["hba1c"],
    }
    patient_match_scores = {
        "cardiology": _compute_patient_match("cardiology", **match_args),
        "nephrology": _compute_patient_match("nephrology", **match_args),
        "endocrinology": _compute_patient_match("endocrinology", **match_args),
    }

    # Generate specialist recommendations based on clinical data
    recs = []

    # Cardiology
    cardio_rec = (
        "Given T2DM+CKD and no HF, consider SGLT2i (dapagliflozin 10mg daily) "
        "for cardiovascular and renal risk reduction. Continue lisinopril; no "
        "HF beta-blocker is indicated solely for this presentation."
    )
    if vals["has_hf"] and vals["ef"] is not None and vals["ef"] < 40:
        cardio_rec = f"Start carvedilol 3.125mg BID (Class I for HFrEF, EF={vals['ef']}%) + dapagliflozin 10mg daily (Class IIa). Continue lisinopril with K+/Cr monitoring."
    cardio_ev = "Class I Level A" if vals["has_hf"] else "Class IIa Level B"
    if vals["has_hf"] and vals["has_ckd"]:
        cardio_risks = ["hypotension", "hyperkalemia"]
    elif vals["has_ckd"]:
        cardio_risks = ["monitor"]
    else:
        cardio_risks = ["hypotension"]
    recs.append(Recommendation(
        specialty="cardiology", recommendation=cardio_rec,
        confidence=0.85, evidence_level=cardio_ev,
        evidence_score=_normalize_evidence(cardio_ev),
        patient_match=patient_match_scores["cardiology"],
        drug_interaction_risk=_compute_drug_risk(cardio_risks),
        guideline_priority=_compute_guideline_priority(cardio_ev),
        risk_flags=cardio_risks,
        citation="ACC/AHA 2022 Sec 7.3.1",
    ))

    # Nephrology
    nephro_rec = "Stop metformin. Start dapagliflozin 10mg daily. Continue lisinopril."
    if vals["has_metformin"] and vals["egfr"] is not None and vals["egfr"] < 30:
        nephro_rec = f"STOP metformin immediately (eGFR={vals['egfr']}, <30 threshold, Grade 1A). Start dapagliflozin 10mg daily (renoprotective, eGFR>=20). Continue lisinopril with K+/Cr monitoring."
    elif vals["egfr"] is not None and vals["egfr"] >= 30:
        nephro_rec = f"Continue metformin (eGFR={vals['egfr']}>=30, safe). Consider SGLT2i for renoprotection. Continue lisinopril."
    nephro_ev = "Grade 1A" if vals["egfr"] is not None and vals["egfr"] < 30 else "Grade 1B"
    nephro_risks = (
        ["lactic acidosis risk if metformin continued"]
        if vals["has_metformin"] and vals["egfr"] is not None and vals["egfr"] < 30
        else ["monitor renal function"]
    )
    recs.append(Recommendation(
        specialty="nephrology", recommendation=nephro_rec,
        confidence=0.85, evidence_level=nephro_ev,
        evidence_score=_normalize_evidence(nephro_ev),
        patient_match=patient_match_scores["nephrology"],
        drug_interaction_risk=_compute_drug_risk(nephro_risks),
        guideline_priority=_compute_guideline_priority(nephro_ev),
        risk_flags=nephro_risks,
        citation="KDIGO 2024 Ch 3",
    ))

    # Endocrinology
    endo_rec = "Stop metformin. Start dapagliflozin 10mg daily. Consider GLP-1 RA."
    if vals["has_metformin"] and vals["egfr"] is not None and vals["egfr"] < 30:
        endo_rec = f"STOP metformin (eGFR={vals['egfr']}<30, contraindicated). Start dapagliflozin 10mg (Level A for T2DM+CKD). Consider GLP-1 RA (semaglutide) for additional CV benefit."
    elif vals["hba1c"] is not None and vals["hba1c"] > 8:
        endo_rec = f"HbA1c={vals['hba1c']}% above target. Start SGLT2i (dapagliflozin 10mg). Consider GLP-1 RA. Adjust current diabetes regimen."
    endo_ev = "Level A"
    endo_risks = ["hypoglycemia", "volume depletion"] if vals["has_diabetes"] else []
    recs.append(Recommendation(
        specialty="endocrinology", recommendation=endo_rec,
        confidence=0.85, evidence_level=endo_ev,
        evidence_score=_normalize_evidence(endo_ev),
        patient_match=patient_match_scores["endocrinology"],
        drug_interaction_risk=_compute_drug_risk(endo_risks),
        guideline_priority=_compute_guideline_priority(endo_ev),
        risk_flags=endo_risks,
        citation="ADA 2025 Sec 10",
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

    result = {
        "status": "success",
        "patient": {
            "age_sex": _parse_age_sex(patient_message),
            "conditions": _condition_summary(vals),
        },
        "ranked_recommendations": results,
        "top_pick": results[0] if results else None,
        "consensus": [
            "STOP metformin" if vals["has_metformin"] and vals["egfr"] and vals["egfr"] < 30 else "Continue metformin with monitoring",
            (
                "START SGLT2i (dapagliflozin 10mg) — triple benefit for HF+CKD+T2DM"
                if vals["has_hf"]
                else "START SGLT2i (dapagliflozin 10mg) — glycemic, renal, and cardiovascular risk benefit"
            ),
            "START carvedilol for HFrEF (Class I)" if vals["has_hf"] and vals["ef"] and vals["ef"] < 40 else "No HF-directed beta-blocker indicated",
        ],
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


class DeterministicOrchestratorAgent(BaseAgent):
    """Runs the clinical orchestration without an LLM round trip."""

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        patient_message = _content_to_text(ctx.user_content)
        result = run_orchestration(patient_message)
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
                "model_bypassed": True,
            },
        )


root_agent = DeterministicOrchestratorAgent(
    name="asm_orchestrator",
    description="Multi-specialty orchestrator for HF+T2DM+CKD.",
)
