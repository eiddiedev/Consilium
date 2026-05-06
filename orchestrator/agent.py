"""asm_orchestrator — Multi-specialty clinical orchestrator.

Single LLM call generates all 3 specialist recommendations with structured
clinical scores, then TOPSIS ranks them deterministically.

Scoring is DYNAMIC — computed from actual agent outputs, not hardcoded:
- evidence_score: normalized from guideline class/grade string
- patient_match: LLM-assessed fit to patient's specific comorbidities
- drug_interaction_risk: computed from risk_flags count and severity
- guideline_priority: LLM-assessed strength of guideline endorsement
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

from shared.fhir_hook import extract_fhir_context
from tools.topsis import Recommendation, Preferences, score_topsis, _normalize_evidence

_model_name = os.getenv("ORCHESTRATOR_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)


def _compute_drug_risk(risk_flags: list[str]) -> float:
    """Compute drug interaction risk score from risk flags.

    More flags and more severe keywords → higher risk (0.0 = none, 1.0 = severe).
    """
    if not risk_flags:
        return 0.0
    severity_keywords = {
        "high": 0.3, "severe": 0.3, "contraindicated": 0.4, "avoid": 0.25,
        "stop": 0.3, "danger": 0.3, "critical": 0.35, "immediate": 0.3,
        "moderate": 0.15, "caution": 0.1, "monitor": 0.05, "watch": 0.05,
    }
    score = 0.0
    for flag in risk_flags:
        flag_lower = flag.lower()
        for keyword, weight in severity_keywords.items():
            if keyword in flag_lower:
                score += weight
                break
        else:
            score += 0.1  # default weight for unrecognized flags
    return min(score, 1.0)


def generate_decision(
    # Cardiology
    cardiology_recommendation: str,
    cardiology_confidence: float,
    cardiology_evidence: str,
    cardiology_patient_match: float,
    cardiology_guideline_priority: float,
    cardiology_risks: list[str],
    cardiology_citation: str,
    # Nephrology
    nephrology_recommendation: str,
    nephrology_confidence: float,
    nephrology_evidence: str,
    nephrology_patient_match: float,
    nephrology_guideline_priority: float,
    nephrology_risks: list[str],
    nephrology_citation: str,
    # Endocrinology
    endocrinology_recommendation: str,
    endocrinology_confidence: float,
    endocrinology_evidence: str,
    endocrinology_patient_match: float,
    endocrinology_guideline_priority: float,
    endocrinology_risks: list[str],
    endocrinology_citation: str,
) -> dict:
    """Receive 3 specialist recommendations, compute TOPSIS scores dynamically, return ranked decision.

    All 21 fields are filled by the LLM from its clinical analysis.
    TOPSIS scores are computed deterministically from these structured inputs.
    """
    specs = [
        ("cardiology", cardiology_recommendation, cardiology_confidence,
         cardiology_evidence, cardiology_patient_match, cardiology_guideline_priority,
         cardiology_risks, cardiology_citation),
        ("nephrology", nephrology_recommendation, nephrology_confidence,
         nephrology_evidence, nephrology_patient_match, nephrology_guideline_priority,
         nephrology_risks, nephrology_citation),
        ("endocrinology", endocrinology_recommendation, endocrinology_confidence,
         endocrinology_evidence, endocrinology_patient_match, endocrinology_guideline_priority,
         endocrinology_risks, endocrinology_citation),
    ]

    recs = []
    for (specialty, rec, conf, evidence, match, priority, risks, citation) in specs:
        recs.append(Recommendation(
            specialty=specialty,
            recommendation=rec,
            confidence=conf,
            evidence_level=evidence,
            evidence_score=_normalize_evidence(evidence),      # dynamic from guideline class
            patient_match=match,                                # LLM-assessed (0-1)
            drug_interaction_risk=_compute_drug_risk(risks),    # computed from flags
            guideline_priority=priority,                        # LLM-assessed (0-1)
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
            "confidence": s.recommendation.confidence,
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
        "and call generate_decision ONCE with all 21 fields filled in.\n\n"
        "SPECIALIST PERSPECTIVES:\n\n"
        "1. CARDIOLOGY (ACC/AHA 2022 HF Guidelines):\n"
        "   - HFrEF (EF<40%): beta-blocker (Class I, Sec 7.3.1) + ACEi/ARB (Class I) + SGLT2i (Class IIa, Sec 7.3.2)\n"
        "   - NSAIDs contraindicated (Class III Harm, Sec 7.6)\n"
        "   - MRA: caution if eGFR <30 (hyperkalemia risk, Sec 7.3.3)\n"
        "   - ARNI (sacubitril/valsartan) preferred over ACEi if tolerated (Class IIa, Sec 7.3.1)\n\n"
        "2. NEPHROLOGY (KDIGO 2024 CKD Guidelines):\n"
        "   - Metformin: CONTRAINDICATED if eGFR <30 (Grade 1A, Ch 4)\n"
        "   - SGLT2i: renoprotective if eGFR ≥20 (Grade 1A, Ch 3)\n"
        "   - NSAIDs: avoid (Grade 2A, Ch 4)\n"
        "   - ACEi/ARB: renoprotective, monitor K+/Cr (Grade 1B, Ch 3)\n"
        "   - Finerenone: consider for CKD+T2DM (Grade 2A, Ch 3)\n\n"
        "3. ENDOCRINOLOGY (ADA 2025 Standards of Care):\n"
        "   - SGLT2i preferred if CKD or HF (Level A, Sec 10)\n"
        "   - GLP-1 RA: CV benefit, no renal dose adjustment (Level A, Sec 10)\n"
        "   - Metformin contraindicated if eGFR <30 (Sec 9)\n"
        "   - TZDs: contraindicated in HF — fluid retention (Level B, Sec 10)\n"
        "   - HbA1c target: individualize, <7% for most (Level A, Sec 6)\n\n"
        "For EACH specialist, provide:\n"
        "- recommendation: 2-3 sentences of specific, actionable clinical advice with drug names and doses\n"
        "- confidence: 0.0-1.0 (how confident you are in this recommendation given the data)\n"
        "- evidence: guideline class/grade string (e.g. 'Class I', '1A', 'Level A')\n"
        "- patient_match: 0.0-1.0 (how well this recommendation addresses THIS patient's specific situation)\n"
        "- guideline_priority: 0.0-1.0 (how strongly the guideline endorses this action for this scenario)\n"
        "- risks: list of specific safety concerns with drug names\n"
        "- citation: specific guideline section (e.g. 'ACC/AHA 2022 Sec 7.3.1', 'KDIGO 2024 Ch 3')\n\n"
        "IMPORTANT:\n"
        "- patient_match should reflect how critical this recommendation is for THIS patient's specific eGFR, LVEF, HbA1c\n"
        "- guideline_priority should reflect the strength of the guideline's endorsement (Class I > IIa > IIb > III)\n"
        "- risks should name specific drugs and specific interactions\n"
        "- Be precise with citations — include section numbers\n"
        "- Call generate_decision exactly ONCE as your ONLY tool call"
    ),
    tools=[
        FunctionTool(func=generate_decision),
    ],
    before_model_callback=extract_fhir_context,
)
