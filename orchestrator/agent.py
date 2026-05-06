"""asm_orchestrator — Multi-specialty clinical orchestrator.

Production mode: single LLM call generates all 3 specialist recommendations
plus TOPSIS scoring and explanation. Optimized for <30s response time.

Sub-agent code is preserved in cardiology_agent/, nephrology_agent/,
endocrinology_agent/ for architecture demonstration and testing.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

from shared.fhir_hook import extract_fhir_context
from tools.topsis import Recommendation, Preferences, score_topsis

_model_name = os.getenv("ORCHESTRATOR_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)


def generate_decision(
    cardiology_recommendation: str,
    cardiology_confidence: float,
    cardiology_evidence: str,
    cardiology_risks: list[str],
    nephrology_recommendation: str,
    nephrology_confidence: float,
    nephrology_evidence: str,
    nephrology_risks: list[str],
    endocrinology_recommendation: str,
    endocrinology_confidence: float,
    endocrinology_evidence: str,
    endocrinology_risks: list[str],
) -> dict:
    """Receive all 3 specialist recommendations, run TOPSIS, return ranked decision.

    The LLM fills in all 12 fields from its clinical analysis.
    This function runs TOPSIS deterministically on the inputs.
    """
    recs = [
        Recommendation(
            specialty="cardiology",
            recommendation=cardiology_recommendation,
            confidence=cardiology_confidence,
            evidence_level=cardiology_evidence,
            evidence_score=0.85,  # ACC/AHA generally strong
            patient_match=0.7,
            drug_interaction_risk=0.3,
            guideline_priority=0.9,
            risk_flags=cardiology_risks,
            citation="ACC/AHA 2022 HF Guidelines",
        ),
        Recommendation(
            specialty="nephrology",
            recommendation=nephrology_recommendation,
            confidence=nephrology_confidence,
            evidence_level=nephrology_evidence,
            evidence_score=0.9,  # KDIGO 1A
            patient_match=0.9,
            drug_interaction_risk=0.1,
            guideline_priority=0.85,
            risk_flags=nephrology_risks,
            citation="KDIGO 2024 CKD Guidelines",
        ),
        Recommendation(
            specialty="endocrinology",
            recommendation=endocrinology_recommendation,
            confidence=endocrinology_confidence,
            evidence_level=endocrinology_evidence,
            evidence_score=0.8,  # ADA Level A
            patient_match=0.75,
            drug_interaction_risk=0.2,
            guideline_priority=0.8,
            risk_flags=endocrinology_risks,
            citation="ADA 2025 Standards of Care",
        ),
    ]

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
        "and call generate_decision ONCE with all 12 fields filled in.\n\n"
        "SPECIALIST PERSPECTIVES:\n\n"
        "1. CARDIOLOGY (ACC/AHA 2022 HF Guidelines):\n"
        "   - HFrEF (EF<40%): beta-blocker + ACEi/ARB + SGLT2i are Class I/IIa\n"
        "   - NSAIDs contraindicated in HF\n"
        "   - MRA: caution if eGFR <30 (hyperkalemia)\n\n"
        "2. NEPHROLOGY (KDIGO 2024 CKD Guidelines):\n"
        "   - Metformin: CONTRAINDICATED if eGFR <30\n"
        "   - SGLT2i: renoprotective, start if eGFR ≥20 (Grade 1A)\n"
        "   - NSAIDs: avoid (nephrotoxic)\n"
        "   - ACEi: renoprotective but monitor K+/Cr\n"
        "   - Sulfonylureas: accumulation risk in CKD\n\n"
        "3. ENDOCRINOLOGY (ADA 2025 Standards):\n"
        "   - SGLT2i preferred if CKD or HF present (Level A)\n"
        "   - GLP-1 RA: CV benefit, no renal dose adjustment\n"
        "   - Metformin contraindicated if eGFR <30\n"
        "   - TZDs: contraindicated in HF (fluid retention)\n\n"
        "For each specialist, provide:\n"
        "- recommendation: 2-3 sentence actionable advice\n"
        "- confidence: 0.0-1.0\n"
        "- evidence_level: guideline class/grade\n"
        "- risks: list of safety concerns\n\n"
        "Then present the TOPSIS-ranked results with a clinical decision summary.\n"
        "Include a disclaimer that all recommendations are advisory."
    ),
    tools=[
        FunctionTool(func=generate_decision),
    ],
    before_model_callback=extract_fhir_context,
)
