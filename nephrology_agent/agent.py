"""nephrology_agent — Chronic Kidney Disease specialist sub-agent.

Follows KDIGO 2024 CKD Guidelines.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

from shared.tools import (
    get_active_conditions,
    get_active_medications,
    get_patient_demographics,
    get_recent_observations,
)

_model_name = os.getenv("NEPHROLOGY_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)


def _build_nephrology_recommendation(
    recommendation: str,
    confidence: float,
    evidence_level: str,
    risk_flags: list[str],
    fhir_refs: list[str],
    citation: str,
) -> dict:
    """Package the nephrology recommendation into a structured dict."""
    return {
        "specialty": "nephrology",
        "recommendation": recommendation,
        "confidence": confidence,
        "evidence_level": evidence_level,
        "risk_flags": risk_flags,
        "fhir_refs": fhir_refs,
        "citation": citation,
    }


nephrology_recommendation_tool = FunctionTool(
    func=_build_nephrology_recommendation,
)

root_agent = Agent(
    name="nephrology_agent",
    model=_model,
    description=(
        "A nephrology specialist agent focused on chronic kidney disease (CKD) management. "
        "Follows KDIGO 2024 CKD Guidelines for recommendations."
    ),
    instruction=(
        "You are a nephrology specialist focusing on chronic kidney disease (CKD).\n\n"
        "CLINICAL GUIDELINES:\n"
        "- Follow KDIGO 2024 CKD Guidelines\n"
        "- CKD Stages: G1(eGFR≥90), G2(60-89), G3a(45-59), G3b(30-44), G4(15-29), G5(<15)\n"
        "- Key metric: eGFR (CKD-EPI equation), urine albumin-to-creatinine ratio (UACR)\n"
        "- CKD progression risk depends on eGFR + albuminuria category\n\n"
        "KEY DRUG INTERACTIONS TO WATCH:\n"
        "- Metformin: contraindicated if eGFR <30, reduce dose if 30-45\n"
        "- NSAIDs: avoid in CKD (worsen renal function, increase CV risk)\n"
        "- ACEi/ARB: renoprotective but monitor K+ and creatinine; hold if K+ >5.5\n"
        "- SGLT2 inhibitors: renoprotective in CKD (Class I if eGFR ≥20)\n"
        "- Contrast dye: avoid if eGFR <30 unless essential\n"
        "- Lithium, aminoglycosides: nephrotoxic — flag if present\n\n"
        "YOUR TASK:\n"
        "1. Use the available FHIR tools to get the patient's conditions, medications, and observations\n"
        "2. Assess CKD stage based on eGFR and albuminuria\n"
        "3. Check for nephrotoxic medications and drug interactions\n"
        "4. Call build_nephrology_recommendation with your structured recommendation\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- recommendation: specific, actionable clinical advice (2-3 sentences)\n"
        "- confidence: 0.0-1.0 based on data completeness\n"
        "- evidence_level: KDIGO grade (1A, 1B, 2A, 2B, ungraded)\n"
        "- risk_flags: list of nephrotoxicity or safety concerns\n"
        "- fhir_refs: list of FHIR resource IDs you referenced\n"
        "- citation: specific guideline reference (e.g., 'KDIGO 2024 CKD Guideline, Chapter 3')\n\n"
        "IMPORTANT: Always call build_nephrology_recommendation as your final step."
    ),
    tools=[
        get_patient_demographics,
        get_active_medications,
        get_active_conditions,
        get_recent_observations,
        nephrology_recommendation_tool,
    ],
)
