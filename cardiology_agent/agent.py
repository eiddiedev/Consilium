"""cardiology_agent — Heart Failure specialist sub-agent.

Receives patient FHIR context via shared session state (extracted by
the orchestrator's before_model_callback).  Queries FHIR tools and
returns a structured cardiology recommendation JSON.

Instruction follows ACC/AHA 2022 Heart Failure Guidelines.
"""
import json
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

_model_name = os.getenv("CARDIOLOGY_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)


def _build_cardiology_recommendation(
    recommendation: str,
    confidence: float,
    evidence_level: str,
    risk_flags: list[str],
    fhir_refs: list[str],
    citation: str,
) -> dict:
    """Package the cardiology recommendation into a structured dict.

    This is a passthrough — the LLM calls this tool with its analysis
    and the tool simply returns the same data as a well-formed dict.
    """
    return {
        "specialty": "cardiology",
        "recommendation": recommendation,
        "confidence": confidence,
        "evidence_level": evidence_level,
        "risk_flags": risk_flags,
        "fhir_refs": fhir_refs,
        "citation": citation,
    }


cardiology_recommendation_tool = FunctionTool(
    func=_build_cardiology_recommendation,
)

root_agent = Agent(
    name="cardiology_agent",
    model=_model,
    description=(
        "A cardiology specialist agent focused on heart failure (HF) management. "
        "Follows ACC/AHA 2022 HF Guidelines for recommendations."
    ),
    instruction=(
        "You are a cardiology specialist focusing on heart failure (HF).\n\n"
        "CLINICAL GUIDELINES:\n"
        "- Follow ACC/AHA 2022 HF Guidelines (Class I, IIa, IIb, III)\n"
        "- Key metrics: LVEF (Ejection Fraction), BNP/NT-proBNP, NYHA class\n"
        "- HFpEF (EF≥50%) vs HFrEF (EF<40%) have different treatment paths\n\n"
        "KEY DRUG INTERACTIONS TO WATCH:\n"
        "- NSAIDs worsen HF fluid retention — avoid in all HF patients\n"
        "- Metformin: caution if eGFR <30 (CKD interaction)\n"
        "- ACEi/ARB + CKD: monitor K+ and creatinine closely\n"
        "- SGLT2 inhibitors (dapagliflozin/empagliflozin): beneficial in HF regardless of diabetes status (Class IIa)\n\n"
        "YOUR TASK:\n"
        "1. Use the available FHIR tools to get the patient's conditions, medications, and observations\n"
        "2. Analyze the heart failure status based on LVEF, BNP, and symptoms\n"
        "3. Check for drug interactions with the patient's other conditions (CKD, T2DM)\n"
        "4. Call build_cardiology_recommendation with your structured recommendation\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- recommendation: specific, actionable clinical advice (2-3 sentences)\n"
        "- confidence: 0.0-1.0 based on data completeness\n"
        "- evidence_level: ACC/AHA class (Class I, Class IIa, Class IIb, Class III)\n"
        "- risk_flags: list of drug interaction or safety concerns\n"
        "- fhir_refs: list of FHIR resource IDs you referenced\n"
        "- citation: specific guideline reference (e.g., 'ACC/AHA 2022 HF Guidelines, Section 7.3')\n\n"
        "IMPORTANT: Always call build_cardiology_recommendation as your final step."
    ),
    tools=[
        get_patient_demographics,
        get_active_medications,
        get_active_conditions,
        get_recent_observations,
        cardiology_recommendation_tool,
    ],
)
