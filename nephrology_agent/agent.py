"""nephrology_agent — CKD specialist sub-agent.

Follows KDIGO 2024 CKD Guidelines.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

_model_name = os.getenv("NEPHROLOGY_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)


def build_nephrology_recommendation(
    recommendation: str,
    confidence: float,
    evidence_level: str,
    risk_flags: list[str],
    citation: str,
) -> dict:
    """Return the structured nephrology recommendation."""
    return {
        "specialty": "nephrology",
        "recommendation": recommendation,
        "confidence": confidence,
        "evidence_level": evidence_level,
        "risk_flags": risk_flags,
        "citation": citation,
    }


root_agent = Agent(
    name="nephrology_agent",
    model=_model,
    description="CKD specialist (KDIGO 2024).",
    instruction=(
        "You are a nephrology specialist for chronic kidney disease (CKD). "
        "You will receive a patient summary in the message. "
        "Analyze it and call build_nephrology_recommendation ONCE with your structured output.\n\n"
        "Key rules:\n"
        "- CKD Stage 4 (eGFR 15-29): high risk, nephrotoxic drugs must be stopped\n"
        "- Metformin: CONTRAINDICATED if eGFR <30 (lactic acidosis)\n"
        "- SGLT2i: renoprotective, start if eGFR ≥20 (KDIGO 1A)\n"
        "- NSAIDs: avoid (nephrotoxic)\n"
        "- ACEi/ARB: renoprotective, but monitor K+ and Cr\n"
        "- Glipizide: accumulation risk in CKD → hypoglycemia\n\n"
        "Output fields:\n"
        "- recommendation: 2-3 sentence actionable advice\n"
        "- confidence: 0.0-1.0\n"
        "- evidence_level: KDIGO grade (1A, 1B, 2A, 2B)\n"
        "- risk_flags: nephrotoxicity concerns\n"
        "- citation: specific guideline reference\n\n"
        "IMPORTANT: Call build_nephrology_recommendation exactly ONCE as your ONLY action."
    ),
    tools=[
        FunctionTool(func=build_nephrology_recommendation),
    ],
)
