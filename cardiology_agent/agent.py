"""cardiology_agent — Heart Failure specialist sub-agent.

Receives patient context directly from the orchestrator prompt.
Returns a structured cardiology recommendation JSON.
Follows ACC/AHA 2022 HF Guidelines.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

_model_name = os.getenv("CARDIOLOGY_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)


def build_cardiology_recommendation(
    recommendation: str,
    confidence: float,
    evidence_level: str,
    risk_flags: list[str],
    citation: str,
) -> dict:
    """Return the structured cardiology recommendation."""
    return {
        "specialty": "cardiology",
        "recommendation": recommendation,
        "confidence": confidence,
        "evidence_level": evidence_level,
        "risk_flags": risk_flags,
        "citation": citation,
    }


root_agent = Agent(
    name="cardiology_agent",
    model=_model,
    description="Heart failure specialist (ACC/AHA 2022).",
    instruction=(
        "You are a cardiology specialist for heart failure (HF). "
        "You will receive a patient summary in the message. "
        "Analyze it and call build_cardiology_recommendation ONCE with your structured output.\n\n"
        "Key rules:\n"
        "- HFrEF (EF<40%): needs beta-blocker + ACEi/ARB + SGLT2i (Class I/IIa)\n"
        "- NSAIDs are contraindicated in HF\n"
        "- SGLT2i benefits HF regardless of diabetes\n"
        "- If eGFR <30: avoid nephrotoxic drugs, monitor K+ with ACEi\n\n"
        "Output fields:\n"
        "- recommendation: 2-3 sentence actionable advice\n"
        "- confidence: 0.0-1.0\n"
        "- evidence_level: ACC/AHA class (Class I, IIa, IIb, III)\n"
        "- risk_flags: safety concerns list\n"
        "- citation: specific guideline reference\n\n"
        "IMPORTANT: Call build_cardiology_recommendation exactly ONCE as your ONLY action."
    ),
    tools=[
        FunctionTool(func=build_cardiology_recommendation),
    ],
)
