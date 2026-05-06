"""nephrology_agent — CKD specialist.

Single LLM call: analyze patient → call build_nephrology_recommendation.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

_model_name = os.getenv("NEPHROLOGY_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)


def build_nephrology_recommendation(
    recommendation: str,
    evidence: str,
    risks: list[str],
    citation: str,
) -> dict:
    return {
        "specialty": "nephrology",
        "recommendation": recommendation,
        "evidence": evidence,
        "risks": risks,
        "citation": citation,
    }


root_agent = Agent(
    name="nephrology_agent",
    model=_model,
    description="CKD specialist (KDIGO 2024).",
    instruction=(
        "You are a nephrology specialist. The message contains a patient summary.\n"
        "Analyze it for CKD management and call build_nephrology_recommendation.\n\n"
        "Rules: Metformin CONTRAINDICATED if eGFR<30(1A). SGLT2i if eGFR>=20(1A). "
        "ACEi renoprotective but monitor K+/Cr(1B). NSAIDs avoid(2A).\n\n"
        "Fields: recommendation(2-3 sentences with drug/dose), evidence(e.g.'1A'), "
        "risks(specific safety concerns), citation(e.g.'KDIGO 2024 Ch 3').\n"
        "Call build_nephrology_recommendation as your ONLY action."
    ),
    tools=[FunctionTool(func=build_nephrology_recommendation)],
)
