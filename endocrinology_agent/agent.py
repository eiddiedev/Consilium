"""endocrinology_agent — T2DM specialist.

Single LLM call: analyze patient → call build_endocrinology_recommendation.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

_model_name = os.getenv("ENDOCRINOLOGY_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)


def build_endocrinology_recommendation(
    recommendation: str,
    evidence: str,
    risks: list[str],
    citation: str,
) -> dict:
    return {
        "specialty": "endocrinology",
        "recommendation": recommendation,
        "evidence": evidence,
        "risks": risks,
        "citation": citation,
    }


root_agent = Agent(
    name="endocrinology_agent",
    model=_model,
    description="T2DM specialist (ADA 2025).",
    instruction=(
        "You are an endocrinology specialist. The message contains a patient summary.\n"
        "Analyze it for T2DM management and call build_endocrinology_recommendation.\n\n"
        "Rules: Metformin contraindicated if eGFR<30. SGLT2i preferred if CKD/HF(Level A). "
        "GLP-1 RA CV benefit(Level A). TZDs contraindicated in HF.\n\n"
        "Fields: recommendation(2-3 sentences with drug/dose), evidence(e.g.'Level A'), "
        "risks(specific safety concerns), citation(e.g.'ADA 2025 Sec 10').\n"
        "Call build_endocrinology_recommendation as your ONLY action."
    ),
    tools=[FunctionTool(func=build_endocrinology_recommendation)],
)
