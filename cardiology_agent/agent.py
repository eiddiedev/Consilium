"""cardiology_agent — Heart Failure specialist.

Single LLM call: analyze patient → call build_cardiology_recommendation.
No FHIR tools — patient context comes from orchestrator prompt.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

_model_name = os.getenv("CARDIOLOGY_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)


def build_cardiology_recommendation(
    recommendation: str,
    evidence: str,
    risks: list[str],
    citation: str,
) -> dict:
    return {
        "specialty": "cardiology",
        "recommendation": recommendation,
        "evidence": evidence,
        "risks": risks,
        "citation": citation,
    }


root_agent = Agent(
    name="cardiology_agent",
    model=_model,
    description="Heart failure specialist (ACC/AHA 2022).",
    instruction=(
        "You are a cardiology specialist. The message contains a patient summary.\n"
        "Analyze it for HF management and call build_cardiology_recommendation.\n\n"
        "Rules: HFrEF(EF<40%) needs beta-blocker(Class I)+ACEi(Class I)+SGLT2i(Class IIa). "
        "NSAIDs contraindicated. MRA caution if eGFR<30.\n\n"
        "Fields: recommendation(2-3 sentences with drug/dose), evidence(e.g.'Class I'), "
        "risks(specific safety concerns), citation(e.g.'ACC/AHA 2022 Sec 7.3.1').\n"
        "Call build_cardiology_recommendation as your ONLY action."
    ),
    tools=[FunctionTool(func=build_cardiology_recommendation)],
)
