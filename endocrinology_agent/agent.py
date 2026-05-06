"""endocrinology_agent — T2DM specialist sub-agent.

Follows ADA 2025 Standards of Care for Diabetes.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

_model_name = os.getenv("ENDOCRINOLOGY_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)


def build_endocrinology_recommendation(
    recommendation: str,
    confidence: float,
    evidence_level: str,
    risk_flags: list[str],
    citation: str,
) -> dict:
    """Return the structured endocrinology recommendation."""
    return {
        "specialty": "endocrinology",
        "recommendation": recommendation,
        "confidence": confidence,
        "evidence_level": evidence_level,
        "risk_flags": risk_flags,
        "citation": citation,
    }


root_agent = Agent(
    name="endocrinology_agent",
    model=_model,
    description="T2DM specialist (ADA 2025).",
    instruction=(
        "You are an endocrinology specialist for type 2 diabetes (T2DM). "
        "You will receive a patient summary in the message. "
        "Analyze it and call build_endocrinology_recommendation ONCE with your structured output.\n\n"
        "Key rules:\n"
        "- Metformin: first-line BUT contraindicated if eGFR <30\n"
        "- SGLT2i: preferred if CKD or HF present (ADA Level A)\n"
        "- GLP-1 RA: CV benefit, no renal dose adjustment needed\n"
        "- Sulfonylureas (Glipizide): hypoglycemia risk in CKD (reduced clearance)\n"
        "- TZDs (pioglitazone): fluid retention → contraindicated in HF\n"
        "- HbA1c target: <7% for most, individualize for comorbidities\n\n"
        "Output fields:\n"
        "- recommendation: 2-3 sentence actionable advice\n"
        "- confidence: 0.0-1.0\n"
        "- evidence_level: ADA grade (A, B, C, E)\n"
        "- risk_flags: drug interaction concerns\n"
        "- citation: specific guideline reference\n\n"
        "IMPORTANT: Call build_endocrinology_recommendation exactly ONCE as your ONLY action."
    ),
    tools=[
        FunctionTool(func=build_endocrinology_recommendation),
    ],
)
