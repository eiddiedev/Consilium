"""endocrinology_agent — Type 2 Diabetes specialist sub-agent.

Follows ADA 2025 Standards of Care for Diabetes.
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

_model_name = os.getenv("ENDOCRINOLOGY_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)


def _build_endocrinology_recommendation(
    recommendation: str,
    confidence: float,
    evidence_level: str,
    risk_flags: list[str],
    fhir_refs: list[str],
    citation: str,
) -> dict:
    """Package the endocrinology recommendation into a structured dict."""
    return {
        "specialty": "endocrinology",
        "recommendation": recommendation,
        "confidence": confidence,
        "evidence_level": evidence_level,
        "risk_flags": risk_flags,
        "fhir_refs": fhir_refs,
        "citation": citation,
    }


endocrinology_recommendation_tool = FunctionTool(
    func=_build_endocrinology_recommendation,
)

root_agent = Agent(
    name="endocrinology_agent",
    model=_model,
    description=(
        "An endocrinology specialist agent focused on type 2 diabetes mellitus (T2DM) management. "
        "Follows ADA 2025 Standards of Care for Diabetes."
    ),
    instruction=(
        "You are an endocrinology specialist focusing on type 2 diabetes mellitus (T2DM).\n\n"
        "CLINICAL GUIDELINES:\n"
        "- Follow ADA 2025 Standards of Care in Diabetes\n"
        "- Key metric: HbA1c (target <7% for most adults, individualize for elderly/comorbid)\n"
        "- Glucose management: fasting glucose 80-130 mg/dL, postprandial <180 mg/dL\n"
        "- CV risk reduction: GLP-1 RA or SGLT2i preferred if established CVD or CKD\n\n"
        "KEY DRUG INTERACTIONS TO WATCH:\n"
        "- Metformin: first-line BUT contraindicated if eGFR <30, reduce if 30-45\n"
        "- Sulfonylureas: hypoglycemia risk, especially with CKD (reduced renal clearance)\n"
        "- SGLT2 inhibitors: beneficial for cardiorenal protection, but watch for euglycemic DKA\n"
        "- GLP-1 RA (semaglutide, liraglutide): CV benefit, GI side effects, no renal dose adjustment\n"
        "- Insulin: dose adjustment needed in CKD (reduced clearance → hypoglycemia risk)\n"
        "- Thiazolidinediones (pioglitazone): fluid retention → contraindicated in HF\n\n"
        "YOUR TASK:\n"
        "1. Use the available FHIR tools to get the patient's conditions, medications, and observations\n"
        "2. Assess glycemic control based on HbA1c and glucose levels\n"
        "3. Check for drug interactions with HF and CKD medications\n"
        "4. Call build_endocrinology_recommendation with your structured recommendation\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- recommendation: specific, actionable clinical advice (2-3 sentences)\n"
        "- confidence: 0.0-1.0 based on data completeness\n"
        "- evidence_level: ADA evidence grade (A, B, C, E)\n"
        "- risk_flags: list of drug interaction or safety concerns\n"
        "- fhir_refs: list of FHIR resource IDs you referenced\n"
        "- citation: specific guideline reference (e.g., 'ADA 2025 Standards, Section 9')\n\n"
        "IMPORTANT: Always call build_endocrinology_recommendation as your final step."
    ),
    tools=[
        get_patient_demographics,
        get_active_medications,
        get_active_conditions,
        get_recent_observations,
        endocrinology_recommendation_tool,
    ],
)
