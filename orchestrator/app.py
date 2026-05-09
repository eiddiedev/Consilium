"""asm_orchestrator — A2A application entry point.

Start the server with:
    uvicorn orchestrator.app:a2a_app --host 0.0.0.0 --port 8003

The agent card is served publicly at:
    GET http://localhost:8003/.well-known/agent-card.json
"""
import os

from a2a.types import AgentSkill
from shared.app_factory import create_a2a_app

from .agent import root_agent

PO_PLATFORM_BASE_URL = os.getenv("PO_PLATFORM_BASE_URL", "https://app.promptopinion.ai").rstrip("/")

a2a_app = create_a2a_app(
    agent=root_agent,
    name="Consilium",
    description=(
        "Multi-specialty clinical orchestrator for HF+T2DM+CKD patients. "
        "Routes to cardiology, nephrology, and endocrinology specialist agents, "
        "scores recommendations using TOPSIS multi-criteria ranking, and produces "
        "an explainable ranked clinical decision with evidence citations."
    ),
    url=os.getenv("ORCHESTRATOR_URL", os.getenv("BASE_URL", "http://localhost:8003")),
    port=8003,
    fhir_extension_uri=f"{PO_PLATFORM_BASE_URL}/schemas/a2a/v1/fhir-context",
    fhir_scopes=[
        {"name": "patient/Patient.rs",           "required": True},
        {"name": "patient/MedicationRequest.rs", "required": True},
        {"name": "patient/Condition.rs",         "required": True},
        {"name": "patient/Observation.rs",       "required": True},
    ],
    skills=[
        AgentSkill(
            id="multi-specialty-orchestration",
            name="Multi-Specialty Orchestration",
            description=(
                "Orchestrates cardiology, nephrology, and endocrinology specialists "
                "for complex chronic disease patients (HF+T2DM+CKD). "
                "Scores recommendations with TOPSIS and provides explainable clinical decisions."
            ),
            examples=[
                "Run the full multi-specialty orchestration for this patient.",
                "Resolve guideline conflicts for a patient with HFrEF, CKD, and T2DM.",
            ],
            input_modes=["text/plain"],
            output_modes=["text/markdown"],
            tags=["clinical", "orchestrator", "multi-specialty", "HF", "T2DM", "CKD", "TOPSIS"],
        ),
        AgentSkill(
            id="fhir-clinical-context-summary",
            name="FHIR Clinical Context Summary",
            description=(
                "Reads Prompt Opinion FHIR context when available and summarizes "
                "patient demographics, active conditions, medications, and observations "
                "before specialist review."
            ),
            examples=[
                "Use the connected FHIR patient context to summarize active problems and medications.",
                "Build a clinical summary from Patient, Condition, MedicationRequest, and Observation resources.",
            ],
            input_modes=["text/plain"],
            output_modes=["text/markdown"],
            tags=["FHIR", "SMART-on-FHIR", "clinical-summary", "patient-context"],
        ),
        AgentSkill(
            id="specialist-consult-panel",
            name="Specialist Consult Panel",
            description=(
                "Invokes structured cardiology, nephrology, and endocrinology consult agents "
                "and validates each specialist recommendation before reconciliation."
            ),
            examples=[
                "Ask the cardiology, nephrology, and endocrinology agents for structured recommendations.",
                "Generate specialty-specific treatment recommendations with risks and citations.",
            ],
            input_modes=["text/plain"],
            output_modes=["text/markdown"],
            tags=["cardiology", "nephrology", "endocrinology", "consult"],
        ),
        AgentSkill(
            id="medication-safety-reconciliation",
            name="Medication Safety Reconciliation",
            description=(
                "Detects cross-specialty medication conflicts such as metformin in advanced CKD, "
                "hyperkalemia risk, and overlapping HF/diabetes treatment opportunities."
            ),
            examples=[
                "Identify medication conflicts for a patient with eGFR below 30.",
                "Explain whether metformin, MRA, ACEi/ARB, or SGLT2i recommendations conflict.",
            ],
            input_modes=["text/plain"],
            output_modes=["text/markdown"],
            tags=["medication-safety", "CKD", "metformin", "hyperkalemia", "SGLT2i"],
        ),
        AgentSkill(
            id="topsis-clinical-ranking",
            name="TOPSIS Clinical Ranking",
            description=(
                "Ranks validated specialist recommendations with deterministic TOPSIS scoring "
                "across evidence strength, patient match, medication risk, and guideline priority."
            ),
            examples=[
                "Rank the specialist recommendations and explain the top clinical priority.",
                "Show the TOPSIS scores for cardiology, nephrology, and endocrinology recommendations.",
            ],
            input_modes=["text/plain"],
            output_modes=["text/markdown"],
            tags=["TOPSIS", "clinical-ranking", "evidence", "guidelines", "explainability"],
        ),
    ],
)
