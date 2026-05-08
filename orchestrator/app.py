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
    name="asm_orchestrator",
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
            name="multi-specialty-orchestration",
            description=(
                "Orchestrates cardiology, nephrology, and endocrinology specialists "
                "for complex chronic disease patients (HF+T2DM+CKD). "
                "Scores recommendations with TOPSIS and provides explainable clinical decisions."
            ),
            tags=["clinical", "orchestrator", "multi-specialty", "HF", "T2DM", "CKD", "TOPSIS"],
        ),
    ],
)
