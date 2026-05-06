"""asm_orchestrator — Multi-specialty clinical orchestrator.

This is the core agent for the Agents Assemble hackathon.
It delegates to 3 specialty sub-agents (cardiology, nephrology, endocrinology),
collects their structured recommendations, scores them with TOPSIS, and
generates an explainable clinical decision.

Sub-agents run in-process via ADK's AgentTool (same Python process).
Session state is shared — FHIR credentials extracted by extract_fhir_context
are available to all sub-agents' tools.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool

from cardiology_agent.agent import root_agent as cardiology_agent
from nephrology_agent.agent import root_agent as nephrology_agent
from endocrinology_agent.agent import root_agent as endocrinology_agent
from shared.fhir_hook import extract_fhir_context
from tools.score_tool import score_recommendations
from tools.explain_tool import explain_decision

_model_name = os.getenv("ORCHESTRATOR_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)

root_agent = Agent(
    name="asm_orchestrator",
    model=_model,
    description=(
        "A multi-specialty clinical orchestrator for complex chronic disease patients "
        "(HF + T2DM + CKD). Routes to cardiology, nephrology, and endocrinology "
        "specialist agents, scores their recommendations using TOPSIS, and produces "
        "an explainable ranked clinical decision."
    ),
    instruction=(
        "You are a multi-specialty clinical orchestrator for patients with complex "
        "chronic conditions — specifically heart failure (HF), type 2 diabetes (T2DM), "
        "and chronic kidney disease (CKD).\n\n"
        "WORKFLOW — Follow these steps in order:\n\n"
        "Step 1: COLLECT RECOMMENDATIONS\n"
        "Call ALL THREE specialist agents to get their recommendations:\n"
        "  - cardiology_agent: heart failure management\n"
        "  - nephrology_agent: chronic kidney disease management\n"
        "  - endocrinology_agent: type 2 diabetes management\n"
        "Each agent will access the patient's FHIR record and return a structured "
        "JSON recommendation with specialty, recommendation, confidence, evidence_level, "
        "risk_flags, fhir_refs, and citation.\n\n"
        "Step 2: SCORE RECOMMENDATIONS\n"
        "Collect the JSON outputs from all three agents and pass them as a JSON array "
        "to the score_recommendations tool. This runs TOPSIS multi-criteria ranking "
        "across four dimensions: evidence level, patient match, drug interaction risk, "
        "and guideline priority.\n\n"
        "Step 3: EXPLAIN THE DECISION\n"
        "Take the scoring result and pass it to the explain_decision tool to generate "
        "a clinician-friendly explanation.\n\n"
        "Step 4: PRESENT THE FINAL OUTPUT\n"
        "Return the full decision summary to the user, including:\n"
        "  - The ranked recommendations (best first)\n"
        "  - The TOPSIS scores and breakdown\n"
        "  - The natural-language explanation\n"
        "  - All citations and risk flags\n\n"
        "IMPORTANT:\n"
        "- Always call all three specialist agents — do not skip any.\n"
        "- Always use score_recommendations to rank them — do not rank manually.\n"
        "- Always use explain_decision for the final summary.\n"
        "- All recommendations are advisory. Final decisions rest with the treating physician."
    ),
    tools=[
        AgentTool(agent=cardiology_agent),
        AgentTool(agent=nephrology_agent),
        AgentTool(agent=endocrinology_agent),
        score_recommendations,
        explain_decision,
    ],
    before_model_callback=extract_fhir_context,
)
