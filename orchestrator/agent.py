"""asm_orchestrator — Multi-specialty clinical orchestrator.

Sub-agents return JSON directly (no tools = no second LLM call).
Orchestrator parses results, runs deterministic TOPSIS, formats output.
"""
import json
import os
import re

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool

from cardiology_agent.agent import root_agent as cardiology_agent
from nephrology_agent.agent import root_agent as nephrology_agent
from endocrinology_agent.agent import root_agent as endocrinology_agent
from shared.fhir_hook import extract_fhir_context
from tools.topsis import Recommendation, score_topsis, _normalize_evidence

_model_name = os.getenv("ORCHESTRATOR_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)


def _extract_json(text: str) -> dict:
    """Extract JSON from agent response (handles markdown wrapping)."""
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try extracting JSON from markdown code block
    match = re.search(r'\{[^{}]*"specialty"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"specialty": "unknown", "recommendation": str(text), "evidence": "", "risks": [], "citation": ""}


def _compute_patient_match(specialty: str, ctx: str) -> float:
    score = 0.5
    ef_m = re.search(r'(?:ef|lvef)\D*(\d+)', ctx)
    egfr_m = re.search(r'egfr\D*(\d+)', ctx)
    hba1c_m = re.search(r'hba1c\D*(\d+\.?\d*)', ctx)
    ef = float(ef_m.group(1)) if ef_m else None
    egfr = float(egfr_m.group(1)) if egfr_m else None
    hba1c = float(hba1c_m.group(1)) if hba1c_m else None

    if specialty == "cardiology" and ef is not None and ef < 40:
        score += 0.4
    elif specialty == "nephrology" and egfr is not None and egfr < 30:
        score += 0.4
    elif specialty == "endocrinology" and hba1c is not None and hba1c > 8:
        score += 0.3
    return min(score, 1.0)


def _compute_drug_risk(risks: list[str]) -> float:
    if not risks:
        return 0.0
    high = {"contraindicated", "stop", "severe", "avoid", "critical", "lactic acidosis"}
    mod = {"caution", "monitor", "reduce", "adjust"}
    s = 0.0
    for r in risks:
        rl = r.lower()
        if any(k in rl for k in high):
            s += 0.2
        elif any(k in rl for k in mod):
            s += 0.05
        else:
            s += 0.08
    return min(s, 1.0)


def _compute_guideline_priority(evidence: str) -> float:
    e = evidence.strip().lower()
    for k, v in [("class i", 1.0), ("class iia", 0.75), ("class iib", 0.5),
                  ("class iii", 0.25), ("1a", 1.0), ("1b", 0.85), ("2a", 0.6),
                  ("level a", 1.0), ("level b", 0.75)]:
        if k in e:
            return v
    return 0.5


def score_and_explain(cardiology: str, nephrology: str, endocrinology: str, patient: str) -> dict:
    """Parse 3 specialist JSON responses, run TOPSIS, return ranked decision."""
    raw = [_extract_json(cardiology), _extract_json(nephrology), _extract_json(endocrinology)]

    recs = []
    for r in raw:
        sp = r.get("specialty", "unknown")
        ev = r.get("evidence", "")
        risks = r.get("risks", [])
        recs.append(Recommendation(
            specialty=sp,
            recommendation=r.get("recommendation", ""),
            confidence=0.85,
            evidence_level=ev,
            evidence_score=_normalize_evidence(ev),
            patient_match=_compute_patient_match(sp, patient),
            drug_interaction_risk=_compute_drug_risk(risks),
            guideline_priority=_compute_guideline_priority(ev),
            risk_flags=risks,
            citation=r.get("citation", ""),
        ))

    scored = score_topsis(recs)
    results = []
    for s in scored:
        results.append({
            "rank": s.rank,
            "specialty": s.recommendation.specialty,
            "recommendation": s.recommendation.recommendation,
            "total_score": s.total_score,
            "breakdown": s.breakdown,
            "evidence_level": s.recommendation.evidence_level,
            "risk_flags": s.recommendation.risk_flags,
            "citation": s.recommendation.citation,
        })

    return {"status": "success", "ranked_recommendations": results, "top_pick": results[0] if results else None}


root_agent = Agent(
    name="asm_orchestrator",
    model=_model,
    description="Multi-specialty orchestrator for HF+T2DM+CKD. Routes to 3 specialists, scores with TOPSIS.",
    instruction=(
        "You are the ASM Orchestrator. The message has a patient summary.\n\n"
        "Steps:\n"
        "1. Call cardiology_agent with the patient message\n"
        "2. Call nephrology_agent with the patient message\n"
        "3. Call endocrinology_agent with the patient message\n"
        "4. Call score_and_explain(cardiology=response1, nephrology=response2, endocrinology=response3, patient=original_message)\n"
        "5. Present the TOPSIS-ranked results with a unified action plan\n\n"
        "Keep your final presentation concise. Show ranking, top pick, and action plan."
    ),
    tools=[
        AgentTool(agent=cardiology_agent),
        AgentTool(agent=nephrology_agent),
        AgentTool(agent=endocrinology_agent),
        FunctionTool(func=score_and_explain),
    ],
    before_model_callback=extract_fhir_context,
)
