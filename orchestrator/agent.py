"""asm_orchestrator — Multi-specialty clinical orchestrator.

True multi-agent architecture:
1. Call 3 specialist sub-agents via AgentTool (in-process, parallel-capable)
2. Collect structured recommendations
3. Score with deterministic TOPSIS
4. Generate explainable decision

Sub-agents are optimized for single LLM call each (no FHIR tools).
"""
import json
import os

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


def _compute_patient_match(specialty: str, patient_context: str) -> float:
    """Compute patient match from clinical thresholds in the patient context."""
    ctx = patient_context.lower()
    score = 0.5

    # Extract EF
    ef = None
    for token in ctx.split():
        if "ef" in token or "lvef" in token:
            for t in ctx.split():
                try:
                    val = float(t.replace("%", ""))
                    if 10 <= val <= 80:
                        ef = val
                        break
                except ValueError:
                    pass
    # Try to find standalone number near "ef" or "lvef"
    import re
    ef_match = re.search(r'(?:ef|lvef)\s*[:=]?\s*(\d+\.?\d*)', ctx)
    if ef_match:
        ef = float(ef_match.group(1))

    egfr_match = re.search(r'(?:egfr)\s*[:=]?\s*(\d+\.?\d*)', ctx)
    egfr = float(egfr_match.group(1)) if egfr_match else None

    hba1c_match = re.search(r'(?:hba1c)\s*[:=]?\s*(\d+\.?\d*)', ctx)
    hba1c = float(hba1c_match.group(1)) if hba1c_match else None

    has_hf = any(kw in ctx for kw in ["heart failure", "hf", "hfref", "hfpef", "lvef"])
    has_ckd = any(kw in ctx for kw in ["ckd", "kidney", "egfr", "renal"])
    has_dm = any(kw in ctx for kw in ["diabetes", "t2dm", "hba1c", "diabetic"])

    if specialty == "cardiology":
        if has_hf and ef is not None and ef < 40:
            score += 0.4
        elif has_hf:
            score += 0.15

    elif specialty == "nephrology":
        if has_ckd and egfr is not None:
            if egfr < 30:
                score += 0.4
            elif egfr < 60:
                score += 0.2

    elif specialty == "endocrinology":
        if has_dm and hba1c is not None:
            if hba1c > 8:
                score += 0.3
            elif hba1c > 7:
                score += 0.15

    return min(score, 1.0)


def _compute_drug_risk(risk_flags: list[str]) -> float:
    if not risk_flags:
        return 0.0
    high = {"contraindicated", "stop", "severe", "avoid", "critical", "immediate", "danger", "lactic acidosis"}
    moderate = {"caution", "monitor", "watch", "reduce", "adjust", "titrate"}
    score = 0.0
    for flag in risk_flags:
        flag_lower = flag.lower()
        if any(kw in flag_lower for kw in high):
            score += 0.2
        elif any(kw in flag_lower for kw in moderate):
            score += 0.05
        else:
            score += 0.08
    return min(score, 1.0)


def _compute_guideline_priority(evidence_level: str) -> float:
    level = evidence_level.strip().lower()
    mapping = {
        "class i": 1.0, "class ii": 0.75, "class iia": 0.75, "class iib": 0.5,
        "class iii": 0.25, "1a": 1.0, "1b": 0.85, "2a": 0.6, "2b": 0.4,
        "a": 1.0, "level a": 1.0, "b": 0.75, "level b": 0.75,
        "c": 0.5, "level c": 0.5, "e": 0.3, "level e": 0.3,
    }
    for key, val in mapping.items():
        if key in level:
            return val
    return 0.5


def score_and_explain(
    cardiology_json: str,
    nephrology_json: str,
    endocrinology_json: str,
    patient_context: str,
) -> dict:
    """Parse 3 specialist recommendations, run deterministic TOPSIS, return ranked decision."""
    recs_raw = []
    for raw in [cardiology_json, nephrology_json, endocrinology_json]:
        try:
            # Try to parse as JSON (from tool return)
            if isinstance(raw, str):
                data = json.loads(raw)
            else:
                data = raw
            recs_raw.append(data)
        except (json.JSONDecodeError, TypeError):
            # If it's not JSON, create a minimal dict
            recs_raw.append({
                "specialty": "unknown",
                "recommendation": str(raw),
                "evidence": "unknown",
                "risks": [],
                "citation": "",
            })

    recs = []
    for r in recs_raw:
        specialty = r.get("specialty", "unknown")
        evidence = r.get("evidence", "")
        risks = r.get("risks", [])
        recs.append(Recommendation(
            specialty=specialty,
            recommendation=r.get("recommendation", ""),
            confidence=0.85,
            evidence_level=evidence,
            evidence_score=_normalize_evidence(evidence),
            patient_match=_compute_patient_match(specialty, patient_context),
            drug_interaction_risk=_compute_drug_risk(risks),
            guideline_priority=_compute_guideline_priority(evidence),
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
            "reasoning": s.reasoning,
        })

    return {
        "status": "success",
        "ranked_recommendations": results,
        "top_pick": results[0] if results else None,
    }


root_agent = Agent(
    name="asm_orchestrator",
    model=_model,
    description=(
        "Multi-specialty clinical orchestrator for HF+T2DM+CKD patients. "
        "Routes to 3 specialist agents, scores with TOPSIS, explains decision."
    ),
    instruction=(
        "You are the ASM Clinical Orchestrator for HF+T2DM+CKD patients.\n\n"
        "WORKFLOW:\n"
        "1. Call cardiology_agent with the patient summary\n"
        "2. Call nephrology_agent with the patient summary\n"
        "3. Call endocrinology_agent with the patient summary\n"
        "4. Call score_and_explain with all 3 results + the original patient message\n\n"
        "For steps 1-3, pass the FULL patient message as the input to each agent.\n"
        "For step 4, pass each agent's response as a string, plus the patient message.\n\n"
        "After score_and_explain returns, present the ranked results clearly:\n"
        "- Show TOPSIS scores and ranking\n"
        "- Show each specialist's recommendation\n"
        "- Show the unified action plan\n"
        "- Include disclaimer that recommendations are advisory\n"
    ),
    tools=[
        AgentTool(agent=cardiology_agent),
        AgentTool(agent=nephrology_agent),
        AgentTool(agent=endocrinology_agent),
        FunctionTool(func=score_and_explain),
    ],
    before_model_callback=extract_fhir_context,
)
