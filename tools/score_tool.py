"""ADK tool wrapper for TOPSIS scoring.

The orchestrator calls this after collecting recommendations from
the 3 specialty sub-agents.
"""
import json
import logging

from google.adk.tools import ToolContext

from .topsis import Recommendation, Preferences, score_topsis

logger = logging.getLogger(__name__)


def score_recommendations(recommendations_json: str, tool_context: ToolContext) -> dict:
    """Score a list of specialty recommendations using TOPSIS multi-criteria ranking.

    Args:
        recommendations_json: A JSON string containing a list of recommendations.
            Each item should have: specialty, recommendation, confidence,
            evidence_level, risk_flags, fhir_refs, citation.
            Optionally: evidence_score (0-1), patient_match (0-1),
            drug_interaction_risk safety score (0-1, higher is safer), guideline_priority (0-1).

    Returns:
        A dict with ranked recommendations and scoring details.
    """
    logger.info("tool_score_recommendations entered")

    try:
        recs_data = json.loads(recommendations_json)
    except json.JSONDecodeError as e:
        return {"status": "error", "detail": f"Invalid JSON: {e}"}

    if not isinstance(recs_data, list):
        return {"status": "error", "detail": "Expected a JSON list of recommendations"}

    # Parse into Recommendation objects
    recommendations = []
    for item in recs_data:
        if not isinstance(item, dict):
            continue
        rec = Recommendation(
            specialty=item.get("specialty", "unknown"),
            recommendation=item.get("recommendation", ""),
            confidence=item.get("confidence", 0.5),
            evidence_level=item.get("evidence_level", ""),
            evidence_score=item.get("evidence_score", 0.5),
            patient_match=item.get("patient_match", 0.5),
            drug_interaction_risk=item.get("drug_interaction_risk", 1.0),
            guideline_priority=item.get("guideline_priority", 0.5),
            risk_flags=item.get("risk_flags", []),
            fhir_refs=item.get("fhir_refs", []),
            citation=item.get("citation", ""),
        )
        recommendations.append(rec)

    if len(recommendations) < 2:
        return {
            "status": "warning",
            "detail": "Need at least 2 recommendations for meaningful ranking",
            "recommendations": [r.__dict__ for r in recommendations],
        }

    # Run TOPSIS
    scored = score_topsis(recommendations)

    # Build output
    results = []
    for s in scored:
        results.append({
            "rank": s.rank,
            "specialty": s.recommendation.specialty,
            "recommendation": s.recommendation.recommendation,
            "total_score": s.total_score,
            "breakdown": s.breakdown,
            "confidence": s.recommendation.confidence,
            "evidence_level": s.recommendation.evidence_level,
            "risk_flags": s.recommendation.risk_flags,
            "fhir_refs": s.recommendation.fhir_refs,
            "citation": s.recommendation.citation,
            "reasoning": s.reasoning,
        })

    logger.info("tool_score_recommendations completed, ranked %d recommendations", len(results))
    return {
        "status": "success",
        "ranked_recommendations": results,
        "summary": results[0]["reasoning"] if results else "",
    }
