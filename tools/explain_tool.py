"""ADK tool for generating an explainable decision summary.

Takes the TOPSIS scoring result and produces a natural-language
decision explanation suitable for a clinician audience.
"""
import json
import logging

from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


def explain_decision(score_result_json: str, tool_context: ToolContext) -> dict:
    """Generate an explainable clinical decision summary from TOPSIS scoring results.

    Args:
        score_result_json: JSON string from score_recommendations output.

    Returns:
        A dict with a natural-language explanation of the ranking.
    """
    logger.info("tool_explain_decision entered")

    try:
        data = json.loads(score_result_json)
    except json.JSONDecodeError as e:
        return {"status": "error", "detail": f"Invalid JSON: {e}"}

    ranked = data.get("ranked_recommendations", [])
    if not ranked:
        return {"status": "error", "detail": "No ranked recommendations found"}

    # Build the explanation
    top = ranked[0]
    lines = []
    lines.append("## Multi-Specialty Decision Summary")
    lines.append("")
    lines.append(f"**Primary recommendation (Rank #1): {top['specialty'].upper()}**")
    lines.append(f"> {top['recommendation']}")
    lines.append("")
    lines.append(f"- TOPSIS score: {top['total_score']:.3f}")
    lines.append(f"- Evidence level: {top.get('evidence_level', 'N/A')}")
    lines.append(f"- Confidence: {top.get('confidence', 'N/A')}")
    if top.get("citation"):
        lines.append(f"- Guideline reference: {top['citation']}")
    if top.get("risk_flags"):
        lines.append(f"- Risk flags: {', '.join(top['risk_flags'])}")
    lines.append("")

    # Explain why this one won
    breakdown = top.get("breakdown", {})
    if breakdown:
        best_dim = max(breakdown, key=breakdown.get)
        lines.append(f"**Why this ranked highest:** Strongest performance in *{best_dim}* "
                     f"(score: {breakdown[best_dim]:.2f}).")
        lines.append("")

    # Show runners-up
    if len(ranked) > 1:
        lines.append("### Supporting recommendations")
        lines.append("")
        for rec in ranked[1:]:
            lines.append(f"**Rank #{rec['rank']}: {rec['specialty'].upper()}** (score: {rec['total_score']:.3f})")
            lines.append(f"> {rec['recommendation']}")
            if rec.get("risk_flags"):
                lines.append(f"- Risk flags: {', '.join(rec['risk_flags'])}")
            lines.append("")

    # Decision rationale
    lines.append("### Decision rationale")
    lines.append("")
    lines.append("This ranking was computed using TOPSIS (Technique for Order Preference by "
                 "Similarity to Ideal Solution) across four clinical dimensions:")
    lines.append("1. **Evidence level** — strength of clinical evidence per ACC/AHA, KDIGO, ADA guidelines")
    lines.append("2. **Patient match** — how well the recommendation fits this patient's specific conditions")
    lines.append("3. **Drug interaction risk** — potential for adverse interactions with existing medications")
    lines.append("4. **Guideline priority** — how strongly the relevant clinical guideline recommends this action")
    lines.append("")
    lines.append("All recommendations are advisory. Final clinical decisions rest with the treating physician.")

    explanation = "\n".join(lines)
    logger.info("tool_explain_decision completed")

    return {
        "status": "success",
        "explanation": explanation,
        "top_recommendation": {
            "specialty": top["specialty"],
            "recommendation": top["recommendation"],
            "score": top["total_score"],
        },
    }
