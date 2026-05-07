"""TOPSIS Multi-Criteria Decision Engine — Healthcare Edition.

Ported from ARC's scorer/scorer.py and adapted for medical use.
Original dimensions: cost / quality / speed / reliability
New dimensions: evidence_level / patient_match / drug_safety / guideline_priority

ARC TOPSIS reference: /Users/a1234/Documents/ARC/scorer/scorer.py (lines 308-410)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Recommendation:
    """A structured recommendation from a specialty agent."""
    specialty: str
    recommendation: str
    confidence: float = 0.5
    evidence_level: str = ""
    evidence_score: float = 0.5  # normalized 0-1
    patient_match: float = 0.5   # how well it matches current patient state
    drug_interaction_risk: float = 1.0  # medication safety score: 1=lowest risk, 0=highest risk
    guideline_priority: float = 0.5     # how strongly guideline recommends it
    risk_flags: list[str] = field(default_factory=list)
    fhir_refs: list[str] = field(default_factory=list)
    citation: str = ""


@dataclass
class Preferences:
    """Weights for TOPSIS scoring dimensions. Must sum to 1.0."""
    evidence_level: float = 0.30      # higher = better (stronger evidence)
    patient_match: float = 0.30       # higher = better (more relevant to patient)
    drug_interaction_risk: float = 0.20  # higher = better (fewer risks)
    guideline_priority: float = 0.20  # higher = better (guideline-preferred)

    def __post_init__(self):
        total = (
            self.evidence_level
            + self.patient_match
            + self.drug_interaction_risk
            + self.guideline_priority
        )
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")


@dataclass
class ScoredRecommendation:
    """A recommendation with its TOPSIS score and breakdown."""
    recommendation: Recommendation
    total_score: float
    breakdown: dict[str, float]
    rank: int = 0
    reasoning: str = ""


# ──────────────────────────────────────────────
# Evidence Level Mapping (normalized 0-1)
# ──────────────────────────────────────────────

EVIDENCE_SCORE_TABLE = {
    # ACC/AHA grades
    "Class I Level A": 1.00,
    "Class I Level B-R": 0.90,
    "Class I Level B-NR": 0.85,
    "Class I Level B": 0.85,
    "Class I Level C": 0.75,
    "Class I": 0.80,
    "Class IIa Level A": 0.70,
    "Class IIa Level B": 0.60,
    "Class IIa Level C": 0.55,
    "Class IIa": 0.60,
    "Class IIb Level A": 0.45,
    "Class IIb Level B": 0.35,
    "Class IIb Level C": 0.25,
    "Class IIb": 0.35,
    "Class III": 0.05,
    # KDIGO grades
    "Grade 1A": 1.00,
    "Grade 1B": 0.85,
    "Grade 1C": 0.70,
    "Grade 1D": 0.55,
    "Grade 2A": 0.45,
    "Grade 2B": 0.35,
    "Grade 2C": 0.25,
    "Grade 2D": 0.15,
    # ADA levels
    "Level A": 1.00,
    "Level B": 0.75,
    "Level C": 0.50,
    "Level E": 0.30,
}


def _normalize_evidence(level: str) -> float:
    """Map a clinical evidence level string to a deterministic 0-1 score."""
    if level in EVIDENCE_SCORE_TABLE:
        return EVIDENCE_SCORE_TABLE[level]

    key = (level or "").strip().lower()
    for evidence_key, score in sorted(EVIDENCE_SCORE_TABLE.items(), key=lambda item: len(item[0]), reverse=True):
        if evidence_key.lower() in key:
            return score
    return 0.55


def _calibrate_scores(results: list[ScoredRecommendation]) -> None:
    """Map TOPSIS coefficients into a realistic clinical display range."""
    if not results:
        return
    if len(results) == 1:
        results[0].total_score = 0.500
        return

    coefficients = [r.total_score for r in results]
    if max(coefficients) == min(coefficients):
        for r in results:
            r.total_score = 0.500
        return

    high = 0.900
    low = 0.350
    steps = len(results) - 1
    previous_coefficient = None
    previous_score = None
    for index, result in enumerate(results):
        if previous_coefficient is not None and math.isclose(
            result.total_score, previous_coefficient, abs_tol=1e-9
        ):
            result.total_score = previous_score
        else:
            result.total_score = round(high - ((high - low) * index / steps), 3)
        previous_coefficient = coefficients[index]
        previous_score = result.total_score


# ──────────────────────────────────────────────
# TOPSIS Scorer
# ──────────────────────────────────────────────

def score_topsis(
    recommendations: list[Recommendation],
    preferences: Preferences | None = None,
) -> list[ScoredRecommendation]:
    """Rank recommendations using TOPSIS (Technique for Order Preference by
    Similarity to Ideal Solution).

    Dimensions:
      1. evidence_score    — benefit (higher is better)
      2. patient_match     — benefit (higher is better)
      3. drug_interaction_risk — benefit safety score (higher is better)
      4. guideline_priority — benefit (higher is better)

    Steps (same as ARC):
      1. Build decision matrix
      2. Vector normalization
      3. Apply weights
      4. Find positive ideal (A+) and negative ideal (A-)
      5. Calculate distances
      6. Closeness coefficient C = d- / (d+ + d-)
      7. Rank by C (higher = better)
    """
    if not recommendations:
        return []

    if preferences is None:
        preferences = Preferences()

    n = len(recommendations)

    # Normalize evidence levels to scores
    for r in recommendations:
        if r.evidence_score == 0.5 and r.evidence_level:
            r.evidence_score = _normalize_evidence(r.evidence_level)

    # Step 1: Decision matrix (n x 4)
    raw = []
    for r in recommendations:
        raw.append([
            r.evidence_score,          # benefit: higher is better
            r.patient_match,           # benefit: higher is better
            r.drug_interaction_risk,   # benefit: higher safety score is better
            r.guideline_priority,      # benefit: higher is better
        ])

    is_benefit = [True, True, True, True]
    weights = [
        preferences.evidence_level,
        preferences.patient_match,
        preferences.drug_interaction_risk,
        preferences.guideline_priority,
    ]

    # Step 2: Min-max normalization (more stable than vector normalization
    # when values have extreme differences, e.g. drug_interaction_risk 0.0 vs 0.35)
    num_criteria = 4
    norm = [[0.0] * num_criteria for _ in range(n)]
    for j in range(num_criteria):
        col = [raw[i][j] for i in range(n)]
        col_min = min(col)
        col_max = max(col)
        if col_max == col_min:
            for i in range(n):
                norm[i][j] = 1.0
        else:
            for i in range(n):
                norm[i][j] = (raw[i][j] - col_min) / (col_max - col_min)

    # Step 3: Weighted normalized matrix
    weighted = [[norm[i][j] * weights[j] for j in range(num_criteria)] for i in range(n)]

    # Step 4: Ideal solutions
    a_pos = []
    a_neg = []
    for j in range(num_criteria):
        col = [weighted[i][j] for i in range(n)]
        if is_benefit[j]:
            a_pos.append(max(col))
            a_neg.append(min(col))
        else:
            a_pos.append(min(col))
            a_neg.append(max(col))

    # Step 5: Distances
    d_pos = []
    d_neg = []
    for i in range(n):
        dp = math.sqrt(sum((weighted[i][j] - a_pos[j]) ** 2 for j in range(num_criteria)))
        dn = math.sqrt(sum((weighted[i][j] - a_neg[j]) ** 2 for j in range(num_criteria)))
        d_pos.append(dp)
        d_neg.append(dn)

    # Step 6 & 7: Closeness coefficient and ranking
    results = []
    labels = ["evidence_level", "patient_match", "drug_interaction_risk", "guideline_priority"]
    for i, r in enumerate(recommendations):
        denom = d_pos[i] + d_neg[i]
        c = d_neg[i] / denom if denom > 0 else 0.5

        breakdown = {
            label: round(raw[i][j], 4)
            for j, label in enumerate(labels)
        }

        results.append(ScoredRecommendation(
            recommendation=r,
            total_score=round(c, 6),
            breakdown=breakdown,
        ))

    results.sort(key=lambda x: x.total_score, reverse=True)
    _calibrate_scores(results)
    for i, r in enumerate(results):
        r.rank = i + 1
        r.reasoning = _generate_reasoning(r, preferences)
    return results


def _generate_reasoning(scored: ScoredRecommendation, prefs: Preferences) -> str:
    """Generate a human-readable reasoning for the score."""
    r = scored.recommendation
    top_dim = max(scored.breakdown, key=lambda k: scored.breakdown[k])
    return (
        f"{r.specialty} recommendation scored {scored.total_score:.3f} "
        f"(evidence={scored.breakdown['evidence_level']:.2f}, "
        f"patient_match={scored.breakdown['patient_match']:.2f}, "
        f"drug_safety={scored.breakdown['drug_interaction_risk']:.2f}, "
        f"guideline={scored.breakdown['guideline_priority']:.2f}). "
        f"Strongest dimension: {top_dim}."
    )
