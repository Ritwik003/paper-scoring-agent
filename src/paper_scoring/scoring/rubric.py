"""Turn the 7 individual scores into one weighted total and a verdict.

Weights and cut-offs live in config/settings.yaml, so the team can tune them
without touching code.
"""
from paper_scoring.scoring.state import CRITERIA


def weighted_total(scores: dict, weights: dict) -> float:
    """Combine the 7 scores (each 1-10) into one number on the same 1-10 scale."""
    total = sum(float(scores.get(c, 0)) * float(weights[c]) for c in CRITERIA)
    return round(total, 2)


def decide_verdict(total: float, verdict_cfg: dict) -> str:
    """Map the weighted total to read / skim / skip."""
    if total >= verdict_cfg["read_min"]:
        return "read"
    if total >= verdict_cfg["skim_min"]:
        return "skim"
    return "skip"
