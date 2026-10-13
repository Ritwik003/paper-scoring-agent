"""The shared 'clipboard' that passes down the agent pipeline.

Agent A fills paper_object (structured Paper Understanding).
Agent B fills context      (Research Context - repo, citations, similar work, discussion).
Agent C reads BOTH and writes scores + verdict. Agent C never sees raw PDF text.
"""
from typing import Optional, TypedDict


CRITERIA = [
    "significance", "novelty", "methodology", "empirical",
    "reproducibility", "presentation", "ethics",
]


class ReviewState(TypedDict, total=False):
    # ----- input -----
    arxiv_id: str
    title: str
    abstract: str
    authors: list
    pdf_url: str
    pdf_bytes: Optional[bytes]
    categories: list
    published: str

    # ----- Agent A output -----
    paper_object: dict     # {title, abstract, introduction, ..., tables, figures, references, facts}
    gate_failed: bool
    gate_reason: Optional[str]

    # ----- Agent B output -----
    context: dict          # {citations, repository, similar_work, discussion, dataset}

    # ----- Agent C output -----
    scores: dict           # {"significance": 8.0, ...}
    evidence: dict         # {"reason": "..."}

    # ----- final -----
    weighted_total: Optional[float]
    verdict: Optional[str] # "read" | "skim" | "skip"
    notes: list


def new_state(paper: dict) -> ReviewState:
    """Build a fresh state from a metadata row (title, abstract, pdf_url, pdf_bytes...)."""
    return ReviewState(
        arxiv_id=paper.get("arxiv_id", ""),
        title=paper.get("title", ""),
        abstract=paper.get("abstract", ""),
        authors=paper.get("authors", []),
        pdf_url=paper.get("pdf_url", ""),
        pdf_bytes=paper.get("pdf_bytes"),
        categories=paper.get("categories", []),
        published=paper.get("published", ""),
        paper_object={},
        gate_failed=False,
        gate_reason=None,
        context={},
        scores={},
        evidence={},
        weighted_total=None,
        verdict=None,
        notes=[],
    )
