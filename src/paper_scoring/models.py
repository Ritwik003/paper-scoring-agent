"""The shape of a single paper's data.

Pydantic checks each field's type, so we never store half-broken records.
"""
from datetime import datetime
from pydantic import BaseModel


class Paper(BaseModel):
    arxiv_id: str                 # e.g. "2507.01234v1"
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]         # e.g. ["cs.LG", "cs.AI"]
    published: datetime           # when it was submitted to arXiv
    pdf_url: str
    pdf_path: str | None = None   # filled in once the PDF is downloaded