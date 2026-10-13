"""Fetch paper metadata from arXiv for one week.

This asks the arXiv API for every paper in our chosen categories that was
submitted within a given date range, and returns them as clean `Paper`
objects. It does NOT download PDFs — that is a separate step.
"""
from datetime import date, datetime, time

import arxiv

from paper_scoring.models import Paper


def _build_query(categories: list[str], start: date, end: date) -> str:
    """Build the arXiv search string: our categories AND the date window."""
    cats = " OR ".join(f"cat:{c}" for c in categories)
    # arXiv wants the date window as YYYYMMDDHHMM.
    date_window = f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    return f"({cats}) AND {date_window}"


def fetch_week(
    categories: list[str],
    start: date,
    end: date,
    delay_seconds: float = 3.0,
    page_size: int = 100,
    num_retries: int = 5,
    max_results: int = 5000,
) -> list[Paper]:
    """Return a list of Paper objects submitted between start and end (inclusive)."""
    query = _build_query(categories, start, end)
    print(f"arXiv query: {query}")

    client = arxiv.Client(
        page_size=page_size,
        delay_seconds=delay_seconds,   # polite pause between requests
        num_retries=num_retries,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    # Compare dates safely: arXiv timestamps are timezone-aware.
    start_dt = datetime.combine(start, time.min).astimezone()
    end_dt = datetime.combine(end, time.max).astimezone()

    papers: list[Paper] = []
    seen = 0
    for result in client.results(search):
        seen += 1
        if seen % 100 == 0:
            print(f"  ...scanned {seen} results so far")

        # Extra safety check: keep only papers truly inside our window.
        if not (start_dt <= result.published <= end_dt):
            continue
        papers.append(
            Paper(
                arxiv_id=result.get_short_id(),
                title=" ".join(result.title.split()),
                authors=[a.name for a in result.authors],
                abstract=" ".join(result.summary.split()),
                categories=list(result.categories),
                published=result.published,
                pdf_url=result.pdf_url,
            )
        )
    return papers