"""Celery worker + task for parallel paper scoring.

Manager's diagram: run_weekly.py creates one Celery task per paper. Redis is
the queue. Workers pull tasks in parallel, each downloads the PDF into
memory, runs Agents A/B/C, and returns the score row.

Start Redis (once):        redis-server
Start a worker (separate):
    celery -A paper_scoring.workers.celery_app worker --loglevel=info --concurrency=4

Then run scripts/run_weekly.py normally - it fans out tasks to the workers.
"""
import os
from pathlib import Path

# .env is loaded before importing anything that needs GROQ / TAVILY keys.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except Exception:
    pass

from celery import Celery

from paper_scoring.config import load_settings
from paper_scoring.ingestion.downloader import download_pdf_bytes
from paper_scoring.scoring.graph import build_graph, score_paper


BROKER_URL  = os.environ.get("CELERY_BROKER_URL",  "redis://localhost:6379/0")
RESULT_URL  = os.environ.get("CELERY_RESULT_URL",  "redis://localhost:6379/1")


celery_app = Celery(
    "paper_scoring",
    broker=BROKER_URL,
    backend=RESULT_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_time_limit=600,        # kill a task if it runs longer than 10 minutes
    task_soft_time_limit=540,
    worker_prefetch_multiplier=1,   # be fair: one task at a time per worker slot
)


# Compile the LangGraph pipeline ONCE per worker process, not per task,
# so we don't pay the setup cost on every paper.
_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph(load_settings())
    return _GRAPH


@celery_app.task(name="score_paper_task", bind=True, max_retries=2, default_retry_delay=30)
def score_paper_task(self, paper: dict) -> dict:
    """Score one paper end-to-end. Runs on a Celery worker."""
    # Metadata carried through to the result (needed by the publisher).
    meta = {
        "authors":    paper.get("authors", []),
        "abstract":   paper.get("abstract", ""),
        "published":  paper.get("published", ""),
        "categories": paper.get("categories", []),
    }

    try:
        paper["pdf_bytes"] = download_pdf_bytes(paper["pdf_url"])
    except Exception as e:
        return {
            "arxiv_id": paper.get("arxiv_id", ""),
            "title":    paper.get("title", ""),
            "pdf_url":  paper.get("pdf_url", ""),
            "error":    f"download_failed: {e}",
            **meta,
        }

    try:
        final = score_paper(_get_graph(), paper)
    finally:
        paper.pop("pdf_bytes", None)

    return {
        # scoring output
        "arxiv_id":       final.get("arxiv_id", "") or paper.get("arxiv_id", ""),
        "title":          final.get("title", "")    or paper.get("title", ""),
        "pdf_url":        final.get("pdf_url", "")  or paper.get("pdf_url", ""),
        "scores":         final.get("scores", {}),
        "weighted_total": final.get("weighted_total"),
        "verdict":        final.get("verdict"),
        "reason":         (final.get("evidence", {}) or {}).get("reason", ""),
        "notes":          final.get("notes", []),
        # original metadata (needed by publish_week.py to render the markdown)
        **meta,
    }