"""Download the PDF for each paper and record where it was saved.

Downloads directly from each paper's pdf_url using Python's built-in urllib,
so it doesn't depend on the arxiv library's download helpers. Safe to re-run:
papers already on disk are skipped, and one failure doesn't stop the rest.
"""
import time
import urllib.request
from pathlib import Path

from paper_scoring.models import Paper

# arXiv blocks requests with no User-Agent; identify ourselves politely.
_HEADERS = {"User-Agent": "paper-scoring-agent/0.1 (research use)"}


def download_pdfs(papers: list[Paper], pdf_dir: Path, delay_seconds: float = 3.0) -> list[Paper]:
    """Download each paper's PDF into pdf_dir. Returns papers with pdf_path set."""
    pdf_dir.mkdir(parents=True, exist_ok=True)

    total = len(papers)
    for i, paper in enumerate(papers, start=1):
        filename = paper.arxiv_id.replace("/", "_") + ".pdf"
        target = pdf_dir / filename

        # Skip if we already have it — makes re-runs cheap and resumable.
        if target.exists():
            paper.pdf_path = str(target)
            print(f"[{i}/{total}] already have {filename}")
            continue

        try:
            req = urllib.request.Request(paper.pdf_url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            target.write_bytes(data)
            paper.pdf_path = str(target)
            print(f"[{i}/{total}] downloaded {filename}")
            time.sleep(delay_seconds)   # polite pause between downloads
        except Exception as e:
            paper.pdf_path = None
            print(f"[{i}/{total}] FAILED {filename}: {e}")

    return papers


def download_pdf_bytes(pdf_url: str) -> bytes:
    """
    Download a PDF into memory.

    Returns:
        bytes: Raw PDF bytes.
    Raises:
        Exception if download fails.
    """

    req = urllib.request.Request(pdf_url, headers=_HEADERS)

    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()