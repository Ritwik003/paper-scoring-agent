"""Fetch and download one week of AI/ML papers from arXiv.

Usage (from the project root):
    python scripts/fetch_week.py                 # last completed week (auto)
    python scripts/fetch_week.py 2026 27         # a specific year and week
    python scripts/fetch_week.py 2026 27 --no-pdf   # metadata only, skip PDFs
"""
import sys
from pathlib import Path

# Make the 'paper_scoring' package importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_scoring.config import load_settings, get_paths
from paper_scoring.dates import week_range, last_completed_week
from paper_scoring.ingestion.arxiv_source import fetch_week
#from paper_scoring.ingestion.downloader import download_pdfs
from paper_scoring.ingestion.storage import save_papers


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    #no_pdf = "--no-pdf" in sys.argv

    # Decide which week to fetch.
    if len(args) >= 2:
        year, week = int(args[0]), int(args[1])
    else:
        year, week = last_completed_week()

    settings = load_settings()
    paths = get_paths(settings)
    categories = settings["arxiv"]["categories"]
    arxiv_cfg = settings["arxiv"]
    start, end = week_range(year, week)

    print(f"Week       : {year}-W{week:02d}")
    print(f"Date range : {start} to {end}")
    print(f"Categories : {', '.join(categories)}")
    print("-" * 50)

    # 1. Fetch metadata.
    papers = fetch_week(
        categories, start, end,
        delay_seconds=arxiv_cfg["delay_seconds"],
        page_size=arxiv_cfg["page_size"],
        num_retries=arxiv_cfg["num_retries"],
        max_results=arxiv_cfg["max_results"],
    )
    print(f"Fetched {len(papers)} papers.")

    # 2. Download PDFs (unless --no-pdf).
    '''if not no_pdf and papers:
        download_pdfs(papers, paths["pdf_dir"], delay_seconds=arxiv_cfg["delay_seconds"])'''

    # 3. Save the metadata index.
    out_path = save_papers(papers, paths["metadata_dir"], year, week)
    print("-" * 50)
    print(f"Saved metadata -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()