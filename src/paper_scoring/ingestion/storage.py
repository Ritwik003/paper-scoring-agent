"""Save and load paper metadata as JSONL (one JSON object per line).

JSONL is ideal here: easy to append to, and easy to read back line by line
without loading everything into memory at once.
"""
import json
from pathlib import Path

from paper_scoring.models import Paper


def save_papers(papers: list[Paper], metadata_dir: Path, year: int, week: int) -> Path:
    """Write papers to metadata_dir/<year>-W<week>.jsonl. Returns the file path."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    out_path = metadata_dir / f"{year}-W{week:02d}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for paper in papers:
            # model_dump_json() turns the Paper into a JSON string safely.
            f.write(paper.model_dump_json() + "\n")

    return out_path