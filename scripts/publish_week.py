"""Publish the qualified papers (score >= threshold) for one week.

Usage (from the project root):
    python scripts/publish_week.py 2026 27           # write the weekly_picks folder
    python scripts/publish_week.py 2026 27 --push     # also push to the GitHub repo

Reads data/scores/<year>-W<week>.jsonl, keeps papers scoring >= publish_min,
writes weekly_picks/<year>-W<week>/, and (with --push) commits the markdown to
the GitHub repo named in GITHUB_REPO using GITHUB_TOKEN.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

from paper_scoring.config import load_settings, get_paths
from paper_scoring.dates import current_week
from paper_scoring.publishing.weekly_repo import write_week_folder, github_publish


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    push = "--push" in sys.argv

    if len(args) >= 2:
        year, week = int(args[0]), int(args[1])
    else:
        year, week = current_week()

    settings = load_settings()
    paths = get_paths(settings)
    label = f"{year}-W{week:02d}"
    publish_min = settings["scoring"]["publish_min"]

    scores_file = paths["scores_dir"] / f"{label}.jsonl"
    if not scores_file.exists():
        print(f"No scores for {label}. Run scripts/score_week.py first.")
        return

    rows = [json.loads(line) for line in open(scores_file, encoding="utf-8")]
    picks = [
        {
            "arxiv_id": r["arxiv_id"],
            "title": r["title"],
            "authors": r.get("authors", []),
            "abstract": r.get("abstract", ""),
            "published": r.get("published", ""),
            "pdf_url": r.get("pdf_url", ""),
            "score": r["weighted_total"],
            "verdict": r["verdict"],
            "reason": r.get("reason", ""),
        }
        for r in rows
        if r.get("weighted_total") is not None and r["weighted_total"] >= publish_min
    ]

    print(f"Week {label}: {len(picks)} of {len(rows)} papers score >= {publish_min}.")

    md_path = write_week_folder(picks, paths["picks_dir"], year, week)
    print(f"Wrote -> {md_path}")

    if push:
        token = os.environ.get("GITHUB_TOKEN")
        repo = os.environ.get("GITHUB_REPO")
        if not token or not repo:
            print("--push set but GITHUB_TOKEN / GITHUB_REPO missing in .env. Skipped.")
            return
        result = github_publish(md_path, repo, token, year, week)
        print(f"GitHub: {result}")


if __name__ == "__main__":
    main()
