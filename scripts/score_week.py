"""Score one week of AI/ML papers.

Two modes:
  * DEFAULT: fan out one Celery task per paper (parallel via Redis).
      Requires:  a running redis-server AND a running Celery worker.
  * --sequential: run the pipeline in-process, paper by paper (no Redis).

Usage:
    python scripts/score_week.py                 # current week, parallel
    python scripts/score_week.py 2026 27         # specific week, parallel
    python scripts/score_week.py 2026 27 --limit 3
    python scripts/score_week.py 2026 27 --sequential   # no Redis/Celery
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

from paper_scoring.config import load_settings, get_paths
from paper_scoring.dates import current_week
from paper_scoring.ingestion.downloader import download_pdf_bytes


def _parse_args():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sequential = "--sequential" in sys.argv
    limit = None
    for a in sys.argv:
        if a.startswith("--limit"):
            if "=" in a:
                limit = int(a.split("=")[1])
            else:
                limit = int(sys.argv[sys.argv.index(a) + 1])
    if len(args) >= 2:
        year, week = int(args[0]), int(args[1])
    else:
        year, week = current_week()
    return year, week, limit, sequential


def _row_from(paper: dict, final: dict) -> dict:
    """Build the score row we write to JSONL.

    Metadata fields (authors, abstract, published, categories) are carried
    over from the original paper dict, so the publisher can render them into
    the weekly markdown without needing to re-open the metadata file.
    """
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
        # original metadata (needed by publish_week.py)
        "authors":        paper.get("authors", []),
        "abstract":       paper.get("abstract", ""),
        "published":      paper.get("published", ""),
        "categories":     paper.get("categories", []),
    }


def _run_sequential(papers, settings, out_file):
    """In-process fallback: no Redis, no Celery. Runs paper by paper."""
    from paper_scoring.scoring.graph import build_graph, score_paper
    graph = build_graph(settings)
    with open(out_file, "w", encoding="utf-8") as out:
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] {paper['arxiv_id']}  {paper['title'][:70]}")
            try:
                paper["pdf_bytes"] = download_pdf_bytes(paper["pdf_url"])
            except Exception as e:
                print(f"  ! download failed: {e}")
                continue
            try:
                final = score_paper(graph, paper)
            finally:
                paper.pop("pdf_bytes", None)
            row = _row_from(paper, final)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"    -> {row['verdict']} {row['weighted_total']}")


def _run_parallel(papers, out_file):
    """Fan out one Celery task per paper. Requires Redis + a running worker."""
    from paper_scoring.workers.celery_app import score_paper_task
    print(f"Submitting {len(papers)} tasks to Celery ...")
    async_results = [score_paper_task.delay(p) for p in papers]

    remaining = list(enumerate(async_results, 1))
    done = 0
    with open(out_file, "w", encoding="utf-8") as out:
        while remaining:
            still = []
            for idx, ar in remaining:
                if ar.ready():
                    done += 1
                    try:
                        row = ar.get(timeout=1)
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        vt = row.get("weighted_total")
                        vd = row.get("verdict")
                        print(f"  [{done}/{len(async_results)}] "
                              f"{vd or 'err':<5} {vt if vt is not None else '-':<5} "
                              f"{row.get('title','')[:60]}")
                    except Exception as e:
                        print(f"  [{done}/{len(async_results)}] task error: {e}")
                else:
                    still.append((idx, ar))
            remaining = still
            if remaining:
                time.sleep(1.0)


def main():
    year, week, limit, sequential = _parse_args()
    settings = load_settings()
    paths = get_paths(settings)
    label = f"{year}-W{week:02d}"

    meta_file = paths["metadata_dir"] / f"{label}.jsonl"
    if not meta_file.exists():
        print(f"No metadata for {label}. Run scripts/fetch_week.py first.")
        return

    papers = [json.loads(line) for line in open(meta_file, encoding="utf-8")]
    if limit:
        papers = papers[:limit]

    paths["scores_dir"].mkdir(parents=True, exist_ok=True)
    out_file = paths["scores_dir"] / f"{label}.jsonl"
    print(f"Week {label}: {len(papers)} papers   mode={'sequential' if sequential else 'celery'}")
    print("-" * 60)

    if sequential:
        _run_sequential(papers, settings, out_file)
    else:
        try:
            _run_parallel(papers, out_file)
        except Exception as e:
            print(f"\n! Celery/Redis unavailable ({e}). Falling back to --sequential.\n")
            _run_sequential(papers, settings, out_file)

    print("-" * 60)
    print(f"Saved scores -> {out_file}")


if __name__ == "__main__":
    main()