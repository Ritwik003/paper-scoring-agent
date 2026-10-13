"""Publish the qualified papers for a week.

Two steps, kept separate so you can run the first without the second:
  1. write_week_folder(): write a folder like weekly_picks/2026-W29/ with a
     tidy markdown list + a machine-readable JSONL of the qualifying papers.
  2. github_publish(): (optional) push that markdown file into a GitHub repo
     using the Contents API — no local clone needed, just a token.
"""
import base64
import json
import os
import urllib.request
from pathlib import Path
from datetime import datetime


def _abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def _format_date(published) -> str:
    """Turn arXiv's 'YYYY-MM-DDTHH:MM:SSZ' string into a human-readable date."""
    if not published:
        return ""
    if isinstance(published, str):
        try:
            return datetime.fromisoformat(
                published.replace("Z", "+00:00")
            ).strftime("%d %B %Y")
        except Exception:
            return published
    if hasattr(published, "strftime"):
        return published.strftime("%d %B %Y")
    return str(published)


def write_week_folder(picks: list[dict], picks_root: Path, year: int, week: int) -> Path:
    """Write weekly_picks/<year>-W<week>/ with markdown + jsonl. Returns the .md path."""
    label = f"{year}-W{week:02d}"
    folder = picks_root / label
    folder.mkdir(parents=True, exist_ok=True)

    # Machine-readable copy.
    with open(folder / "qualified.jsonl", "w", encoding="utf-8") as f:
        for p in picks:
            f.write(json.dumps(p) + "\n")

    # Sort highest score first (descending).  Ties preserved by title.
    picks_sorted = sorted(picks, key=lambda x: (-float(x.get("score", 0)), x.get("title", "")))

    # Human-readable markdown — this is what gets pushed to the repo.
    md_path = folder / "qualified.md"
    lines = [
        f"# AI/ML Research papers — {label}",
        "",
        f"Total Papers: **{len(picks_sorted)}**  (sorted by score, highest first)",
        "",
    ]

    for p in picks_sorted:
        score = p.get("score")
        score_txt = f"{float(score):.2f}/10" if score is not None else "—"

        lines.append("---")
        lines.append("")
        lines.append(f"## {p.get('title', 'Untitled')}")
        lines.append("")
        #lines.append(f"**Score:** {score_txt}")
        #lines.append("")

        authors = p.get("authors") or []
        if authors:
            lines.append(f"**Authors:** {', '.join(authors)}")
            lines.append("")

        published = _format_date(p.get("published"))
        if published:
            lines.append(f"**Published:** {published}")
            lines.append("")

        lines.append("### Description")
        lines.append("")
        lines.append(p.get("abstract") or "No description available.")
        lines.append("")

        lines.append(f"**Paper:** {_abs_url(p['arxiv_id'])}")
        pdf = p.get("pdf_url")
        if pdf:
            lines.append(f"**PDF:** {pdf}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def github_publish(md_path: Path, repo: str, token: str, year: int, week: int,
                   api_base: str = "https://api.github.com") -> dict:
    """Upsert the week's markdown into <repo> at weekly_picks/<year>-W<week>.md.

    Uses the GitHub Contents API (PUT). Returns a small status dict.
    """
    label = f"{year}-W{week:02d}"
    path_in_repo = f"weekly_picks/{label}.md"
    content_b64 = base64.b64encode(md_path.read_bytes()).decode("ascii")
    url = f"{api_base}/repos/{repo}/contents/{path_in_repo}"
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "paper-scoring-agent",
        "Accept": "application/vnd.github+json",
    }

    # If the file already exists we must pass its sha to update it.
    sha = None
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            sha = json.load(resp).get("sha")
    except Exception:
        pass  # file doesn't exist yet — that's fine

    body = {"message": f"Add qualified papers for {label}", "content": content_b64}
    if sha:
        body["sha"] = sha

    try:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="PUT"
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
        return {"status": "pushed", "path": path_in_repo,
                "commit": data.get("commit", {}).get("sha")}
    except Exception as e:
        return {"status": "error", "error": str(e), "path": path_in_repo}