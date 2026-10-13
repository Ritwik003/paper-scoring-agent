# Paper Scoring Agent — final architecture

Sequential 3-agent LangGraph pipeline, fanned out in parallel via **Redis + Celery**,
scheduled by a **weekly cron**, publishing qualified papers to a **GitHub repo**.

```
        Cron (Sunday)
              |
              v
        run_weekly.py
              |
      fetch metadata (arXiv)
              |
              v
      Create Celery tasks (1 per paper)
              |
              v
        Redis queue
              |
      Celery workers (parallel)
      /       |        \
  Agent A   Agent A   Agent A     paper_object (sections, tables, figures,
     |        |         |         references, deterministic facts)
     v        v         v
  Agent B   Agent B   Agent B     context (repo, citations x3 sources,
     |        |         |         similar work, discussion, dataset)
     v        v         v
  Agent C   Agent C   Agent C     7-criteria scores + verdict (Groq LLM,
     |        |         |         JSON mode, per-criterion prompts)
     \_______|________/
              v
    Save individual scores
              v
    Publish qualified (>=8) papers
              v
       GitHub repository
```

## The three agents

| Agent | Role | LLM? |
|---|---|---|
| **A** Paper Understanding | Parse PDF -> `paper_object` (structured sections, tables, figures, references, facts) | No |
| **B** Research Context | Verify repo, gather citations (OpenAlex / Semantic Scholar / CrossRef), similar work, discussion (Tavily), dataset visibility | No |
| **C** Research Reviewer | Reads `paper_object` + `context` only (never raw PDF). Scores each of the 7 criteria independently via Groq (JSON mode). Falls back deterministically if any call fails. | Yes |

## Setup (once)

```bash
pip install -r requirements.txt
cp .env.example .env    # then paste GROQ_API_KEY, TAVILY_API_KEY, GITHUB_TOKEN, GITHUB_REPO
```

## Running

**Sequential (no Redis, quick smoke test):**
```bash
python scripts/fetch_week.py 2026 27
python scripts/score_week.py 2026 27 --sequential --limit 5
python scripts/publish_week.py 2026 27 --push
```

**Parallel (Redis + Celery, production):**
```bash
# Terminal 1
redis-server

# Terminal 2
celery -A paper_scoring.workers.celery_app worker --loglevel=info --concurrency=4

# Terminal 3
python scripts/run_weekly.py
```

`run_weekly.py` uses `last_completed_week()` — on Sunday of W28, it processes W27.

## Weekly cron

```cron
0 6 * * 0  cd /full/path/paper-scoring-agent && .venv/bin/python scripts/run_weekly.py >> logs/weekly.log 2>&1
```

Runs every Sunday 06:00. Uses the venv's Python so `.env`, deps, and keys all resolve.

## Config (`config/settings.yaml`)

* `scoring.weights` — the 7 criterion weights (sum to 1.0)
* `scoring.verdict` — `read_min: 8.0`, `skim_min: 5.0`
* `scoring.publish_min` — only papers scoring >= this get pushed to GitHub

Tune without touching code.
