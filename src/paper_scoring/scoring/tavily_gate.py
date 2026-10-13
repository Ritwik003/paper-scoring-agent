"""Tavily call-reduction gate for Agent B.

Every other web call Agent B makes (GitHub, OpenAlex, Semantic Scholar,
CrossRef) is free. Tavily is not, so before Agent B spends a Tavily call on
a paper's "community discussion" signal, it runs this gate.

The gate is a strict funnel, cheapest/most-decisive checks first:

  Level 1 - hard skips     : paper clearly isn't worth analyzing at all
  Level 2 - "worth it" gate: nothing suggests Tavily would change the score
  Level 3 - monthly budget : a hard cap tracked in data/tavily_usage.json
  Level 4 - rate limiting  : per-week burst cap + auto-disable on errors

Only Agent A / Agent B data already computed earlier in the same run is
used - no new network calls are made just to decide whether to call Tavily.

Usage counters live in small JSON files instead of a database, since this is
a single-box batch job. A short-lived lockfile keeps concurrent Celery
workers from racing on the same counter.
"""
import contextlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from paper_scoring.config import ROOT
from paper_scoring.dates import current_week


# ---------------------------------------------------------------- locking --

@contextlib.contextmanager
def _locked(path: Path, timeout: float = 5.0):
    """A tiny cross-platform advisory lock using exclusive file creation.

    Good enough for a handful of Celery workers incrementing a JSON counter;
    not meant to survive heavy contention.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() > deadline:
                # Don't let a stuck lock (e.g. crashed worker) block the
                # whole pipeline forever - steal it.
                with contextlib.suppress(FileNotFoundError):
                    os.remove(lock_path)
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            os.remove(lock_path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------- usage counters --

def _month_key(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now.year}-{now.month:02d}"


def _week_key(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    year, week = current_week(now.date())
    return f"{year}-W{week:02d}"


def get_monthly_usage(usage_file: str) -> int:
    """Read-only peek at this month's Tavily call count."""
    path = ROOT / usage_file
    data = _read_json(path)
    if data.get("month") != _month_key():
        return 0
    return int(data.get("used", 0))


def _increment_monthly_usage(usage_file: str) -> int:
    path = ROOT / usage_file
    with _locked(path):
        data = _read_json(path)
        month = _month_key()
        if data.get("month") != month:
            data = {"month": month, "used": 0}
        data["used"] = int(data.get("used", 0)) + 1
        _write_json(path, data)
        return data["used"]


def get_weekly_usage(weekly_usage_file: str) -> int:
    path = ROOT / weekly_usage_file
    data = _read_json(path)
    return int(data.get(_week_key(), 0))


def _increment_weekly_usage(weekly_usage_file: str) -> int:
    path = ROOT / weekly_usage_file
    with _locked(path):
        data = _read_json(path)
        wk = _week_key()
        data[wk] = int(data.get(wk, 0)) + 1
        _write_json(path, data)
        return data[wk]


def record_tavily_call(cfg: dict) -> None:
    """Call this once, immediately after an actual Tavily API call is made
    (success or failure) - it's what makes the budget real."""
    _increment_monthly_usage(cfg.get("usage_file", "data/tavily_usage.json"))
    _increment_weekly_usage(cfg.get("weekly_usage_file", "data/tavily_weekly_usage.json"))


# --------------------------------------------- per-run error auto-disable --
# Process-local (per worker process). Resets whenever the worker restarts,
# which is the natural boundary for "the rest of that run" in Celery mode.

_consecutive_errors = 0
_disabled_for_run = False


def record_tavily_result(ok: bool, max_consecutive_errors: int = 2) -> None:
    global _consecutive_errors, _disabled_for_run
    if ok:
        _consecutive_errors = 0
        return
    _consecutive_errors += 1
    if _consecutive_errors >= max_consecutive_errors:
        _disabled_for_run = True


def reset_run_state() -> None:
    """Call at the start of a run/worker if you want a clean slate."""
    global _consecutive_errors, _disabled_for_run
    _consecutive_errors = 0
    _disabled_for_run = False


# --------------------------------------------------------------- helpers --

def _parse_published(published: str):
    if not published:
        return None
    try:
        return datetime.fromisoformat(published.replace("Z", "+00:00"))
    except Exception:
        return None


def _days_since(published: str) -> "int | None":
    dt = _parse_published(published)
    if not dt:
        return None
    now = datetime.now(timezone.utc)
    return (now - dt).days


def _is_math_stat_only(categories: list, prefixes: list) -> bool:
    if not categories:
        return False
    prefixes = tuple(f"{p}." for p in prefixes)
    return all(c.startswith(prefixes) for c in categories)


def _title_matches_keywords(title: str, keywords: list) -> bool:
    t = (title or "").lower()
    return any(kw.lower() in t for kw in keywords)


# ----------------------------------------------------------------- gate --

def decide(state: dict, paper_object: dict, context_so_far: dict, cfg: dict,
           force_important: bool = False) -> dict:
    """Decide whether Agent B should spend a Tavily call on this paper.

    `context_so_far` must already contain "repository", "citations" and
    "similar_work" (Agent B computes these before deciding on Tavily).

    Returns {"allow": bool, "reason": str, "level": str}.
    """
    if not cfg.get("enabled", True):
        return {"allow": False, "reason": "tavily_gate_disabled", "level": "config"}

    facts = (paper_object or {}).get("facts", {})
    figures = (paper_object or {}).get("figures", [])
    char_count = (paper_object or {}).get("char_count", 0)
    title = state.get("title", "")
    categories = state.get("categories", []) or []
    published = state.get("published", "")

    # ---------- Level 1: hard skips ----------
    if state.get("gate_failed"):
        return {"allow": False, "reason": "agent_a_gate_failed", "level": "1_hard"}

    if facts.get("sections_present", 0) < cfg.get("min_sections_present", 5):
        return {"allow": False, "reason": "too_few_sections", "level": "1_hard"}

    n_tables = facts.get("num_tables", 0)
    n_figures = len(figures)
    min_tf = cfg.get("min_tables_or_figures", 2)
    if n_tables < min_tf and n_figures < min_tf:
        return {"allow": False, "reason": "too_few_tables_and_figures", "level": "1_hard"}

    if facts.get("num_references", 0) < cfg.get("min_references", 10):
        return {"allow": False, "reason": "too_few_references", "level": "1_hard"}

    if char_count < cfg.get("min_char_count", 3000):
        return {"allow": False, "reason": "paper_too_short", "level": "1_hard"}

    # ---------- Level 1: cheap-signal skips ----------
    openalex = (context_so_far.get("citations") or {}).get("openalex", {})
    days_old = _days_since(published)
    if (openalex.get("status") == "not_found" and days_old is not None
            and days_old > cfg.get("obscure_not_found_days", 30)):
        return {"allow": False, "reason": "obscure_old_paper", "level": "1_cheap"}

    similar_work = context_so_far.get("similar_work") or []
    if len(similar_work) < cfg.get("min_similar_work", 3):
        return {"allow": False, "reason": "niche_topic_no_similar_work", "level": "1_cheap"}

    # ---------- Level 1: domain skips ----------
    if _is_math_stat_only(categories, cfg.get("math_stat_only_categories", ["math", "stat"])):
        return {"allow": False, "reason": "theoretical_math_stat_only", "level": "1_domain"}

    # ---------- Level 2: only call if it will plausibly change the score ----------
    cited_by = openalex.get("cited_by_count") or 0
    semantic = (context_so_far.get("citations") or {}).get("semantic_scholar", {})
    influential = semantic.get("influential_citations") or 0
    repo = context_so_far.get("repository") or {}
    repo_live_and_popular = (repo.get("status") == "live"
                             and (repo.get("stars") or 0) >= cfg.get("min_github_stars", 10))
    is_hot_recent = (any(c in cfg.get("hot_categories", []) for c in categories)
                     and days_old is not None
                     and days_old <= cfg.get("hot_recency_days", 14))
    keyword_hit = _title_matches_keywords(title, cfg.get("high_value_keywords", []))

    level2_pass = (
        cited_by >= cfg.get("min_cited_by_count", 3)
        or influential >= cfg.get("min_influential_citations", 1)
        or repo_live_and_popular
        or is_hot_recent
        or keyword_hit
    )
    if not level2_pass:
        return {"allow": False, "reason": "no_signal_tavily_unlikely_to_help", "level": "2"}

    # ---------- Level 3: hard monthly budget cap ----------
    used = get_monthly_usage(cfg.get("usage_file", "data/tavily_usage.json"))
    monthly_budget = cfg.get("monthly_budget", 1000)
    soft_cap = cfg.get("soft_cap", 700)
    hard_cap = cfg.get("hard_cap", 900)
    reserved = cfg.get("reserved_for_important", 100)

    if used >= monthly_budget:
        return {"allow": False, "reason": "monthly_budget_exhausted", "level": "3"}

    if used >= monthly_budget - reserved:
        # We're in the reserved tail - only known-important papers get through.
        if not force_important:
            return {"allow": False, "reason": "reserved_for_important_papers", "level": "3"}

    elif used >= hard_cap:
        # Between hard_cap and the reserved tail: same as above in effect,
        # since hard_cap <= monthly_budget - reserved by construction, but
        # keep this branch explicit for clarity / future re-tuning.
        if not force_important:
            return {"allow": False, "reason": "over_hard_cap", "level": "3"}

    elif used >= soft_cap:
        # 700-900 band: only "top decile" signal papers get through.
        td = cfg.get("top_decile", {})
        is_top_decile = (
            facts.get("num_references", 0) >= td.get("min_references", 30)
            and n_tables >= td.get("min_tables", 5)
            and cited_by >= td.get("min_cited_by_count", 10)
        )
        if not is_top_decile:
            return {"allow": False, "reason": "over_soft_cap_not_top_decile", "level": "3"}

    # ---------- Level 4: per-run rate limit ----------
    if _disabled_for_run:
        return {"allow": False, "reason": "disabled_after_repeated_errors", "level": "4"}

    weekly_used = get_weekly_usage(cfg.get("weekly_usage_file", "data/tavily_weekly_usage.json"))
    if weekly_used >= cfg.get("weekly_run_cap", 50):
        return {"allow": False, "reason": "weekly_run_cap_reached", "level": "4"}

    return {"allow": True, "reason": "ok", "level": "allow"}
