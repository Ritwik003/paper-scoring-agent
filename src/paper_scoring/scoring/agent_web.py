"""Agent B - Research Context Agent.

Never reads the PDF. Instead answers: how does this paper fit the research
ecosystem? Returns a `context` dict:

  {
    "repository":   {...}   # verified repo details
    "citations":    {...}   # OpenAlex + Semantic Scholar + CrossRef
    "similar_work": [...]   # top similar / related papers
    "discussion":   {...}   # web discussion signal (Tavily)
    "dataset":      {...}   # basic dataset visibility signal
  }

Every call is wrapped and short-timed. Web facts are a BONUS, not a blocker.
"""
import json
import os
import urllib.parse
import urllib.request
from paper_scoring.scoring import tavily_gate



def _get_json(url: str, headers: dict | None = None, timeout: int = 12):
    """GET JSON or return None on any problem (network, parse, rate-limit)."""
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except Exception:
        return None


# ---------- 1. Repository verification ----------

def _verify_repo(code_url: str | None, title: str, arxiv_id: str,
                 api_github: str) -> dict:
    """Verify or discover the code repo.

    If Agent A found a github.com URL we verify that one. Otherwise we do a
    GitHub code search on the paper title / arXiv id to try to find it.
    """
    headers = {"User-Agent": "paper-scoring-agent"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def _fetch_meta(owner_repo: str) -> dict:
        meta = _get_json(f"{api_github}/repos/{owner_repo}", headers=headers)
        if not meta or "full_name" not in meta:
            return {"status": "not_found_or_rate_limited"}
        return {
            "status":       "live",
            "url":          meta.get("html_url"),
            "stars":        meta.get("stargazers_count"),
            "forks":        meta.get("forks_count"),
            "open_issues":  meta.get("open_issues_count"),
            "license":      (meta.get("license") or {}).get("spdx_id"),
            "last_push":    meta.get("pushed_at"),
            "archived":     meta.get("archived", False),
            "has_readme":   meta.get("size", 0) > 0,
        }

    # (a) explicit link Agent A found in the PDF
    if code_url and "github.com/" in code_url:
        try:
            parts = code_url.split("github.com/")[1].strip("/").split("/")
            owner_repo = f"{parts[0]}/{parts[1].replace('.git', '')}"
            return _fetch_meta(owner_repo)
        except Exception:
            pass

    # (b) fall back to a GitHub code search by title / arxiv id
    query = f"{title} {arxiv_id}".strip()
    if query:
        q = urllib.parse.quote(query[:80])
        data = _get_json(f"{api_github}/search/repositories?q={q}&per_page=1",
                         headers=headers)
        items = (data or {}).get("items") or []
        if items:
            return _fetch_meta(items[0]["full_name"])

    return {"status": "no_link"}


# ---------- 2. Citations (three sources, best-effort) ----------

def _openalex_citations(title: str, api_openalex: str) -> dict:
    if not title:
        return {"status": "no_title"}
    q = urllib.parse.quote(title[:200])
    data = _get_json(f"{api_openalex}/works?search={q}&per_page=1")
    results = (data or {}).get("results") or []
    if not results:
        return {"status": "not_found"}
    top = results[0]
    return {
        "status":               "ok",
        "cited_by_count":       top.get("cited_by_count", 0),
        "venue":                (top.get("primary_location") or {}).get("source", {}).get("display_name"),
        "doi":                  top.get("doi"),
        "openalex_id":          top.get("id"),
    }


def _semantic_scholar(title: str) -> dict:
    if not title:
        return {"status": "no_title"}
    q = urllib.parse.quote(title[:200])
    url = (f"https://api.semanticscholar.org/graph/v1/paper/search?"
           f"query={q}&limit=1&fields=citationCount,influentialCitationCount,venue,year")
    data = _get_json(url, headers={"User-Agent": "paper-scoring-agent"})
    results = (data or {}).get("data") or []
    if not results:
        return {"status": "not_found"}
    top = results[0]
    return {
        "status":                "ok",
        "citation_count":        top.get("citationCount"),
        "influential_citations": top.get("influentialCitationCount"),
        "venue":                 top.get("venue"),
        "year":                  top.get("year"),
    }


def _crossref(title: str) -> dict:
    if not title:
        return {"status": "no_title"}
    q = urllib.parse.quote(title[:200])
    data = _get_json(f"https://api.crossref.org/works?query.title={q}&rows=1")
    items = ((data or {}).get("message") or {}).get("items") or []
    if not items:
        return {"status": "not_found"}
    top = items[0]
    return {
        "status":                   "ok",
        "doi":                      top.get("DOI"),
        "is_referenced_by_count":   top.get("is-referenced-by-count"),
        "container":                (top.get("container-title") or [None])[0],
    }


# ---------- 3. Similar work ----------

def _similar_work(title: str, api_openalex: str, k: int = 5) -> list:
    """Top-k related works from OpenAlex - a lightweight novelty context."""
    if not title:
        return []
    q = urllib.parse.quote(title[:200])
    data = _get_json(f"{api_openalex}/works?search={q}&per_page={k}")
    out = []
    for w in (data or {}).get("results", [])[:k]:
        out.append({
            "title":          w.get("title", "")[:200],
            "year":           w.get("publication_year"),
            "cited_by_count": w.get("cited_by_count", 0),
            "id":             w.get("id"),
        })
    return out


# ---------- 4. Community discussion ----------

def _discussion(title: str, max_results: int) -> dict:
    key = os.environ.get("TAVILY_API_KEY")
    if not key or not title:
        return {"status": "skipped"}
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)
        res = client.search(query=title[:200], max_results=max_results,include_answer=True,search_depth="advanced")
        hits = res.get("results", []) or []
        return {"status":"ok","n_results":len(hits),"summary":res.get("answer", ""),   # Tavily's AI-generated overall summary
                "top_urls":     [h.get("url") for h in hits[:3]],"top_snippets": [{"url":     h.get("url"),"title":   h.get("title", ""),"content": (h.get("content") or "")[:500],  # first 500 chars per page
                                                                                   }
        for h in hits[:3]],}
        '''return {"status": "ok", "n_results": len(hits),
                "top_urls": [h.get("url") for h in hits[:3]]}'''
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------- 5. Dataset visibility ----------

_KNOWN_DATASETS = {
    "imagenet", "coco", "cifar", "mnist", "glue", "squad", "wikitext",
    "librispeech", "audioset", "kinetics", "openwebtext", "the pile", "c4",
    "commoncrawl", "arxiv", "pubmed", "mimic", "kitti", "nuscenes",
}


def _dataset_signal(paper_object: dict) -> dict:
    text = " ".join(str(paper_object.get(k, "")) for k in
                    ("abstract", "methodology", "experiments", "results")).lower()
    hits = sorted({d for d in _KNOWN_DATASETS if d in text})
    return {"public_benchmarks_mentioned": hits,
            "any_public_benchmark": bool(hits)}


# ---------- assembled Agent B ----------
'''
def make_agent_b(web_cfg: dict):
    api_github    = web_cfg["github_api"]
    api_openalex  = web_cfg["openalex_api"]
    tavily_max    = web_cfg.get("tavily_max_results", 3)

    def agent_b(state: dict) -> dict:
        print("\n" + "=" * 30)
        print("Agent B - Research Context")
        print("=" * 30)

        notes = list(state.get("notes", []))
        paper_object = state.get("paper_object", {})
        facts = paper_object.get("facts", {})
        title = state.get("title", "")
        arxiv_id = state.get("arxiv_id", "")

        context = {}
        print("  * verifying repository ...")
        context["repository"] = _verify_repo(facts.get("code_url"), title, arxiv_id, api_github)
        print(f"    -> {context['repository'].get('status')}")

        print("  * citations (OpenAlex / Semantic Scholar / CrossRef) ...")
        context["citations"] = {
            "openalex":         _openalex_citations(title, api_openalex),
            "semantic_scholar": _semantic_scholar(title),
            "crossref":         _crossref(title),
        }
        cites = context["citations"]["openalex"].get("cited_by_count")
        print(f"    -> openalex cited_by = {cites}")

        print("  * similar work ...")
        context["similar_work"] = _similar_work(title, api_openalex)
        print(f"    -> {len(context['similar_work'])} similar papers")

        print("  * community discussion ...")
        context["discussion"] = _discussion(title, tavily_max)
        print(f"    -> {context['discussion'].get('status')}")

        print("  * dataset visibility ...")
        context["dataset"] = _dataset_signal(paper_object)
        print(f"    -> {context['dataset']['public_benchmarks_mentioned']}")

        notes.append(
            f"Agent B: repo={context['repository'].get('status')}, "
            f"cites={cites}, similar={len(context['similar_work'])}."
        )
        return {"context": context, "notes": notes}

    return agent_b
'''


def make_agent_b(web_cfg: dict, gate_cfg: dict | None = None):
    api_github    = web_cfg["github_api"]
    api_openalex  = web_cfg["openalex_api"]
    tavily_max    = web_cfg.get("tavily_max_results", 3)
    gate_cfg      = gate_cfg or {}

    def agent_b(state: dict) -> dict:
        print("\n" + "=" * 30)
        print("Agent B - Research Context")
        print("=" * 30)

        notes = list(state.get("notes", []))
        paper_object = state.get("paper_object", {})
        facts = paper_object.get("facts", {})
        title = state.get("title", "")
        arxiv_id = state.get("arxiv_id", "")

        context = {}
        print("  * verifying repository ...")
        context["repository"] = _verify_repo(facts.get("code_url"), title, arxiv_id, api_github)
        print(f"    -> {context['repository'].get('status')}")

        print("  * citations (OpenAlex / Semantic Scholar / CrossRef) ...")
        context["citations"] = {
            "openalex":         _openalex_citations(title, api_openalex),
            "semantic_scholar": _semantic_scholar(title),
            "crossref":         _crossref(title),
        }
        cites = context["citations"]["openalex"].get("cited_by_count")
        print(f"    -> openalex cited_by = {cites}")

        print("  * similar work ...")
        context["similar_work"] = _similar_work(title, api_openalex)
        print(f"    -> {len(context['similar_work'])} similar papers")

        print("  * community discussion ...")
        gate = tavily_gate.decide(state, paper_object, context, gate_cfg,
                                  force_important=bool(state.get("force_important")))
        if gate["allow"]:
            context["discussion"] = _discussion(title, tavily_max)
            tavily_gate.record_tavily_call(gate_cfg)
            tavily_gate.record_tavily_result(
                context["discussion"].get("status") in ("ok", "skipped"),
                max_consecutive_errors=gate_cfg.get("max_consecutive_errors", 2),
            )
        else:
            context["discussion"] = {"status": "skipped", "gate_reason": gate["reason"],
                                      "gate_level": gate["level"]}
        print(f"    -> {context['discussion'].get('status')}"
              f"{' (' + gate['reason'] + ')' if not gate['allow'] else ''}")

        print("  * dataset visibility ...")
        context["dataset"] = _dataset_signal(paper_object)
        print(f"    -> {context['dataset']['public_benchmarks_mentioned']}")

        notes.append(
            f"Agent B: repo={context['repository'].get('status')}, "
            f"cites={cites}, similar={len(context['similar_work'])}, "
            f"tavily={'called' if gate['allow'] else 'skipped:' + gate['reason']}."
        )
        return {"context": context, "notes": notes}

    return agent_b
