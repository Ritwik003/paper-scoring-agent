"""Agent C - Research Reviewer.

The only reasoning agent. Per the plan:
  - Never sees raw PDF text.
  - Receives the paper_object from Agent A and the context from Agent B.
  - Scores each of the 7 criteria independently, in sequence.
  - Explains its reasoning, then produces the final verdict.

Design note: earlier we hit `tool_use_failed` from Groq when the paper text
contained raw LaTeX. Fix: we use plain JSON-mode (a text response containing a
single JSON block), not the strict function-call structured-output path. We
parse it with a lenient regex so unicode/latex fragments do not break scoring.
"""
import json
import os
import re
from typing import Optional

from paper_scoring.scoring.state import CRITERIA


# ---------- prompt builders (one per criterion) ----------

_SHARED_HEADER = """You are reviewing an AI/ML research paper.
Use ONLY the structured evidence below. Do not invent facts.
Return a compact JSON object with EXACTLY these keys:
  {"score": <integer 1-10>, "reason": "<2-3 sentence explanation>"}
No prose outside the JSON. No code fences.Start your response with { and end with }.
"""


def _clip(s: Optional[str], n: int) -> str:
    if not s:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + " ..."


def _prompt_significance(po: dict, ctx: dict) -> str:
    return _SHARED_HEADER + f"""
CRITERION: Significance (is the problem important & clearly motivated?)

TITLE: {po.get('title','')}

ABSTRACT: {_clip(po.get('abstract'), 1500)}

INTRODUCTION: {_clip(po.get('introduction'), 2000)}

PUBLIC BENCHMARKS MENTIONED: {ctx.get('dataset',{}).get('public_benchmarks_mentioned', [])}
CITATIONS (openalex): {ctx.get('citations',{}).get('openalex',{}).get('cited_by_count')}

COMMUNITY DISCUSSION SUMMARY: {_clip(ctx.get('discussion', {}).get('summary'), 800)}
"""


def _prompt_novelty(po: dict, ctx: dict) -> str:
    similar = [s["title"] for s in ctx.get("similar_work", [])][:5]
    return _SHARED_HEADER + f"""
CRITERION: Novelty (how different from prior work is the contribution?)

TITLE: {po.get('title','')}

RELATED WORK: {_clip(po.get('related_work'), 2000)}

METHODOLOGY (proposed approach): {_clip(po.get('methodology'), 2000)}

TOP SIMILAR PAPERS (from OpenAlex): {similar}
"""


def _prompt_methodology(po: dict, ctx: dict) -> str:
    facts = po.get("facts", {})
    return _SHARED_HEADER + f"""
CRITERION: Methodology (scientific quality: baselines, seeds, variance, metrics).

METHODOLOGY: {_clip(po.get('methodology'), 2500)}

EXPERIMENTS: {_clip(po.get('experiments'), 2000)}

DETERMINISTIC FACTS from Agent A:
  seeds reported            : {facts.get('n_seeds')}
  variance / std hits       : {facts.get('variance_hits')}
  has p-value / t-test      : {facts.get('has_p_value')}
  has confidence interval   : {facts.get('has_confidence_interval')}
  #baseline mentions        : {facts.get('num_baselines_mentioned')}
  #dataset mentions         : {facts.get('num_datasets_mentioned')}
  #tables                   : {facts.get('num_tables')}
"""


def _prompt_empirical(po: dict, ctx: dict) -> str:
    table_titles = [t["title"] for t in po.get("tables", [])][:6]
    return _SHARED_HEADER + f"""
CRITERION: Empirical evidence (do the results back the abstract's claims?).

ABSTRACT: {_clip(po.get('abstract'), 1200)}

RESULTS: {_clip(po.get('results'), 2000)}

DISCUSSION / LIMITATIONS:
{_clip(po.get('discussion'), 1200)}
{_clip(po.get('limitations'), 1200)}

TABLES DETECTED: {table_titles}
"""


def _prompt_reproducibility(po: dict, ctx: dict) -> str:
    f = po.get("facts", {})
    repo = ctx.get("repository", {})
    return _SHARED_HEADER + f"""
CRITERION: Reproducibility (code, data, seeds, compute, license).
Base your score on the concrete evidence below (mostly deterministic).

  code_url               : {f.get('code_url')}
  repo status (verified) : {repo.get('status')}   stars={repo.get('stars')}   license={repo.get('license')}
  env file mentioned     : {f.get('env_file_mentioned')}
  seeds fixed / stated   : {f.get('n_seeds')}
  GPU / compute stated   : {f.get('gpu_mentioned')}
  training time stated   : {f.get('training_time_mentioned')}
  checkpoint mentioned   : {f.get('checkpoint_mentioned')}
  API model pinned       : {f.get('api_model_pinned')}
  DOI                    : {f.get('doi')}
  license mentioned      : {f.get('license_mentioned')}
"""


def _prompt_presentation(po: dict, ctx: dict) -> str:
    f = po.get("facts", {})
    return _SHARED_HEADER + f"""
CRITERION: Presentation (structure, figures, tables, notation, clarity).

sections detected: {f.get('sections_present')} of 10
#tables: {f.get('num_tables')}    #figures: {len(po.get('figures', []))}
#references: {f.get('num_references')}

FIRST FEW FIGURE CAPTIONS: {[f['caption'][:120] for f in po.get('figures', [])[:5]]}
"""


def _prompt_ethics(po: dict, ctx: dict) -> str:
    f = po.get("facts", {})
    return _SHARED_HEADER + f"""
CRITERION: Ethics & responsible AI (impact, bias, privacy, licensing).

DISCUSSION / LIMITATIONS:
{_clip(po.get('discussion'), 1500)}
{_clip(po.get('limitations'), 1500)}

license mentioned in paper : {f.get('license_mentioned')}
license on repo            : {ctx.get('repository',{}).get('license')}
"""


_PROMPTS = {
    "significance":    _prompt_significance,
    "novelty":         _prompt_novelty,
    "methodology":     _prompt_methodology,
    "empirical":       _prompt_empirical,
    "reproducibility": _prompt_reproducibility,
    "presentation":    _prompt_presentation,
    "ethics":          _prompt_ethics,
}


# ---------- robust JSON extraction ----------

_JSON_RE = re.compile(r"\{[^{}]*\"score\"\s*:\s*\d+[^{}]*\}", re.DOTALL)


def _parse_json_score(raw: str) -> dict:
    """Pull {'score': N, 'reason': '...'} out of an LLM reply, robustly."""
    if not raw:
        return {}
    # Try direct JSON parse first.
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Fall back: find the first JSON-looking blob that contains a "score".
    m = _JSON_RE.search(raw)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


# ---------- deterministic fallback (when no key or LLM errors) ----------

def _fallback(po: dict, ctx: dict) -> dict:
    """Rough scores from facts alone, so the pipeline always produces a result."""
    f = po.get("facts", {})
    repo = ctx.get("repository", {})

    method = 3
    if f.get("n_seeds", 0) >= 3:            method += 3
    if f.get("variance_hits", 0) > 0:       method += 2
    if f.get("has_p_value"):                method += 2
    method = min(method, 10)

    repro_items = [
        bool(f.get("code_url")),
        repo.get("status") == "live",
        bool(f.get("env_file_mentioned")),
        f.get("n_seeds", 0) >= 1,
        bool(f.get("gpu_mentioned")),
        bool(f.get("checkpoint_mentioned")),
        bool(f.get("api_model_pinned")),
        bool(f.get("license_mentioned")),
    ]
    repro = 1 + round(sum(1 for x in repro_items if x) / 8 * 9)

    presentation = min(3 + f.get("sections_present", 0), 10)

    cites = (ctx.get("citations", {}).get("openalex") or {}).get("cited_by_count") or 0
    significance = 6 + (2 if cites >= 20 else 1 if cites >= 5 else 0)
    significance = min(significance, 10)

    novelty = 5
    empirical = 5 + (1 if f.get("variance_hits", 0) > 0 else 0)
    ethics = 5 + (1 if f.get("license_mentioned") else 0)

    scores = {
        "significance":    significance,
        "novelty":         novelty,
        "methodology":     method,
        "empirical":       empirical,
        "reproducibility": repro,
        "presentation":    presentation,
        "ethics":          ethics,
    }
    reason = ("Deterministic fallback based on parsed facts (LLM unavailable). "
              f"seeds={f.get('n_seeds')}, variance={f.get('variance_hits')}, "
              f"code_url={'yes' if f.get('code_url') else 'no'}, "
              f"citations={cites}.")
    return {"scores": scores, "reason": reason}


# ---------- assembled Agent C ----------

def make_agent_c(llm_cfg: dict):
    model_name = llm_cfg["model"]
    temperature = llm_cfg.get("temperature", 0.0)
    ''''''
    def _call_groq(prompt: str) -> str:
        """One Groq call in plain JSON-mode (no function calling)."""
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        resp = llm.invoke(prompt)
        # ChatGroq returns an AIMessage; content is the JSON string.
        return getattr(resp, "content", str(resp))
    '''
    def _call_llm(prompt: str) -> str:
        """One NVIDIA NIM call."""
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        llm = ChatNVIDIA(
            model=model_name,
            temperature=temperature,
            max_tokens=1024,)
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp))'''

    def agent_c(state: dict) -> dict:
        print("\n" + "=" * 30)
        print("Agent C - Research Reviewer")
        print("=" * 30)
        notes = list(state.get("notes", []))
        po = state.get("paper_object", {}) or {}
        ctx = state.get("context", {}) or {}

        key = os.environ.get("GROQ_API_KEY")
        print(f"  GROQ key present: {'YES' if key else 'NO'}")

        scores: dict = {}
        reasons: dict = {}
        overall_reason = ""

        # Try LLM per-criterion. If any single call fails, fill in the fallback for it.
        if key:
            for crit in CRITERIA:
                prompt_fn = _PROMPTS[crit]
                try:
                    reply = _call_groq(prompt_fn(po, ctx))
                    parsed = _parse_json_score(reply)
                    score = int(parsed.get("score", 0))
                    if 1 <= score <= 10:
                        scores[crit] = float(score)
                        reasons[crit] = str(parsed.get("reason", ""))[:400]
                        print(f"  {crit:<15} -> {score}")
                    else:
                        raise ValueError(f"bad score value: {parsed}")
                except Exception as e:
                    print(f"  {crit:<15} !! LLM failed ({type(e).__name__}) - falling back")
            overall_reason = " ".join(reasons.get(c, "") for c in CRITERIA if reasons.get(c))

        # Fill any missing criteria with the deterministic fallback.
        if len(scores) < len(CRITERIA):
            fb = _fallback(po, ctx)
            for crit in CRITERIA:
                scores.setdefault(crit, float(fb["scores"][crit]))
            if not overall_reason:
                overall_reason = fb["reason"]
            notes.append("Agent C: partial/full fallback used.")
        else:
            notes.append("Agent C: LLM scored all 7 criteria.")

        return {
            "scores":   scores,
            "evidence": {"reason": overall_reason[:1200], "per_criterion": reasons},
            "notes":    notes,
        }

    return agent_c
