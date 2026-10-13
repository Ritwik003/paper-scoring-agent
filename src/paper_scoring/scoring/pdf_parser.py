"""Deep PDF parsing for Agent A.

Turns raw PDF bytes into a structured 'paper_object':
  {title, abstract, introduction, related_work, methodology, experiments,
   results, discussion, limitations, conclusion, references,
   tables[], figures[], facts{}}

Uses PyMuPDF only (no LLM, no network).
"""
import io
import re
from typing import Optional

import fitz  # PyMuPDF


# ---------- section splitter ----------

# Map of paper_object key -> list of patterns to look for at the start of a line.
_SECTION_HEADERS = {
    "abstract":     [r"^\s*abstract\b"],
    "introduction": [r"^\s*\d+\.?\s*introduction\b", r"^\s*introduction\b"],
    "related_work": [r"^\s*\d+\.?\s*related\s+work\b", r"^\s*related\s+work\b",
                     r"^\s*\d+\.?\s*background\b"],
    "methodology":  [r"^\s*\d+\.?\s*method(s|ology)?\b", r"^\s*method(s|ology)?\b",
                     r"^\s*\d+\.?\s*approach\b", r"^\s*\d+\.?\s*proposed\s+method\b"],
    "experiments":  [r"^\s*\d+\.?\s*experiment(s|al\s+setup)?\b",
                     r"^\s*experiments\b", r"^\s*\d+\.?\s*setup\b"],
    "results":      [r"^\s*\d+\.?\s*results?\b", r"^\s*results?\b",
                     r"^\s*\d+\.?\s*evaluation\b"],
    "discussion":   [r"^\s*\d+\.?\s*discussion\b", r"^\s*discussion\b"],
    "limitations":  [r"^\s*\d+\.?\s*limitations?\b", r"^\s*limitations?\b"],
    "conclusion":   [r"^\s*\d+\.?\s*conclusions?\b", r"^\s*conclusions?\b"],
    "references":   [r"^\s*references\b", r"^\s*bibliography\b"],
}


def _match_section(line: str) -> Optional[str]:
    """Return the section key if this line is a heading, else None."""
    for key, patterns in _SECTION_HEADERS.items():
        for pat in patterns:
            if re.match(pat, line, flags=re.IGNORECASE):
                return key
    return None


def split_sections(text: str) -> dict:
    """Walk the text line by line, cutting it into named sections.

    Any line that matches a known heading starts a new section; text between
    two headings is assigned to the earlier one. Anything before the first
    heading is dropped (usually just author list / affiliations).
    """
    lines = text.split("\n")
    sections = {k: "" for k in _SECTION_HEADERS}
    current = None
    buf = []
    for line in lines:
        match = _match_section(line.strip())
        if match:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = match
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


# ---------- table + figure extraction ----------

_TABLE_HDR = re.compile(r"^\s*Table\s+(\d+)[.:]?\s*(.*)$", re.IGNORECASE)
_FIGURE_HDR = re.compile(r"^\s*Figure\s+(\d+)[.:]?\s*(.*)$", re.IGNORECASE)


def extract_tables(text: str) -> list:
    """Find every 'Table N: ...' caption and take the following ~8 lines as content."""
    tables = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = _TABLE_HDR.match(line)
        if m:
            body = "\n".join(lines[i:i + 10]).strip()
            tables.append({"title": f"Table {m.group(1)}", "content": body[:1200]})
    return tables


def extract_figures(text: str) -> list:
    """Find every 'Figure N: ...' caption line."""
    figures = []
    for line in text.split("\n"):
        m = _FIGURE_HDR.match(line)
        if m:
            caption = line.strip()
            figures.append({"caption": caption[:400]})
    return figures


# ---------- references ----------

# Very rough reference-line pattern: starts with [N] or "1." and mentions a year.
_REF_LINE = re.compile(r"^\s*(?:\[\d+\]|\d+\.)\s+.+?(19|20)\d{2}")
_YEAR = re.compile(r"(19|20)\d{2}")


def extract_references(references_text: str, limit: int = 60) -> list:
    """Pull short structured records out of the references section text."""
    refs = []
    if not references_text:
        return refs
    for line in references_text.split("\n"):
        line = line.strip()
        if not line or len(line) < 20:
            continue
        if _REF_LINE.match(line) or _YEAR.search(line):
            year_match = _YEAR.search(line)
            year = int(year_match.group(0)) if year_match else None
            # Authors = the chunk before the first year, roughly.
            authors_part = line.split(str(year))[0] if year else line
            authors = [a.strip() for a in re.split(r",| and ", authors_part) if a.strip()][:6]
            refs.append({"title": line[:300], "year": year, "authors": authors})
        if len(refs) >= limit:
            break
    return refs


# ---------- deterministic facts ----------

_GITHUB   = re.compile(r"https?://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+")
_DOI      = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_HF       = re.compile(r"huggingface\.co/[A-Za-z0-9_.\-/]+", re.IGNORECASE)
_N_SEEDS  = re.compile(r"\b(\d+)\s+(?:random\s+)?(?:seeds|runs|trials)\b", re.IGNORECASE)
_SEED     = re.compile(r"\bseed(s)?\b", re.IGNORECASE)
_VARIANCE = re.compile(r"±|\bstd\b|standard deviation|confidence interval|95% ci", re.IGNORECASE)
_PVAL     = re.compile(r"p\s*[<=]\s*0?\.\d+|p-value|t-test|wilcoxon|bootstrap", re.IGNORECASE)
_CI       = re.compile(r"confidence interval|95% ci", re.IGNORECASE)
_GPU      = re.compile(r"\b(A100|V100|H100|A6000|RTX\s?\d+|TPU|GPU[- ]?hours?|GPU[- ]?days?)\b",
                       re.IGNORECASE)
_TRAINTIME = re.compile(r"\btrain(?:ing|ed)?\s+time\b|\btook\s+\d+\s+(?:hours|days|minutes)\b",
                        re.IGNORECASE)
_CHECKPOINT = re.compile(r"checkpoint|pretrained weights|model weights", re.IGNORECASE)
_APIMODEL   = re.compile(r"gpt-4[\w.-]*\d|gpt-3\.5[\w.-]*|claude-[\w.-]*\d|gemini-[\w.-]*\d",
                         re.IGNORECASE)
_ENVFILE    = re.compile(r"requirements\.txt|environment\.yml|dockerfile|pyproject\.toml",
                         re.IGNORECASE)
_LICENSE    = re.compile(r"\blicen[cs]e\b|creative commons|apache|mit licen", re.IGNORECASE)
_DATASETS   = re.compile(r"\bdataset[s]?\b", re.IGNORECASE)
_BASELINES  = re.compile(r"\bbaseline[s]?\b", re.IGNORECASE)


def _first(pat, text):
    m = pat.search(text)
    return m.group(0) if m else None


def extract_facts(text: str, sections: dict, tables: list, references: list) -> dict:
    """All the deterministic checks the plan lists — no opinions."""
    n_seeds_match = _N_SEEDS.search(text)
    n_seeds = int(n_seeds_match.group(1)) if n_seeds_match else (1 if _SEED.search(text) else 0)
    code_urls = list(dict.fromkeys(m.group(0) for m in _GITHUB.finditer(text)))

    return {
        "num_datasets_mentioned":  len(_DATASETS.findall(text)),
        "num_baselines_mentioned": len(_BASELINES.findall(text)),
        "num_experiments":         sum(1 for s in ("experiments", "results") if sections.get(s)),
        "num_tables":              len(tables),
        "num_references":          len(references),
        "n_seeds":                 n_seeds,
        "variance_hits":           len(_VARIANCE.findall(text)),
        "has_p_value":             bool(_PVAL.search(text)),
        "has_confidence_interval": bool(_CI.search(text)),
        "gpu_mentioned":           bool(_GPU.search(text)),
        "training_time_mentioned": bool(_TRAINTIME.search(text)),
        "code_url":                code_urls[0] if code_urls else None,
        "doi":                     _first(_DOI, text),
        "huggingface_url":         _first(_HF, text),
        "license_mentioned":       bool(_LICENSE.search(text)),
        "checkpoint_mentioned":    bool(_CHECKPOINT.search(text)),
        "env_file_mentioned":      bool(_ENVFILE.search(text)),
        "api_model_pinned":        _first(_APIMODEL, text),
        "sections_present":        sum(1 for v in sections.values() if v),
    }


# ---------- entry point ----------

def parse_pdf(pdf_bytes: Optional[bytes], meta: dict, max_pages: int = 40) -> dict:
    """Turn PDF bytes into the paper_object described in the plan."""
    if not pdf_bytes:
        return {}

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = "\n".join(doc[i].get_text() for i in range(min(len(doc), max_pages)))
    except Exception as e:
        print(f"  ! PDF parse error: {e}")
        return {}

    sections = split_sections(text)
    tables = extract_tables(text)
    figures = extract_figures(text)
    references = extract_references(sections.get("references", ""))
    facts = extract_facts(text, sections, tables, references)

    # Fall back to arXiv metadata for title/abstract if we didn't detect them cleanly.
    return {
        "title":         meta.get("title", ""),
        "abstract":      sections.get("abstract") or meta.get("abstract", ""),
        "introduction":  sections.get("introduction", ""),
        "related_work":  sections.get("related_work", ""),
        "methodology":   sections.get("methodology", ""),
        "experiments":   sections.get("experiments", ""),
        "results":       sections.get("results", ""),
        "discussion":    sections.get("discussion", ""),
        "limitations":   sections.get("limitations", ""),
        "conclusion":    sections.get("conclusion", ""),
        "references":    references,
        "tables":        tables,
        "figures":       figures,
        "facts":         facts,
        "char_count":    len(text),
    }
