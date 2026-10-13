"""Agent A - Paper Understanding Agent.

Reads the PDF (in memory) and produces the paper_object described in the plan:
  {title, abstract, introduction, related_work, methodology, experiments,
   results, discussion, limitations, conclusion, references,
   tables[], figures[], facts{}}

No LLM. No web. Deterministic parsing only. Also owns the early-exit gate: if
the PDF is unreadable or too short, we stop the pipeline here.
"""
from paper_scoring.scoring.pdf_parser import parse_pdf


def agent_a(state: dict) -> dict:
    print("\n" + "=" * 30)
    print("Agent A - Paper Understanding")
    print("=" * 30)

    notes = list(state.get("notes", []))
    meta = {"title": state.get("title", ""), "abstract": state.get("abstract", "")}
    paper_object = parse_pdf(state.get("pdf_bytes"), meta)

    # Gate: unreadable or too short -> stop the pipeline early.
    if not paper_object or paper_object.get("char_count", 0) < 500:
        notes.append("Agent A: PDF unreadable or too short.")
        print("  ! PDF unreadable - gate failed")
        return {"paper_object": paper_object or {},
                "gate_failed": True, "gate_reason": "unreadable_pdf", "notes": notes}

    facts = paper_object["facts"]
    # A concise dashboard so the run log is easy to scan.
    print(f"  Chars       : {paper_object['char_count']}")
    print(f"  Sections    : {facts['sections_present']}/10 detected")
    print(f"  Tables      : {facts['num_tables']}")
    print(f"  Figures     : {len(paper_object['figures'])}")
    print(f"  References  : {facts['num_references']}")
    print(f"  Seeds       : {facts['n_seeds']}     Variance hits : {facts['variance_hits']}")
    print(f"  Code URL    : {facts['code_url']}")
    print(f"  DOI         : {facts['doi']}")
    print(f"  GPU/compute : {facts['gpu_mentioned']}")

    notes.append(
        f"Agent A: {facts['sections_present']} sections, {facts['num_tables']} tables, "
        f"{facts['num_references']} refs, seeds={facts['n_seeds']}."
    )
    return {"paper_object": paper_object, "gate_failed": False, "notes": notes}
