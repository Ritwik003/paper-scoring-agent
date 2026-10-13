"""Wire the three agents into one sequential LangGraph pipeline.

Flow:
    START -> Agent A -> (gate failed? -> finalize)   [early exit]
                     -> Agent B -> Agent C -> finalize -> END

Agent A can stop the line: if the paper has a fatal problem, we skip the
web and LLM steps entirely and go straight to the verdict.
"""
from langgraph.graph import StateGraph, START, END

from paper_scoring.scoring.state import ReviewState, new_state
from paper_scoring.scoring.agent_deterministic import agent_a
from paper_scoring.scoring.agent_web import make_agent_b
from paper_scoring.scoring.agent_llm import make_agent_c
from paper_scoring.scoring.rubric import weighted_total, decide_verdict


def _route_after_a(state: dict) -> str:
    """After Agent A: stop early if a gate failed, else continue to Agent B."""
    return "stop" if state.get("gate_failed") else "continue"


def build_graph(settings: dict):
    """Compile and return the runnable scoring pipeline."""
    scoring = settings["scoring"]
    weights = scoring["weights"]
    verdict_cfg = scoring["verdict"]

    agent_b = make_agent_b(scoring["web"],scoring.get("tavily_gate", {}))
    agent_c = make_agent_c(scoring["llm"])

    def finalize(state: dict) -> dict:
        notes = list(state.get("notes", []))
        if state.get("gate_failed"):
            notes.append(f"Verdict: skip (gate: {state.get('gate_reason')}).")
            return {"weighted_total": 0.0, "verdict": "skip", "notes": notes}

        total = weighted_total(state.get("scores", {}), weights)
        verdict = decide_verdict(total, verdict_cfg)
        notes.append(f"Verdict: {verdict} (score {total}/10).")
        return {"weighted_total": total, "verdict": verdict, "notes": notes}

    builder = StateGraph(ReviewState)
    builder.add_node("agent_a", agent_a)
    builder.add_node("agent_b", agent_b)
    builder.add_node("agent_c", agent_c)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "agent_a")
    builder.add_conditional_edges(
        "agent_a", _route_after_a, {"stop": "finalize", "continue": "agent_b"}
    )
    builder.add_edge("agent_b", "agent_c")
    builder.add_edge("agent_c", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


def score_paper(graph, paper: dict) -> dict:
    """Run one paper through the pipeline and return the final state."""
    return graph.invoke(new_state(paper))
