"""Google ADK agent: diagnoses an injection-molding defect from a knowledge graph.

The LLM does three things: maps the operator's words to graph symptoms, calls the
deterministic inference tool, and explains the result. Every probability and
confidence label it reports is produced by `kg.py`; the instruction forbids it
from inventing or adjusting them.

Evidence accumulates in session state, so a follow-up such as "moisture test
came back positive" updates the same case.
"""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.tools import ToolContext

from .kg import DefectKG, Evidence

KG = DefectKG.load()
MODEL = os.environ.get("DEFECT_AGENT_MODEL", "claude-opus-5-5")
STATE_KEY = "defect_case"


def _case(tool_context: ToolContext) -> dict:
    case = tool_context.state.get(STATE_KEY)
    if not case:
        case = {"present": [], "absent": [], "checks": {}}
    return case


# --------------------------------------------------------------------- tools
def list_known_symptoms() -> dict:
    """Lists every defect symptom and diagnostic check in the knowledge graph.

    Call this when the operator's description does not obviously match a
    symptom name, so you can map their words onto the graph's vocabulary.

    Returns:
        dict with 'symptoms' (id -> name and aliases) and 'checks' (id -> name, what a positive result means).
    """
    return {
        "symptoms": {sid: {"name": s["name"], "aliases": s.get("aliases", [])} for sid, s in KG.symptoms.items()},
        "checks": {kid: {"name": k["name"], "positive_means": k["positive_means"]} for kid, k in KG.checks.items()},
    }


def diagnose_defect(
    observed_symptoms: list[str],
    absent_symptoms: list[str],
    positive_checks: list[str],
    negative_checks: list[str],
    tool_context: ToolContext,
) -> dict:
    """Adds evidence to the current defect case and returns a ranked diagnosis with calibrated confidence.

    Evidence is cumulative across turns: pass only what is NEW in this turn
    (pass empty lists for categories with nothing new). Symptoms can be given
    as graph ids ('splay_marks') or plain words ('silver streaks').

    Args:
        observed_symptoms: Symptoms the operator says ARE present on the parts.
        absent_symptoms: Symptoms the operator explicitly says are NOT present. Do not guess these.
        positive_checks: Check ids (from list_known_symptoms) whose result was positive / abnormal.
        negative_checks: Check ids whose result was negative / normal.

    Returns:
        dict with the top diagnosis, its probability, a confidence label
        (high / medium / low / none) with the reason, perturbation stability,
        ranked candidates with per-evidence likelihood ratios, remedies, the
        next most informative questions or checks, and any input terms that
        could not be matched to the graph.
    """
    case = _case(tool_context)
    unrecognized: list[str] = []

    def add_symptoms(terms, bucket, other):
        for t in terms or []:
            sid = KG.resolve_symptom(t)
            if sid is None:
                unrecognized.append(t)
                continue
            if sid in case[other]:          # operator corrected themselves; latest wins
                case[other].remove(sid)
            if sid not in case[bucket]:
                case[bucket].append(sid)

    add_symptoms(observed_symptoms, "present", "absent")
    add_symptoms(absent_symptoms, "absent", "present")
    for terms, result in ((positive_checks, True), (negative_checks, False)):
        for t in terms or []:
            kid = KG.resolve_check(t)
            if kid is None:
                unrecognized.append(t)
            else:
                case["checks"][kid] = result

    tool_context.state[STATE_KEY] = case
    ev = Evidence(tuple(case["present"]), tuple(case["absent"]), tuple(case["checks"].items()))
    result = KG.diagnose(ev)
    result["case_evidence_so_far"] = {
        "present": [KG.symptoms[s]["name"] for s in case["present"]],
        "absent": [KG.symptoms[s]["name"] for s in case["absent"]],
        "checks": {KG.checks[k]["name"]: ("positive" if v else "negative") for k, v in case["checks"].items()},
    }
    result["unrecognized_terms"] = unrecognized
    result["confidence_line"] = (
        f"Confidence: {result['confidence'].upper()} - {result['diagnosis']} at "
        f"{result['probability']:.0%} ({result['confidence_reason']})"
    )
    return result


def explore_graph(node_id: str) -> dict:
    """Shows a node of the knowledge graph and its direct relationships.

    Use it to answer 'why' questions, e.g. which symptoms a root cause produces
    or which causes produce a symptom.

    Args:
        node_id: Id of a root cause, symptom, check or process parameter.

    Returns:
        dict with the node's type and attributes plus its incoming and outgoing edges.
    """
    nid = node_id if node_id in KG.graph else (KG.resolve_symptom(node_id) or KG.resolve_check(node_id))
    if nid is None or nid not in KG.graph:
        return {"error": f"'{node_id}' is not in the knowledge graph."}
    g = KG.graph

    def edge(u, v, d):
        out = {"from": u, "to": v, "relation": d["rel"]}
        if "p" in d:
            out["p_symptom_given_cause"] = d["p"]
        if "sensitivity" in d:
            out["sensitivity"] = d["sensitivity"]
        return out

    return {
        "id": nid,
        **g.nodes[nid],
        "outgoing": [edge(u, v, d) for u, v, d in g.out_edges(nid, data=True)],
        "incoming": [edge(u, v, d) for u, v, d in g.in_edges(nid, data=True)],
    }


def reset_case(tool_context: ToolContext) -> dict:
    """Clears all evidence so a new, unrelated defect can be diagnosed from scratch.

    Returns:
        dict confirming the reset.
    """
    tool_context.state[STATE_KEY] = {"present": [], "absent": [], "checks": {}}
    return {"status": "case cleared"}


INSTRUCTION = """You are a manufacturing quality engineer's assistant that diagnoses injection-molding defects.

How to work:
1. Map the operator's description to knowledge-graph symptoms. If a phrase is ambiguous, call
   list_known_symptoms. Only put a symptom in absent_symptoms if the operator explicitly said it is not present.
2. Call diagnose_defect with only the NEW evidence from this turn. Evidence from earlier turns is already stored.
   If the operator starts describing a different part or defect, call reset_case first.
3. If unrecognized_terms is non-empty, tell the operator which words you could not map and ask them to rephrase.

How to answer (keep it short, plain text, for someone on the shop floor):
- First line: the diagnosis and its probability exactly as returned.
- Second line: the confidence_line field, verbatim.
- 'Why': the 2-3 strongest evidence items from the top candidate, citing their likelihood ratios.
- 'Also possible': the next one or two candidates with probabilities.
- 'Next step': the first item in next_best_actions (and its 'how' if it is a check). If confidence is high,
  instead give the remedy and say which check would confirm it.
- If out_of_graph_warning is true, say the pattern may not match any known cause and recommend escalation.

Hard rules:
- Never state, round differently, or adjust any probability or confidence label that the tool did not return.
- Never claim higher confidence than the tool's label, even if the answer seems obvious to you.
- If confidence is 'none' or 'low', say so plainly and ask for the most informative next observation.
"""

# ADK routes bare "claude-*" strings to Claude on Vertex AI; AnthropicLlm calls the
# Anthropic API directly (reads ANTHROPIC_API_KEY). Gemini names still go to Vertex.
root_agent = Agent(
    name="defect_diagnoser",
    model=AnthropicLlm(model=MODEL) if MODEL.startswith("claude-") else MODEL,
    description="Diagnoses injection-molding defects from a knowledge graph and reports calibrated confidence.",
    instruction=INSTRUCTION,
    tools=[list_known_symptoms, diagnose_defect, explore_graph, reset_case],
)
