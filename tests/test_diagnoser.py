from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from defect_agent.kg import DefectKG, Evidence

KG = DefectKG.load()


# ------------------------------------------------------------------ engine
def test_posterior_sums_to_one():
    post = KG.posterior(Evidence(present=("flash", "sink_marks")))
    assert abs(sum(post.values()) - 1) < 1e-9


@pytest.mark.parametrize("present,expected", [
    (("splay_marks", "brittle_parts"), "wet_resin"),
    (("sink_marks", "voids"), "insufficient_pack_pressure"),
    (("burn_marks", "short_shot"), "blocked_vents"),
    (("black_specks",), "contaminated_resin"),
    (("warpage", "dims_out_of_tolerance"), "uneven_mold_cooling"),
    (("short_shot", "flow_lines", "weld_lines"), "low_melt_temp"),
])
def test_textbook_cases(present, expected):
    assert KG.diagnose(Evidence(present=present))["diagnosis_id"] == expected


def test_ambiguous_evidence_is_low_confidence():
    # Flash alone is produced by three causes; the agent must not be sure.
    r = KG.diagnose(Evidence(present=("flash",)))
    assert r["confidence"] == "low"
    assert r["next_best_actions"][0]["type"] == "run_check"


def test_check_result_disambiguates():
    before = KG.diagnose(Evidence(present=("flash",)))
    after = KG.diagnose(Evidence(present=("flash",), checks=(("parting_line_blue_check", True),)))
    assert after["diagnosis_id"] == "worn_parting_line"
    assert after["probability"] > before["probability"]


def test_confirming_check_raises_confidence():
    r = KG.diagnose(Evidence(present=("splay_marks", "brittle_parts"), checks=(("moisture_test", True),)))
    assert r["confidence"] == "high" and r["probability"] > 0.95


def test_negative_check_lowers_belief():
    pos = KG.posterior(Evidence(present=("splay_marks",)))["wet_resin"]
    neg = KG.posterior(Evidence(present=("splay_marks",), checks=(("moisture_test", False),)))["wet_resin"]
    assert neg < pos / 3


def test_incoherent_symptoms_flag_unmodeled_cause():
    r = KG.diagnose(Evidence(present=("black_specks", "warpage", "flash", "short_shot")))
    assert r["out_of_graph_warning"] and r["confidence"] == "low"


def test_no_evidence_returns_none_confidence():
    assert KG.diagnose(Evidence())["confidence"] == "none"


def test_alias_resolution():
    assert KG.resolve_symptom("silver streaks") == "splay_marks"
    assert KG.resolve_symptom("warpping") == "warpage"      # typo
    assert KG.resolve_symptom("purple polka dots") is None


def test_stability_is_deterministic():
    ev = Evidence(present=("flash",))
    assert KG.stability(ev, "insufficient_clamp_force") == KG.stability(ev, "insufficient_clamp_force")


# ------------------------------------------------- full agent loop, offline
class ScriptedLlm(BaseLlm):
    """Plays the model's role: one scripted tool call per user turn, then a
    reply that echoes the tool's numbers (as the real instruction requires)."""

    script: list = []

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        last = llm_request.contents[-1]
        fr = next((p.function_response for p in last.parts if p.function_response), None)
        if fr is not None:
            res = fr.response
            text = f"{res['diagnosis']} ({res['probability']:.0%}).\n{res['confidence_line']}"
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))
            return
        args = self.script.pop(0)
        call = types.FunctionCall(name="diagnose_defect", args=args)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=call)]))


def _run(turns_and_args):
    from defect_agent import agent as mod

    fake = ScriptedLlm(model="scripted")
    fake.script = [a for _, a in turns_and_args]
    test_agent = mod.root_agent.clone(update={"model": fake})
    runner = InMemoryRunner(agent=test_agent, app_name="t")

    async def go():
        s = await runner.session_service.create_session(app_name="t", user_id="u")
        replies, tool_results = [], []
        for msg, _ in turns_and_args:
            content = types.Content(role="user", parts=[types.Part(text=msg)])
            async for ev in runner.run_async(user_id="u", session_id=s.id, new_message=content):
                for p in (ev.content.parts if ev.content else []):
                    if p.function_response:
                        tool_results.append(p.function_response.response)
                    if p.text:
                        replies.append(p.text)
        s = await runner.session_service.get_session(app_name="t", user_id="u", session_id=s.id)
        return replies, tool_results, s.state

    return asyncio.run(go())


def test_agent_loop_accumulates_evidence_across_turns():
    empty = {"observed_symptoms": [], "absent_symptoms": [], "positive_checks": [], "negative_checks": []}
    replies, results, state = _run([
        ("silver streaks and parts snap", {**empty, "observed_symptoms": ["silver streaks", "parts snap"]}),
        ("moisture test failed", {**empty, "positive_checks": ["moisture_test"]}),
    ])
    first, second = results
    assert first["diagnosis_id"] == second["diagnosis_id"] == "wet_resin"
    assert second["probability"] > first["probability"]            # evidence was merged, not replaced
    assert state["defect_case"]["present"] == ["splay_marks", "brittle_parts"]
    assert "Confidence: HIGH" in replies[-1]


def test_agent_reports_unrecognized_terms():
    empty = {"observed_symptoms": [], "absent_symptoms": [], "positive_checks": [], "negative_checks": []}
    _, results, _ = _run([("purple dots", {**empty, "observed_symptoms": ["purple polka dots"]})])
    assert results[0]["unrecognized_terms"] == ["purple polka dots"]
    assert results[0]["confidence"] == "none"
