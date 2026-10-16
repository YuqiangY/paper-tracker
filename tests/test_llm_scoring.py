import json
from unittest.mock import patch

from models import Paper
from filter.llm_scoring import score_papers, build_scoring_prompt, coerce_scores

INTERESTS = [{"name": "底层视觉", "description": "超分、去噪"}, {"name": "AI Agent", "description": "智能体"}]


def _paper(pid, **kw):
    return Paper(id=pid, title=f"T{pid}", authors=["A"], abstract=f"abs {pid}", url="",
                 source="arxiv", published="2026-10-14", categories=["cs.CV"], **kw)


def _entry(pid, rel=36, nov=24, qual=25, cat="底层视觉"):
    return {"paper_id": pid, "relevance": rel, "novelty": nov, "quality": qual, "primary_category": cat}


def _fake_llm(responses_by_call):
    """responses_by_call: list of callables(prompt)->list[dict], consumed in order."""
    calls = []

    async def fake(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(responses_by_call[len(calls) - 1](prompt))
    return fake, calls


def test_coerce_scores():
    assert coerce_scores(_entry("x")) == (36, 24, 25)
    assert coerce_scores({"relevance": "38", "novelty": 31.6, "quality": -3}) == (38, 30, 0)
    assert coerce_scores({"relevance": "high", "novelty": 1, "quality": 1}) is None
    assert coerce_scores({"relevance": 1, "novelty": 1}) is None


def test_prompt_contains_context_and_no_penalty_rule():
    p = _paper("1", conclusion="We conclude X.", code_url="https://github.com/a/b",
               authors_enriched=[{"name": "A", "affiliation": "MIT", "h_index": None}])
    q = _paper("2")
    prompt = build_scoring_prompt([p, q], INTERESTS)
    assert "We conclude X." in prompt and "https://github.com/a/b" in prompt and "MIT" in prompt
    assert "full text unavailable" in prompt
    assert "0-40" in prompt and "0-30" in prompt
    assert "Do not penalize" in prompt


@patch("filter.llm_scoring.call_llm_async")
def test_score_sets_fields_and_total(mock_call):
    fake, _ = _fake_llm([lambda _p: [_entry("1"), _entry("2", 10, 5, 5, "AI Agent")]])
    mock_call.side_effect = fake
    papers = [_paper("1"), _paper("2")]
    out = score_papers(papers, INTERESTS, model="m", api_key="k", base_url=None, batch_size=50)
    assert [p.id for p in out.scored] == ["1", "2"] and out.failed == []
    p1 = out.scored[0]
    assert (p1.score_relevance, p1.score_novelty, p1.score_quality, p1.relevance_score) == (36, 24, 25, 85)
    assert out.scored[1].primary_category == "AI Agent"


@patch("filter.llm_scoring.call_llm_async")
def test_missing_ids_retried_once_then_failed(mock_call):
    fake, calls = _fake_llm([
        lambda _p: [_entry("1")],
        lambda _p: [_entry("2")],
    ])
    mock_call.side_effect = fake
    out = score_papers([_paper("1"), _paper("2"), _paper("3")], INTERESTS,
                       model="m", api_key="k", base_url=None, batch_size=50)
    assert sorted(p.id for p in out.scored) == ["1", "2"]
    assert [p.id for p in out.failed] == ["3"]
    assert len(calls) == 2
    assert "T3" in calls[1] and "T1" not in calls[1]


@patch("filter.llm_scoring.call_llm_async")
def test_batches_grouped_by_area(mock_call):
    seen_batches = []

    async def fake(prompt, **kwargs):
        ids = [line[4:] for line in prompt.splitlines() if line.startswith("ID: ")]
        seen_batches.append(ids)
        return json.dumps([_entry(i) for i in ids])
    mock_call.side_effect = fake
    papers = [_paper("a1"), _paper("b1"), _paper("a2"), _paper("b2")]
    area_of = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    score_papers(papers, INTERESTS, model="m", api_key="k", base_url=None, batch_size=2, area_of=area_of)
    assert sorted(seen_batches) == [["a1", "a2"], ["b1", "b2"]]


@patch("filter.llm_scoring.call_llm_async")
def test_garbage_scores_count_as_missing(mock_call):
    fake, _ = _fake_llm([
        lambda _p: [{"paper_id": "1", "relevance": "n/a", "novelty": 1, "quality": 1}],
        lambda _p: [_entry("1")],
    ])
    mock_call.side_effect = fake
    out = score_papers([_paper("1")], INTERESTS, model="m", api_key="k", base_url=None, batch_size=50)
    assert [p.id for p in out.scored] == ["1"]
