import json
from unittest.mock import patch

from models import Paper
from filter.llm_summary import summarize_papers

INTERESTS = [{"name": "底层视觉", "description": "超分"}]


def _p(pid):
    p = Paper(id=pid, title=f"T{pid}", authors=[], abstract="a", url="", source="arxiv",
              published="", categories=[], conclusion="We conclude.")
    p.primary_category = "底层视觉"
    return p


@patch("filter.llm_summary.call_llm_async")
def test_summaries_applied_and_missing_retried(mock_call):
    calls = []

    async def fake(prompt, **kw):
        calls.append(prompt)
        ids = ["1"] if len(calls) == 1 else ["2"]
        return json.dumps([{"paper_id": i, "summary_zh": f"摘要{i}", "why_relevant": "r", "tags": ["sr"]} for i in ids])
    mock_call.side_effect = fake
    papers = [_p("1"), _p("2"), _p("3")]
    missing = summarize_papers(papers, INTERESTS, model="m", api_key="k", base_url=None, batch_size=20)
    assert [p.summary_zh for p in papers] == ["摘要1", "摘要2", None]
    assert papers[0].tags == ["sr"]
    assert missing == 1
    assert "We conclude." in calls[0]


@patch("filter.llm_summary.call_llm_async")
def test_api_error_in_one_batch_does_not_raise(mock_call):
    async def fake(prompt, **kw):
        if "T1" in prompt:
            raise RuntimeError("529 overloaded")
        ids = [line[4:] for line in prompt.splitlines() if line.startswith("ID: ")]
        return json.dumps([{"paper_id": i, "summary_zh": "s", "why_relevant": "r", "tags": []} for i in ids])
    mock_call.side_effect = fake
    papers = [_p("1"), _p("2")]
    missing = summarize_papers(papers, INTERESTS, model="m", api_key="k", base_url=None, batch_size=1)
    assert missing == 1
    assert papers[1].summary_zh == "s"


def test_prompt_forbids_ascii_quotes_inside_text():
    from filter.llm_summary import _build_prompt
    assert "「」" in _build_prompt([_p("1")], INTERESTS)
