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
