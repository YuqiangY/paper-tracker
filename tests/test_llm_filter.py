from unittest.mock import patch

from models import Paper
from filter.llm_filter import llm_filter
from filter.llm_scoring import ScoringOutcome

INTERESTS = [{"name": "底层视觉", "keywords": ["sr"], "description": "超分"}]


def _p(pid, total):
    p = Paper(id=pid, title=pid, authors=[], abstract="", url="", source="arxiv",
              published="", categories=[])
    p.relevance_score, p.primary_category = total, " 底层视觉"
    return p


@patch("filter.llm_filter.summarize_papers", return_value=0)
@patch("filter.llm_filter.score_papers")
def test_facade_thresholds_normalizes_and_summarizes_passed(mock_score, mock_sum):
    hi, lo = _p("hi", 80), _p("lo", 40)
    mock_score.return_value = ScoringOutcome(scored=[hi, lo], failed=[])
    out = llm_filter([hi, lo], INTERESTS, "m", threshold=60, batch_size=20, api_key="k", base_url=None)
    assert out == [hi]
    assert hi.primary_category == "底层视觉"
    assert mock_sum.call_args.args[0] == [hi]
    assert mock_score.call_args.kwargs["batch_size"] == 50


@patch("filter.llm_filter.summarize_papers", return_value=0)
@patch("filter.llm_filter.score_papers")
def test_facade_threshold_zero_keeps_all_scored(mock_score, _sum):
    a, b = _p("a", 5), _p("b", 0)
    mock_score.return_value = ScoringOutcome(scored=[a, b], failed=[])
    assert llm_filter([a, b], INTERESTS, "m", 0, 20, "k", None) == [a, b]
