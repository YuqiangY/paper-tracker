import json
from unittest.mock import patch, MagicMock
from models import Paper
from filter.llm_filter import llm_filter, _build_prompt, _parse_llm_response


def _paper(id="2404.12345", title="Test", abstract="Abstract"):
    return Paper(
        id=id,
        title=title,
        authors=["Author"],
        abstract=abstract,
        url=f"https://arxiv.org/abs/{id}",
        source="arxiv",
        published="2026-04-17",
        categories=["cs.CV"],
    )


INTERESTS = [
    {
        "name": "底层视觉",
        "keywords": ["super-resolution"],
        "description": "图像超分、去噪等底层视觉任务",
    },
]


def test_build_prompt_includes_paper_info():
    papers = [_paper(title="Super-Res Paper", abstract="About SR.")]
    prompt = _build_prompt(papers, INTERESTS)
    assert "Super-Res Paper" in prompt
    assert "About SR." in prompt
    assert "底层视觉" in prompt
    assert "Author" in prompt


def test_build_prompt_includes_enriched_authors():
    p = _paper()
    p.authors_enriched = [
        {"name": "Alice", "affiliation": "MIT", "h_index": 45, "semantic_scholar_id": None},
    ]
    prompt = _build_prompt([p], INTERESTS)
    assert "Alice" in prompt
    assert "MIT" in prompt
    assert "h-index:45" in prompt


def test_parse_llm_response_valid():
    response_text = json.dumps([
        {
            "paper_id": "2404.12345",
            "relevance_score": 8,
            "primary_category": "底层视觉",
            "summary_zh": "超分方法",
            "why_relevant": "directly related",
            "tags": ["super-resolution"],
        }
    ])
    results = _parse_llm_response(response_text)
    assert len(results) == 1
    assert results[0]["relevance_score"] == 8


def test_parse_llm_response_extracts_json_from_markdown():
    response_text = """Here are the results:
```json
[{"paper_id": "2404.12345", "relevance_score": 9, "primary_category": "底层视觉", "summary_zh": "摘要", "why_relevant": "reason", "tags": ["tag"]}]
```"""
    results = _parse_llm_response(response_text)
    assert len(results) == 1
    assert results[0]["relevance_score"] == 9


def test_parse_llm_response_handles_malformed():
    results = _parse_llm_response("not valid json at all")
    assert results == []


@patch("filter.llm_filter.anthropic")
def test_call_claude_skips_thinking_blocks(mock_anthropic):
    from filter.llm_filter import _call_claude

    thinking_block = MagicMock()
    thinking_block.type = "thinking"

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = '[{"paper_id": "123", "relevance_score": 8}]'

    mock_msg = MagicMock()
    mock_msg.content = [thinking_block, text_block]
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

    result = _call_claude("test prompt", "model", "key", None)
    assert result == '[{"paper_id": "123", "relevance_score": 8}]'


@patch("filter.llm_filter.anthropic")
def test_call_claude_returns_empty_if_no_text_block(mock_anthropic):
    from filter.llm_filter import _call_claude

    thinking_block = MagicMock()
    thinking_block.type = "thinking"

    mock_msg = MagicMock()
    mock_msg.content = [thinking_block]
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

    result = _call_claude("test prompt", "model", "key", None)
    assert result == ""


@patch("filter.llm_filter._call_claude")
def test_llm_filter_end_to_end(mock_call):
    mock_call.return_value = json.dumps([
        {
            "paper_id": "2404.12345",
            "relevance_score": 8,
            "primary_category": "底层视觉",
            "summary_zh": "超分方法",
            "why_relevant": "reason",
            "tags": ["sr"],
        },
        {
            "paper_id": "2404.99999",
            "relevance_score": 3,
            "primary_category": "底层视觉",
            "summary_zh": "不太相关",
            "why_relevant": "weak",
            "tags": [],
        },
    ])
    papers = [
        _paper("2404.12345", "SR Paper", "About SR"),
        _paper("2404.99999", "Other Paper", "Other topic"),
    ]
    results = llm_filter(
        papers=papers,
        interests=INTERESTS,
        model="claude-sonnet-4-6",
        threshold=7,
        batch_size=20,
        api_key="test-key",
        base_url=None,
    )
    assert len(results) == 1
    assert results[0].id == "2404.12345"
    assert results[0].relevance_score == 8
    assert results[0].summary_zh == "超分方法"
