from unittest.mock import patch, MagicMock
from output.summary import generate_summary, _build_prompt, _call_llm


SAMPLE_PAPERS = [
    {
        "title": "SR Method",
        "primary_category": "底层视觉",
        "summary_zh": "一种新超分方法",
        "tags": '["super-resolution"]',
        "relevance_score": 8.5,
        "url": "https://arxiv.org/abs/2404.12345",
    },
    {
        "title": "Video Model",
        "primary_category": "视频算法",
        "summary_zh": "视频生成模型",
        "tags": '["video-generation", "diffusion"]',
        "relevance_score": 9.0,
        "url": "https://arxiv.org/abs/2404.67890",
    },
]

INTERESTS = [
    {"name": "底层视觉", "description": "图像超分、去噪等底层视觉任务"},
    {"name": "视频算法", "description": "视频生成、编辑、理解"},
]


def test_build_prompt_includes_categories():
    prompt = _build_prompt(SAMPLE_PAPERS, INTERESTS)
    assert "底层视觉" in prompt
    assert "视频算法" in prompt
    assert "SR Method" in prompt
    assert "Video Model" in prompt


def test_build_prompt_includes_interests():
    prompt = _build_prompt(SAMPLE_PAPERS, INTERESTS)
    assert "图像超分" in prompt
    assert "视频生成" in prompt


def test_build_prompt_includes_tags():
    prompt = _build_prompt(SAMPLE_PAPERS, INTERESTS)
    assert "super-resolution" in prompt
    assert "video-generation" in prompt


def test_build_prompt_includes_scores():
    prompt = _build_prompt(SAMPLE_PAPERS, INTERESTS)
    assert "8.5分" in prompt
    assert "9.0分" in prompt


@patch("output.summary._call_llm")
def test_generate_summary_returns_tuple(mock_llm):
    mock_llm.return_value = "**底层视觉**：今日主要关注超分辨率方向。"
    html, text = generate_summary(SAMPLE_PAPERS, INTERESTS, "model", "key", None)
    assert "<strong>" in html
    assert "超分辨率" in html
    assert "**底层视觉**" in text
    assert "超分辨率" in text


@patch("output.summary._call_llm")
def test_generate_summary_empty_on_failure(mock_llm):
    mock_llm.side_effect = Exception("API error")
    html, text = generate_summary(SAMPLE_PAPERS, INTERESTS, "model", "key", None)
    assert html == ""
    assert text == ""


def test_generate_summary_empty_papers():
    html, text = generate_summary([], INTERESTS, "model", "key", None)
    assert html == ""
    assert text == ""


@patch("output.summary._call_llm")
def test_generate_summary_empty_llm_response(mock_llm):
    mock_llm.return_value = ""
    html, text = generate_summary(SAMPLE_PAPERS, INTERESTS, "model", "key", None)
    assert html == ""
    assert text == ""


@patch("output.summary.anthropic")
def test_call_llm_skips_thinking_blocks(mock_anthropic):
    thinking_block = MagicMock()
    thinking_block.type = "thinking"

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "### Summary content"

    mock_msg = MagicMock()
    mock_msg.content = [thinking_block, text_block]
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

    result = _call_llm("prompt", "model", "key", None)
    assert result == "### Summary content"
