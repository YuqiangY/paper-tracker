import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from filter.llm_client import call_llm_async, parse_json_array


def _msg(blocks, stop_reason="end_turn"):
    m = MagicMock()
    m.content = blocks
    m.stop_reason = stop_reason
    m.usage.input_tokens = 10
    m.usage.output_tokens = 20
    return m


def _block(kind, text=""):
    b = MagicMock()
    b.type = kind
    b.text = text
    return b


@patch("filter.llm_client.anthropic")
def test_call_returns_first_text_block_after_thinking(mock_anthropic):
    client = mock_anthropic.AsyncAnthropic.return_value
    client.close = AsyncMock()
    client.messages.create = AsyncMock(return_value=_msg([_block("thinking"), _block("text", "[1]")]))
    out = asyncio.run(call_llm_async("p", model="m", api_key="k", base_url=None, system="s"))
    assert out == "[1]"


@patch("filter.llm_client.anthropic")
def test_call_warns_on_truncation(mock_anthropic, caplog):
    client = mock_anthropic.AsyncAnthropic.return_value
    client.close = AsyncMock()
    client.messages.create = AsyncMock(return_value=_msg([_block("text", "[")], stop_reason="max_tokens"))
    asyncio.run(call_llm_async("p", model="m", api_key="k", base_url=None, system="s"))
    assert "truncated" in caplog.text


def test_parse_plain_and_fenced_and_think_wrapped():
    data = [{"paper_id": "1"}]
    assert parse_json_array(json.dumps(data)) == data
    assert parse_json_array(f"Here:\n```json\n{json.dumps(data)}\n```") == data
    assert parse_json_array(f"<think>hmm</think>\n{json.dumps(data)}") == data


def test_parse_garbage_returns_empty():
    assert parse_json_array("not json") == []
    assert parse_json_array('{"a": 1}') == []


@patch("filter.llm_client.anthropic")
def test_call_closes_client_even_on_error(mock_anthropic):
    import pytest
    client = mock_anthropic.AsyncAnthropic.return_value
    client.close = AsyncMock()
    client.messages.create = AsyncMock(return_value=_msg([_block("text", "[]")]))
    asyncio.run(call_llm_async("p", model="m", api_key="k", base_url=None, system="s"))
    client.messages.create = AsyncMock(side_effect=RuntimeError("529"))
    with pytest.raises(RuntimeError):
        asyncio.run(call_llm_async("p", model="m", api_key="k", base_url=None, system="s"))
    assert client.close.await_count == 2


def test_parse_repairs_unescaped_quotes_inside_strings():
    # Real 2026-10-19 Stage B output: ASCII quotes inside a Chinese summary broke the whole batch.
    text = (
        '```json\n[\n  {\n    "paper_id": "2610.23610",\n'
        '    "summary_zh": "主张幻觉评估应区分"提及"与"断言"。",\n'
        '    "why_relevant": "Challenges a \\"core\\" assumption.",\n'
        '    "tags": ["多模态", "评测"]\n  },\n'
        '  {"paper_id": "2", "summary_zh": "正常", "why_relevant": "ok", "tags": []}\n]\n```'
    )
    data = parse_json_array(text)
    assert [d["paper_id"] for d in data] == ["2610.23610", "2"]
    assert data[0]["summary_zh"] == '主张幻觉评估应区分"提及"与"断言"。'
    assert data[0]["why_relevant"] == 'Challenges a "core" assumption.'
    assert data[0]["tags"] == ["多模态", "评测"]


def test_parse_real_stage_b_output_file_if_present():
    import os
    path = "/tmp/pt-stageb/raw.txt"
    if not os.path.exists(path):
        return
    with open(path) as f:
        assert len(parse_json_array(f.read())) == 20
