"""Shared Anthropic-SDK call and JSON-array parsing for the LLM stages."""
from __future__ import annotations
import json
import logging
import re

import anthropic

log = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 16384
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


async def call_llm_async(
    prompt: str,
    *,
    model: str,
    api_key: str,
    base_url: str | None,
    system: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.AsyncAnthropic(**kwargs)
    try:
        message = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
    finally:
        # Close inside this event loop; a GC'd client would try after asyncio.run() closed it.
        await client.close()
    log.info("LLM tokens: input=%d, output=%d", message.usage.input_tokens, message.usage.output_tokens)
    if message.stop_reason == "max_tokens":
        log.warning("LLM response truncated (hit max_tokens=%d)", max_tokens)
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""


def parse_json_array(text: str) -> list[dict]:
    text = _THINK_RE.sub("", text).strip()
    candidates = [text]
    match = _FENCE_RE.search(text)
    if match:
        candidates.append(match.group(1).strip())
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    return []
