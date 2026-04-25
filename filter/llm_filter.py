from __future__ import annotations
import asyncio
import json
import logging
import re

import anthropic
from models import Paper

MAX_CONCURRENT = 5
log = logging.getLogger(__name__)

SYSTEM_MESSAGE = (
    "You are an expert research paper reviewer specializing in computer vision, "
    "machine learning, and AI. You evaluate papers for relevance to specific research "
    "interests and return structured JSON assessments. Be precise and consistent in scoring."
)


# ---------------------------------------------------------------------------
# Public entry point (synchronous)
# ---------------------------------------------------------------------------

def llm_filter(
    papers: list[Paper],
    interests: list[dict],
    model: str,
    threshold: float,
    batch_size: int,
    api_key: str,
    base_url: str | None,
) -> list[Paper]:
    if not papers:
        return []

    batches = [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]
    log.info(
        "LLM filtering %d papers in %d batches (concurrency=%d)",
        len(papers),
        len(batches),
        MAX_CONCURRENT,
    )

    batch_results = asyncio.run(
        _process_all_batches(batches, interests, model, api_key, base_url)
    )

    results = []
    for batch, parsed in batch_results:
        score_map = {r["paper_id"]: r for r in parsed}
        for paper in batch:
            if paper.id in score_map:
                info = score_map[paper.id]
                paper.relevance_score = info.get("relevance_score", 0)
                paper.primary_category = info.get("primary_category", "")
                paper.summary_zh = info.get("summary_zh", "")
                paper.why_relevant = info.get("why_relevant", "")
                paper.tags = info.get("tags", [])
                if paper.relevance_score >= threshold:
                    results.append(paper)

    return results


# ---------------------------------------------------------------------------
# Async batch orchestration
# ---------------------------------------------------------------------------

async def _process_all_batches(batches, interests, model, api_key, base_url):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    total = len(batches)
    tasks = [
        _process_batch_async(b, interests, model, api_key, base_url, semaphore, i + 1, total)
        for i, b in enumerate(batches)
    ]
    return await asyncio.gather(*tasks)


async def _process_batch_async(batch, interests, model, api_key, base_url, semaphore, batch_idx, total_batches):
    async with semaphore:
        prompt = _build_prompt(batch, interests)
        for attempt in range(3):
            response_text = await _call_claude_async(prompt, model, api_key, base_url)
            parsed = _parse_llm_response(response_text)
            if parsed:
                return batch, parsed
            log.warning(
                "LLM parse failed for batch %d/%d (attempt %d/3)",
                batch_idx,
                total_batches,
                attempt + 1,
            )
        log.error(
            "LLM batch %d/%d failed after 3 attempts, %d papers dropped",
            batch_idx,
            total_batches,
            len(batch),
        )
        return batch, []


async def _call_claude_async(prompt: str, model: str, api_key: str, base_url: str | None) -> str:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.AsyncAnthropic(**kwargs)
    message = await client.messages.create(
        model=model,
        max_tokens=16384,
        system=SYSTEM_MESSAGE,
        messages=[{"role": "user", "content": prompt}],
    )
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    log.info("LLM batch tokens: input=%d, output=%d", input_tokens, output_tokens)
    if output_tokens >= 16384:
        log.warning("LLM response may be truncated (hit max_tokens)")
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(papers: list[Paper], interests: list[dict]) -> str:
    interest_desc = "\n".join(
        f"- **{i['name']}**: {i.get('description', '')}" for i in interests
    )

    paper_entries = []
    for p in papers:
        lines = [f"ID: {p.id}", f"Title: {p.title}"]
        if p.authors_enriched:
            name_parts = []
            for a in p.authors_enriched[:5]:
                s = a["name"]
                if a.get("h_index") is not None:
                    s += f" h-index:{a['h_index']}"
                name_parts.append(s)
            lines.append(f"Authors: {', '.join(name_parts)}")
            affils = []
            seen: set[str] = set()
            for a in p.authors_enriched[:5]:
                aff = a.get("affiliation")
                if aff and aff not in seen:
                    seen.add(aff)
                    affils.append(aff)
            if affils:
                lines.append(f"Affiliations: {'; '.join(affils)}")
        elif p.authors:
            lines.append(f"Authors: {', '.join(p.authors[:5])}")
        lines.append(f"Abstract: {p.abstract}")
        paper_entries.append("\n".join(lines) + "\n")
    papers_text = "\n---\n".join(paper_entries)

    return f"""You are a research paper relevance scorer. Evaluate each paper's relevance to the following research interest areas:

{interest_desc}

Scoring rubric:
- 9-10: Directly addresses a core topic, novel method or significant advance, potentially from a strong group
- 7-8: Clearly relevant to an interest area, solid contribution worth reading
- 5-6: Tangentially related, or relevant method applied to a different domain
- 3-4: Weak connection, mostly a different topic
- 1-2: Not relevant to any interest area

Example scoring (do not include these in output):
- "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data" → relevance_score: 9, primary_category: "底层视觉"
- "GPT-4 Technical Report" → relevance_score: 4, primary_category: "多模态大模型"

Consider author reputation (institution affiliation, h-index) as a quality signal when available. Papers from top institutions or by high h-index authors may warrant slightly higher scores.

For each paper below, return a JSON array with one object per paper:
{{
  "paper_id": "<the paper ID>",
  "relevance_score": <1-10 integer>,
  "primary_category": "<most relevant interest area name, or 'other'>",
  "summary_zh": "<one-sentence Chinese summary of the paper>",
  "why_relevant": "<brief English explanation of relevance>",
  "tags": ["<keyword1>", "<keyword2>"]
}}

Return ONLY the JSON array, no other text.

Papers to evaluate:

{papers_text}"""


# ---------------------------------------------------------------------------
# Synchronous Claude call (kept for backward compatibility)
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, model: str, api_key: str, base_url: str | None) -> str:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**kwargs)
    message = client.messages.create(
        model=model,
        max_tokens=16384,
        system=SYSTEM_MESSAGE,
        messages=[{"role": "user", "content": prompt}],
    )
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    log.info("LLM batch tokens: input=%d, output=%d", input_tokens, output_tokens)
    if output_tokens >= 16384:
        log.warning("LLM response may be truncated (hit max_tokens)")
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(text: str) -> list[dict]:
    # Try direct JSON parse first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []
