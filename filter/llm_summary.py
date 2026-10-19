"""Stage B: Chinese summary, rationale and tags — only for papers that were selected."""
from __future__ import annotations
import asyncio
import logging

from models import Paper
from .llm_client import call_llm_async, parse_json_array

log = logging.getLogger(__name__)

MAX_CONCURRENT = 5
PARSE_ATTEMPTS = 2
SYSTEM_MESSAGE = (
    "You summarize research papers for a Chinese-speaking researcher. Return strict JSON."
)


def summarize_papers(
    papers: list[Paper],
    interests: list[dict],
    *,
    model: str,
    api_key: str,
    base_url: str | None,
    batch_size: int,
) -> int:
    pending = list(papers)
    for round_idx in range(2):
        if not pending:
            break
        if round_idx:
            log.warning("Stage B: %d papers missing summaries, retrying once", len(pending))
        batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
        asyncio.run(_summarize_batches(batches, interests, model, api_key, base_url))
        pending = [p for p in pending if not p.summary_zh]
    if pending:
        log.warning("Stage B: %d papers left without summary", len(pending))
    return len(pending)


async def _summarize_batches(batches, interests, model, api_key, base_url):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def one(batch):
        async with semaphore:
            prompt = _build_prompt(batch, interests)
            for _ in range(PARSE_ATTEMPTS):
                try:
                    text = await call_llm_async(prompt, model=model, api_key=api_key,
                                                base_url=base_url, system=SYSTEM_MESSAGE)
                except Exception as e:  # noqa: BLE001 - a failed batch is retried as missing
                    log.warning("Stage B batch failed: %s", e)
                    return
                entries = parse_json_array(text)
                if entries:
                    _apply(batch, entries)
                    return

    await asyncio.gather(*(one(b) for b in batches))


def _apply(batch: list[Paper], entries: list[dict]) -> None:
    by_id = {p.id: p for p in batch}
    for entry in entries:
        paper = by_id.get(str(entry.get("paper_id", "")))
        summary = str(entry.get("summary_zh") or "").strip()
        if paper is None or not summary:
            continue
        paper.summary_zh = summary
        paper.why_relevant = str(entry.get("why_relevant") or "")
        tags = entry.get("tags")
        paper.tags = [str(t) for t in tags] if isinstance(tags, list) else []


def _build_prompt(papers: list[Paper], interests: list[dict]) -> str:
    interest_desc = "\n".join(f"- **{i['name']}**: {i.get('description', '')}" for i in interests)
    blocks = []
    for p in papers:
        lines = [f"ID: {p.id}", f"Title: {p.title}", f"Area: {p.primary_category}",
                 f"Abstract: {p.abstract}"]
        if p.conclusion:
            lines.append(f"Conclusion: {p.conclusion}")
        blocks.append("\n".join(lines) + "\n")
    papers_text = "\n---\n".join(blocks)
    return f"""Interest areas:

{interest_desc}

For each paper return a JSON array with one object per paper:
[{{"paper_id": "<ID>", "summary_zh": "<一句话中文摘要，说明做了什么、效果如何>",
  "why_relevant": "<brief English reason it matters for its area>", "tags": ["<keyword>", "<keyword>"]}}]

Return ONLY the JSON array. Inside text values never use ASCII double quotes; quote terms with 「」 instead.

Papers:

{papers_text}"""
