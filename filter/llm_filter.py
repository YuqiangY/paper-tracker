from __future__ import annotations
import json
import re
import anthropic
from models import Paper


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

    results = []
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        prompt = _build_prompt(batch, interests)
        response_text = _call_claude(prompt, model, api_key, base_url)
        parsed = _parse_llm_response(response_text)

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


def _build_prompt(papers: list[Paper], interests: list[dict]) -> str:
    interest_desc = "\n".join(
        f"- **{i['name']}**: {i.get('description', '')}"
        for i in interests
    )

    paper_entries = []
    for p in papers:
        lines = [f"ID: {p.id}", f"Title: {p.title}"]
        if p.authors_enriched:
            author_strs = []
            for a in p.authors_enriched[:5]:
                parts = [a["name"]]
                if a.get("affiliation"):
                    parts.append(f'({a["affiliation"]})')
                if a.get("h_index") is not None:
                    parts.append(f'h-index:{a["h_index"]}')
                author_strs.append(" ".join(parts))
            lines.append(f"Authors: {'; '.join(author_strs)}")
        elif p.authors:
            lines.append(f"Authors: {', '.join(p.authors[:5])}")
        lines.append(f"Abstract: {p.abstract}")
        paper_entries.append("\n".join(lines) + "\n")
    papers_text = "\n---\n".join(paper_entries)

    return f"""You are a research paper relevance scorer. Evaluate each paper's relevance to the following research interest areas:

{interest_desc}

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


def _call_claude(
    prompt: str, model: str, api_key: str, base_url: str | None
) -> str:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**kwargs)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""


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
