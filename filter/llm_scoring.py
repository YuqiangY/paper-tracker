"""Stage A: score keyword-matched papers with a 0-100 sub-score rubric (no summaries)."""
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass

from models import Paper
from .llm_client import call_llm_async, parse_json_array

log = logging.getLogger(__name__)

MAX_CONCURRENT = 5
PARSE_ATTEMPTS = 2
SCORE_MAX = {"relevance": 40, "novelty": 30, "quality": 30}
MAX_AUTHORS_IN_PROMPT = 5

SYSTEM_MESSAGE = (
    "You are an expert reviewer in computer vision, machine learning and AI. "
    "You score papers against a fixed rubric and return strict JSON. "
    "Compare the papers within a batch so that scores are consistent with each other."
)


@dataclass(frozen=True)
class ScoringOutcome:
    scored: list[Paper]
    failed: list[Paper]


def score_papers(
    papers: list[Paper],
    interests: list[dict],
    *,
    model: str,
    api_key: str,
    base_url: str | None,
    batch_size: int,
    area_of: dict[str, str] | None = None,
) -> ScoringOutcome:
    if not papers:
        return ScoringOutcome(scored=[], failed=[])
    area_of = area_of or {}
    ordered = sorted(papers, key=lambda p: area_of.get(p.id, ""))
    batches = _chunk(ordered, batch_size)
    log.info("Stage A scoring %d papers in %d batches", len(papers), len(batches))

    scored_ids = _run_round(batches, interests, model, api_key, base_url)
    missing = [p for p in ordered if p.id not in scored_ids]
    if missing:
        log.warning("Stage A: %d papers missing from LLM output, retrying once", len(missing))
        scored_ids |= _run_round(_chunk(missing, batch_size), interests, model, api_key, base_url)

    scored = [p for p in papers if p.id in scored_ids]
    failed = [p for p in papers if p.id not in scored_ids]
    if failed:
        log.warning("Stage A: %d papers could not be scored: %s",
                    len(failed), ", ".join(p.id for p in failed[:10]))
    return ScoringOutcome(scored=scored, failed=failed)


def coerce_scores(entry: dict) -> tuple[int, int, int] | None:
    values = []
    for key, upper in SCORE_MAX.items():
        try:
            value = round(float(entry[key]))
        except (KeyError, TypeError, ValueError):
            return None
        values.append(min(max(value, 0), upper))
    return tuple(values)


def _chunk(papers: list[Paper], size: int) -> list[list[Paper]]:
    return [papers[i : i + size] for i in range(0, len(papers), size)]


def _run_round(batches, interests, model, api_key, base_url) -> set[str]:
    results = asyncio.run(_score_batches(batches, interests, model, api_key, base_url))
    scored_ids: set[str] = set()
    for batch, entries in results:
        by_id = {p.id: p for p in batch}
        for entry in entries:
            paper = by_id.get(str(entry.get("paper_id", "")))
            scores = coerce_scores(entry)
            if paper is None or scores is None:
                continue
            paper.score_relevance, paper.score_novelty, paper.score_quality = scores
            paper.relevance_score = sum(scores)
            paper.primary_category = str(entry.get("primary_category") or "")
            scored_ids.add(paper.id)
    return scored_ids


async def _score_batches(batches, interests, model, api_key, base_url):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def one(batch, idx):
        async with semaphore:
            prompt = build_scoring_prompt(batch, interests)
            for attempt in range(PARSE_ATTEMPTS):
                text = await call_llm_async(prompt, model=model, api_key=api_key,
                                            base_url=base_url, system=SYSTEM_MESSAGE)
                entries = parse_json_array(text)
                if entries:
                    return batch, entries
                log.warning("Stage A batch %d/%d unparseable (attempt %d/%d)",
                            idx + 1, len(batches), attempt + 1, PARSE_ATTEMPTS)
            return batch, []

    return await asyncio.gather(*(one(b, i) for i, b in enumerate(batches)))


def build_scoring_prompt(papers: list[Paper], interests: list[dict]) -> str:
    interest_desc = "\n".join(f"- **{i['name']}**: {i.get('description', '')}" for i in interests)
    names = ", ".join(f'"{i["name"]}"' for i in interests)
    entries = "\n---\n".join(_paper_block(p) for p in papers)
    return f"""Score each paper for a researcher with these interest areas:

{interest_desc}

Rubric (integers; total = relevance + novelty + quality, max 100):
- relevance (0-40): 36-40 a core problem of one interest area; 28-35 clearly relevant;
  16-27 tangential, or a relevant method applied elsewhere; 0-15 weak or unrelated.
- novelty (0-30): 26-30 new problem formulation or paradigm; 18-25 a clearly new method
  component; 10-17 incremental combination of known parts; 0-9 minor variant or engineering report.
- quality (0-30): strength of evidence — breadth and rigor of experiments and comparisons,
  released code, scale, credible concrete results stated in the conclusion, and the
  institutions' track record in this area.
  Do not penalize a paper for missing conclusion, affiliations or code link: some papers
  have no HTML full text. Judge from what is given; with only an abstract, use the abstract.

primary_category must be exactly one of: {names}, or "other" if none fits.

Return ONLY a JSON array, one object per paper, no other text:
[{{"paper_id": "<ID>", "relevance": <int>, "novelty": <int>, "quality": <int>, "primary_category": "<name>"}}]

Papers:

{entries}"""


def _paper_block(p: Paper) -> str:
    enriched = p.authors_enriched or []
    names = [a["name"] for a in enriched][:MAX_AUTHORS_IN_PROMPT] or p.authors[:MAX_AUTHORS_IN_PROMPT]
    affils = list(dict.fromkeys(a["affiliation"] for a in enriched if a.get("affiliation")))
    lines = [f"ID: {p.id}", f"Title: {p.title}"]
    if names:
        lines.append(f"Authors: {', '.join(names)}")
    if affils:
        lines.append(f"Affiliations: {'; '.join(affils[:MAX_AUTHORS_IN_PROMPT])}")
    lines.append(f"Code: {p.code_url or 'not found'}")
    lines.append(f"Abstract: {p.abstract}")
    lines.append(f"Conclusion: {p.conclusion}" if p.conclusion else "Conclusion: (full text unavailable)")
    return "\n".join(lines) + "\n"
