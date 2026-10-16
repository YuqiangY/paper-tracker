"""Facade kept for conference_pipeline.py: Stage A scoring -> threshold -> Stage B summaries."""
from __future__ import annotations
import logging

from models import Paper
from .llm_scoring import score_papers
from .llm_summary import summarize_papers
from .selection import normalize_categories

log = logging.getLogger(__name__)

SCORE_BATCH_SIZE = 50


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
    outcome = score_papers(papers, interests, model=model, api_key=api_key,
                           base_url=base_url, batch_size=SCORE_BATCH_SIZE)
    normalize_categories(outcome.scored, [i["name"] for i in interests])
    passed = [p for p in outcome.scored if (p.relevance_score or 0) >= threshold]
    summarize_papers(passed, interests, model=model, api_key=api_key,
                     base_url=base_url, batch_size=batch_size)
    return passed
