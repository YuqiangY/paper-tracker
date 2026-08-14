"""Combined award-paper fetcher: Paper Copilot first (rich, has status +
abstracts), falling back to the FeijiangHan markdown list when Paper Copilot
has no data for that conference/year yet. Results are merged and de-duplicated
by normalized title.
"""
from __future__ import annotations
import logging
import re

from models import Paper
from .papercopilot_fetcher import fetch_papercopilot
from .best_paper_md_fetcher import fetch_best_papers_md

log = logging.getLogger(__name__)


def fetch_awards(
    conf: str,
    year: int = 2026,
    statuses: list[str] | None = None,
) -> list[Paper]:
    """Fetch award / candidate papers for ``conf`` ``year`` from both sources."""
    pc = fetch_papercopilot(conf, year, statuses)
    md = fetch_best_papers_md(conf, year)

    merged: list[Paper] = []
    seen: set[str] = set()
    # Paper Copilot first (richer records), then markdown-only extras.
    for paper in pc + md:
        key = _norm(paper.title)
        if key in seen:
            continue
        seen.add(key)
        merged.append(paper)

    log.info("Awards %s%d: %d total (%d paper-copilot, %d markdown, deduped)",
             conf, year, len(merged), len(pc), len(md))
    return merged


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())
