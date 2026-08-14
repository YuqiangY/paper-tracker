"""Fetch a conference's FULL accepted-paper list from Paper Copilot's open data
(github.com/papercopilot/paperlists), for conferences whose proceedings are only
published there (e.g. ICML/ICLR, sourced from OpenReview).

Unlike ``papercopilot_fetcher`` (which keeps only award/highlight statuses and
pins them as a 🏆 section), this returns EVERY accepted paper as an ordinary
``Paper`` so the interest filter + LLM scoring run over the whole conference.

Two conference-shape differences are handled here:
  * OpenReview dumps (ICLR) include rejected/withdrawn submissions — an
    ``accept_statuses`` whitelist keeps only the accepted ones.
  * ICLR's JSON is git-LFS backed; the shared ``_get_json`` transparently falls
    back to the media host (see papercopilot_fetcher).
"""
from __future__ import annotations
import logging

from models import Paper
from .cache import DiskCache
from .papercopilot_fetcher import (
    RAW_URL,
    MEDIA_URL,
    CACHE_TTL_SECONDS,
    _get_json,
    _slug,
    _default_cache_dir,
)

log = logging.getLogger(__name__)


def fetch_papercopilot_full(
    conf: str,
    year: int = 2026,
    timeout: int = 180,
    concurrency: int = 4,  # accepted for interface uniformity; one JSON fetch, no threading
    limit: int | None = None,
    accept_statuses: list[str] | None = None,
    cache_dir: str | None = None,
) -> list[Paper]:
    """Fetch the full accepted-paper list for ``conf`` ``year``.

    Args:
        conf: Paper Copilot conference key (e.g. "icml", "iclr").
        year: Conference year.
        timeout: Per-request timeout (files are tens of MB; LFS files ~90MB).
        accept_statuses: If given, keep only papers whose ``status`` matches one
            of these (case-insensitive substring, so "Poster" also matches
            "ICLR 2026 ConditionalPoster"). If None, keep every record — use for
            conferences whose dump contains only accepted papers (e.g. ICML).
        limit: Keep only the first N papers (smoke testing).
    """
    cache = DiskCache(cache_dir or _default_cache_dir(), ttl_seconds=CACHE_TTL_SECONDS)

    url = RAW_URL.format(conf=conf, year=year)
    media_url = MEDIA_URL.format(conf=conf, year=year)
    raw = _get_json(url, cache, timeout, media_url=media_url)
    if raw is None:
        log.warning("Paper Copilot data not available for %s%d (not published yet?)", conf, year)
        return []

    papers = _parse_full(raw, conf, year, accept_statuses)
    log.info("Paper Copilot %s%d: %d accepted papers (of %d records, statuses=%s)",
             conf, year, len(papers), len(raw), accept_statuses or "ALL")

    if limit is not None:
        papers = papers[:limit]
        log.info("Limited to %d papers", len(papers))
    return papers


# ---------------------------------------------------------------------------
# Pure parsing (offline-testable)
# ---------------------------------------------------------------------------

def _parse_full(
    data: list[dict], conf: str, year: int, accept_statuses: list[str] | None,
) -> list[Paper]:
    wanted = [s.lower() for s in accept_statuses] if accept_statuses else None
    papers: list[Paper] = []
    for item in data:
        if wanted is not None and not _status_accepted(item.get("status"), wanted):
            continue
        paper = _build_full_paper(item, conf, year)
        if paper is not None:
            papers.append(paper)
    return papers


def _status_accepted(status: str | None, wanted: list[str]) -> bool:
    """True if ``status`` matches any wanted value (case-insensitive substring).

    Substring so "Poster"/"Oral" also match OpenReview's decorated variants like
    "ICLR 2026 ConditionalPoster". ``wanted`` is pre-lowercased.
    """
    s = (status or "").lower()
    if not s:
        return False
    return any(w in s for w in wanted)


def _build_full_paper(item: dict, conf: str, year: int) -> Paper | None:
    title = (item.get("title") or "").strip()
    if not title:
        return None

    authors = [a.strip() for a in (item.get("author") or "").split(";") if a.strip()]
    abstract = (item.get("abstract") or "").strip()
    arxiv = (item.get("arxiv") or "").strip()

    # url priority: OA page > conference/OpenReview site > arxiv abstract page
    url = (item.get("oa") or item.get("site") or item.get("openreview") or "").strip()
    if not url and arxiv:
        url = f"https://arxiv.org/abs/{arxiv}"

    pdf_url = (item.get("pdf") or "").strip() or None
    ident = arxiv or _slug(title)

    return Paper(
        id=f"{conf}{year}:pc:{ident}",
        title=title,
        authors=authors,
        abstract=abstract,
        url=url,
        source=conf,
        published=f"{year}-01-01",
        categories=["cs.LG"],
        pdf_url=pdf_url,
        primary_category=None,  # left for the LLM filter to assign an interest area
        venue=f"{conf.upper()} {year}",
    )
