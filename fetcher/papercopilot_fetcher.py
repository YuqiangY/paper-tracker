"""Fetch award / highlight papers from Paper Copilot's open data
(github.com/papercopilot/paperlists), which annotates each paper with a
``status`` field (e.g. "Award Candidate", "Highlight", "Oral", "Poster").

Used to surface a conference's best-paper / award-candidate "final list" as a
pinned section, independent of the interest filter. Abstracts are inline in the
JSON (no extra arXiv fetch needed).

Note: data for a just-concluded conference may not be published yet — a missing
file returns [] gracefully rather than erroring.
"""
from __future__ import annotations
import json
import logging
import re

from models import Paper
from .cache import DiskCache
from .retry import request_with_retry

log = logging.getLogger(__name__)

RAW_URL = "https://raw.githubusercontent.com/papercopilot/paperlists/main/{conf}/{conf}{year}.json"
# Some paperlists files (e.g. iclr) are git-LFS backed: the raw.githubusercontent
# host returns a small pointer instead of the JSON. The real bytes live behind
# the media host, which serves the LFS object directly.
MEDIA_URL = "https://media.githubusercontent.com/media/papercopilot/paperlists/main/{conf}/{conf}{year}.json"
LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"
CACHE_TTL_SECONDS = 7 * 86400
DEFAULT_STATUSES = ["Award Candidate", "Best Paper", "Best Student Paper"]


def fetch_papercopilot(
    conf: str,
    year: int = 2026,
    statuses: list[str] | None = None,
    timeout: int = 90,
    cache_dir: str | None = None,
) -> list[Paper]:
    """Fetch papers of the given ``statuses`` for ``conf`` ``year``.

    Args:
        conf: Paper Copilot conference key (e.g. "cvpr", "aaai", "iclr").
        year: Conference year.
        statuses: Status values to keep (default: award-level statuses).
            Pass e.g. ["Award Candidate", "Highlight"] to widen.
        timeout: Per-request timeout (files are several MB).
    """
    wanted = statuses or DEFAULT_STATUSES
    cache = DiskCache(cache_dir or _default_cache_dir(), ttl_seconds=CACHE_TTL_SECONDS)

    url = RAW_URL.format(conf=conf, year=year)
    media_url = MEDIA_URL.format(conf=conf, year=year)
    raw = _get_json(url, cache, timeout, media_url=media_url)
    if raw is None:
        log.warning("Paper Copilot data not available for %s%d (not published yet?)", conf, year)
        return []

    records = _parse(raw, wanted, conf, year)
    log.info("Paper Copilot %s%d: %d papers matching statuses %s",
             conf, year, len(records), wanted)
    return records


# ---------------------------------------------------------------------------
# Pure parsing (offline-testable)
# ---------------------------------------------------------------------------

def _parse(data: list[dict], statuses: list[str], conf: str, year: int) -> list[Paper]:
    wanted = set(statuses)
    papers: list[Paper] = []
    for item in data:
        status = (item.get("status") or "").strip()
        if status not in wanted:
            continue
        paper = _build_paper(item, status, conf, year)
        if paper is not None:
            papers.append(paper)
    return papers


def _build_paper(item: dict, status: str, conf: str, year: int) -> Paper | None:
    title = (item.get("title") or "").strip()
    if not title:
        return None

    authors = [a.strip() for a in (item.get("author") or "").split(";") if a.strip()]
    abstract = (item.get("abstract") or "").strip()
    arxiv = (item.get("arxiv") or "").strip()

    # url priority: OpenReview/OA page > conference site > arxiv abstract page
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
        source="award",
        published=f"{year}-01-01",
        categories=[conf.upper()],
        pdf_url=pdf_url,
        primary_category=f"🏆 {status}",
        venue=f"{conf.upper()} {year}",
    )


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _get_json(url: str, cache: DiskCache, timeout: int, media_url: str | None = None):
    """Fetch + parse a Paper Copilot JSON, with git-LFS pointer fallback.

    If ``url`` (raw host) returns an LFS pointer instead of JSON, re-fetch from
    ``media_url`` (media host), which serves the LFS object directly. Results are
    cached under the original ``url`` key so the caller need not know which host won.
    """
    cached = cache.get(url)
    if cached is not None:
        return cached

    text = _fetch_text(url, timeout)
    if text is None:
        return None
    if text.lstrip().startswith(LFS_POINTER_PREFIX):
        if not media_url:
            log.warning("Paper Copilot %s is an LFS pointer but no media URL given", url)
            return None
        log.info("Paper Copilot %s is LFS-backed; fetching via media host", url)
        text = _fetch_text(media_url, timeout)
        if text is None:
            return None

    if text.strip().startswith("404"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("Paper Copilot JSON parse failed for %s: %s", url, e)
        return None
    cache.set(url, data)
    return data


def _fetch_text(url: str, timeout: int) -> str | None:
    try:
        raw = request_with_retry(url, timeout=timeout)
    except Exception as e:  # noqa: BLE001 - 404 / network → treat as unavailable
        log.debug("Paper Copilot fetch failed for %s: %s", url, e)
        return None
    return raw.decode("utf-8", errors="replace")


def _default_cache_dir() -> str:
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, ".cache", "papercopilot")
