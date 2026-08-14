"""Fetch ECCV low-level-vision papers from the community-curated GitHub repo
Kobaayyy/Awesome-ECCV2026...-Low-Level-Vision (ECCV{year}.md).

The markdown groups papers by sub-task (super-resolution, deraining, ...), each
as a ``### {title}`` block with an optional ``- Paper: <arxiv>`` link. We parse
titles + arxiv ids + section category, then batch-fetch abstracts from the arXiv
API for papers that have an arxiv link (reusing arxiv_fetcher's XML parser).

Note: this is a curated subset (ECCV 2026 acceptances aren't fully public yet),
so coverage is partial by design — logged at fetch time.
"""
from __future__ import annotations
import logging
import re
import urllib.parse

from models import Paper
from .cache import DiskCache
from .retry import request_with_retry
from .arxiv_fetcher import _parse_arxiv_response

log = logging.getLogger(__name__)

RAW_MD_URL = (
    "https://raw.githubusercontent.com/Kobaayyy/"
    "Awesome-ECCV2026-ECCV2024-ECCV2020-Low-Level-Vision/master/ECCV{year}.md"
)
ARXIV_API_URL = "http://export.arxiv.org/api/query"
CACHE_TTL_SECONDS = 7 * 86400  # curated list updates over time; refresh weekly
ARXIV_BATCH = 50


def fetch_eccv(
    year: int = 2026,
    timeout: int = 30,
    concurrency: int = 4,  # accepted for interface uniformity; arXiv is batched, not threaded
    limit: int | None = None,
    cache_dir: str | None = None,
) -> list[Paper]:
    """Fetch ECCV ``year`` low-level-vision papers from the curated repo."""
    cache = DiskCache(cache_dir or _default_cache_dir(), ttl_seconds=CACHE_TTL_SECONDS)

    md_url = RAW_MD_URL.format(year=year)
    log.info("Fetching ECCV%d curated list: %s", year, md_url)
    md = _get_text(md_url, cache, timeout)
    records = _parse_markdown(md)
    log.info("Parsed %d paper entries (%d with arXiv links)",
             len(records), sum(1 for r in records if r["arxiv_id"]))

    if limit is not None:
        records = records[:limit]

    # Batch-fetch abstracts for arxiv-linked papers
    abstracts = _fetch_arxiv_abstracts(
        [r["arxiv_id"] for r in records if r["arxiv_id"]], cache, timeout,
    )

    papers: list[Paper] = []
    for r in records:
        papers.append(_build_paper(r, year, abstracts.get(r["arxiv_id"] or "")))
    log.info("ECCV%d: built %d papers (%d with abstract)",
             year, len(papers), sum(1 for p in papers if p.abstract))
    return papers


# ---------------------------------------------------------------------------
# Pure parsing helpers (offline-testable)
# ---------------------------------------------------------------------------

def _parse_markdown(md: str) -> list[dict]:
    """Parse ECCV{year}.md into records: {title, arxiv_id, code_url, category}.

    Sections are ``# N.名称(English)``; papers are ``### {title}`` blocks whose
    following lines may contain ``- Paper: https://arxiv.org/abs/{id}`` and/or
    ``- Code: https://github.com/...``.
    """
    records: list[dict] = []
    category = ""
    cur: dict | None = None

    for line in md.splitlines():
        sec = re.match(r"^#+\s*\d+\.\s*(.+)$", line)
        if sec:
            category = _clean_category(sec.group(1))
            cur = None
            continue

        if line.startswith("### "):
            title = line[4:].strip()
            if title.lower() in ("xxx", "tbd", ""):
                cur = None
                continue
            cur = {"title": title, "arxiv_id": None, "code_url": None, "category": category}
            records.append(cur)
            continue

        if cur is not None:
            m = re.search(r"arxiv\.org/abs/(\d+\.\d+)", line)
            if m:
                cur["arxiv_id"] = m.group(1)
            c = re.search(r"Code:\s*(https?://\S+)", line)
            if c:
                cur["code_url"] = c.group(1).rstrip(")")

    return records


def _clean_category(raw: str) -> str:
    """`超分辨率(Super-Resolution)` -> `超分辨率`; strip anchors/links."""
    raw = re.split(r"[\(（]", raw, maxsplit=1)[0].strip()
    return raw


def _build_paper(rec: dict, year: int, arxiv_paper: Paper | None) -> Paper:
    arxiv_id = rec["arxiv_id"]
    code_url = rec.get("code_url")
    if arxiv_paper is not None:
        # Reuse arXiv-provided abstract/authors/pdf but keep ECCV identity.
        abstract = arxiv_paper.abstract
        authors = arxiv_paper.authors
        pdf_url = arxiv_paper.pdf_url
        url = arxiv_paper.url
    else:
        abstract = ""
        authors = []
        pdf_url = None
        # No abstract page: link to arXiv if we have an id, else the code repo.
        if arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        else:
            url = code_url or ""

    paper_id = f"eccv{year}:{arxiv_id}" if arxiv_id else f"eccv{year}:{_slug(rec['title'])}"
    return Paper(
        id=paper_id,
        title=rec["title"],
        authors=authors,
        abstract=abstract,
        url=url,
        source="eccv",
        published=f"{year}-01-01",
        categories=["cs.CV"],
        pdf_url=pdf_url,
        venue=f"ECCV {year}",
    )


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _fetch_arxiv_abstracts(arxiv_ids: list[str], cache: DiskCache, timeout: int) -> dict[str, Paper]:
    """Batch-fetch arXiv metadata by id_list; return {arxiv_id: Paper}."""
    result: dict[str, Paper] = {}
    ids = [i for i in arxiv_ids if i]
    for start in range(0, len(ids), ARXIV_BATCH):
        batch = ids[start:start + ARXIV_BATCH]
        key = "arxiv:" + ",".join(batch)
        cached = cache.get(key)
        if cached is not None:
            xml = cached
        else:
            params = urllib.parse.urlencode({"id_list": ",".join(batch), "max_results": len(batch)})
            try:
                raw = request_with_retry(f"{ARXIV_API_URL}?{params}", timeout=timeout,
                                         max_attempts=5, base_delay=15.0, max_delay=180.0)
                xml = raw.decode("utf-8")
                cache.set(key, xml)
            except Exception as e:  # noqa: BLE001
                log.warning("arXiv batch fetch failed (%d ids): %s", len(batch), e)
                continue
        for paper in _parse_arxiv_response(xml):
            result[paper.id] = paper
    return result


def _get_text(url: str, cache: DiskCache, timeout: int) -> str:
    cached = cache.get(url)
    if cached is not None:
        return cached
    raw = request_with_retry(url, timeout=timeout)
    text = raw.decode("utf-8", errors="replace")
    cache.set(url, text)
    return text


def _default_cache_dir() -> str:
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, ".cache", "eccv")
