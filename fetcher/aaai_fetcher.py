"""Fetch accepted papers from AAAI proceedings on OJS (ojs.aaai.org).

Three-level structure:
  archive page  → AAAI-{yy} Technical Track issues
  issue page    → article/view/{id} links
  article page  → Highwire meta (title/author/pdf/date) + abstract in body

Results are cached on disk so re-runs are near-free.
"""
from __future__ import annotations
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup

from models import Paper
from .cache import DiskCache
from .retry import request_with_retry

log = logging.getLogger(__name__)

BASE_URL = "https://ojs.aaai.org/index.php/AAAI"
ARCHIVE_URL = f"{BASE_URL}/issue/archive"
CACHE_TTL_SECONDS = 30 * 86400
DEFAULT_REQUEST_DELAY = 0.2


def fetch_aaai(
    year: int = 2026,
    timeout: int = 15,
    concurrency: int = 4,
    limit: int | None = None,
    cache_dir: str | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> list[Paper]:
    """Fetch AAAI ``year`` accepted papers (Technical Tracks) with abstracts."""
    cache = DiskCache(cache_dir or _default_cache_dir(), ttl_seconds=CACHE_TTL_SECONDS)

    log.info("Fetching AAAI%d archive: %s", year, ARCHIVE_URL)
    archive_html = _get_html(ARCHIVE_URL, cache, timeout)
    issue_ids = _parse_archive(archive_html, year)
    log.info("Found %d AAAI-%d Technical Track issues", len(issue_ids), year % 100)

    # Collect article URLs across all issues
    article_urls: list[str] = []
    seen: set[str] = set()
    for issue_id in issue_ids:
        issue_html = _get_html(f"{BASE_URL}/issue/view/{issue_id}", cache, timeout)
        for url in _parse_issue(issue_html):
            if url not in seen:
                seen.add(url)
                article_urls.append(url)
    log.info("Collected %d article URLs", len(article_urls))

    if limit is not None:
        article_urls = article_urls[:limit]
        log.info("Limited to %d articles", len(article_urls))

    papers = _fetch_articles(article_urls, year, cache, timeout, concurrency, request_delay)
    log.info("AAAI%d: built %d papers (%d with abstract)",
             year, len(papers), sum(1 for p in papers if p.abstract))
    return papers


# ---------------------------------------------------------------------------
# Pure parsing helpers (offline-testable)
# ---------------------------------------------------------------------------

def _parse_archive(html: str, year: int) -> list[str]:
    """Return issue ids for ``AAAI-{yy}`` Technical Track issues."""
    soup = BeautifulSoup(html, "html.parser")
    yy = f"AAAI-{year % 100}"
    ids: list[str] = []
    seen: set[str] = set()
    for a in soup.select('a[href*="issue/view/"]'):
        text = a.get_text(strip=True)
        if yy in text and "Technical Track" in text:
            issue_id = a["href"].rstrip("/").split("/")[-1]
            if issue_id not in seen:
                seen.add(issue_id)
                ids.append(issue_id)
    return ids


def _parse_issue(html: str) -> list[str]:
    """Return absolute article URLs from an issue's table of contents."""
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    for a in soup.select('a[href*="article/view/"]'):
        href = a["href"]
        # Normalize: keep only .../article/view/{id}, drop /download etc.
        m = re.search(r"(https?://[^\s\"']*?/article/view/\d+)", href)
        url = m.group(1) if m else href
        if "/article/view/" in url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _parse_article(html: str) -> dict:
    """Parse an article page into a record dict.

    Uses Highwire ``citation_*`` meta tags plus the abstract in the page body.
    """
    soup = BeautifulSoup(html, "html.parser")

    def meta(name: str) -> str:
        tag = soup.find("meta", attrs={"name": name})
        return (tag.get("content") or "").strip() if tag else ""

    def meta_all(name: str) -> list[str]:
        return [
            (t.get("content") or "").strip()
            for t in soup.find_all("meta", attrs={"name": name})
            if t.get("content")
        ]

    abstract_node = soup.find("section", class_="item abstract") or soup.find(class_="abstract")
    abstract = ""
    if abstract_node:
        # Drop the leading "Abstract" heading text if present
        abstract = abstract_node.get_text(" ", strip=True)
        abstract = re.sub(r"^Abstract\s*", "", abstract, count=1)

    return {
        "title": meta("citation_title"),
        "authors": meta_all("citation_author"),
        "abstract": abstract,
        "pdf_url": meta("citation_pdf_url") or None,
        "published": (meta("DC.Date.issued") or meta("citation_publication_date"))[:10],
        "doi": meta("citation_doi") or None,
    }


def _article_id(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _fetch_articles(
    urls: list[str],
    year: int,
    cache: DiskCache,
    timeout: int,
    concurrency: int,
    request_delay: float,
) -> list[Paper]:
    papers: list[Paper] = []

    def _one(url: str) -> Paper | None:
        try:
            html = _get_html(url, cache, timeout, request_delay)
            rec = _parse_article(html)
        except Exception as e:  # noqa: BLE001 - one bad page shouldn't abort the run
            log.warning("Failed to fetch AAAI article %s: %s", url, e)
            return None
        if not rec["title"]:
            return None
        return _build_paper(rec, url, year)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(_one, u): u for u in urls}
        for i, fut in enumerate(as_completed(futures), 1):
            paper = fut.result()
            if paper is not None:
                papers.append(paper)
            if i % 200 == 0:
                log.info("Fetched %d/%d articles", i, len(urls))

    return papers


def _build_paper(rec: dict, url: str, year: int) -> Paper:
    return Paper(
        id=f"aaai{year}:{_article_id(url)}",
        title=rec["title"],
        authors=rec["authors"],
        abstract=rec["abstract"],
        url=url,
        source="aaai",
        published=rec["published"] or f"{year}-01-01",
        categories=["cs.AI"],
        pdf_url=rec.get("pdf_url"),
        venue=f"AAAI {year}",
        doi=rec.get("doi"),
    )


def _get_html(url: str, cache: DiskCache, timeout: int, request_delay: float = 0.0) -> str:
    cached = cache.get(url)
    if cached is not None:
        return cached
    raw = request_with_retry(url, timeout=timeout, headers={"Accept-Encoding": "gzip"})
    html = _decode(raw)
    cache.set(url, html)
    if request_delay > 0:
        import time
        time.sleep(request_delay)
    return html


def _decode(raw: bytes) -> str:
    """Decode response bytes, transparently gunzipping if needed."""
    if raw[:2] == b"\x1f\x8b":  # gzip magic
        import gzip
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def _default_cache_dir() -> str:
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, ".cache", "aaai")
