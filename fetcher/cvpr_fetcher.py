"""Fetch accepted papers from the CVF Open Access repository (e.g. CVPR 2026).

Two-step: parse the ``?day=all`` listing page for title/authors/links, then
concurrently fetch each paper's detail page for its abstract. Results are cached
on disk so re-runs are near-free.
"""
from __future__ import annotations
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup

from models import Paper
from .cache import DiskCache
from .retry import request_with_retry

log = logging.getLogger(__name__)

BASE_URL = "https://openaccess.thecvf.com"
CACHE_TTL_SECONDS = 30 * 86400  # conference lists are static; cache a month
DEFAULT_REQUEST_DELAY = 0.2  # polite per-request throttle for detail pages


def fetch_cvpr(
    year: int = 2026,
    timeout: int = 15,
    concurrency: int = 4,
    limit: int | None = None,
    cache_dir: str | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> list[Paper]:
    """Fetch CVPR ``year`` accepted papers with abstracts.

    Args:
        year: Conference year (drives the CVPR{year} URL path).
        timeout: Per-request timeout in seconds.
        concurrency: Max concurrent detail-page fetches.
        limit: If set, only process the first N papers (for smoke tests).
        cache_dir: DiskCache directory; defaults to ``.cache/cvpr`` under repo.
        request_delay: Seconds to sleep after each detail-page fetch (polite
            throttle). Skipped for cache hits, which do no network I/O.
    """
    cache = DiskCache(cache_dir or _default_cache_dir(), ttl_seconds=CACHE_TTL_SECONDS)

    listing_url = f"{BASE_URL}/CVPR{year}?day=all"
    log.info("Fetching CVPR%d listing: %s", year, listing_url)
    listing_html = _get_html(listing_url, cache, timeout)
    records = _parse_listing(listing_html)
    log.info("Parsed %d paper records from listing", len(records))

    if limit is not None:
        records = records[:limit]
        log.info("Limited to %d records", len(records))

    papers = _fetch_abstracts(records, year, cache, timeout, concurrency, request_delay)
    log.info("CVPR%d: built %d papers (%d with abstract)",
             year, len(papers), sum(1 for p in papers if p.abstract))
    return papers


# ---------------------------------------------------------------------------
# Pure parsing helpers (offline-testable)
# ---------------------------------------------------------------------------

def _parse_listing(html: str) -> list[dict]:
    """Parse the ``?day=all`` listing into records.

    Each record: {title, authors, detail_url, pdf_url, slug}.
    Layout: ``dt.ptitle`` (title + detail link), then a ``dd`` with author
    forms, then a ``dd`` with PDF/supplemental links.
    """
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict] = []

    for dt in soup.select("dt.ptitle"):
        link = dt.find("a")
        if not link or not link.get("href"):
            continue
        title = link.get_text(strip=True)
        detail_path = link["href"]
        slug = _slug_from_path(detail_path)
        if not title or not slug:
            continue

        authors = _parse_authors_dd(dt.find_next_sibling("dd"))
        pdf_url = _find_pdf_url(dt)

        records.append({
            "title": title,
            "authors": authors,
            "detail_url": _abs_url(detail_path),
            "pdf_url": pdf_url,
            "slug": slug,
        })

    return records


def _parse_authors_dd(dd) -> list[str]:
    if dd is None:
        return []
    authors: list[str] = []
    for form in dd.find_all("form", class_="authsearch"):
        inp = form.find("input", {"name": "query_author"})
        if inp and inp.get("value"):
            authors.append(inp["value"].strip())
    return authors


def _find_pdf_url(dt) -> str | None:
    """The main paper PDF lives in the links ``dd`` two siblings down."""
    for sib in dt.find_next_siblings("dd"):
        for a in sib.find_all("a"):
            href = a.get("href") or ""
            if href.endswith("_paper.pdf") or "/papers/" in href:
                return _abs_url(href)
    return None


def _parse_abstract(html: str) -> str:
    """Extract the abstract text from a paper detail page (``div#abstract``)."""
    soup = BeautifulSoup(html, "html.parser")
    node = soup.find(id="abstract")
    return node.get_text(strip=True) if node else ""


def _slug_from_path(path: str) -> str:
    """Derive a stable slug from a detail/pdf path.

    ``/content/CVPR2026/html/Xiao_Foo_CVPR_2026_paper.html`` -> ``Xiao_Foo``.
    """
    name = path.rsplit("/", 1)[-1]
    name = re.sub(r"\.(html|pdf)$", "", name)
    name = re.sub(r"_CVPR_\d{4}_paper$", "", name)
    return name


def _abs_url(path: str) -> str:
    if path.startswith("http"):
        return path
    return f"{BASE_URL}{path}"


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _fetch_abstracts(
    records: list[dict],
    year: int,
    cache: DiskCache,
    timeout: int,
    concurrency: int,
    request_delay: float,
) -> list[Paper]:
    papers: list[Paper] = []

    def _one(rec: dict) -> Paper:
        try:
            detail_html = _get_html(rec["detail_url"], cache, timeout, request_delay)
            abstract = _parse_abstract(detail_html)
        except Exception as e:  # noqa: BLE001 - one bad page shouldn't abort the run
            log.warning("Failed to fetch abstract for %s: %s", rec["slug"], e)
            abstract = ""
        return _build_paper(rec, year, abstract)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(_one, rec): rec for rec in records}
        for i, fut in enumerate(as_completed(futures), 1):
            papers.append(fut.result())
            if i % 200 == 0:
                log.info("Fetched %d/%d abstracts", i, len(records))

    return papers


def _build_paper(rec: dict, year: int, abstract: str) -> Paper:
    return Paper(
        id=f"cvpr{year}:{rec['slug']}",
        title=rec["title"],
        authors=rec["authors"],
        abstract=abstract,
        url=rec["detail_url"],
        source="cvpr",
        published=f"{year}-01-01",  # placeholder; CVF lists no per-paper date
        categories=["cs.CV"],
        pdf_url=rec.get("pdf_url"),
        venue=f"CVPR {year}",
    )


def _get_html(url: str, cache: DiskCache, timeout: int, request_delay: float = 0.0) -> str:
    cached = cache.get(url)
    if cached is not None:
        return cached
    raw = request_with_retry(url, timeout=timeout)
    html = raw.decode("utf-8", errors="replace")
    cache.set(url, html)
    if request_delay > 0:
        time.sleep(request_delay)
    return html


def _default_cache_dir() -> str:
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, ".cache", "cvpr")
