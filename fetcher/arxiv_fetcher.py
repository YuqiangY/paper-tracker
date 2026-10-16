from __future__ import annotations
import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from models import Paper
from .retry import request_with_retry

log = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"

PAGE_SIZE = 200
MAX_PAGES = 30
EMPTY_PAGE_RETRIES = 3
REQUEST_DELAY_S = 3.0


@dataclass(frozen=True)
class CategoryStats:
    count: int
    complete: bool
    note: str = ""


def fetch_arxiv(
    categories: list[str],
    since_date: str,
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
) -> tuple[list[Paper], dict[str, CategoryStats]]:
    seen: dict[str, Paper] = {}
    stats: dict[str, CategoryStats] = {}
    for cat in dict.fromkeys(categories):
        try:
            papers, stats[cat] = _fetch_category(cat, since_date, page_size, max_pages)
        except Exception as e:  # noqa: BLE001 - one category must not abort the rest
            log.warning("arXiv category %s failed: %s", cat, e)
            stats[cat] = CategoryStats(count=0, complete=False, note=f"error: {e}")
            continue
        for p in papers:
            seen.setdefault(p.id, p)
        log.info("arXiv %s: %d papers since %s (complete=%s%s)", cat, stats[cat].count,
                 since_date, stats[cat].complete, f", {stats[cat].note}" if stats[cat].note else "")
    return list(seen.values()), stats


def _fetch_category(
    category: str, since_date: str, page_size: int, max_pages: int,
) -> tuple[list[Paper], CategoryStats]:
    kept: list[Paper] = []
    start = 0
    for _ in range(max_pages):
        entries, total = _fetch_page_with_retry(category, start, page_size)
        if not entries:
            # Past the reported total is a normal end; anything else is an API glitch.
            if 0 < total <= start:
                return kept, CategoryStats(len(kept), True)
            return kept, CategoryStats(len(kept), False, f"empty page at start={start} (total={total})")
        kept.extend(p for p in entries if p.published >= since_date)
        start += len(entries)
        if entries[-1].published < since_date or start >= total:
            return kept, CategoryStats(len(kept), True)
    return kept, CategoryStats(len(kept), False, f"hit max_pages={max_pages}")


def _fetch_page_with_retry(category: str, start: int, page_size: int) -> tuple[list[Paper], int]:
    """arXiv intermittently serves an empty feed; retry before trusting it."""
    entries: list[Paper] = []
    total = 0
    for attempt in range(1 + EMPTY_PAGE_RETRIES):
        time.sleep(REQUEST_DELAY_S * (attempt + 1))
        xml_text = _query_arxiv_api(category, start, page_size)
        entries, total = _parse_arxiv_response(xml_text), _parse_total_results(xml_text)
        if entries:
            return entries, total
        log.warning("arXiv %s returned empty page at start=%d (total=%d), attempt %d",
                    category, start, total, attempt + 1)
    return entries, total


def _parse_total_results(xml_text: str) -> int:
    root = ET.fromstring(xml_text)
    value = root.findtext(f"{{{OPENSEARCH_NS}}}totalResults", "0")
    try:
        return int(value.strip())
    except ValueError:
        return 0


def _query_arxiv_api(category: str, start: int, max_results: int) -> str:
    params = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    data = request_with_retry(
        f"{ARXIV_API_URL}?{params}",
        timeout=60,
        max_attempts=5,
        base_delay=15.0,
        max_delay=180.0,
    )
    return data.decode("utf-8")


def _parse_arxiv_response(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers = []

    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        raw_id = entry.findtext(f"{{{ATOM_NS}}}id", "")
        paper_id = re.sub(r"v\d+$", "", raw_id.split("/abs/")[-1])

        title = entry.findtext(f"{{{ATOM_NS}}}title", "").strip()
        title = re.sub(r"\s+", " ", title)

        abstract = entry.findtext(f"{{{ATOM_NS}}}summary", "").strip()
        abstract = re.sub(r"\s+", " ", abstract)

        authors = []
        authors_enriched = []
        for a in entry.findall(f"{{{ATOM_NS}}}author"):
            name_el = a.find(f"{{{ATOM_NS}}}name")
            if name_el is None or not name_el.text:
                continue
            author_name = name_el.text.strip()
            authors.append(author_name)
            affil_el = a.find(f"{{{ARXIV_NS}}}affiliation")
            affiliation = affil_el.text.strip() if affil_el is not None and affil_el.text else None
            authors_enriched.append({
                "name": author_name,
                "affiliation": affiliation,
                "h_index": None,
                "semantic_scholar_id": None,
            })

        published_raw = entry.findtext(f"{{{ATOM_NS}}}published", "")
        published = published_raw[:10] if published_raw else ""

        url = ""
        pdf_url = None
        for link in entry.findall(f"{{{ATOM_NS}}}link"):
            href = link.get("href", "")
            if link.get("rel") == "alternate":
                url = href
            elif link.get("title") == "pdf":
                pdf_url = href

        categories = [
            c.get("term", "")
            for c in entry.findall(f"{{{ATOM_NS}}}category")
            if c.get("term")
        ]

        papers.append(Paper(
            id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            source="arxiv",
            published=published,
            categories=categories,
            pdf_url=pdf_url,
            authors_enriched=authors_enriched,
        ))

    return papers
