from __future__ import annotations
import logging
import time
import urllib.error
from models import Paper
from .author_enrichment import _s2_get

log = logging.getLogger(__name__)

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_SEARCH_FIELDS = "title,authors,abstract,year,externalIds,publicationDate,venue,citationCount,tldr"

CIRCUIT_BREAKER_THRESHOLD = 3


def fetch_s2_search(
    queries: list[str],
    year: str | None = None,
    limit_per_query: int = 20,
    api_key: str | None = None,
    timeout: int = 10,
) -> list[Paper]:
    if not queries:
        return []

    seen: dict[str, Paper] = {}
    consecutive_429 = 0

    for query in queries:
        if consecutive_429 >= CIRCUIT_BREAKER_THRESHOLD:
            log.warning(
                "S2 circuit breaker tripped after %d consecutive 429s, skipping remaining %d queries",
                consecutive_429, len(queries) - len(seen),
            )
            break

        papers, was_429 = _search_one_query(query, year, limit_per_query, api_key, timeout)
        if was_429:
            consecutive_429 += 1
        else:
            consecutive_429 = 0
        for p in papers:
            if p.id not in seen:
                seen[p.id] = p
        time.sleep(3.5)

    log.info("S2 search: %d unique papers from %d queries", len(seen), len(queries))
    return list(seen.values())


def _search_one_query(
    query: str,
    year: str | None,
    limit: int,
    api_key: str | None,
    timeout: int,
) -> tuple[list[Paper], bool]:
    """Returns (papers, was_429)."""
    params = {
        "query": query,
        "fields": S2_SEARCH_FIELDS,
        "limit": str(limit),
    }
    if year:
        params["year"] = year

    try:
        data = _s2_get(S2_SEARCH_URL, params, api_key, timeout)
    except urllib.error.HTTPError as e:
        is_429 = e.code == 429
        log.warning("S2 search failed for query '%s': %s", query, e)
        return [], is_429
    except Exception as e:
        log.warning("S2 search failed for query '%s': %s", query, e)
        return [], False

    if not data or "data" not in data:
        return [], False

    return _parse_s2_papers(data["data"]), False


def _parse_s2_papers(items: list[dict]) -> list[Paper]:
    papers = []
    for item in items:
        if not item:
            continue

        external_ids = item.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv")
        s2_paper_id = item.get("paperId", "")

        paper_id = arxiv_id if arxiv_id else f"s2:{s2_paper_id}"
        if not paper_id or paper_id == "s2:":
            continue

        title = (item.get("title") or "").strip()
        abstract = (item.get("abstract") or "").strip()
        if not title:
            continue

        authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
        published = (item.get("publicationDate") or "")[:10]
        year_val = item.get("year")

        url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"https://www.semanticscholar.org/paper/{s2_paper_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None

        venue = (item.get("venue") or "").strip() or None
        citation_count = item.get("citationCount")
        tldr_obj = item.get("tldr")
        tldr = tldr_obj.get("text") if isinstance(tldr_obj, dict) else None
        doi = external_ids.get("DOI")

        papers.append(Paper(
            id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            source="s2_search",
            published=published or (str(year_val) if year_val else ""),
            categories=[],
            pdf_url=pdf_url,
            venue=venue,
            citation_count=citation_count,
            tldr=tldr,
            doi=doi,
        ))

    return papers
