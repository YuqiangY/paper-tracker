"""Fetch ACL accepted papers from the official ACL Anthology XML dump.

The Anthology publishes one XML file per collection (e.g. ``2026.acl.xml``) in
its GitHub repo — a clean, complete source with inline abstracts, so no HTML
scraping or per-paper detail fetch is needed. Each ``<volume>`` groups papers
(long/short/findings/industry/srw/demo/tutorials); we keep research volumes and
skip tutorials.

Title/abstract may contain inline markup (``<fixed-case>``, ``<i>`` …), so text
is extracted with ``itertext()`` rather than ``.text``.
"""
from __future__ import annotations
import logging
import os
import xml.etree.ElementTree as ET

from models import Paper
from .cache import DiskCache
from .retry import request_with_retry

log = logging.getLogger(__name__)

RAW_XML_URL = (
    "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/{year}.acl.xml"
)
CACHE_TTL_SECONDS = 30 * 86400  # proceedings are static; cache a month
# Research volumes worth surfacing; "tutorials" excluded (not papers).
DEFAULT_VOLUMES = ["long", "short", "findings", "industry", "srw", "demo"]
ANTHOLOGY_BASE = "https://aclanthology.org"


def fetch_acl(
    year: int = 2026,
    timeout: int = 60,
    concurrency: int = 4,  # accepted for interface uniformity; single XML fetch, no threading
    limit: int | None = None,
    volumes: list[str] | None = None,
    cache_dir: str | None = None,
) -> list[Paper]:
    """Fetch ACL ``year`` accepted papers from the Anthology XML dump.

    Args:
        year: Conference year (drives the ``{year}.acl.xml`` path).
        timeout: Per-request timeout (the XML is a few MB).
        volumes: Volume ids to keep (default: research volumes, excl. tutorials).
        limit: Keep only the first N papers (smoke testing).
    """
    wanted = set(volumes or DEFAULT_VOLUMES)
    cache = DiskCache(cache_dir or _default_cache_dir(), ttl_seconds=CACHE_TTL_SECONDS)

    url = RAW_XML_URL.format(year=year)
    xml_text = _get_xml(url, cache, timeout)
    if xml_text is None:
        log.warning("ACL Anthology XML not available for %d (not published yet?)", year)
        return []

    papers = _parse(xml_text, year, wanted)
    log.info("ACL %d: parsed %d papers from volumes %s", year, len(papers), sorted(wanted))

    if limit is not None:
        papers = papers[:limit]
        log.info("Limited to %d papers", len(papers))
    return papers


# ---------------------------------------------------------------------------
# Pure parsing (offline-testable)
# ---------------------------------------------------------------------------

def _parse(xml_text: str, year: int, wanted_volumes: set[str]) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for volume in root.findall("volume"):
        if volume.get("id") not in wanted_volumes:
            continue
        for paper_el in volume.findall("paper"):
            paper = _build_paper(paper_el, year)
            if paper is not None:
                papers.append(paper)
    return papers


def _itertext(el) -> str:
    """Join all descendant text, flattening inline markup like <fixed-case>."""
    return "".join(el.itertext()).strip() if el is not None else ""


def _build_paper(paper_el, year: int) -> Paper | None:
    title = _itertext(paper_el.find("title"))
    if not title:
        return None

    authors = []
    for author_el in paper_el.findall("author"):
        first = _itertext(author_el.find("first"))
        last = _itertext(author_el.find("last"))
        name = f"{first} {last}".strip()
        if name:
            authors.append(name)

    abstract = _itertext(paper_el.find("abstract"))

    url_el = paper_el.find("url")
    anthology_id = url_el.text.strip() if url_el is not None and url_el.text else ""
    if not anthology_id:
        return None
    url = f"{ANTHOLOGY_BASE}/{anthology_id}/"
    pdf_url = f"{ANTHOLOGY_BASE}/{anthology_id}.pdf"

    doi_el = paper_el.find("doi")
    doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

    return Paper(
        id=f"acl{year}:{anthology_id}",
        title=title,
        authors=authors,
        abstract=abstract,
        url=url,
        source="acl",
        published=f"{year}-01-01",
        categories=["cs.CL"],
        pdf_url=pdf_url,
        primary_category=None,  # left for the LLM filter to assign an interest area
        venue=f"ACL {year}",
        doi=doi,
    )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _get_xml(url: str, cache: DiskCache, timeout: int) -> str | None:
    cached = cache.get(url)
    if cached is not None:
        return cached
    try:
        raw = request_with_retry(url, timeout=timeout)
    except Exception as e:  # noqa: BLE001 - 404 / network → treat as unavailable
        log.debug("ACL Anthology fetch failed for %s: %s", url, e)
        return None
    text = raw.decode("utf-8", errors="replace")
    if text.lstrip().startswith("404"):
        return None
    cache.set(url, text)
    return text


def _default_cache_dir() -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, ".cache", "acl")
