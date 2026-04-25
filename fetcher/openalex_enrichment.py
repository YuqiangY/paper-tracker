from __future__ import annotations
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from models import Paper

log = logging.getLogger(__name__)

OPENALEX_API = "https://api.openalex.org"


def enrich_from_openalex(
    papers: list[Paper],
    timeout: int = 10,
    email: str = "",
) -> None:
    authors_to_lookup: list[tuple[Paper, int, str]] = []
    for paper in papers:
        if not paper.authors_enriched:
            continue
        for idx, author in enumerate(paper.authors_enriched):
            if not author.get("affiliation") and author.get("name"):
                authors_to_lookup.append((paper, idx, author["name"]))

    if not authors_to_lookup:
        return

    log.info("OpenAlex: looking up %d authors missing affiliations", len(authors_to_lookup))
    found = 0
    for paper, idx, name in authors_to_lookup:
        try:
            result = _lookup_author(name, timeout, email)
            if result:
                affil, h_index = result
                if affil:
                    paper.authors_enriched[idx]["affiliation"] = affil
                    found += 1
                if h_index is not None and paper.authors_enriched[idx].get("h_index") is None:
                    paper.authors_enriched[idx]["h_index"] = h_index
        except Exception as e:
            log.debug("OpenAlex lookup failed for '%s': %s", name, e)
        time.sleep(0.15)

    log.info("OpenAlex: filled %d/%d author affiliations", found, len(authors_to_lookup))


def _lookup_author(
    name: str, timeout: int, email: str
) -> tuple[str | None, int | None] | None:
    params = {
        "filter": f"display_name.search:{name}",
        "per_page": "3",
        "select": "display_name,last_known_institutions,summary_stats,works_count",
    }
    query = urllib.parse.urlencode(params)
    url = f"{OPENALEX_API}/authors?{query}"

    req = urllib.request.Request(url)
    ua = "PaperTracker/1.0"
    if email:
        ua += f" (mailto:{email})"
    req.add_header("User-Agent", ua)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None

    results = data.get("results", [])
    if not results:
        return None

    # Disambiguation: pick best match
    name_lower = name.lower().strip()
    for r in results:
        display = (r.get("display_name") or "").lower().strip()
        works = r.get("works_count", 0) or 0
        if display == name_lower and works >= 5:
            institutions = r.get("last_known_institutions") or []
            affil = institutions[0].get("display_name") if institutions else None
            stats = r.get("summary_stats") or {}
            h_index = stats.get("h_index")
            return (affil, h_index)

    return None
