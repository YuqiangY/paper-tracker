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

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_BATCH_URL = f"{S2_API_BASE}/paper/batch"
S2_AUTHOR_URL = f"{S2_API_BASE}/author"


def enrich_authors(
    papers: list[Paper],
    api_key: str | None = None,
    timeout: int = 10,
    max_authors_per_paper: int = 5,
) -> list[Paper]:
    arxiv_papers = [p for p in papers if not p.id.startswith("rss:")]
    if not arxiv_papers:
        log.info("No arXiv papers to enrich")
        return papers

    # Stage A: Fetch affiliations from arXiv HTML pages
    _enrich_affiliations_from_arxiv_html(arxiv_papers, timeout)

    # Stage B: Fetch h-index from Semantic Scholar
    try:
        paper_s2_data = _batch_fetch_papers(arxiv_papers, api_key, timeout)
    except Exception as e:
        log.warning("S2 batch fetch failed entirely: %s", e)
        return papers

    author_ids_to_fetch: set[str] = set()
    paper_author_map: dict[str, list[dict]] = {}

    for paper in arxiv_papers:
        s2_data = paper_s2_data.get(paper.id)
        if not s2_data or "authors" not in s2_data:
            continue
        s2_authors = s2_data["authors"]
        paper_author_map[paper.id] = s2_authors
        for s2_author in s2_authors[:max_authors_per_paper]:
            if s2_author.get("authorId"):
                author_ids_to_fetch.add(s2_author["authorId"])

    h_index_map = _batch_fetch_h_indices(author_ids_to_fetch, api_key, timeout)

    for paper in arxiv_papers:
        s2_authors = paper_author_map.get(paper.id, [])
        if not s2_authors:
            continue
        enriched = paper.authors_enriched or [
            {"name": name, "affiliation": None, "h_index": None, "semantic_scholar_id": None}
            for name in paper.authors
        ]
        _merge_s2_into_enriched(enriched, s2_authors, h_index_map)
        paper.authors_enriched = enriched

    return papers


def _enrich_affiliations_from_arxiv_html(papers: list[Paper], timeout: int):
    for paper in papers:
        try:
            affiliations = _fetch_arxiv_html_affiliations(paper.id, timeout)
            if not affiliations:
                continue
            if not paper.authors_enriched:
                paper.authors_enriched = [
                    {"name": name, "affiliation": None, "h_index": None, "semantic_scholar_id": None}
                    for name in paper.authors
                ]
            for idx, affil in enumerate(affiliations):
                if idx < len(paper.authors_enriched) and affil:
                    paper.authors_enriched[idx]["affiliation"] = affil
            log.debug("Parsed affiliations for %s from arXiv HTML", paper.id)
        except Exception as e:
            log.debug("Failed to parse arXiv HTML for %s: %s", paper.id, e)
        time.sleep(0.5)


def _fetch_arxiv_html_affiliations(arxiv_id: str, timeout: int) -> list[str | None]:
    url = f"https://arxiv.org/html/{arxiv_id}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "PaperTracker/1.0")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8")

    m = re.search(r'<div class="ltx_authors">(.*?)</div>', html, re.DOTALL)
    if not m:
        return []

    authors_html = m.group(1)

    # Type 1: per-author blocks with structured affiliation
    author_blocks = re.findall(
        r'<span class="ltx_creator ltx_role_author">(.*?)</span>\s*(?:<span class="ltx_author_before|$)',
        authors_html, re.DOTALL,
    )
    if len(author_blocks) > 1:
        result = []
        for block in author_blocks:
            inst_m = re.search(r'ltx_affiliation_institution">(.*?)</span>', block)
            result.append(inst_m.group(1).strip() if inst_m else None)
        return result

    # Type 2: single block with free-text affiliation notes
    notes_m = re.search(r'<span class="ltx_author_notes">(.*?)</span>\s*</span>', authors_html, re.DOTALL)
    if not notes_m:
        return []

    notes_text = re.sub(r"<[^>]+>", " ", notes_m.group(1))
    names = re.findall(r'ltx_personname">\s*(.*?)\s*</span>', authors_html, re.DOTALL)
    if not names:
        return []
    clean_names = [re.sub(r"<[^>]+>", "", n).strip().rstrip(",").strip() for n in names]
    author_list = []
    for n in clean_names:
        parts = [x.strip() for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", n) if x.strip()]
        author_list.extend([re.sub(r"[#†‡§∗*¶\d]+$", "", p).strip() for p in parts])

    affiliation_map: dict[str, str] = {}
    for sentence in re.split(r"\.\s+|\n\s*\n", notes_text):
        sent_m = re.match(r"(.+?)\s+(?:is|are)\s+with\s+(?:the\s+)?(.+)", sentence, re.IGNORECASE)
        if not sent_m:
            continue
        name_part = sent_m.group(1).strip()
        inst_part = sent_m.group(2).strip()
        # Extract university/institute name
        inst_name = _extract_institution(inst_part)
        if not inst_name:
            continue
        matched_names = re.split(r",\s*(?:and\s+)?|\s+and\s+", name_part)
        for mn in matched_names:
            mn = mn.strip()
            if mn:
                affiliation_map[mn] = inst_name

    return [affiliation_map.get(name) for name in author_list]


def _extract_institution(text: str) -> str | None:
    patterns = [
        r"([A-Z][\w\s]+?University(?:\s+of\s+[\w\s]+)?)",
        r"((?:National\s+)?[\w\s]+?Institute\s+of\s+[\w\s]+)",
        r"([\w\s]+(?:Inc|Corp|Ltd|Co)\b\.?)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    # Fallback: take text up to first comma
    return text.split(",")[0].strip() if text else None


def _batch_fetch_papers(
    papers: list[Paper],
    api_key: str | None,
    timeout: int,
) -> dict[str, dict]:
    result = {}
    BATCH_SIZE = 100
    arxiv_ids = [f"ArXiv:{p.id}" for p in papers]

    for i in range(0, len(arxiv_ids), BATCH_SIZE):
        batch_ids = arxiv_ids[i:i + BATCH_SIZE]
        try:
            data = _s2_post(
                S2_BATCH_URL,
                params={"fields": "authors.authorId,authors.name,authors.affiliations"},
                body={"ids": batch_ids},
                api_key=api_key,
                timeout=timeout,
            )
            if data and isinstance(data, list):
                for item, paper in zip(data, papers[i:i + BATCH_SIZE]):
                    if item is not None:
                        result[paper.id] = item
        except Exception as e:
            log.warning("S2 batch fetch failed for batch %d: %s", i, e)

        if i + BATCH_SIZE < len(arxiv_ids):
            time.sleep(1)

    return result


def _batch_fetch_h_indices(
    author_ids: set[str],
    api_key: str | None,
    timeout: int,
) -> dict[str, int]:
    result = {}
    for author_id in author_ids:
        try:
            url = f"{S2_AUTHOR_URL}/{author_id}"
            data = _s2_get(url, params={"fields": "hIndex"}, api_key=api_key, timeout=timeout)
            if data and "hIndex" in data and data["hIndex"] is not None:
                result[author_id] = data["hIndex"]
        except Exception as e:
            log.debug("S2 author fetch failed for %s: %s", author_id, e)
        time.sleep(0.1)
    return result


def _merge_s2_into_enriched(
    enriched: list[dict],
    s2_authors: list[dict],
    h_index_map: dict[str, int],
):
    for idx, s2_auth in enumerate(s2_authors):
        if idx >= len(enriched):
            enriched.append({
                "name": s2_auth.get("name", ""),
                "affiliation": None, "h_index": None, "semantic_scholar_id": None,
            })
        target = enriched[idx]
        author_id = s2_auth.get("authorId")
        if author_id:
            target["semantic_scholar_id"] = author_id
        s2_affiliations = s2_auth.get("affiliations") or []
        if not target.get("affiliation") and s2_affiliations:
            target["affiliation"] = s2_affiliations[0]
        if author_id and author_id in h_index_map:
            target["h_index"] = h_index_map[author_id]


def _s2_get(url: str, params: dict, api_key: str | None, timeout: int) -> dict | None:
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}" if query else url
    for attempt in range(4):
        try:
            req = urllib.request.Request(full_url)
            req.add_header("User-Agent", "PaperTracker/1.0")
            if api_key:
                req.add_header("x-api-key", api_key)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            elif e.code == 404:
                return None
            raise
        except (TimeoutError, OSError):
            if attempt < 3:
                time.sleep(3)
                continue
            return None
    return None


def _s2_post(url: str, params: dict, body: dict, api_key: str | None, timeout: int) -> list | None:
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}" if query else url
    payload = json.dumps(body).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(full_url, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("User-Agent", "PaperTracker/1.0")
            if api_key:
                req.add_header("x-api-key", api_key)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(4 * (attempt + 1))
                continue
            raise
        except (TimeoutError, OSError):
            if attempt < 2:
                time.sleep(2)
                continue
            return None
    return None
