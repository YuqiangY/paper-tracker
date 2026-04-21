from __future__ import annotations
import re
import time
import urllib.error
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from models import Paper

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"


def fetch_arxiv(
    categories: list[str],
    max_results: int = 100,
    lookback_days: int = 1,
) -> list[Paper]:
    seen: dict[str, Paper] = {}
    unique_cats = list(dict.fromkeys(categories))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    for cat in unique_cats:
        xml_text = _query_arxiv_api(cat, max_results)
        for paper in _parse_arxiv_response(xml_text):
            if paper.id not in seen and paper.published >= cutoff:
                seen[paper.id] = paper
        time.sleep(3)

    return list(seen.values())


def _query_arxiv_api(category: str, max_results: int) -> str:
    params = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API_URL}?{params}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "PaperTracker/1.0")
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.HTTPError, TimeoutError, OSError) as e:
            if attempt < 2:
                time.sleep(15 * (attempt + 1))
                continue
            raise
    return ""


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
