"""Fetch a paper's arXiv HTML once and extract what the scorer needs."""
from __future__ import annotations
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from bs4 import BeautifulSoup
from models import Paper
from .author_enrichment import apply_affiliations, parse_affiliations_from_soup

log = logging.getLogger(__name__)

ARXIV_HTML_URL = "https://arxiv.org/html/{}"
CONCLUSION_MAX_WORDS = 300
ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
CODE_URL_RE = re.compile(r"^https?://(?:www\.)?(github|gitlab)\.com/([\w.-]+)/([\w.-]+)")
# Repos linked from arXiv's HTML page chrome, not from the paper.
BOILERPLATE_REPO_OWNERS = {"arxiv", "brucemiller"}
BIBLIOGRAPHY_CLASS_RE = re.compile(r"ltx_bibliography|ltx_bibitem")
SECTION_PATTERNS = (
    re.compile(r"conclu", re.I),
    re.compile(r"summary|discussion|closing remarks", re.I),
)


@dataclass(frozen=True)
class PaperContext:
    affiliations: list[str | None]
    author_raw_text: str
    conclusion: str
    code_url: str | None


def parse_paper_context(html: str) -> PaperContext:
    soup = BeautifulSoup(html, "html.parser")
    affiliations, raw_text = parse_affiliations_from_soup(soup)
    return PaperContext(
        affiliations=affiliations,
        author_raw_text=raw_text,
        conclusion=_extract_conclusion(soup),
        code_url=_extract_code_url(soup),
    )


def fetch_paper_context(arxiv_id: str, timeout: int) -> PaperContext | None:
    req = urllib.request.Request(ARXIV_HTML_URL.format(arxiv_id))
    req.add_header("User-Agent", "PaperTracker/1.0")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return parse_paper_context(html)


def attach_paper_contexts(papers: list[Paper], timeout: int, delay_s: float = 0.5) -> int:
    with_html = 0
    for paper in papers:
        if not ARXIV_ID_RE.match(paper.id):
            continue
        try:
            ctx = fetch_paper_context(paper.id, timeout)
        except Exception as e:  # noqa: BLE001 - transient; enrichment may retry later
            log.debug("arXiv HTML fetch failed for %s: %s", paper.id, e)
            time.sleep(delay_s)
            continue
        paper.html_fetched = True
        if ctx is not None:
            with_html += 1
            if any(ctx.affiliations):
                apply_affiliations(paper, ctx.affiliations)
            paper.author_raw_text = ctx.author_raw_text or None
            paper.conclusion = ctx.conclusion or None
            paper.code_url = ctx.code_url
        time.sleep(delay_s)
    log.info("arXiv HTML context: %d/%d papers have full text", with_html, len(papers))
    return with_html


def _extract_conclusion(soup: BeautifulSoup) -> str:
    headings = soup.find_all(["h2", "h3"], class_=re.compile("ltx_title_section"))
    for pattern in SECTION_PATTERNS:
        for h in headings:
            if pattern.search(h.get_text()):
                section = h.find_parent("section") or h.parent
                words = section.get_text(" ", strip=True).split()
                return " ".join(words[:CONCLUSION_MAX_WORDS])
    return ""


def _extract_code_url(soup: BeautifulSoup) -> str | None:
    for a in soup.find_all("a", href=True):
        # Bibliography links point at cited baselines, not this paper's code.
        if a.find_parent(class_=BIBLIOGRAPHY_CLASS_RE):
            continue
        m = CODE_URL_RE.match(a["href"])
        if m and m.group(2).lower() not in BOILERPLATE_REPO_OWNERS:
            repo = re.sub(r"\.git$", "", m.group(3).rstrip("."))
            return f"https://{m.group(1)}.com/{m.group(2)}/{repo}"
    return None
