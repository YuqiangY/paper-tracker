"""Fallback award-paper source: the community-maintained markdown list
FeijiangHan/Top-Conference-Best-Papers.

Used when Paper Copilot has no JSON for a given conference/year yet (e.g. a
just-concluded conference). Parses the ``## CVPR`` / ``## AAAI`` sections'
``### Best Papers {year}`` subsections into Paper objects.

Two entry formats are supported:
  CVPR:  * Title (CVPR 2026)\n  [[Paper](url)]\n  *Authors: a, b, c*
  AAAI:  **Title**\n  *a, b, c*
grouped under ``#### {award category}`` headings.
"""
from __future__ import annotations
import logging
import re

from models import Paper
from .cache import DiskCache
from .retry import request_with_retry

log = logging.getLogger(__name__)

RAW_URL = "https://raw.githubusercontent.com/FeijiangHan/Top-Conference-Best-Papers/main/README.md"
CACHE_TTL_SECONDS = 7 * 86400


def fetch_best_papers_md(conf: str, year: int = 2026, cache_dir: str | None = None,
                         timeout: int = 30) -> list[Paper]:
    """Fetch award papers for ``conf`` ``year`` from the markdown list."""
    cache = DiskCache(cache_dir or _default_cache_dir(), ttl_seconds=CACHE_TTL_SECONDS)
    md = _get_text(RAW_URL, cache, timeout)
    if md is None:
        return []
    records = _parse_section(md, conf, year)
    log.info("Best-paper markdown %s%d: %d award papers", conf, year, len(records))
    return records


def _parse_section(md: str, conf: str, year: int) -> list[Paper]:
    """Extract award papers for one conference+year from the full README."""
    lines = md.splitlines()
    conf_upper = conf.upper()

    # 1. Locate the conference section (## CVPR / # CVPR / ## AAAI).
    start = _find_conf_start(lines, conf_upper)
    if start is None:
        return []
    end = _find_next_conf(lines, start)

    # 2. Within it, locate the "### Best Papers {year}" subsection.
    sub_start = None
    for i in range(start, end):
        if re.match(rf"^###\s+Best Papers\s+{year}\b", lines[i]):
            sub_start = i
            break
    if sub_start is None:
        return []
    sub_end = end
    for i in range(sub_start + 1, end):
        if re.match(r"^###\s+Best Papers\s+\d{4}", lines[i]):
            sub_end = i
            break

    return _parse_awards(lines[sub_start + 1:sub_end], conf, year)


def _parse_awards(lines: list[str], conf: str, year: int) -> list[Paper]:
    papers: list[Paper] = []
    category = "Award"
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()

        # award category heading
        m_cat = re.match(r"^####\s+(.+)$", line)
        if m_cat:
            category = _clean_category(m_cat.group(1))
            i += 1
            continue

        # CVPR format: * Title (CVPR 2026) ...
        m_star = re.match(r"^\*\s+(.+?)\s*\((?:CVPR|AAAI|ECCV|ICCV|NeurIPS|ICLR|ICML)\s+\d{4}\)\s*$", line)
        # AAAI format: **Title**
        m_bold = re.match(r"^\*\*(.+?)\*\*\s*$", line)

        if m_star:
            title = m_star.group(1).strip()
            arxiv = _find_arxiv(lines, i + 1)
            authors = _find_authors(lines, i + 1)
            papers.append(_build(conf, year, category, title, arxiv, authors))
        elif m_bold:
            title = m_bold.group(1).strip()
            authors = _find_authors(lines, i + 1)
            papers.append(_build(conf, year, category, title, None, authors))
        i += 1

    return papers


def _find_arxiv(lines: list[str], start: int) -> str | None:
    for j in range(start, min(start + 3, len(lines))):
        m = re.search(r"arxiv\.org/abs/(\d+\.\d+)", lines[j])
        if m:
            return m.group(1)
        if lines[j].strip().startswith(("*", "#", "**")):
            break
    return None


def _find_authors(lines: list[str], start: int) -> list[str]:
    for j in range(start, min(start + 3, len(lines))):
        s = lines[j].strip()
        # *Authors: a, b* or *a, b, c*
        m = re.match(r"^\*(?:Authors:)?\s*(.+?)\*$", s)
        if m and "," in m.group(1) or (m and " " in m.group(1)):
            names = m.group(1).replace("Authors:", "").strip()
            return [a.strip() for a in names.split(",") if a.strip()]
        if s.startswith(("####", "* ", "**")):
            break
    return []


def _build(conf: str, year: int, category: str, title: str,
           arxiv: str | None, authors: list[str]) -> Paper:
    url = f"https://arxiv.org/abs/{arxiv}" if arxiv else ""
    ident = arxiv or _slug(title)
    return Paper(
        id=f"{conf}{year}:md:{ident}",
        title=title,
        authors=authors,
        abstract="",
        url=url,
        source="award",
        published=f"{year}-01-01",
        categories=[conf.upper()],
        pdf_url=f"https://arxiv.org/pdf/{arxiv}" if arxiv else None,
        primary_category=f"🏆 {category}",
        venue=f"{conf.upper()} {year}",
    )


def _find_conf_start(lines: list[str], conf_upper: str) -> int | None:
    for i, l in enumerate(lines):
        if re.match(rf"^#{{1,2}}\s+{re.escape(conf_upper)}\s*$", l.strip()):
            return i
    return None


def _find_next_conf(lines: list[str], start: int) -> int:
    for i in range(start + 1, len(lines)):
        if re.match(r"^#{1,2}\s+[A-Z]", lines[i]) and not lines[i].strip().startswith("###"):
            # a new top-level conference heading
            if re.match(r"^#{1,2}\s+[A-Z][A-Za-z]*\s*$", lines[i].strip()):
                return i
    return len(lines)


def _clean_category(raw: str) -> str:
    return re.sub(r"[:：]\s*$", "", raw.strip())


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]


def _get_text(url: str, cache: DiskCache, timeout: int) -> str | None:
    cached = cache.get(url)
    if cached is not None:
        return cached
    try:
        raw = request_with_retry(url, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        log.debug("Best-paper markdown fetch failed: %s", e)
        return None
    text = raw.decode("utf-8", errors="replace")
    cache.set(url, text)
    return text


def _default_cache_dir() -> str:
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, ".cache", "bestpaper_md")
