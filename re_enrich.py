"""Re-run author enrichment on existing daily JSON files and report coverage."""
from __future__ import annotations
import json
import os
import sys
import logging

from dotenv import load_dotenv
load_dotenv(override=True)

from models import Paper, papers_from_json, papers_to_json
from fetcher import enrich_authors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "daily")
DATES = sys.argv[1:] or ["2026-04-21", "2026-04-22", "2026-04-23"]

s2_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
llm_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
llm_base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
llm_model = "xiaomi/mimo-v2-pro"


def report(date: str, papers: list[Paper]):
    total = len(papers)
    if not total:
        print(f"  {date}: 无论文数据")
        return
    papers_with_affil = 0
    total_authors = 0
    authors_with_affil = 0
    for p in papers:
        if not p.authors_enriched:
            continue
        has = False
        for a in p.authors_enriched:
            total_authors += 1
            if a.get("affiliation"):
                authors_with_affil += 1
                has = True
        if has:
            papers_with_affil += 1
    print(f"  论文: {total}  |  有机构论文: {papers_with_affil}/{total} ({100*papers_with_affil/total:.1f}%)"
          f"  |  有机构作者: {authors_with_affil}/{total_authors} ({100*authors_with_affil/total_authors:.1f}%)" if total_authors else
          f"  论文: {total}  |  无enriched数据")


for dt in DATES:
    path = os.path.join(DATA_DIR, f"{dt}.json")
    if not os.path.exists(path):
        print(f"\n=== {dt}: 文件不存在 ===")
        continue

    with open(path) as f:
        papers = papers_from_json(f.read())

    if not papers:
        print(f"\n=== {dt}: 空数据 ===")
        continue

    print(f"\n=== {dt} (改进前) ===")
    report(dt, papers)

    for p in papers:
        if p.authors_enriched:
            for a in p.authors_enriched:
                a["affiliation"] = None
                a["h_index"] = None
                a["semantic_scholar_id"] = None

    arxiv_papers = [p for p in papers if not p.id.startswith("rss:")]
    log.info("Re-enriching %d arXiv papers for %s", len(arxiv_papers), dt)

    enrich_authors(
        papers=arxiv_papers,
        api_key=s2_api_key,
        timeout=10,
        max_authors_per_paper=5,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )

    print(f"\n=== {dt} (改进后) ===")
    report(dt, papers)

    with open(path, "w") as f:
        f.write(papers_to_json(papers))
    log.info("Saved updated data to %s", path)
