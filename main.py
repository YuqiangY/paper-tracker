from __future__ import annotations
import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from datetime import date

import yaml
from dotenv import load_dotenv

from models import Paper, papers_to_json, papers_from_json
from fetcher import fetch_arxiv, fetch_hf_daily, fetch_s2_search, fetch_rss, enrich_authors
from filter.keyword_filter import keyword_filter
from filter.llm_filter import llm_filter
from output.summary import generate_summary
from storage.db import PaperDB
from output.html_output import generate_daily_page, generate_index_page
from output.feishu_output import generate_feishu_doc
from output.feishu_bot import send_daily_bot_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def stage_fetch(config: dict) -> list[Paper]:
    all_papers: list[Paper] = []

    if config["sources"]["arxiv"]["enabled"]:
        try:
            all_categories = []
            for interest in config["interests"]:
                all_categories.extend(interest.get("arxiv_categories", []))
            unique_categories = list(dict.fromkeys(all_categories))

            log.info("Fetching from arXiv: categories=%s", unique_categories)
            arxiv_papers = fetch_arxiv(
                categories=unique_categories,
                max_results=config["sources"]["arxiv"]["max_results_per_category"],
                lookback_days=config["sources"]["arxiv"]["lookback_days"],
            )
            log.info("arXiv returned %d papers", len(arxiv_papers))
            all_papers.extend(arxiv_papers)
        except Exception as e:
            log.warning("arXiv fetch failed: %s. Continuing with other sources.", e)

    if config["sources"].get("hf_daily", {}).get("enabled", False):
        try:
            log.info("Fetching from HuggingFace Daily Papers")
            hf_papers = fetch_hf_daily(
                lookback_days=config["sources"]["hf_daily"].get("lookback_days", 3),
            )
            log.info("HF Daily returned %d papers", len(hf_papers))
            all_papers.extend(hf_papers)
        except Exception as e:
            log.warning("HF Daily fetch failed: %s. Continuing with other sources.", e)

    if config["sources"].get("s2_search", {}).get("enabled", False):
        try:
            search_queries = []
            for interest in config["interests"]:
                search_queries.extend(interest.get("search_queries", []))
            if search_queries:
                from datetime import date as _date
                current_year = str(_date.today().year)
                log.info("Fetching from Semantic Scholar search: %d queries", len(search_queries))
                s2_papers = fetch_s2_search(
                    queries=search_queries,
                    year=current_year,
                    limit_per_query=config["sources"]["s2_search"].get("limit_per_query", 20),
                    api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
                    timeout=config.get("enrichment", {}).get("timeout", 10),
                )
                log.info("S2 search returned %d papers", len(s2_papers))
                all_papers.extend(s2_papers)
        except Exception as e:
            log.warning("S2 search fetch failed: %s. Continuing with other sources.", e)

    if config["sources"]["rss"]["enabled"]:
        try:
            feeds = config["sources"]["rss"]["feeds"]
            log.info("Fetching from %d RSS feeds", len(feeds))
            rss_papers = fetch_rss(feeds)
            log.info("RSS returned %d papers", len(rss_papers))
            all_papers.extend(rss_papers)
        except Exception as e:
            log.warning("RSS fetch failed: %s. Continuing with other sources.", e)

    log.info("Total fetched: %d papers", len(all_papers))
    return all_papers


def stage_dedup(papers: list[Paper], db: PaperDB) -> list[Paper]:
    new_papers = []
    for p in papers:
        if not db.exists(p.id):
            db.insert(p)
            new_papers.append(p)
    log.info("After dedup: %d new papers (of %d total)", len(new_papers), len(papers))
    return new_papers


def stage_filter(
    papers: list[Paper], config: dict,
) -> list[Paper]:
    kw_results = keyword_filter(
        papers,
        config["interests"],
        threshold=config["filter"]["keyword_threshold"],
    )
    kw_papers = [paper for paper, _ in kw_results]
    log.info("After keyword filter: %d papers", len(kw_papers))

    if not kw_papers:
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL") or None

    try:
        scored_papers = llm_filter(
            papers=kw_papers,
            interests=config["interests"],
            model=config["filter"]["llm_model"],
            threshold=config["filter"]["llm_relevance_threshold"],
            batch_size=config["filter"]["llm_batch_size"],
            api_key=api_key,
            base_url=base_url,
        )
        log.info("After LLM filter: %d papers (threshold=%s)",
                 len(scored_papers), config["filter"]["llm_relevance_threshold"])
    except Exception as e:
        log.warning("LLM filter failed: %s. Falling back to keyword-only results.", e)
        scored_papers = kw_papers

    return scored_papers


def stage_enrich(papers: list[Paper], config: dict, db: PaperDB) -> list[Paper]:
    for p in papers:
        if p.relevance_score is not None:
            db.update_filter_result(
                p.id, p.relevance_score, p.primary_category or "",
                p.summary_zh or "", p.why_relevant or "", p.tags or [],
            )

    enrichment_cfg = config.get("enrichment", {})
    if enrichment_cfg.get("enabled", False) and papers:
        s2_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        log.info("Enriching %d filtered papers with author data", len(papers))
        try:
            enrich_authors(
                papers,
                api_key=s2_api_key,
                timeout=enrichment_cfg.get("timeout", 10),
                max_authors_per_paper=enrichment_cfg.get("max_authors_per_paper", 5),
            )
            for p in papers:
                if p.authors_enriched:
                    db.update_authors_enriched(p.id, p.authors_enriched)
                if any([p.venue, p.citation_count, p.tldr, p.doi]):
                    db.update_metadata(p.id, p.venue, p.citation_count, p.tldr, p.doi)
            log.info("Author enrichment complete")
        except Exception as e:
            log.warning("Author enrichment failed: %s. Continuing without.", e)

    return papers


def stage_output(
    config: dict, db: PaperDB,
    scored_papers: list[Paper],
    data_dir: str, site_dir: str, today: str,
):
    json_path = os.path.join(data_dir, f"{today}.json")
    with open(json_path, "w") as f:
        f.write(papers_to_json(scored_papers))
    log.info("Saved %d papers to %s", len(scored_papers), json_path)

    paper_dicts = db.get_papers_by_date(today, min_score=config["filter"]["llm_relevance_threshold"])
    if not paper_dicts:
        paper_dicts = [json.loads(papers_to_json([p]))[0] for p in scored_papers]

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL") or None

    summary_html = ""
    summary_text = ""
    if config.get("summary", {}).get("enabled", False) and paper_dicts:
        log.info("Generating domain hotspot summary for %d papers", len(paper_dicts))
        try:
            summary_html, summary_text = generate_summary(
                papers=paper_dicts,
                interests=config["interests"],
                model=config["filter"]["llm_model"],
                api_key=api_key,
                base_url=base_url,
            )
            if summary_html:
                log.info("Summary generated (%d chars)", len(summary_html))
            else:
                log.warning("Summary generation returned empty")
        except Exception as e:
            log.warning("Summary generation failed: %s", e)

    if config["output"]["html"]["enabled"]:
        generate_daily_page(paper_dicts, today, site_dir, config["output"]["html"]["title"], summary=summary_html)

        existing_dates = [
            f.replace(".html", "")
            for f in os.listdir(site_dir)
            if f.endswith(".html") and f != "index.html"
        ]
        generate_index_page(existing_dates, site_dir, config["output"]["html"]["title"])
        log.info("HTML site updated at %s", site_dir)

        deploy_cmd = config["output"]["html"].get("deploy_cmd")
        if deploy_cmd:
            try:
                result = subprocess.run(shlex.split(deploy_cmd), capture_output=True, text=True, timeout=120)
                if result.returncode != 0:
                    log.warning("Deploy failed (rc=%d): %s", result.returncode, result.stderr[:500])
            except Exception as e:
                log.warning("Deploy command failed: %s", e)

    feishu_cfg = config.get("output", {}).get("feishu", {})
    if feishu_cfg.get("enabled", False):
        generate_feishu_doc(
            papers=paper_dicts,
            date=today,
            wiki_space=feishu_cfg.get("wiki_space", ""),
        )

    bot_cfg = config.get("output", {}).get("feishu_bot", {})
    if bot_cfg.get("enabled", False) and summary_text:
        webhook_url = os.environ.get(bot_cfg.get("webhook_url_env", ""), "")
        if webhook_url:
            pages_base = bot_cfg.get("pages_base_url", "")
            pages_url = f"{pages_base}/{today}.html" if pages_base else ""
            send_daily_bot_message(
                webhook_url=webhook_url,
                summary_text=summary_text,
                date=today,
                paper_count=len(paper_dicts),
                pages_url=pages_url,
            )
        else:
            log.warning("FEISHU_BOT_WEBHOOK_URL not set, skipping bot notification")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def _resolve_paths(config: dict):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "daily")
    db_path = os.path.join(base_dir, "data", "papers.db")
    site_dir = os.path.join(base_dir, config["output"]["html"]["output_dir"])
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return data_dir, db_path, site_dir


def run_pipeline(
    config: dict,
    data_dir: str | None = None,
    db_path: str | None = None,
    site_dir: str | None = None,
):
    resolved = _resolve_paths(config)
    data_dir = data_dir or resolved[0]
    db_path = db_path or resolved[1]
    site_dir = site_dir or resolved[2]

    db = PaperDB(db_path)
    try:
        today = date.today().isoformat()

        all_papers = stage_fetch(config)
        new_papers = stage_dedup(all_papers, db)

        if not new_papers:
            log.info("No new papers today. Exiting.")
            with open(os.path.join(data_dir, f"{today}.json"), "w") as f:
                f.write("[]")
            return

        scored_papers = stage_filter(new_papers, config)

        if not scored_papers:
            log.info("No papers passed filtering. Exiting.")
            with open(os.path.join(data_dir, f"{today}.json"), "w") as f:
                f.write("[]")
            return

        scored_papers = stage_enrich(scored_papers, config, db)
        stage_output(config, db, scored_papers, data_dir, site_dir, today)
    finally:
        db.close()
        log.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def cmd_fetch(config: dict):
    data_dir, db_path, _ = _resolve_paths(config)
    today = date.today().isoformat()
    papers = stage_fetch(config)

    db = PaperDB(db_path)
    try:
        new_papers = stage_dedup(papers, db)
    finally:
        db.close()

    json_path = os.path.join(data_dir, f"{today}_raw.json")
    with open(json_path, "w") as f:
        f.write(papers_to_json(new_papers))
    log.info("Fetch complete: %d new papers saved to %s", len(new_papers), json_path)


def cmd_filter(config: dict):
    data_dir, db_path, _ = _resolve_paths(config)
    today = date.today().isoformat()

    raw_path = os.path.join(data_dir, f"{today}_raw.json")
    if not os.path.exists(raw_path):
        log.error("No raw data found at %s. Run 'fetch' first.", raw_path)
        return

    with open(raw_path) as f:
        papers = papers_from_json(f.read())

    scored = stage_filter(papers, config)

    db = PaperDB(db_path)
    try:
        scored = stage_enrich(scored, config, db)
    finally:
        db.close()

    json_path = os.path.join(data_dir, f"{today}.json")
    with open(json_path, "w") as f:
        f.write(papers_to_json(scored))
    log.info("Filter complete: %d papers saved to %s", len(scored), json_path)


def cmd_output(config: dict):
    data_dir, db_path, site_dir = _resolve_paths(config)
    today = date.today().isoformat()

    json_path = os.path.join(data_dir, f"{today}.json")
    if not os.path.exists(json_path):
        log.error("No data found at %s. Run the pipeline first.", json_path)
        return

    with open(json_path) as f:
        papers = papers_from_json(f.read())

    db = PaperDB(db_path)
    try:
        stage_output(config, db, papers, data_dir, site_dir, today)
    finally:
        db.close()
    log.info("Output generation complete.")


def main():
    load_dotenv(override=True)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")

    parser = argparse.ArgumentParser(description="Paper Tracker Pipeline")
    parser.add_argument("--config", default=config_path, help="Path to config.yaml")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("fetch", help="Fetch papers and dedup into DB")
    subparsers.add_parser("filter", help="Run keyword + LLM filtering on today's raw data")
    subparsers.add_parser("output", help="Generate HTML and Feishu output from today's data")
    subparsers.add_parser("run-all", help="Run full pipeline (default)")

    args = parser.parse_args()
    config = load_config(args.config)
    command = args.command or "run-all"

    if command == "fetch":
        cmd_fetch(config)
    elif command == "filter":
        cmd_filter(config)
    elif command == "output":
        cmd_output(config)
    else:
        run_pipeline(config)


if __name__ == "__main__":
    main()
