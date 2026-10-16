from __future__ import annotations
import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

import yaml
from dotenv import load_dotenv

from models import Paper, papers_to_json, papers_from_json
from fetcher import fetch_arxiv, fetch_hf_daily, fetch_s2_search, fetch_rss, enrich_authors
from fetcher.arxiv_fetcher import MAX_PAGES_NOTE
from fetcher.fulltext import attach_paper_contexts
from filter.keyword_filter import keyword_filter
from filter.llm_scoring import score_papers
from filter.llm_summary import summarize_papers
from filter.selection import normalize_categories, select_papers
from output.summary import generate_summary
from run_report import RunReport
from storage.db import PaperDB, STATUS_KEYWORD_REJECTED, STATUS_SCORE_FAILED
from storage.run_state import compute_arxiv_cutoff, load_run_state, save_run_state
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

def stage_fetch(config: dict, since_date: str, report: RunReport) -> list[Paper]:
    all_papers: list[Paper] = []

    if config["sources"]["arxiv"]["enabled"]:
        try:
            all_categories = []
            for interest in config["interests"]:
                all_categories.extend(interest.get("arxiv_categories", []))
            unique_categories = list(dict.fromkeys(all_categories))

            log.info("Fetching from arXiv since %s: categories=%s", since_date, unique_categories)
            arxiv_papers, cat_stats = fetch_arxiv(categories=unique_categories, since_date=since_date)
            report.source_counts["arXiv"] = len(arxiv_papers)
            for cat, st in cat_stats.items():
                if st.complete:
                    continue
                if report.cutoff_clamped and st.note.startswith(MAX_PAGES_NOTE):
                    # Loss is already accepted for a clamped window; blocking state would repeat forever.
                    report.warnings.append(f"arXiv {cat} 达到翻页上限（{st.note}），窗口内较早的论文未抓取")
                    continue
                report.arxiv_complete = False
                report.alerts.append(f"arXiv {cat} 抓取不完整（{st.note}），下次运行自动补抓")
            log.info("arXiv returned %d papers", len(arxiv_papers))
            all_papers.extend(arxiv_papers)
        except Exception as e:
            log.warning("arXiv fetch failed: %s. Continuing with other sources.", e)
            report.arxiv_complete = False
            report.errors.append(f"arXiv 抓取异常: {e}")

    if config["sources"].get("hf_daily", {}).get("enabled", False):
        try:
            log.info("Fetching from HuggingFace Daily Papers")
            hf_papers = fetch_hf_daily(
                lookback_days=config["sources"]["hf_daily"].get("lookback_days", 3),
            )
            log.info("HF Daily returned %d papers", len(hf_papers))
            report.source_counts["HF"] = len(hf_papers)
            all_papers.extend(hf_papers)
        except Exception as e:
            log.warning("HF Daily fetch failed: %s. Continuing with other sources.", e)
            report.warnings.append(f"HF Daily 抓取异常: {e}")

    if config["sources"].get("s2_search", {}).get("enabled", False):
        try:
            search_queries = []
            for interest in config["interests"]:
                search_queries.extend(interest.get("search_queries", []))
            if search_queries:
                current_year = str(date.today().year)
                log.info("Fetching from Semantic Scholar search: %d queries", len(search_queries))
                s2_papers = fetch_s2_search(
                    queries=search_queries,
                    year=current_year,
                    limit_per_query=config["sources"]["s2_search"].get("limit_per_query", 20),
                    api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
                    timeout=config.get("enrichment", {}).get("timeout", 10),
                )
                # S2 search is not date-ordered; keep only papers inside the arXiv window.
                # Year-only dates ("2026") sort before any full date and are dropped too.
                fetched_count = len(s2_papers)
                s2_papers = [p for p in s2_papers if p.published >= since_date]
                log.info("S2 search returned %d papers, %d within window since %s",
                         fetched_count, len(s2_papers), since_date)
                report.source_counts["S2"] = len(s2_papers)
                all_papers.extend(s2_papers)
        except Exception as e:
            log.warning("S2 search fetch failed: %s. Continuing with other sources.", e)

    if config["sources"]["rss"]["enabled"]:
        try:
            feeds = config["sources"]["rss"]["feeds"]
            log.info("Fetching from %d RSS feeds", len(feeds))
            rss_papers = fetch_rss(feeds)
            log.info("RSS returned %d papers", len(rss_papers))
            report.source_counts["RSS"] = len(rss_papers)
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
    papers: list[Paper], config: dict, db: PaperDB, report: RunReport, today: str,
) -> list[Paper]:
    fcfg = config["filter"]
    interests = config["interests"]
    names = [i["name"] for i in interests]

    kw_results = keyword_filter(papers, interests, threshold=fcfg["keyword_threshold"])
    kw_papers = [paper for paper, _ in kw_results]
    area_of = {paper.id: areas[0] for paper, areas in kw_results if areas}
    kw_ids = {p.id for p in kw_papers}
    db.set_filter_status([p.id for p in papers if p.id not in kw_ids], STATUS_KEYWORD_REJECTED)
    report.keyword_count = len(kw_papers)
    log.info("After keyword filter: %d papers", len(kw_papers))
    if not kw_papers:
        return []

    timeout = config.get("enrichment", {}).get("timeout", 10)
    html_delay_s = config.get("enrichment", {}).get("html_delay_s", 1.0)
    report.html_count = attach_paper_contexts(kw_papers, timeout=timeout, delay_s=html_delay_s)

    llm_kwargs = {
        "model": fcfg["llm_model"],
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "base_url": os.environ.get("ANTHROPIC_BASE_URL") or None,
    }
    outcome = score_papers(kw_papers, interests, batch_size=fcfg["score_batch_size"],
                           area_of=area_of, **llm_kwargs)
    report.other_count = normalize_categories(outcome.scored, names)
    for p in outcome.scored:
        db.update_scores(p)
    db.set_filter_status([p.id for p in outcome.failed], STATUS_SCORE_FAILED)
    report.scored_count, report.score_failed = len(outcome.scored), len(outcome.failed)
    if outcome.failed:
        report.warnings.append(f"{len(outcome.failed)} 篇打分失败，下次运行自动重试")
    if not outcome.scored:
        report.errors.append("LLM 打分全部失败")
        return []

    selection = select_papers(
        outcome.scored, interest_names=names,
        threshold=fcfg["llm_relevance_threshold"],
        min_relevance=fcfg["min_relevance"], max_per_area=fcfg["max_per_area"],
    )
    report.selected_count, report.dropped_by_cap = len(selection.selected), selection.dropped_by_cap
    report.summary_missing = summarize_papers(
        selection.selected, interests, batch_size=fcfg["llm_batch_size"], **llm_kwargs)
    if report.summary_missing:
        report.warnings.append(f"{report.summary_missing} 篇未生成中文摘要")
    db.mark_selected([p.id for p in selection.selected], today)
    return selection.selected


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
        llm_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        llm_base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
        llm_model = config.get("filter", {}).get("llm_model", "")
        log.info("Enriching %d filtered papers with author data", len(papers))
        try:
            enrich_authors(
                papers,
                api_key=s2_api_key,
                timeout=enrichment_cfg.get("timeout", 10),
                max_authors_per_paper=enrichment_cfg.get("max_authors_per_paper", 5),
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
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
    report: RunReport, publish: bool = True,
) -> str:
    """Write JSON/HTML (+ deploy and Feishu doc when publishing); return bot summary text."""
    json_path = os.path.join(data_dir, f"{today}.json")
    with open(json_path, "w") as f:
        f.write(papers_to_json(scored_papers))
    log.info("Saved %d papers to %s", len(scored_papers), json_path)

    paper_dicts = db.get_papers_by_date(today)
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
                report.warnings.append("热点总结生成为空")
        except Exception as e:
            log.warning("Summary generation failed: %s", e)
            report.warnings.append(f"热点总结生成失败: {e}")

    if config["output"]["html"]["enabled"]:
        generate_daily_page(paper_dicts, today, site_dir, config["output"]["html"]["title"], summary=summary_html)

        existing_dates = [
            f.replace(".html", "")
            for f in os.listdir(site_dir)
            if f.endswith(".html") and f != "index.html"
        ]
        generate_index_page(
            existing_dates, site_dir, config["output"]["html"]["title"],
            features=config["output"]["html"].get("features", []),
        )
        log.info("HTML site updated at %s", site_dir)

        deploy_cmd = config["output"]["html"].get("deploy_cmd")
        if deploy_cmd and publish:
            report.deploy = _run_deploy(deploy_cmd)
            if report.deploy.startswith("failed"):
                report.errors.append(f"网页部署失败: {report.deploy}")
        elif deploy_cmd:
            report.deploy = "skipped (--no-publish)"

    feishu_cfg = config.get("output", {}).get("feishu", {})
    if feishu_cfg.get("enabled", False) and publish:
        report.feishu_doc = generate_feishu_doc(
            papers=paper_dicts,
            date=today,
            wiki_space=feishu_cfg.get("wiki_space", ""),
        )
        if report.feishu_doc.startswith("failed"):
            report.warnings.append(f"飞书文档失败: {report.feishu_doc}")

    return summary_text


def _run_deploy(deploy_cmd: str) -> str:
    try:
        result = subprocess.run(shlex.split(deploy_cmd), capture_output=True, text=True, timeout=120)
    except Exception as e:  # noqa: BLE001
        log.warning("Deploy command failed: %s", e)
        return f"failed: {e}"
    if result.returncode != 0:
        log.warning("Deploy failed (rc=%d): %s", result.returncode, result.stderr[:500])
        return f"failed: rc={result.returncode} {result.stderr.strip()[:200]}"
    return "ok"


def stage_notify(config: dict, report: RunReport, summary_text: str, paper_count: int, today: str):
    bot_cfg = config.get("output", {}).get("feishu_bot", {})
    if not bot_cfg.get("enabled", False):
        return
    webhook_url = os.environ.get(bot_cfg.get("webhook_url_env", ""), "")
    if not webhook_url:
        log.warning("FEISHU_BOT_WEBHOOK_URL not set, skipping bot notification")
        return
    pages_base = bot_cfg.get("pages_base_url", "")
    send_daily_bot_message(
        webhook_url=webhook_url,
        summary_text=summary_text,
        date=today,
        paper_count=paper_count,
        pages_url=f"{pages_base}/{today}.html" if pages_base and paper_count else "",
        status_lines=report.status_lines(),
        is_alert=report.is_alert,
    )


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


def _arxiv_cutoff(config: dict, state_path: str, now: datetime) -> tuple[datetime, bool]:
    acfg = config["sources"]["arxiv"]
    return compute_arxiv_cutoff(
        load_run_state(state_path), now, acfg["overlap_days"],
        acfg["lookback_days"], acfg["max_lookback_days"],
    )


def run_pipeline(
    config: dict,
    data_dir: str | None = None,
    db_path: str | None = None,
    site_dir: str | None = None,
    state_path: str | None = None,
    publish: bool = True,
) -> RunReport:
    resolved = _resolve_paths(config)
    data_dir = data_dir or resolved[0]
    db_path = db_path or resolved[1]
    site_dir = site_dir or resolved[2]
    state_path = state_path or os.path.join(os.path.dirname(db_path), "run_state.json")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(site_dir, exist_ok=True)

    today = date.today().isoformat()
    started = datetime.now(timezone.utc)
    report = RunReport(date=today)
    summary_text = ""
    candidates: list[Paper] = []
    db = None
    try:
        fcfg = config["filter"]
        max_lookback = config["sources"]["arxiv"]["max_lookback_days"]
        since, report.cutoff_clamped = _arxiv_cutoff(config, state_path, started)
        if report.cutoff_clamped:
            report.warnings.append(
                f"距上次成功运行超过 {max_lookback} 天，只回补到 {since.date()}，更早的论文可能缺失")
        log.info("arXiv cutoff: %s", since.isoformat())

        db = PaperDB(db_path)
        fetched = stage_fetch(config, since.date().isoformat(), report)
        new_papers = stage_dedup(fetched, db)
        retry = db.get_retry_candidates(
            since_iso=(started - timedelta(days=fcfg["retry_window_days"])).isoformat(),
            before_iso=started.isoformat(), limit=fcfg["retry_limit"],
        )
        if len(retry) >= fcfg["retry_limit"]:
            report.warnings.append(
                f"重试队列已满（{fcfg['retry_limit']} 篇），更早的待打分论文顺延到下次")
        new_ids = {p.id for p in new_papers}
        candidates = new_papers + [p for p in retry if p.id not in new_ids]
        report.new_count, report.retry_count = len(candidates), len(candidates) - len(new_papers)

        selected = stage_filter(candidates, config, db, report, today) if candidates else []
        if selected:
            selected = stage_enrich(selected, config, db)
            summary_text = stage_output(config, db, selected, data_dir, site_dir, today, report, publish)
        else:
            log.info("No papers selected. Writing empty daily JSON.")
            with open(os.path.join(data_dir, f"{today}.json"), "w") as f:
                f.write("[]")

        if config["sources"]["arxiv"]["enabled"] and report.source_counts.get("arXiv", 0) == 0:
            report.errors.append(f"arXiv 返回 0 篇（窗口自 {since.date()}），可能是 API 异常")
        if report.arxiv_complete and not report.is_failed:
            save_run_state(state_path, started, {
                "sources": report.source_counts, "selected": report.selected_count})
    except Exception as e:  # noqa: BLE001 - surface in the bot message instead of dying silently
        log.exception("Pipeline crashed")
        report.errors.append(f"流程异常中断: {e}")
        if db is not None and candidates:
            db.requeue_scored([p.id for p in candidates])
            log.info("Re-queued this run's scored papers for the next run")
    finally:
        if db is not None:
            db.close()

    if publish:
        stage_notify(config, report, summary_text, report.selected_count, today)
    log.info("Pipeline complete: %s", " / ".join(report.status_lines()))
    return report


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def cmd_fetch(config: dict):
    data_dir, db_path, _ = _resolve_paths(config)
    today = date.today().isoformat()
    report = RunReport(date=today)
    state_path = os.path.join(os.path.dirname(db_path), "run_state.json")
    since, _ = _arxiv_cutoff(config, state_path, datetime.now(timezone.utc))
    papers = stage_fetch(config, since.date().isoformat(), report)

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

    db = PaperDB(db_path)
    try:
        scored = stage_filter(papers, config, db, RunReport(date=today), today)
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

    report = RunReport(date=today)
    db = PaperDB(db_path)
    try:
        summary_text = stage_output(config, db, papers, data_dir, site_dir, today, report)
    finally:
        db.close()
    stage_notify(config, report, summary_text, len(papers), today)
    log.info("Output generation complete.")


def cmd_cvpr(config: dict, limit: int | None = None):
    from cvpr_pipeline import run_cvpr
    run_cvpr(config, limit=limit)


def cmd_conference(config: dict, conf_key: str, limit: int | None = None):
    from conference_pipeline import run_conference
    run_conference(config, conf_key, limit=limit)


def cmd_bot_test(config: dict, target_date: str | None = None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    today = target_date or date.today().isoformat()
    data_dir = os.path.join(base_dir, "data", "daily")
    json_path = os.path.join(data_dir, f"{today}.json")

    if not os.path.exists(json_path):
        log.error("No data found at %s. Run the pipeline first.", json_path)
        return

    with open(json_path) as f:
        papers = papers_from_json(f.read())

    bot_cfg = config.get("output", {}).get("feishu_bot", {})
    test_env = bot_cfg.get("test_webhook_url_env", "")
    webhook_url = os.environ.get(test_env, "")
    if not webhook_url:
        log.error("Test webhook not configured (env: %s)", test_env)
        return

    summary_text = f"[测试] 今日共追踪到 {len(papers)} 篇相关论文。"
    pages_base = bot_cfg.get("pages_base_url", "")
    pages_url = f"{pages_base}/{today}.html" if pages_base else ""

    ok = send_daily_bot_message(
        webhook_url=webhook_url,
        summary_text=summary_text,
        date=today,
        paper_count=len(papers),
        pages_url=pages_url,
    )
    log.info("Test bot message %s", "sent successfully" if ok else "failed")


def main():
    load_dotenv(override=True)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")

    parser = argparse.ArgumentParser(description="Paper Tracker Pipeline")
    parser.add_argument("--config", default=config_path, help="Path to config.yaml")
    parser.add_argument("--db", default=None, help="SQLite path (default data/papers.db)")
    parser.add_argument("--data-dir", default=None, help="Daily JSON dir (default data/daily)")
    parser.add_argument("--site-dir", default=None, help="HTML output dir (default from config)")
    parser.add_argument("--state-file", default=None, help="Run state JSON (default next to DB)")
    parser.add_argument("--no-publish", action="store_true",
                        help="Skip deploy, Feishu doc and bot message (for trial runs)")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("fetch", help="Fetch papers and dedup into DB")
    subparsers.add_parser("filter", help="Run keyword + LLM filtering on today's raw data")
    subparsers.add_parser("output", help="Generate HTML and Feishu output from today's data")
    subparsers.add_parser("run-all", help="Run full pipeline (default)")
    cvpr_parser = subparsers.add_parser("cvpr", help="One-shot CVPR conference sweep (independent of daily flow)")
    cvpr_parser.add_argument("--limit", type=int, default=None, help="Only process first N papers (smoke test)")
    for _conf in ("aaai", "eccv", "icml", "iclr", "acl"):
        _p = subparsers.add_parser(_conf, help=f"One-shot {_conf.upper()} conference sweep (independent of daily flow)")
        _p.add_argument("--limit", type=int, default=None, help="Only process first N papers (smoke test)")
    bot_test_parser = subparsers.add_parser("bot-test", help="Send bot notification to test webhook")
    bot_test_parser.add_argument("--date", default=None, help="Date to send (default: today)")

    args = parser.parse_args()
    config = load_config(args.config)
    command = args.command or "run-all"

    if command == "fetch":
        cmd_fetch(config)
    elif command == "filter":
        cmd_filter(config)
    elif command == "output":
        cmd_output(config)
    elif command == "cvpr":
        cmd_cvpr(config, getattr(args, "limit", None))
    elif command in ("aaai", "eccv", "icml", "iclr", "acl"):
        cmd_conference(config, command, getattr(args, "limit", None))
    elif command == "bot-test":
        cmd_bot_test(config, getattr(args, "date", None))
    else:
        report = run_pipeline(
            config, data_dir=args.data_dir, db_path=args.db, site_dir=args.site_dir,
            state_path=args.state_file, publish=not args.no_publish,
        )
        sys.exit(1 if report.is_failed else 0)


if __name__ == "__main__":
    main()
