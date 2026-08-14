"""Generic one-shot conference sweep, reusing the daily pipeline's filter/output.

Unlike ``main.py``'s daily flow, this does NOT touch the persistent SQLite DB
or the date-keyed daily files/site index. Each conference writes to its own
``<slug>`` namespace (e.g. ``cvpr2026``, ``aaai2026``) so daily tracking is
never polluted.

Add a conference by (1) writing a fetcher that returns ``list[Paper]``,
(2) registering it in ``FETCHERS``, (3) adding a ``conferences.<key>`` block to
config.yaml.
"""
from __future__ import annotations
import json
import logging
import os
from typing import Callable

from models import Paper, papers_to_json
from fetcher.cvpr_fetcher import fetch_cvpr
from fetcher.aaai_fetcher import fetch_aaai
from fetcher.eccv_fetcher import fetch_eccv
from fetcher.papercopilot_full_fetcher import fetch_papercopilot_full
from fetcher.acl_fetcher import fetch_acl
from fetcher.awards_fetcher import fetch_awards
from filter.keyword_filter import keyword_filter
from filter.llm_filter import llm_filter
from output.summary import generate_summary
from output.html_output import generate_daily_page, generate_index_page
from output.markdown_output import generate_markdown
from output.feishu_output import generate_feishu_doc

log = logging.getLogger(__name__)

# Registry: config `fetcher` value -> callable(year, timeout, concurrency, limit) -> list[Paper].
# Each fetcher accepts **kwargs and ignores what it doesn't use, so all are called uniformly.
FETCHERS: dict[str, Callable[..., list[Paper]]] = {
    "cvpr": fetch_cvpr,
    "aaai": fetch_aaai,
    "eccv": fetch_eccv,
    "papercopilot_full": fetch_papercopilot_full,
    "acl": fetch_acl,
}


def run_conference(config: dict, conf_key: str, limit: int | None = None) -> None:
    """Run the sweep for one conference identified by ``conf_key``.

    Reads ``config["conferences"][conf_key]`` for: fetcher, year, slug, title.
    """
    conf_cfg = config.get("conferences", {}).get(conf_key)
    if not conf_cfg:
        log.error("No config for conference '%s' (expected conferences.%s)", conf_key, conf_key)
        return

    fetcher_name = conf_cfg.get("fetcher", conf_key)
    fetcher = FETCHERS.get(fetcher_name)
    if fetcher is None:
        log.error("Unknown fetcher '%s' for conference '%s'", fetcher_name, conf_key)
        return

    year = conf_cfg.get("year", 2026)
    slug = conf_cfg.get("slug", f"{conf_key}{year}")
    title = conf_cfg.get("title", f"{conf_key.upper()} {year} 精选")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    site_dir = os.path.join(base_dir, config["output"]["html"]["output_dir"])
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(site_dir, exist_ok=True)

    # 1. Fetch (cached; no DB dedup for this one-shot task)
    fetch_kwargs = {
        "year": year,
        "timeout": conf_cfg.get("timeout", 15),
        "concurrency": conf_cfg.get("concurrency", 4),
        "limit": limit,
    }
    # The generic Paper Copilot fetcher also needs the conf slug + accept-status
    # whitelist; the conference-specific fetchers (cvpr/aaai/eccv) don't take them.
    if fetcher_name == "papercopilot_full":
        fetch_kwargs["conf"] = conf_key
        fetch_kwargs["accept_statuses"] = conf_cfg.get("accept_statuses")
    papers = fetcher(**fetch_kwargs)
    if not papers:
        log.warning("No %s papers fetched. Exiting.", slug)
        return

    # 2. Keyword prefilter (free) -> LLM relevance scoring
    scored = _filter(papers, config)
    if not scored:
        log.info("No papers passed filtering. Exiting.")
        return

    paper_dicts = [json.loads(papers_to_json([p]))[0] for p in scored]

    # 3b. Award / candidate papers (independent of interest filter), pinned on top.
    award_dicts = _fetch_awards(conf_key, year, config)
    paper_dicts = award_dicts + paper_dicts

    # 4. Optional domain summary (over interest-filtered papers only)
    summary_html = _summary(paper_dicts, config)

    # 5. Outputs to independent namespace
    _write_json_dicts(paper_dicts, data_dir, slug)
    _write_markdown(paper_dicts, data_dir, slug, title)
    _write_html(paper_dicts, config, site_dir, slug, title, summary_html)
    _write_feishu(paper_dicts, config, title)

    log.info("%s sweep complete: %d relevant papers.", slug, len(scored))


def _fetch_awards(conf_key: str, year: int, config: dict) -> list[dict]:
    """Fetch award/candidate papers (independent of interest filter) as dicts.

    These bypass keyword+interest filtering entirely — award-worthy papers are
    valuable regardless of the user's interest areas. Optionally enriched with a
    Chinese summary via the LLM (all kept; no threshold).
    """
    conf_cfg = config.get("conferences", {}).get(conf_key, {})
    if not conf_cfg.get("awards", True):
        return []

    statuses = conf_cfg.get("award_statuses")
    try:
        awards = fetch_awards(conf_key, year, statuses)
    except Exception as e:  # noqa: BLE001 - never block the main sweep
        log.warning("Award fetch failed for %s%d: %s", conf_key, year, e)
        return []
    if not awards:
        return []

    # Enrich with Chinese summary/why via LLM, keeping ALL (threshold=0).
    # Preserve the award category — llm_filter overwrites primary_category.
    award_categories = {p.id: p.primary_category for p in awards}
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
    try:
        scored = llm_filter(
            papers=awards,
            interests=config["interests"],
            model=config["filter"]["llm_model"],
            threshold=0,  # keep every award paper regardless of relevance
            batch_size=config["filter"]["llm_batch_size"],
            api_key=api_key,
            base_url=base_url,
        )
        # llm_filter drops papers missing from its response; fall back to originals.
        awards = scored if len(scored) == len(awards) else awards
    except Exception as e:  # noqa: BLE001
        log.warning("Award LLM enrichment failed: %s. Using raw abstracts.", e)

    # Restore award categories (LLM reassigned them to interest areas).
    for p in awards:
        if p.id in award_categories:
            p.primary_category = award_categories[p.id]

    log.info("Injecting %d award papers on top for %s%d", len(awards), conf_key, year)
    return [json.loads(papers_to_json([p]))[0] for p in awards]


def _filter(papers: list[Paper], config: dict) -> list[Paper]:
    kw_results = keyword_filter(
        papers, config["interests"], threshold=config["filter"]["keyword_threshold"],
    )
    kw_papers = [paper for paper, _ in kw_results]
    log.info("After keyword filter: %d papers (of %d)", len(kw_papers), len(papers))
    if not kw_papers:
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
    try:
        scored = llm_filter(
            papers=kw_papers,
            interests=config["interests"],
            model=config["filter"]["llm_model"],
            threshold=config["filter"]["llm_relevance_threshold"],
            batch_size=config["filter"]["llm_batch_size"],
            api_key=api_key,
            base_url=base_url,
        )
        log.info("After LLM filter: %d papers (threshold=%s)",
                 len(scored), config["filter"]["llm_relevance_threshold"])
        return scored
    except Exception as e:  # noqa: BLE001 - mirror stage_filter's graceful fallback
        log.warning("LLM filter failed: %s. Falling back to keyword-only.", e)
        return kw_papers


def _summary(paper_dicts: list[dict], config: dict) -> str:
    if not (config.get("summary", {}).get("enabled", False) and paper_dicts):
        return ""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
    try:
        summary_html, _ = generate_summary(
            papers=paper_dicts,
            interests=config["interests"],
            model=config["filter"]["llm_model"],
            api_key=api_key,
            base_url=base_url,
        )
        return summary_html
    except Exception as e:  # noqa: BLE001
        log.warning("Summary generation failed: %s", e)
        return ""


def _write_json_dicts(paper_dicts: list[dict], data_dir: str, slug: str) -> None:
    path = os.path.join(data_dir, f"{slug}.json")
    with open(path, "w") as f:
        json.dump(paper_dicts, f, ensure_ascii=False, indent=2)
    log.info("Saved %d papers to %s", len(paper_dicts), path)


def _write_markdown(paper_dicts: list[dict], data_dir: str, slug: str, title: str) -> None:
    path = os.path.join(data_dir, f"{slug}.md")
    with open(path, "w") as f:
        f.write(generate_markdown(paper_dicts, title))
    log.info("Saved Markdown digest to %s", path)


def _write_html(
    paper_dicts: list[dict], config: dict, site_dir: str,
    slug: str, title: str, summary_html: str,
) -> None:
    if not config["output"]["html"]["enabled"]:
        return
    generate_daily_page(paper_dicts, slug, site_dir, title, summary=summary_html)
    log.info("HTML page written to %s", os.path.join(site_dir, f"{slug}.html"))

    # Rebuild the index so the homepage 专题精选 entry appears/updates even when
    # a conference is run on its own. Daily pages are discovered from disk;
    # non-date names are filtered out by generate_index_page.
    existing_dates = [
        f[:-5] for f in os.listdir(site_dir)
        if f.endswith(".html") and f != "index.html"
    ]
    generate_index_page(
        existing_dates, site_dir, config["output"]["html"]["title"],
        features=config["output"]["html"].get("features", []),
    )
    log.info("Index rebuilt with %d feature page(s)",
             len(config["output"]["html"].get("features", [])))


def _write_feishu(paper_dicts: list[dict], config: dict, title: str) -> None:
    feishu_cfg = config.get("output", {}).get("feishu", {})
    if not feishu_cfg.get("enabled", False):
        return
    generate_feishu_doc(
        papers=paper_dicts,
        date=title,
        wiki_space=feishu_cfg.get("wiki_space", ""),
    )
