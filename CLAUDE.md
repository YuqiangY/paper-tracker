# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Daily academic paper tracking pipeline that fetches papers from multiple sources, deduplicates via SQLite, filters by keywords + LLM relevance scoring, enriches with author metadata, and outputs to GitHub Pages static site + Feishu (Lark) document/bot notification.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Run individual stages
python main.py fetch      # fetch + dedup only
python main.py filter     # keyword + LLM filter on today's raw data
python main.py output     # regenerate HTML/Feishu from today's data

# Trial run against copies, no deploy/Feishu/bot
python main.py --db /tmp/t/papers.db --data-dir /tmp/t/daily --site-dir /tmp/t/site --state-file /tmp/t/state.json --no-publish

# One-shot CVPR conference sweep (independent of the daily flow; does NOT touch
# the daily DB, daily JSON, or site index). Outputs to cvpr{year} namespace.
python main.py cvpr              # full sweep of CVPR accepted papers
python main.py cvpr --limit 30   # smoke test on first 30 papers

# Run tests
pytest tests/
pytest tests/test_llm_filter.py -v   # single test file

# Deploy site manually (normally auto-triggered by main.py)
bash deploy.sh
```

## Architecture

Pipeline orchestrated by `main.py:run_pipeline()` (returns a `RunReport`; exit code 1 when it has errors):

1. **Fetch** — `fetcher/arxiv_fetcher.py` paginates each category back to a cutoff = last successful run (`data/run_state.json`, `storage/run_state.py`) minus 4 days (submit→announce lag), clamped to 14 days; empty pages are retried and a category that never completes blocks the state from advancing. HF Daily, S2 search and RSS are supplementary.
2. **Dedup** — `storage/db.py` inserts new papers with `filter_status='pending'`; `pending`/`score_failed` papers from the last 14 days are re-queued.
3. **Keyword filter** — `filter/keyword_filter.py`; rejected papers get `keyword_rejected`.
4. **Full-text context** — `fetcher/fulltext.py` fetches arXiv HTML once: affiliations, conclusion (≤300 words), code link (bibliography/page-chrome links ignored).
5. **Stage A scoring** — `filter/llm_scoring.py`: relevance 0-40 + novelty 0-30 + quality 0-30 = 0-100, batches of 50 grouped by keyword area, missing ids retried once. All scored papers store their sub-scores.
6. **Selection** — `filter/selection.py`: category normalized to an interest name or `other` (never selected), total ≥ `llm_relevance_threshold` and relevance ≥ `min_relevance`, at most `max_per_area` (50) per area; selected papers get `selected_on=<today>`.
7. **Stage B summaries** — `filter/llm_summary.py`: `summary_zh`/`why_relevant`/tags only for selected papers.
8. **Author enrichment** — `fetcher/author_enrichment.py` (reuses the HTML already parsed in step 4).
9. **Output** — daily JSON, HTML, deploy, Feishu doc, then the bot message, which is sent on every run (including 0 papers) with source counts, stage funnel, output status and warnings; red header on errors. `--no-publish` skips deploy/Feishu/bot for trial runs.

`filter/llm_filter.py` is a facade (Stage A → threshold → Stage B) kept for `conference_pipeline.py`.

## Key Configuration

- `config.yaml` — research interests (5 areas), source settings, filter thresholds, output toggles
- `.env` — API keys: `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` (custom proxy), `SEMANTIC_SCHOLAR_API_KEY`, `FEISHU_BOT_WEBHOOK_URL`
- LLM calls use `anthropic` SDK with a custom base URL (`model.mify.ai.srv/anthropic`)

## Data Flow Notes

- If LLM scoring fails, papers stay `pending`/`score_failed` and are retried next run; the bot message reports the failure
- Scores are 0-100; legacy 1-10 scores were multiplied by 10 once (`PRAGMA user_version`)
- `Paper` dataclass in `models.py` is the core data structure throughout the pipeline (includes venue, citation_count, tldr, doi fields)
- Database tracks `pushed_feishu` and `pushed_html` flags to avoid duplicate outputs
- Each fetcher is independently wrapped in try/except — one source failure doesn't abort others
- `fetcher/cache.py` provides a `DiskCache` class for TTL-based JSON file caching
- Generated site lives in `site/`; GitHub Actions deploys it on push to `main` when `site/**` changes

## Deployment

`deploy.sh` commits `site/` changes and pushes to `main`. GitHub Actions (`.github/workflows/deploy-pages.yml`) then deploys to GitHub Pages. For cron scheduling, `run.sh` wraps the pipeline with proper PATH/env setup.
