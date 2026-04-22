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

# Run tests
pytest tests/
pytest tests/test_llm_filter.py -v   # single test file

# Deploy site manually (normally auto-triggered by main.py)
bash deploy.sh
```

## Architecture

Seven-stage pipeline orchestrated by `main.py:run_pipeline()`:

1. **Fetch** — `fetcher/` modules pull papers from arXiv API, HuggingFace Daily Papers, Semantic Scholar search, and RSS feeds
2. **Dedup** — `storage/db.py` (PaperDB, SQLite) checks paper IDs against existing records
3. **Keyword filter** — `filter/keyword_filter.py` regex-matches titles+abstracts against interest keywords
4. **LLM filter** — `filter/llm_filter.py` batches papers to Claude API for 1-10 relevance scoring (threshold=7), with async parallel batches (concurrency=5), scoring rubric, and parse-failure retry
5. **Author enrichment** — `fetcher/author_enrichment.py` fetches affiliations from arXiv HTML, h-indices, venue, citation counts, and TLDR from Semantic Scholar (only on filtered papers)
6. **Save** — writes daily JSON to `data/daily/YYYY-MM-DD.json`
7. **Output** — generates HTML (`output/html_output.py` + Jinja2 templates), Feishu doc (`output/feishu_output.py` via `feishu` CLI), domain summary (`output/summary.py`), and bot notification (`output/feishu_bot.py`)

## Key Configuration

- `config.yaml` — research interests (5 areas), source settings, filter thresholds, output toggles
- `.env` — API keys: `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` (custom proxy), `SEMANTIC_SCHOLAR_API_KEY`, `FEISHU_BOT_WEBHOOK_URL`
- LLM calls use `anthropic` SDK with a custom base URL (`model.mify.ai.srv/anthropic`)

## Data Flow Notes

- If LLM filter fails, pipeline falls back to keyword-only results (graceful degradation)
- `Paper` dataclass in `models.py` is the core data structure throughout the pipeline (includes venue, citation_count, tldr, doi fields)
- Database tracks `pushed_feishu` and `pushed_html` flags to avoid duplicate outputs
- Each fetcher is independently wrapped in try/except — one source failure doesn't abort others
- `fetcher/cache.py` provides a `DiskCache` class for TTL-based JSON file caching
- Generated site lives in `site/`; GitHub Actions deploys it on push to `main` when `site/**` changes

## Deployment

`deploy.sh` commits `site/` changes and pushes to `main`. GitHub Actions (`.github/workflows/deploy-pages.yml`) then deploys to GitHub Pages. For cron scheduling, `run.sh` wraps the pipeline with proper PATH/env setup.
